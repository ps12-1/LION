#!/usr/bin/env python3
"""
Swin Transformer Noise2Inverse experiment with custom geometries.

  - Sparse angle:  100 projection angles over 0–360°
  - Limited angle: 60 projection angles over 0–150°
  - Full angle:    360 projection angles over 0–360°  (unchanged, kept for comparison)

Because the model is trained on sinogram-based artifacts tied to a specific geometry,
this script retrains the Swin Transformer from scratch for each new geometry.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Subset

import LION.CTtools.ct_geometry as ctgeo
import LION.experiments.ct_experiments as ct_experiments
from LION.classical_algorithms.fdk import fdk
from LION.models.LIONmodel import LIONModelParameter, LIONmodel, ModelInputType
from LION.optimizers.Noise2InverseSolver import Noise2InverseSolver
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# ── Custom geometry builders ──────────────────────────────────────────────────


def sparse_100_geometry():
    """100 views uniformly over 0–360°."""
    return ctgeo.Geometry(
        image_shape=[1, 512, 512],
        image_size=[300 / 512, 300, 300],
        detector_shape=[1, 900],
        detector_size=[1, 900],
        dso=575,
        dsd=1050,
        mode="fan",
        angles=np.linspace(0, 2 * np.pi, 100, endpoint=False),
    )


def limited_150_geometry():
    """60 views over 0–150° (limited angular range)."""
    return ctgeo.Geometry(
        image_shape=[1, 512, 512],
        image_size=[300 / 512, 300, 300],
        detector_shape=[1, 900],
        detector_size=[1, 900],
        dso=575,
        dsd=1050,
        mode="fan",
        angles=np.linspace(0, np.deg2rad(150), 60, endpoint=False),
    )


def full_360_geometry():
    """360 views over 0–360° (standard full angle)."""
    return None  # Use experiment's default geometry (like run_swin_noise2inverse.py)


# ── Geometry-overriding experiment wrappers ───────────────────────────────────


class CustomGeomExperiment:
    """Wraps a LION Experiment but replaces its geometry."""

    def __init__(self, base_experiment, custom_geometry):
        self._base = base_experiment
        self.geometry = custom_geometry
        # Patch the experiment's internal geometry so data-loaders receive it
        self._base.geometry = custom_geometry
        self._base.experiment_params.geometry = custom_geometry

    def get_training_dataset(self):
        return self._base.get_training_dataset()

    def get_testing_dataset(self):
        return self._base.get_testing_dataset()


# ── SwinIR-style CT Denoiser (Liang et al. 2021, adapted for N2I) ────────────
#
# Architecture (mirrors SwinIR for image denoising):
#   1. Shallow feature extraction  – 3×3 Conv
#   2. Deep feature extraction     – num_rstb × RSTB  +  LayerNorm  +  3×3 Conv
#      Each RSTB = stl_per_rstb × SwinTransformerLayer (W-MSA/SW-MSA)
#                + 3×3 Conv  +  residual connection
#   3. Reconstruction              – (f_shallow + f_deep) → 3×3 Conv
#      Residual learning: output = input + recon(f_shallow + f_deep)
#
# Key SwinIR improvements over a plain Swin stack:
#   • Learnable relative position bias (RPB) in every attention layer
#   • RSTB residual stabilises gradient flow across many STL layers
#   • Long skip from shallow conv bypasses all RSTBs (low-freq preservation)


class WindowAttentionRPB(nn.Module):
    """Window multi-head self-attention with learnable relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # Learnable RPB table  shape: (2W-1)^2 × nH
        self.rpb_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.rpb_table, std=0.02)

        # Pre-computed relative position index  shape: W^2 × W^2
        hs = torch.arange(window_size)
        ws = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(hs, ws, indexing="ij"))  # 2,W,W
        flat = torch.flatten(coords, 1)  # 2, W^2
        rel = flat[:, :, None] - flat[:, None, :]  # 2, W^2, W^2
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("rpb_index", rel.sum(-1))  # W^2, W^2

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        # Add relative position bias
        rpb = self.rpb_table[self.rpb_index.view(-1)]
        rpb = rpb.view(N, N, self.num_heads).permute(2, 0, 1).contiguous()
        attn = attn + rpb.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = torch.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinTransformerLayer(nn.Module):
    """Swin Transformer Layer (STL): W-MSA or SW-MSA + MLP, both with residuals."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttentionRPB(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    # ── static window helpers ─────────────────────────────────────────────────

    @staticmethod
    def _partition(x: torch.Tensor, ws: int) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)

    @staticmethod
    def _reverse(wins: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
        B = int(wins.shape[0] * ws * ws / (H * W))
        x = wins.view(B, H // ws, W // ws, ws, ws, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    def _mask(self, H: int, W: int, device) -> torch.Tensor:
        if self.shift_size == 0:
            return None
        img = torch.zeros(1, H, W, 1, device=device)
        cnt = 0
        for hs in (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        ):
            for ws in (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            ):
                img[:, hs, ws, :] = cnt
                cnt += 1
        m = self._partition(img, self.window_size).squeeze(-1)
        am = m.unsqueeze(1) - m.unsqueeze(2)
        return am.masked_fill(am != 0, -100.0).masked_fill(am == 0, 0.0)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """x: [B, H*W, C]"""
        B, _, C = x.shape
        sc = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        wins = self._partition(x, self.window_size)
        wins = self.attn(wins, self._mask(H, W, wins.device))
        x = self._reverse(wins, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = sc + x.view(B, H * W, C)
        return x + self.mlp(self.norm2(x))


class ResidualSwinTransformerBlock(nn.Module):
    """RSTB: stl_per_rstb STLs → 3×3 Conv → residual (SwinIR Fig. 2a)."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList(
            [
                SwinTransformerLayer(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                )
                for i in range(depth)
            ]
        )
        # Conv after STLs enhances translational equivariance (SwinIR )
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W]"""
        shortcut = x
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                seq = checkpoint(layer, seq, H, W, use_reentrant=False)
            else:
                seq = layer(seq, H, W)
        feat = seq.transpose(1, 2).view(B, C, H, W)
        return self.conv(feat) + shortcut  # RSTB residual


class SwinIRCTDenoiser(LIONmodel):
    """
    SwinIR-style denoiser for CT reconstruction, trained via Noise2Inverse.

    Compared with a plain Swin U-Net this model adds:
      • Relative position bias  – content-adaptive long-range attention
      • RSTB residuals          – gradient highway across many STL layers
      • Long skip connection    – preserves low-frequency anatomy unchanged
      • Residual output         – network learns the noise residual, not the image
    """

    def __init__(self, geometry, model_parameters=None):
        if model_parameters is None:
            model_parameters = SwinIRCTDenoiser.default_parameters()
        super().__init__(model_parameters, geometry)
        p = self.model_parameters
        self.window_size = p.window_size

        # 1. Shallow feature extraction (single conv — stable early gradients)
        self.shallow = nn.Conv2d(1, p.embed_dim, 3, 1, 1)

        # 2. Deep feature extraction: num_rstb RSTBs + norm + conv
        self.rstbs = nn.ModuleList(
            [
                ResidualSwinTransformerBlock(
                    dim=p.embed_dim,
                    depth=p.stl_per_rstb,
                    num_heads=p.num_heads,
                    window_size=p.window_size,
                    use_checkpoint=p.use_checkpoint,
                )
                for _ in range(p.num_rstb)
            ]
        )
        self.norm = nn.LayerNorm(p.embed_dim)
        self.deep_conv = nn.Conv2d(p.embed_dim, p.embed_dim, 3, 1, 1)

        # 3. Reconstruction: fused shallow+deep → 1 channel
        self.recon = nn.Conv2d(p.embed_dim, 1, 3, 1, 1)

    @staticmethod
    def default_parameters():
        p = LIONModelParameter()
        p.model_input_type = ModelInputType.IMAGE
        # SwinIR lightweight settings (similar param count to N2I U-Net ~2-3 M)
        p.embed_dim = 96  # feature channels
        p.window_size = 8  # SwinIR uses 8 for denoising tasks
        p.num_rstb = 4  # number of Residual Swin Transformer Blocks
        p.stl_per_rstb = 4  # Swin Transformer Layers per RSTB
        p.num_heads = 6
        p.use_checkpoint = True
        return p

    def _pad(self, x: torch.Tensor):
        m = self.window_size
        _, _, h, w = x.shape
        ph = (m - h % m) % m
        pw = (m - w % m) % m
        if ph == 0 and pw == 0:
            return x, (0, 0)
        return F.pad(x, (0, pw, 0, ph), mode="reflect"), (ph, pw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != 1:
            x = x[:, :1]

        orig_h, orig_w = x.shape[2], x.shape[3]
        x, (ph, pw) = self._pad(x)
        _, _, H, W = x.shape

        # 1. Shallow features (long skip — low-freq preservation)
        f_shallow = self.shallow(x)

        # 2. Deep features through RSTBs
        f = f_shallow
        for rstb in self.rstbs:
            f = rstb(f)

        # Final LayerNorm in sequence domain then project back to spatial
        B, C, _, _ = f.shape
        f = self.norm(f.flatten(2).transpose(1, 2)).transpose(1, 2).view(B, C, H, W)
        f = self.deep_conv(f)

        # 3. Residual reconstruction: input + recon(shallow + deep)
        out = x + self.recon(f_shallow + f)

        if ph > 0 or pw > 0:
            out = out[:, :, :orig_h, :orig_w]
        return out


# ── Helpers ─────────────────────────────────────────────────────────────────


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Force deterministic behavior across repeated trainings in one process
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def to_np(t):
    if isinstance(t, np.ndarray):
        return t
    return t.detach().cpu().numpy()


def compute_metrics(target, pred):
    t = to_np(target).squeeze()
    p = to_np(pred).squeeze()
    mn, mx = t.min(), t.max()
    if mx > mn:
        t = (t - mn) / (mx - mn)
        p = (p - mn) / (mx - mn)
    return {
        "ssim": float(ssim(t, p, data_range=1.0)),
        "psnr": float(psnr(t, p, data_range=1.0)),
    }


def validate_model(solver, val_loader, device):
    """Validate model on validation set. Returns mean PSNR."""
    solver.model.eval()
    psnr_vals = []
    with torch.no_grad():
        for sino, tgt in val_loader:
            sino, tgt = sino.to(device), tgt.to(device)
            recon = solver.reconstruct(sino)
            metrics = compute_metrics(tgt, recon)
            psnr_vals.append(metrics["psnr"])
    return float(np.mean(psnr_vals)) if psnr_vals else 0.0


def subset_if_needed(dataset, n):
    if n and len(dataset) > n:
        # Deterministic subset (not random.sample)
        return Subset(dataset, list(range(n)))
    return dataset


def prepare_loader(experiment, batch_size, train_n, seed):
    ds = subset_if_needed(experiment.get_training_dataset(), train_n)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def prepare_val_loader(experiment, batch_size, val_n=None):
    """Prepare validation set loader (deterministic, no shuffle)."""
    ds = subset_if_needed(experiment.get_testing_dataset(), val_n)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def train(
    experiment, train_loader, val_loader, device, output_dir, tag, epochs, sino_splits
):
    """Train with Enhanced Model Selection (based on validation PSNR)."""
    model = SwinIRCTDenoiser(experiment.geometry)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    params = Noise2InverseSolver.default_parameters()
    params.sino_split_count = sino_splits
    params.recon_fn = fdk
    params.cali_J = Noise2InverseSolver.X_one_strategy(params.sino_split_count)

    solver = Noise2InverseSolver(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        solver_params=params,
        geometry=experiment.geometry,
        verbose=True,
        device=device,
    )
    solver.set_training(train_loader)
    solver.set_checkpointing(
        checkpoint_fname=f"{tag}_swinir_n2i_check_*.pt",
        checkpoint_freq=1,  # Save every epoch for model selection
        load_checkpoint_if_exists=False,
        save_folder=output_dir,
    )

    # We drive epochs manually via `epoch_step`, so make sure internal history
    # buffers are allocated to the requested epoch count.
    solver.solver_params.epochs = epochs
    if (
        not hasattr(solver, "train_loss")
        or solver.train_loss is None
        or len(solver.train_loss) < epochs
    ):
        solver.train_loss = np.full((epochs,), np.nan, dtype=np.float32)

    if len(train_loader) == 0:
        raise RuntimeError(
            "Training loader is empty. Increase --train-samples or dataset size."
        )

    # Enhanced Model Selection: Track validation metrics
    best_psnr = -np.inf
    best_epoch = 0
    val_psnr_history = []

    print(f"  Training with model selection (validation every epoch)...")
    for epoch in range(epochs):
        solver.current_epoch = epoch
        solver.epoch_step(epoch)

        # Validate at each epoch
        if val_loader is not None:
            val_psnr = validate_model(solver, val_loader, device)
            val_psnr_history.append(val_psnr)
            print(f"    Epoch {epoch+1}/{epochs}: Val PSNR = {val_psnr:.2f} dB", end="")

            # Update best model
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_epoch = epoch
                # Save best checkpoint
                best_ckpt_path = output_dir / f"{tag}_best_model.pt"
                torch.save(
                    {
                        "model_state": solver.model.state_dict(),
                        "epoch": epoch,
                        "val_psnr": val_psnr,
                    },
                    best_ckpt_path,
                )
                print(" (BEST)", flush=True)
            else:
                print(flush=True)

    # Load best model for inference
    best_ckpt_path = output_dir / f"{tag}_best_model.pt"
    if best_ckpt_path.exists():
        print(
            f"  Loading best model from epoch {best_epoch+1} (Val PSNR={best_psnr:.2f} dB)"
        )
        ckpt = torch.load(best_ckpt_path, map_location=device)
        solver.model.load_state_dict(ckpt["model_state"])

    return solver, val_psnr_history, best_epoch, best_psnr


def get_test_sample(experiment, idx):
    ds = experiment.get_testing_dataset()
    idx = max(0, min(idx, len(ds) - 1))
    sino, tgt = ds[idx]
    if sino.dim() == 3:
        sino = sino.unsqueeze(0)
    if tgt.dim() == 3:
        tgt = tgt.unsqueeze(0)
    return sino, tgt


# ── Main ──────────────────────────────────────────────────────────────────────


def run(args):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Custom geometry settings ───────────────────────────────────────────
    #  name          | views | angular range
    #  Sparse-100    |  100  | 0–360°
    #  Limited-150   |   60  | 0–150°
    #  Full-360      |  360  | 0–360°
    settings = [
        (
            "Sparse-100 (100 views, 0-360°)",
            sparse_100_geometry,
            ct_experiments.SparseAngleLowDoseCTRecon,
        ),
        (
            "Limited-150 (60 views, 0-150°)",
            limited_150_geometry,
            ct_experiments.LimitedAngleLowDoseCTRecon,
        ),
        (
            "Full-360 (360 views, 0-360°)",
            full_360_geometry,
            ct_experiments.LowDoseCTRecon,
        ),
    ]

    results = []

    for setting_name, geo_fn, exp_ctor in settings:
        print(f"\n===== {setting_name} =====")

        # Reset RNG before each geometry so training is comparable
        set_seed(args.seed)

        # Build base experiment, then override geometry if custom
        base_exp = exp_ctor(dataset=args.dataset, datafolder=args.datafolder)
        if geo_fn() is not None:
            # Custom geometry: override
            experiment = CustomGeomExperiment(base_exp, geo_fn())
        else:
            # Full angle: use experiment's default geometry (no override)
            experiment = base_exp

        angles_deg = np.degrees(experiment.geometry.angles)
        print(
            f"  Angles: {len(angles_deg)} views, {angles_deg[0]:.1f}° – {angles_deg[-1]:.1f}°"
        )

        train_loader = prepare_loader(
            experiment, args.batch_size, args.train_samples, args.seed
        )
        val_loader = prepare_val_loader(experiment, args.batch_size, args.val_samples)
        tag = setting_name.split(" ")[0].lower().replace("-", "_")

        solver, val_history, best_epoch, best_psnr = train(
            experiment,
            train_loader,
            val_loader,
            device,
            output_dir,
            tag,
            args.epochs,
            args.sino_splits,
        )

        del train_loader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        sino, tgt = get_test_sample(experiment, args.test_index)
        sino, tgt = sino.to(device), tgt.to(device)

        with torch.no_grad():
            recon = solver.reconstruct(sino)

        metrics = compute_metrics(tgt, recon)
        results.append(
            {
                "setting": setting_name,
                "target": tgt.cpu(),
                "reconstruction": recon.cpu(),
                "metrics": metrics,
            }
        )

        del solver, sino, tgt, recon, experiment
        gc.collect()
        torch.cuda.empty_cache()

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(8, 14))

    all_vals = np.concatenate([to_np(r["reconstruction"]).ravel() for r in results])
    vmin, vmax = 0.0, float(np.max(all_vals))

    for r, row in enumerate(results):
        img = to_np(row["reconstruction"]).squeeze()
        m = row["metrics"]
        axes[r].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        axes[r].set_title(
            f"{row['setting']}\nSSIM: {m['ssim']:.4f}, PSNR: {m['psnr']:.2f} dB",
            fontsize=11,
            fontweight="bold",
        )
        axes[r].axis("off")

    plt.tight_layout()
    fig_path = output_dir / "swin_custom_geometry_3x1.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pt_path = output_dir / "swin_custom_geometry_3x1.pt"
    torch.save(results, pt_path)

    print("\n" + "=" * 70)
    print("SWINIR + N2I — CUSTOM GEOMETRY RESULTS")
    print("=" * 70)
    for row in results:
        img = to_np(row["reconstruction"]).squeeze()
        print(f"\n{row['setting']}:")
        print(f"  SSIM : {row['metrics']['ssim']:.6f}")
        print(f"  PSNR : {row['metrics']['psnr']:.2f} dB")
        print(f"  Image min={img.min():.6f}, max={img.max():.6f}")
    print("=" * 70)
    print(f"\nFigure : {fig_path.resolve()}")
    print(f"Tensors: {pt_path.resolve()}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Swin Scratch + N2I with custom geometries"
    )
    parser.add_argument("--dataset", type=str, default="LIDC-IDRI")
    parser.add_argument("--datafolder", type=str, default=None)
    parser.add_argument(
        "--output-dir", type=str, default=str(script_dir / "results_custom_geometry")
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=10)
    parser.add_argument("--test-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--sino-splits", type=int, default=5)
    run(parser.parse_args())
