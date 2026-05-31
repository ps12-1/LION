#!/usr/bin/env python3
"""
Run Noise2Inverse experiments for three angle settings across:
  - Swin Transformer (SwinIR-style)
  - UNet (LION implementation)
  - State Space Model (SSM) denoiser (from scratch)

Outputs:
  - Per-model checkpoints and tensors
  - Summary metrics (csv + json)
  - 3x3 grid of denoised reconstructions (angles x models)
  - FDK reconstructions for each angle (1x3 grid)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Subset

import LION.CTtools.ct_geometry as ctgeo
import LION.experiments.ct_experiments as ct_experiments
from LION.classical_algorithms.fdk import fdk
from LION.models.CNNs.UNets.Unet import UNet
from LION.models.LIONmodel import LIONModelParameter, LIONmodel, ModelInputType
from LION.optimizers.Noise2InverseSolver import Noise2InverseSolver
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# ---- Custom geometry builders -------------------------------------------------


def sparse_100_geometry():
    """100 views uniformly over 0-360 degrees."""
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
    """60 views over 0-150 degrees (limited angular range)."""
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
    """360 views over 0-360 degrees (standard full angle)."""
    return None  # Use experiment's default geometry


class CustomGeomExperiment:
    """Wrap a LION Experiment but replace its geometry."""

    def __init__(self, base_experiment, custom_geometry):
        self._base = base_experiment
        self.geometry = custom_geometry
        self._base.geometry = custom_geometry
        self._base.experiment_params.geometry = custom_geometry

    def get_training_dataset(self):
        return self._base.get_training_dataset()

    def get_testing_dataset(self):
        return self._base.get_testing_dataset()


# ---- SwinIR-style denoiser (from run_swin_custom_geometry.py) ----------------


class WindowAttentionRPB(nn.Module):
    """Window multi-head self-attention with learnable relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.rpb_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.rpb_table, std=0.02)

        hs = torch.arange(window_size)
        ws = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(hs, ws, indexing="ij"))
        flat = torch.flatten(coords, 1)
        rel = flat[:, :, None] - flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("rpb_index", rel.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        rpb = self.rpb_table[self.rpb_index.view(-1)]
        rpb = rpb.view(n, n, self.num_heads).permute(2, 0, 1).contiguous()
        attn = attn + rpb.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(b // nW, nW, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = torch.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class SwinTransformerLayer(nn.Module):
    """Swin Transformer layer (W-MSA / SW-MSA + MLP)."""

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

    @staticmethod
    def _partition(x: torch.Tensor, ws: int) -> torch.Tensor:
        b, h, w, c = x.shape
        x = x.view(b, h // ws, ws, w // ws, ws, c)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, c)

    @staticmethod
    def _reverse(wins: torch.Tensor, ws: int, h: int, w: int) -> torch.Tensor:
        b = int(wins.shape[0] * ws * ws / (h * w))
        x = wins.view(b, h // ws, w // ws, ws, ws, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)

    def _mask(self, h: int, w: int, device) -> torch.Tensor:
        if self.shift_size == 0:
            return None
        img = torch.zeros(1, h, w, 1, device=device)
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

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, _, c = x.shape
        sc = x
        x = self.norm1(x).view(b, h, w, c)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        wins = self._partition(x, self.window_size)
        wins = self.attn(wins, self._mask(h, w, wins.device))
        x = self._reverse(wins, self.window_size, h, w)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = sc + x.view(b, h * w, c)
        return x + self.mlp(self.norm2(x))


class ResidualSwinTransformerBlock(nn.Module):
    """RSTB: stack of Swin layers with 3x3 conv and residual."""

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
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                seq = checkpoint(layer, seq, h, w, use_reentrant=False)
            else:
                seq = layer(seq, h, w)
        feat = seq.transpose(1, 2).view(b, c, h, w)
        return self.conv(feat) + x


class SwinIRCTDenoiser(LIONmodel):
    """SwinIR-style denoiser trained via Noise2Inverse."""

    def __init__(self, geometry, model_parameters=None):
        if model_parameters is None:
            model_parameters = SwinIRCTDenoiser.default_parameters()
        super().__init__(model_parameters, geometry)
        p = self.model_parameters
        self.window_size = p.window_size

        self.shallow = nn.Conv2d(1, p.embed_dim, 3, 1, 1)
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
        self.recon = nn.Conv2d(p.embed_dim, 1, 3, 1, 1)

    @staticmethod
    def default_parameters():
        p = LIONModelParameter()
        p.model_input_type = ModelInputType.IMAGE
        p.embed_dim = 96
        p.window_size = 8
        p.num_rstb = 4
        p.stl_per_rstb = 4
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
        _, _, h, w = x.shape

        f_shallow = self.shallow(x)
        f = f_shallow
        for rstb in self.rstbs:
            f = rstb(f)

        b, c, _, _ = f.shape
        f = self.norm(f.flatten(2).transpose(1, 2)).transpose(1, 2).view(b, c, h, w)
        f = self.deep_conv(f)

        out = x + self.recon(f_shallow + f)

        if ph > 0 or pw > 0:
            out = out[:, :, :orig_h, :orig_w]
        return out


# ---- State Space Model denoiser (from scratch) -------------------------------


class DiagonalSSM1D(nn.Module):
    """Diagonal state space layer scanning along the last dimension."""

    def __init__(self, channels: int):
        super().__init__()
        self.log_a = nn.Parameter(torch.zeros(channels))
        self.b = nn.Parameter(torch.ones(channels))
        self.c = nn.Parameter(torch.ones(channels))
        self.d = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> scan along W
        bsz, ch, h, w = x.shape
        a = torch.exp(self.log_a)
        alpha = torch.exp(-a).view(1, ch, 1)
        b = self.b.view(1, ch, 1)
        c = self.c.view(1, ch, 1)
        d = self.d.view(1, ch, 1)

        h_state = x.new_zeros((bsz, ch, h))
        ys = []
        for t in range(w):
            xt = x[:, :, :, t]
            h_state = alpha * h_state + b * xt
            yt = c * h_state + d * xt
            ys.append(yt)
        return torch.stack(ys, dim=3)


class SpatialSSMBlock(nn.Module):
    """2D SSM block with width and height scans + MLP."""

    def __init__(self, dim: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.ssm_w = DiagonalSSM1D(dim)
        self.ssm_h = DiagonalSSM1D(dim)
        self.norm2 = nn.GroupNorm(1, dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm1(x)
        x = self.ssm_w(x)
        x = self.ssm_h(x.transpose(2, 3)).transpose(2, 3)
        x = res + x
        return x + self.mlp(self.norm2(x))


class SSMCTDenoiser(LIONmodel):
    """SSM-based denoiser using lightweight diagonal SSM scans."""

    def __init__(self, geometry, model_parameters=None):
        if model_parameters is None:
            model_parameters = SSMCTDenoiser.default_parameters()
        super().__init__(model_parameters, geometry)
        p = self.model_parameters
        self.shallow = nn.Conv2d(1, p.embed_dim, 3, 1, 1)
        self.blocks = nn.ModuleList(
            [SpatialSSMBlock(p.embed_dim, p.mlp_ratio) for _ in range(p.depth)]
        )
        self.norm = nn.GroupNorm(1, p.embed_dim)
        self.recon = nn.Conv2d(p.embed_dim, 1, 3, 1, 1)

    @staticmethod
    def default_parameters():
        p = LIONModelParameter()
        p.model_input_type = ModelInputType.IMAGE
        p.embed_dim = 64
        p.depth = 6
        p.mlp_ratio = 2.0
        return p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != 1:
            x = x[:, :1]
        f = self.shallow(x)
        for block in self.blocks:
            f = block(f)
        f = self.norm(f)
        return x + self.recon(f)


# ---- Utilities ---------------------------------------------------------------


@dataclass
class ModelSpec:
    name: str
    tag: str
    ctor: callable
    optimizer: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def to_np(t: torch.Tensor) -> np.ndarray:
    if isinstance(t, np.ndarray):
        return t
    return t.detach().cpu().numpy()


def compute_metrics(target: torch.Tensor, pred: torch.Tensor) -> dict:
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


def subset_if_needed(dataset, n):
    if n and len(dataset) > n:
        return Subset(dataset, list(range(n)))
    return dataset


def prepare_loader(experiment, batch_size, train_n, seed, shuffle=True):
    ds = subset_if_needed(experiment.get_training_dataset(), train_n)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g if shuffle else None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def prepare_val_loader(experiment, batch_size, val_n=None):
    ds = subset_if_needed(experiment.get_testing_dataset(), val_n)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def validate_model(solver, val_loader, device) -> float:
    solver.model.eval()
    psnr_vals = []
    with torch.no_grad():
        for sino, tgt in val_loader:
            sino, tgt = sino.to(device), tgt.to(device)
            recon = solver.reconstruct(sino)
            metrics = compute_metrics(tgt, recon)
            psnr_vals.append(metrics["psnr"])
    return float(np.mean(psnr_vals)) if psnr_vals else 0.0


def allocate_solver_buffers(solver, epochs: int) -> None:
    solver.solver_params.epochs = epochs
    if (
        not hasattr(solver, "train_loss")
        or solver.train_loss is None
        or len(solver.train_loss) < epochs
    ):
        solver.train_loss = np.full((epochs,), np.nan, dtype=np.float32)


def train_model_n2i(
    model: LIONmodel,
    experiment,
    train_loader,
    val_loader,
    device,
    output_dir: Path,
    tag: str,
    epochs: int,
    sino_splits: int,
    patience: int,
    min_delta: float,
    optimizer_name: str,
):
    if optimizer_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    else:
        optimizer = Adam(model.parameters(), lr=1e-4)
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
        checkpoint_fname=f"{tag}_n2i_check_*.pt",
        checkpoint_freq=1,
        load_checkpoint_if_exists=False,
        save_folder=output_dir,
    )

    allocate_solver_buffers(solver, epochs)
    if len(train_loader) == 0:
        raise RuntimeError(
            "Training loader is empty. Increase --train-samples or dataset size."
        )

    best_psnr = -np.inf
    best_epoch = 0
    wait = 0
    history = []

    print("  Training with early stopping (validation every epoch)...")
    for epoch in range(epochs):
        solver.current_epoch = epoch
        solver.epoch_step(epoch)

        if val_loader is not None:
            val_psnr = validate_model(solver, val_loader, device)
            history.append(val_psnr)
            print(
                f"    Epoch {epoch+1}/{epochs}: Val PSNR = {val_psnr:.2f} dB",
                end="",
            )
            if val_psnr > best_psnr + min_delta:
                best_psnr = val_psnr
                best_epoch = epoch
                wait = 0
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
                wait += 1
                print(flush=True)

            if wait >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    best_ckpt_path = output_dir / f"{tag}_best_model.pt"
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        solver.model.load_state_dict(ckpt["model_state"])
        print(
            f"  Loaded best model from epoch {ckpt['epoch']+1} (Val PSNR={ckpt['val_psnr']:.2f} dB)"
        )

    return solver, history, best_epoch, best_psnr


def get_test_sample(experiment, idx: int):
    ds = experiment.get_testing_dataset()
    idx = max(0, min(idx, len(ds) - 1))
    sino, tgt = ds[idx]
    if sino.dim() == 3:
        sino = sino.unsqueeze(0)
    if tgt.dim() == 3:
        tgt = tgt.unsqueeze(0)
    return sino, tgt


def save_metrics_csv(rows: list, out_path: Path) -> None:
    header = "model,setting,ssim,psnr\n"
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['model']},{r['setting']},{r['metrics']['ssim']:.6f},{r['metrics']['psnr']:.4f}\n"
        )
    out_path.write_text("".join(lines))


def plot_3x3_denoised(results, out_path: Path) -> None:
    # rows: angles, cols: models
    rows = ["Sparse-100", "Limited-150", "Full-360"]
    cols = ["SwinIR", "UNet", "SSM"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    all_vals = np.concatenate([to_np(r["reconstruction"]).ravel() for r in results])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    for r_i, row in enumerate(rows):
        for c_i, col in enumerate(cols):
            match = [
                r for r in results if r["setting_tag"] == row and r["model_tag"] == col
            ][0]
            img = to_np(match["reconstruction"]).squeeze()
            ax = axes[r_i, c_i]
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(
                f"{row} | {col}\nSSIM {match['metrics']['ssim']:.4f}, PSNR {match['metrics']['psnr']:.2f}",
                fontsize=9,
            )
            ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fdk(fdk_results, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    all_vals = np.concatenate([to_np(r["reconstruction"]).ravel() for r in fdk_results])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    for i, row in enumerate(fdk_results):
        img = to_np(row["reconstruction"]).squeeze()
        axes[i].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        axes[i].set_title(
            f"{row['setting']}\nSSIM {row['metrics']['ssim']:.4f}, PSNR {row['metrics']['psnr']:.2f}",
            fontsize=9,
        )
        axes[i].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_4x3_grid(results, fdk_results, out_path: Path) -> None:
    rows = ["FDK", "SwinIR", "UNet", "SSM"]
    cols = ["Sparse-100", "Limited-150", "Full-360"]
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))

    all_imgs = []
    for r in fdk_results:
        all_imgs.append(to_np(r["reconstruction"]).ravel())
    for r in results:
        all_imgs.append(to_np(r["reconstruction"]).ravel())
    all_vals = np.concatenate(all_imgs)
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    for r_i, row in enumerate(rows):
        for c_i, col in enumerate(cols):
            ax = axes[r_i, c_i]
            if row == "FDK":
                match = [r for r in fdk_results if r["setting"] == col][0]
                img = to_np(match["reconstruction"]).squeeze()
                title = (
                    f"{row} | {col}\n"
                    f"SSIM {match['metrics']['ssim']:.4f}, PSNR {match['metrics']['psnr']:.2f}"
                )
            else:
                match = [
                    r
                    for r in results
                    if r["setting_tag"] == col and r["model_tag"] == row
                ][0]
                img = to_np(match["reconstruction"]).squeeze()
                title = (
                    f"{row} | {col}\n"
                    f"SSIM {match['metrics']['ssim']:.4f}, PSNR {match['metrics']['psnr']:.2f}"
                )

            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---- Main --------------------------------------------------------------------


def run(args):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    settings = [
        (
            "Sparse-100 (100 views, 0-360)",
            "Sparse-100",
            sparse_100_geometry,
            ct_experiments.SparseAngleLowDoseCTRecon,
        ),
        (
            "Limited-150 (60 views, 0-150)",
            "Limited-150",
            limited_150_geometry,
            ct_experiments.LimitedAngleLowDoseCTRecon,
        ),
        (
            "Full-360 (360 views, 0-360)",
            "Full-360",
            full_360_geometry,
            ct_experiments.LowDoseCTRecon,
        ),
    ]

    models = [
        ModelSpec("SwinIR", "swinir", lambda g: SwinIRCTDenoiser(g), "adamw"),
        ModelSpec("UNet", "unet", lambda g: UNet(UNet.default_parameters()), "adam"),
        ModelSpec("SSM", "ssm", lambda g: SSMCTDenoiser(g), "adam"),
    ]

    results = []
    fdk_results = []
    metrics_rows = []

    for setting_name, setting_tag, geo_fn, exp_ctor in settings:
        print(f"\n===== {setting_name} =====")
        set_seed(args.seed)

        base_exp = exp_ctor(dataset=args.dataset, datafolder=args.datafolder)
        if geo_fn() is not None:
            experiment = CustomGeomExperiment(base_exp, geo_fn())
        else:
            experiment = base_exp

        angles_deg = np.degrees(experiment.geometry.angles)
        print(
            f"  Angles: {len(angles_deg)} views, {angles_deg[0]:.1f} - {angles_deg[-1]:.1f}"
        )

        train_loader = prepare_loader(
            experiment, args.batch_size, args.train_samples, args.seed, shuffle=True
        )
        val_loader = prepare_val_loader(experiment, args.batch_size, args.val_samples)

        sino, tgt = get_test_sample(experiment, args.test_index)
        sino = sino.to(device)
        tgt = tgt.to(device)

        # FDK baseline
        with torch.no_grad():
            fdk_recon = fdk(sino, experiment.geometry)
        fdk_metrics = compute_metrics(tgt, fdk_recon)
        fdk_results.append(
            {
                "setting": setting_tag,
                "reconstruction": fdk_recon.detach().cpu(),
                "metrics": fdk_metrics,
            }
        )

        for model_spec in models:
            print(f"  -> Training {model_spec.name} + Noise2Inverse")
            model_out_dir = output_root / model_spec.tag / setting_tag
            model_out_dir.mkdir(parents=True, exist_ok=True)

            if model_spec.tag == "unet":
                unet_params = UNet.default_parameters()
                unet_params.baseDim = args.unet_base_dim
                model = UNet(unet_params)
            else:
                model = model_spec.ctor(experiment.geometry)

            solver, val_history, best_epoch, best_psnr = train_model_n2i(
                model,
                experiment,
                train_loader,
                val_loader,
                device,
                model_out_dir,
                f"{model_spec.tag}_{setting_tag}",
                args.epochs,
                args.sino_splits,
                args.patience,
                args.min_delta,
                model_spec.optimizer,
            )

            with torch.no_grad():
                recon = solver.reconstruct(sino)
            metrics = compute_metrics(tgt, recon)

            results.append(
                {
                    "model": model_spec.name,
                    "model_tag": model_spec.name,
                    "setting": setting_name,
                    "setting_tag": setting_tag,
                    "target": tgt.detach().cpu(),
                    "reconstruction": recon.detach().cpu(),
                    "metrics": metrics,
                    "best_epoch": best_epoch,
                    "best_psnr": best_psnr,
                    "val_history": val_history,
                }
            )

            metrics_rows.append(
                {
                    "model": model_spec.name,
                    "setting": setting_tag,
                    "metrics": metrics,
                }
            )

            del solver, model, recon
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        del train_loader, val_loader, sino, tgt, experiment
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Save outputs
    torch.save(results, output_root / "all_models_results.pt")
    torch.save(fdk_results, output_root / "fdk_results.pt")

    save_metrics_csv(metrics_rows, output_root / "metrics_summary.csv")
    (output_root / "metrics_summary.json").write_text(
        json.dumps(metrics_rows, indent=2)
    )

    plot_3x3_denoised(results, output_root / "denoised_3x3.png")
    plot_fdk(fdk_results, output_root / "fdk_1x3.png")
    plot_4x3_grid(results, fdk_results, output_root / "fdk_plus_models_4x3.png")

    print("\n" + "=" * 70)
    print("NOISE2INVERSE - ALL MODELS (CUSTOM GEOMETRY)")
    print("=" * 70)
    for row in metrics_rows:
        print(
            f"{row['model']} | {row['setting']}: SSIM={row['metrics']['ssim']:.6f}, PSNR={row['metrics']['psnr']:.2f} dB"
        )
    print("=" * 70)
    print(f"\nOutputs saved to: {output_root.resolve()}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Noise2Inverse experiments for SwinIR, UNet, and SSM (custom geometry)"
    )
    parser.add_argument("--dataset", type=str, default="LIDC-IDRI")
    parser.add_argument("--datafolder", type=str, default=None)
    parser.add_argument(
        "--output-dir", type=str, default=str(script_dir / "results_all_models")
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=10)
    parser.add_argument("--test-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.01)
    parser.add_argument("--sino-splits", type=int, default=5)
    parser.add_argument("--unet-base-dim", type=int, default=32)

    run(parser.parse_args())
