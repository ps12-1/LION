#!/usr/bin/env python3
"""
Swin Transformer Noise2Inverse experiment.

Trains a Swin Transformer-based denoiser using Noise2Inverse self-supervised approach.
Creates a 3x1 reconstruction figure showing results across 3 angle settings.
"""

from __future__ import annotations

import argparse
import os
import gc
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

import LION.experiments.ct_experiments as ct_experiments
from LION.classical_algorithms.fdk import fdk
from LION.models.LIONmodel import LIONModelParameter, LIONmodel, ModelInputType
from LION.optimizers.Noise2InverseSolver import Noise2InverseSolver
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class WindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: [B*nW, N, C]
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            num_windows = attn_mask.shape[0]
            attn = attn.view(b // num_windows, num_windows, self.num_heads, n, n)
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = torch.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class SwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def _window_partition(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        b, h, w, c = x.shape
        x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return windows.view(-1, window_size * window_size, c)

    def _window_reverse(
        self, windows: torch.Tensor, window_size: int, h: int, w: int, b: int
    ) -> torch.Tensor:
        x = windows.view(
            b, h // window_size, w // window_size, window_size, window_size, -1
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(b, h, w, -1)

    def _create_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        if self.shift_size == 0:
            return None

        img_mask = torch.zeros((1, h, w, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        cnt = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = cnt
                cnt += 1

        mask_windows = self._window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C]
        b, h, w, c = x.shape
        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        x_windows = self._window_partition(shifted_x, self.window_size)
        attn_mask = self._create_mask(h, w, x.device)
        attn_windows = self.attn(x_windows, attn_mask=attn_mask)
        shifted_x = self._window_reverse(attn_windows, self.window_size, h, w, b)

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class BasicSwinLayer(nn.Module):
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
        self.blocks = nn.ModuleList(
            [
                SwinBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


class SwinScratchDenoiser(LIONmodel):
    """Swin Transformer denoiser trained fully from scratch (no pretrained weights)."""

    def __init__(self, geometry, model_parameters: LIONModelParameter | None = None):
        if geometry is None:
            raise ValueError("Geometry parameters are required")
        if model_parameters is None:
            model_parameters = SwinScratchDenoiser.default_parameters()
        super().__init__(model_parameters, geometry)

        p = self.model_parameters
        self.window_size = p.window_size

        self.in_proj = nn.Conv2d(1, p.embed_dim, kernel_size=3, padding=1)

        self.stage1 = BasicSwinLayer(
            dim=p.embed_dim,
            depth=p.depth_stage1,
            num_heads=p.heads_stage1,
            window_size=p.window_size,
            use_checkpoint=p.use_checkpoint,
        )

        self.down = nn.Conv2d(
            p.embed_dim, p.embed_dim * 2, kernel_size=3, stride=2, padding=1
        )

        self.stage2 = BasicSwinLayer(
            dim=p.embed_dim * 2,
            depth=p.depth_stage2,
            num_heads=p.heads_stage2,
            window_size=p.window_size,
            use_checkpoint=p.use_checkpoint,
        )

        self.up = nn.ConvTranspose2d(
            p.embed_dim * 2, p.embed_dim, kernel_size=2, stride=2
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(p.embed_dim * 2, p.embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(p.embed_dim, p.embed_dim, kernel_size=3, padding=1),
        )

        self.stage3 = BasicSwinLayer(
            dim=p.embed_dim,
            depth=p.depth_stage3,
            num_heads=p.heads_stage3,
            window_size=p.window_size,
            use_checkpoint=p.use_checkpoint,
        )

        self.out_head = nn.Sequential(
            nn.Conv2d(p.embed_dim, p.embed_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(p.embed_dim // 2, 1, kernel_size=3, padding=1),
        )

    @staticmethod
    def default_parameters() -> LIONModelParameter:
        p = LIONModelParameter()
        p.model_input_type = ModelInputType.IMAGE
        p.embed_dim = 64  # Reduced from 96 for memory efficiency
        p.window_size = 7  # Reduced from 8 for memory efficiency
        p.depth_stage1 = 2  # Reduced from 4
        p.depth_stage2 = 2  # Reduced from 4
        p.depth_stage3 = 1  # Reduced from 2
        p.heads_stage1 = 4
        p.heads_stage2 = 8
        p.heads_stage3 = 4
        p.use_checkpoint = True  # Enable gradient checkpointing
        return p

    def _pad_to_multiple(
        self, x: torch.Tensor, multiple: int
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, h, w = x.shape
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (pad_h, pad_w)

    def _to_bhwc(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1).contiguous()

    def _to_bchw(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor [B,C,H,W], got shape={tuple(x.shape)}"
            )

        if x.shape[1] != 1:
            x = x[:, :1, ...]

        residual = x
        x_pad, (pad_h, pad_w) = self._pad_to_multiple(x, multiple=self.window_size * 2)

        f0 = self.in_proj(x_pad)

        s1 = self._to_bchw(self.stage1(self._to_bhwc(f0)))
        d1 = self.down(s1)
        s2 = self._to_bchw(self.stage2(self._to_bhwc(d1)))

        u1 = self.up(s2)
        if u1.shape[-2:] != s1.shape[-2:]:
            u1 = F.interpolate(
                u1, size=s1.shape[-2:], mode="bilinear", align_corners=False
            )
        f1 = self.fuse(torch.cat([u1, s1], dim=1))

        f2 = self._to_bchw(self.stage3(self._to_bhwc(f1)))
        noise_est = self.out_head(f2)

        if pad_h > 0:
            noise_est = noise_est[:, :, :-pad_h, :]
        if pad_w > 0:
            noise_est = noise_est[:, :, :, :-pad_w]

        return residual + noise_est


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subset_if_needed(dataset, n_samples: int):
    if n_samples is None or n_samples <= 0 or n_samples >= len(dataset):
        return dataset
    return Subset(dataset, torch.arange(n_samples))


def prepare_loaders(experiment, batch_size: int, train_n: int):
    train_ds = subset_if_needed(experiment.get_training_dataset(), train_n)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    return train_loader


def train_swin_n2i(
    experiment,
    train_loader,
    device,
    output_dir: Path,
    tag: str,
    epochs: int,
    sino_splits: int,
):
    """Train Swin Transformer from scratch using Noise2Inverse."""
    model = SwinScratchDenoiser(experiment.geometry)
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
        checkpoint_fname=f"{tag}_swin_scratch_n2i_check_*.pt",
        checkpoint_freq=max(epochs + 1, 99999),
        load_checkpoint_if_exists=False,
        save_folder=output_dir,
    )
    solver.train(epochs)
    return solver


def get_test_sample(experiment, sample_index: int):
    test_ds = experiment.get_testing_dataset()
    sample_index = max(0, min(sample_index, len(test_ds) - 1))
    sinogram, target = test_ds[sample_index]
    if sinogram.dim() == 3:
        sinogram = sinogram.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    return sinogram, target


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().squeeze().numpy()


def compute_metrics(ground_truth: torch.Tensor, prediction: torch.Tensor) -> dict:
    """Compute SSIM and PSNR metrics."""
    gt_np = to_np(ground_truth)
    pred_np = to_np(prediction)

    def normalize(img):
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return img

    gt_norm = normalize(gt_np)
    pred_norm = normalize(pred_np)

    ssim_val = ssim(gt_norm, pred_norm, data_range=1.0)
    psnr_val = psnr(gt_norm, pred_norm, data_range=1.0)

    return {"ssim": ssim_val, "psnr": psnr_val}


def run(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    # Enable memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = [
        ("Sparse angle", ct_experiments.SparseAngleLowDoseCTRecon),
        ("Limited angle", ct_experiments.LimitedAngleLowDoseCTRecon),
        ("Full angle", ct_experiments.LowDoseCTRecon),
    ]

    # Filter to single angle if requested
    if args.angle_setting:
        settings = [
            s
            for s in settings
            if s[0].lower().replace(" ", "_") == args.angle_setting.lower()
        ]
        if not settings:
            raise ValueError(
                f"Invalid angle setting: {args.angle_setting}. Use 'sparse_angle', 'limited_angle', or 'full_angle'"
            )

    results = []

    for setting_name, exp_ctor in settings:
        print(f"\n===== {setting_name} =====")
        experiment = exp_ctor(dataset=args.dataset, datafolder=args.datafolder)

        train_loader = prepare_loaders(
            experiment,
            batch_size=args.batch_size,
            train_n=args.train_samples,
        )

        tag = setting_name.lower().replace(" ", "_")

        print(
            f"Training Swin Transformer (scratch) + Noise2Inverse for {setting_name}..."
        )
        solver = train_swin_n2i(
            experiment=experiment,
            train_loader=train_loader,
            device=device,
            output_dir=output_dir,
            tag=tag,
            epochs=args.epochs,
            sino_splits=args.sino_splits,
        )

        # Clean cache after training
        del train_loader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print(f"Running inference for {setting_name}...")
        sinogram, target = get_test_sample(experiment, args.test_index)
        sinogram = sinogram.to(device)
        target = target.to(device)

        with torch.no_grad():
            recon = solver.reconstruct(sinogram)

        metrics = compute_metrics(target, recon)

        results.append(
            {
                "setting": setting_name,
                "target": target.detach().cpu(),
                "reconstruction": recon.detach().cpu(),
                "metrics": metrics,
            }
        )

        # Clean cache
        del solver, sinogram, target, recon, experiment
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Build 3x1 reconstruction panel
    fig, axes = plt.subplots(3, 1, figsize=(8, 14))

    for r, row in enumerate(results):
        setting_name = row["setting"]
        recon = row["reconstruction"]
        metrics = row["metrics"]

        axes[r].imshow(to_np(recon), cmap="gray")
        axes[r].set_title(
            f"{setting_name} | SwinScratch+N2I\nSSIM: {metrics['ssim']:.4f}, PSNR: {metrics['psnr']:.2f}dB",
            fontsize=12,
            fontweight="bold",
        )
        axes[r].axis("off")

    plt.tight_layout()
    fig_path = output_dir / "swin_scratch_3x1_reconstructions.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure: {fig_path}")

    tensor_path = output_dir / "swin_scratch_3x1_reconstructions.pt"
    torch.save(results, tensor_path)
    print(f"Saved tensors: {tensor_path}")

    # Print metrics summary
    print("\n" + "=" * 70)
    print("SWIN TRANSFORMER (SCRATCH) + NOISE2INVERSE METRICS")
    print("=" * 70)
    for row in results:
        print(f"\n{row['setting'].upper()}:")
        print(
            f"  SSIM: {row['metrics']['ssim']:.6f}, PSNR: {row['metrics']['psnr']:.2f} dB"
        )
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Swin Transformer (from scratch) + Noise2Inverse experiment"
    )
    parser.add_argument("--dataset", type=str, default="LIDC-IDRI")
    parser.add_argument("--datafolder", type=str, default=None)
    parser.add_argument(
        "--output-dir", type=str, default="scripts/angleexp/results_swin"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--test-index", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--sino-splits", type=int, default=5)
    parser.add_argument(
        "--angle-setting",
        type=str,
        default=None,
        help="Run only one angle setting: 'sparse_angle', 'limited_angle', or 'full_angle'",
    )

    run(parser.parse_args())
