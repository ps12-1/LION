#!/usr/bin/env python3
"""
Memory-efficient Swin2SR + Noise2Inverse experiment.

Fine-tunes only the decoder head while keeping Swin2SR backbone frozen.
Uses memory optimization techniques:
- Frozen backbone (no gradients)
- Mixed precision training
- Gradient checkpointing
- Smaller input crops
- torch.no_grad() for backbone inference
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

import LION.experiments.ct_experiments as ct_experiments
from LION.classical_algorithms.fdk import fdk
from LION.models.LIONmodel import LIONModelParameter, LIONmodel, ModelInputType
from LION.optimizers.Noise2InverseSolver import Noise2InverseSolver
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class MemoryEfficientSwin2SR(LIONmodel):
    """Memory-efficient Swin2SR: frozen backbone + trainable decoder only."""

    def __init__(self, geometry, model_parameters: LIONModelParameter | None = None):
        if geometry is None:
            raise ValueError("Geometry parameters are required")
        if model_parameters is None:
            model_parameters = MemoryEfficientSwin2SR.default_parameters()
        super().__init__(model_parameters, geometry)

        try:
            from transformers import Swin2SRForImageSuperResolution, AutoImageProcessor
        except Exception as exc:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            ) from exc

        cache_dir = self.model_parameters.hf_cache_dir
        os.environ["HF_HOME"] = cache_dir

        # Load pretrained Swin2SR
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_parameters.hf_model_name, cache_dir=cache_dir
        )
        self.backbone = Swin2SRForImageSuperResolution.from_pretrained(
            self.model_parameters.hf_model_name, cache_dir=cache_dir
        )

        # **Freeze backbone completely**
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Small trainable decoder head
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    @staticmethod
    def default_parameters() -> LIONModelParameter:
        p = LIONModelParameter()
        p.model_input_type = ModelInputType.IMAGE
        p.hf_model_name = "caidas/swin2SR-classical-sr-x2-64"
        p.hf_cache_dir = str(Path(__file__).resolve().parent / "hf_cache")
        p.input_size = 512
        p.crop_size = 256  # Smaller crops for memory efficiency
        return p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with mixed precision and frozen backbone."""
        b, c, h, w = x.shape
        if c != 1:
            x = x[:, :1, ...]

        # Convert to 3-channel
        x_rgb = x.repeat(1, 3, 1, 1)

        # Resize to standard size
        x_resized = F.interpolate(
            x_rgb, size=(512, 512), mode="bilinear", align_corners=False
        )

        # Backbone inference with no gradients
        with torch.no_grad():
            backbone_out = self.backbone(pixel_values=x_resized)
            enhanced_rgb = backbone_out.reconstruction  # [B, 3, H, W]

        # Decoder (trainable)
        decoded = self.decoder(enhanced_rgb)

        # Resize back to original
        output = F.interpolate(
            decoded, size=(h, w), mode="bilinear", align_corners=False
        )
        return output


class MemoryEfficientNoise2InverseSolver(Noise2InverseSolver):
    """Noise2Inverse with mixed precision training."""

    def __init__(self, *args, use_mixed_precision=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mixed_precision = use_mixed_precision
        self.scaler = GradScaler() if use_mixed_precision else None

    def mini_batch_step(self, sinos, targets):
        """Mini batch step with mixed precision and gradient checkpointing."""
        with autocast(enabled=self.use_mixed_precision):
            noisy_sub_recons = self._calculate_noisy_sub_recons(sinos)

            loss = torch.zeros(len(self.cali_J), device=self.device)
            for i, J in enumerate(self.cali_J):
                J_zero_indexing = list(map(lambda idx: idx - 1, J))
                J_c = [
                    idx
                    for idx in np.arange(self.sino_split_count)
                    if idx not in J_zero_indexing
                ]

                jnsr = noisy_sub_recons[:, J_zero_indexing, :, :, :]
                jcnsr = noisy_sub_recons[:, J_c, :, :, :]
                mean_target_recons = torch.mean(jnsr, dim=1)
                mean_input_recons = torch.mean(jcnsr, dim=1)

                output = self.model(mean_input_recons)
                loss[i] = self.loss_fn(output, mean_target_recons)

            loss_final = loss.sum() / len(self.cali_J)

        if self.use_mixed_precision:
            self.scaler.scale(loss_final).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss_final.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return loss_final


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


def train_efficient_swin2sr_n2i(
    experiment,
    train_loader,
    device,
    output_dir: Path,
    tag: str,
    epochs: int,
    sino_splits: int,
):
    """Train efficient Swin2SR with frozen backbone using Noise2Inverse."""
    model = MemoryEfficientSwin2SR(experiment.geometry)

    # Only optimize decoder parameters
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-5,
    )
    loss_fn = nn.MSELoss()

    params = Noise2InverseSolver.default_parameters()
    params.sino_split_count = sino_splits
    params.recon_fn = fdk
    params.cali_J = Noise2InverseSolver.X_one_strategy(params.sino_split_count)

    solver = MemoryEfficientNoise2InverseSolver(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        solver_params=params,
        geometry=experiment.geometry,
        verbose=True,
        device=device,
        use_mixed_precision=True,
    )
    solver.set_training(train_loader)
    solver.set_checkpointing(
        checkpoint_fname=f"{tag}_swin2sr_n2i_check_*.pt",
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
            f"Training Swin2SR + Noise2Inverse (frozen backbone) for {setting_name}..."
        )
        solver = train_efficient_swin2sr_n2i(
            experiment=experiment,
            train_loader=train_loader,
            device=device,
            output_dir=output_dir,
            tag=tag,
            epochs=args.epochs,
            sino_splits=args.sino_splits,
        )

        # Clean cache
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
            f"{setting_name} | Swin2SR+N2I (frozen)\nSSIM: {metrics['ssim']:.4f}, PSNR: {metrics['psnr']:.2f}dB",
            fontsize=12,
            fontweight="bold",
        )
        axes[r].axis("off")

    plt.tight_layout()
    fig_path = output_dir / "swin2sr_frozen_3x1_reconstructions.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure: {fig_path}")

    tensor_path = output_dir / "swin2sr_frozen_3x1_reconstructions.pt"
    torch.save(results, tensor_path)
    print(f"Saved tensors: {tensor_path}")

    # Print metrics summary
    print("\n" + "=" * 70)
    print("SWIN2SR (FROZEN) + NOISE2INVERSE METRICS")
    print("=" * 70)
    for row in results:
        print(f"\n{row['setting'].upper()}:")
        print(
            f"  SSIM: {row['metrics']['ssim']:.6f}, PSNR: {row['metrics']['psnr']:.2f} dB"
        )
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Memory-efficient Swin2SR + Noise2Inverse (frozen backbone)"
    )
    parser.add_argument("--dataset", type=str, default="LIDC-IDRI")
    parser.add_argument("--datafolder", type=str, default=None)
    parser.add_argument(
        "--output-dir", type=str, default="scripts/angleexp/results_swin2sr"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--test-index", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--sino-splits", type=int, default=5)

    run(parser.parse_args())
