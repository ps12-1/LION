#!/usr/bin/env python3
"""
Compare UNet+N2I vs SwinScratch+N2I reconstructions.

Trains UNet with Noise2Inverse for 3 angle settings and compares with previously trained Swin model.
Creates 3x2 comparison figure with unified color scaling.
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
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

import LION.experiments.ct_experiments as ct_experiments
from LION.classical_algorithms.fdk import fdk
from LION.models.CNNs.UNets.Unet import UNet
from LION.optimizers.Noise2InverseSolver import Noise2InverseSolver
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


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


def train_unet_n2i(
    experiment,
    train_loader,
    device,
    output_dir: Path,
    tag: str,
    epochs: int,
    sino_splits: int,
):
    """Train UNet using Noise2Inverse."""
    model_params = UNet.default_parameters()
    model_params.baseDim = 32
    model = UNet(model_params)
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
        checkpoint_fname=f"{tag}_unet_n2i_check_*.pt",
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

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    swin_output_dir = Path(args.swin_dir)

    settings = [
        ("Sparse angle", ct_experiments.SparseAngleLowDoseCTRecon),
        ("Limited angle", ct_experiments.LimitedAngleLowDoseCTRecon),
        ("Full angle", ct_experiments.LowDoseCTRecon),
    ]

    # Load previously saved Swin results
    swin_results_file = swin_output_dir / "swin_scratch_3x1_reconstructions.pt"
    if not swin_results_file.exists():
        raise FileNotFoundError(f"Swin results not found: {swin_results_file}")

    print(f"Loading Swin results from {swin_results_file}...")
    swin_results = torch.load(swin_results_file)
    print(f"Loaded {len(swin_results)} Swin reconstruction(s)")

    # Train UNet for each setting
    unet_results = []

    for setting_name, exp_ctor in settings:
        print(f"\n===== {setting_name} =====")
        experiment = exp_ctor(dataset=args.dataset, datafolder=args.datafolder)

        train_loader = prepare_loaders(
            experiment,
            batch_size=args.batch_size,
            train_n=args.train_samples,
        )

        tag = setting_name.lower().replace(" ", "_")

        print(f"Training UNet + Noise2Inverse for {setting_name}...")
        solver = train_unet_n2i(
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

        unet_results.append(
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

    # Save UNet results
    tensor_path = output_dir / "unet_3x1_reconstructions.pt"
    torch.save(unet_results, tensor_path)
    print(f"\nSaved UNet tensors: {tensor_path}")

    # Create 3x2 comparison figure with unified color scaling
    print("\nCreating 3x2 comparison figure...")

    # Collect all reconstructions to find unified color range
    all_recons = []
    for i in range(len(unet_results)):
        all_recons.append(to_np(unet_results[i]["reconstruction"]))
        all_recons.append(to_np(swin_results[i]["reconstruction"]))

    all_recons = np.concatenate([r.flatten() for r in all_recons])
    v_min = all_recons.min()
    v_max = all_recons.max()

    print(f"Color range: [{v_min:.4f}, {v_max:.4f}]")

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))

    for r in range(len(unet_results)):
        # UNet column
        unet_recon = to_np(unet_results[r]["reconstruction"])
        unet_metrics = unet_results[r]["metrics"]
        axes[r, 0].imshow(unet_recon, cmap="gray")
        axes[r, 0].clim(v_min, v_max)
        axes[r, 0].set_title(
            f"{unet_results[r]['setting']} | UNet+N2I\nSSIM: {unet_metrics['ssim']:.4f}, PSNR: {unet_metrics['psnr']:.2f}dB",
            fontsize=11,
            fontweight="bold",
        )
        axes[r, 0].axis("off")

        # Swin column
        swin_recon = to_np(swin_results[r]["reconstruction"])
        swin_metrics = swin_results[r]["metrics"]
        axes[r, 1].imshow(swin_recon, cmap="gray")
        axes[r, 1].clim(v_min, v_max)
        axes[r, 1].set_title(
            f"{swin_results[r]['setting']} | SwinScratch+N2I\nSSIM: {swin_metrics['ssim']:.4f}, PSNR: {swin_metrics['psnr']:.2f}dB",
            fontsize=11,
            fontweight="bold",
        )
        axes[r, 1].axis("off")

    plt.tight_layout()
    fig_path = output_dir / "comparison_3x2_unet_vs_swin.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison figure: {fig_path}")
    plt.close()

    # Print metrics summary
    print("\n" + "=" * 80)
    print("RECONSTRUCTION METRICS COMPARISON")
    print("=" * 80)
    for i in range(len(unet_results)):
        setting = unet_results[i]["setting"].upper()
        unet_ssim = unet_results[i]["metrics"]["ssim"]
        unet_psnr = unet_results[i]["metrics"]["psnr"]
        swin_ssim = swin_results[i]["metrics"]["ssim"]
        swin_psnr = swin_results[i]["metrics"]["psnr"]

        print(f"\n{setting}:")
        print(f"  UNet+N2I:         SSIM: {unet_ssim:.6f}, PSNR: {unet_psnr:.2f} dB")
        print(f"  SwinScratch+N2I:  SSIM: {swin_ssim:.6f}, PSNR: {swin_psnr:.2f} dB")
        print(
            f"  Δ SSIM: {swin_ssim - unet_ssim:+.6f}, Δ PSNR: {swin_psnr - unet_psnr:+.2f} dB"
        )
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare UNet+N2I vs SwinScratch+N2I")
    parser.add_argument("--dataset", type=str, default="LIDC-IDRI")
    parser.add_argument("--datafolder", type=str, default=None)
    parser.add_argument(
        "--output-dir", type=str, default="scripts/angleexp/results_comparison"
    )
    parser.add_argument("--swin-dir", type=str, default="scripts/angleexp/results_swin")
    parser.add_argument("--gpu", type=int, default=1, help="GPU ID to use")
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--test-index", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--sino-splits", type=int, default=5)

    run(parser.parse_args())
