#!/usr/bin/env python3
"""
Plot-only script: compare saved UNet+N2I, SwinScratch+N2I, and FDK reconstructions.

Loads:
- scripts/angleexp/results_swin/swin_scratch_3x1_reconstructions.pt
- scripts/angleexp/results_comparison/unet_3x1_reconstructions.pt

Generates FDK reconstructions and creates a 3x3 figure with a shared color range.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import LION.experiments.ct_experiments as ct_experiments
from LION.classical_algorithms.fdk import fdk
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().squeeze().numpy()
    return np.asarray(x).squeeze()


def compute_metrics(ground_truth, prediction):
    """Compute SSIM and PSNR metrics."""
    gt = to_np(ground_truth)
    pred = to_np(prediction)

    def normalize(img):
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return img

    gt_norm = normalize(gt)
    pred_norm = normalize(pred)

    ssim_val = ssim(gt_norm, pred_norm, data_range=1.0)
    psnr_val = psnr(gt_norm, pred_norm, data_range=1.0)

    return {"ssim": ssim_val, "psnr": psnr_val}


def get_fdk_reconstruction(experiment, test_index, device):
    """Get FDK reconstruction for a given experiment."""
    test_ds = experiment.get_testing_dataset()
    test_index = max(0, min(test_index, len(test_ds) - 1))
    sinogram, target = test_ds[test_index]

    if sinogram.dim() == 3:
        sinogram = sinogram.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)

    sinogram = sinogram.to(device)
    target = target.to(device)

    with torch.no_grad():
        fdk_recon = fdk(sinogram, experiment.geometry)

    metrics = compute_metrics(target, fdk_recon)

    return {
        "reconstruction": fdk_recon.detach().cpu(),
        "target": target.detach().cpu(),
        "metrics": metrics,
    }


def run(args):
    swin_path = Path(args.swin_pt)
    unet_path = Path(args.unet_pt)
    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}")

    if not swin_path.exists():
        raise FileNotFoundError(f"Swin file not found: {swin_path}")
    if not unet_path.exists():
        raise FileNotFoundError(f"UNet file not found: {unet_path}")

    print(f"Loading Swin results: {swin_path}")
    swin_results = torch.load(swin_path, map_location="cpu")
    print(f"Loading UNet results: {unet_path}")
    unet_results = torch.load(unet_path, map_location="cpu")

    if len(swin_results) < 3 or len(unet_results) < 3:
        raise ValueError(
            "Expected at least 3 entries in each .pt file (sparse/limited/full)."
        )

    # Define experiments for FDK reconstruction
    exp_configs = [
        ("Sparse angle", ct_experiments.SparseAngleLowDoseCTRecon),
        ("Limited angle", ct_experiments.LimitedAngleLowDoseCTRecon),
        ("Full angle", ct_experiments.LowDoseCTRecon),
    ]

    print("\nGenerating FDK reconstructions...")
    fdk_results = []
    datafolder_arg = args.datafolder if hasattr(args, "datafolder") else None
    for setting_name, exp_ctor in exp_configs:
        print(f"  {setting_name}...")
        if datafolder_arg:
            experiment = exp_ctor(dataset=args.dataset, datafolder=datafolder_arg)
        else:
            experiment = exp_ctor(dataset=args.dataset)
        fdk_result = get_fdk_reconstruction(experiment, args.test_index, device)
        fdk_result["setting"] = setting_name
        fdk_results.append(fdk_result)

    # Align rows by setting name when possible
    swin_by_setting = {str(d.get("setting", i)): d for i, d in enumerate(swin_results)}
    unet_by_setting = {str(d.get("setting", i)): d for i, d in enumerate(unet_results)}
    fdk_by_setting = {str(d.get("setting", i)): d for i, d in enumerate(fdk_results)}

    preferred_order = ["Sparse angle", "Limited angle", "Full angle"]
    rows = []
    for s in preferred_order:
        if s in swin_by_setting and s in unet_by_setting and s in fdk_by_setting:
            rows.append((s, unet_by_setting[s], swin_by_setting[s], fdk_by_setting[s]))

    if len(rows) < 3:
        # fallback to index pairing
        rows = []
        n = min(3, len(unet_results), len(swin_results), len(fdk_results))
        for i in range(n):
            setting = str(unet_results[i].get("setting", f"Setting {i+1}"))
            rows.append((setting, unet_results[i], swin_results[i], fdk_results[i]))

    # Separate color ranges: UNet+Swin share one scale, FDK has its own
    learned_vals = []
    fdk_vals = []
    for _, u, s, f in rows:
        learned_vals.append(to_np(u["reconstruction"]).ravel())
        learned_vals.append(to_np(s["reconstruction"]).ravel())
        fdk_vals.append(to_np(f["reconstruction"]).ravel())

    learned_vals = np.concatenate(learned_vals)
    fdk_vals = np.concatenate(fdk_vals)

    vmin = 0.0
    vmax_learned = float(np.max(learned_vals))
    vmax_fdk = float(np.max(fdk_vals))

    if vmax_learned <= vmin:
        vmax_learned = vmin + 1e-6
    if vmax_fdk <= vmin:
        vmax_fdk = vmin + 1e-6

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    for r, (setting, unet_row, swin_row, fdk_row) in enumerate(rows):
        unet_img = to_np(unet_row["reconstruction"])
        swin_img = to_np(swin_row["reconstruction"])
        fdk_img = to_np(fdk_row["reconstruction"])

        unet_metrics = unet_row.get("metrics", {})
        swin_metrics = swin_row.get("metrics", {})
        fdk_metrics = fdk_row.get("metrics", {})

        # UNet column
        axes[r, 0].imshow(unet_img, cmap="gray", vmin=vmin, vmax=vmax_learned)
        axes[r, 0].set_title(
            f"{setting} | UNet+N2I\nSSIM: {unet_metrics.get('ssim', float('nan')):.4f}, "
            f"PSNR: {unet_metrics.get('psnr', float('nan')):.2f}dB",
            fontsize=11,
            fontweight="bold",
        )
        axes[r, 0].axis("off")

        # Swin column
        axes[r, 1].imshow(swin_img, cmap="gray", vmin=vmin, vmax=vmax_learned)
        axes[r, 1].set_title(
            f"{setting} | SwinScratch+N2I\nSSIM: {swin_metrics.get('ssim', float('nan')):.4f}, "
            f"PSNR: {swin_metrics.get('psnr', float('nan')):.2f}dB",
            fontsize=11,
            fontweight="bold",
        )
        axes[r, 1].axis("off")

        # FDK column
        axes[r, 2].imshow(fdk_img, cmap="gray", vmin=vmin, vmax=vmax_learned)
        axes[r, 2].set_title(
            f"{setting} | FDK\nSSIM: {fdk_metrics.get('ssim', float('nan')):.4f}, "
            f"PSNR: {fdk_metrics.get('psnr', float('nan')):.2f}dB",
            fontsize=11,
            fontweight="bold",
        )
        axes[r, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path.resolve()}")
    print(f"Shared color range (UNet+Swin+FDK): [0, {vmax_learned:.6f}]")
    print("\nReconstruction min/max values:")
    for setting, unet_row, swin_row, fdk_row in rows:
        unet_img = to_np(unet_row["reconstruction"])
        swin_img = to_np(swin_row["reconstruction"])
        fdk_img = to_np(fdk_row["reconstruction"])

        print(
            f"  {setting} | UNet min={float(np.min(unet_img)):.6f}, max={float(np.max(unet_img)):.6f}"
        )
        print(
            f"  {setting} | Swin min={float(np.min(swin_img)):.6f}, max={float(np.max(swin_img)):.6f}"
        )
        print(
            f"  {setting} | FDK  min={float(np.min(fdk_img)):.6f}, max={float(np.max(fdk_img)):.6f}"
        )


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Plot UNet vs Swin vs FDK from saved .pt files"
    )
    parser.add_argument(
        "--swin-pt",
        type=str,
        default=str(
            script_dir
            / "scripts"
            / "angleexp"
            / "results_swin"
            / "swin_scratch_3x1_reconstructions.pt"
        ),
    )
    parser.add_argument(
        "--unet-pt",
        type=str,
        default=str(
            script_dir
            / "scripts"
            / "angleexp"
            / "results_comparison"
            / "unet_3x1_reconstructions.pt"
        ),
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default=str(
            script_dir
            / "scripts"
            / "angleexp"
            / "results_comparison"
            / "comparison_3x3_unet_vs_swin_vs_fdk.png"
        ),
    )
    parser.add_argument("--test-index", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="LIDC-IDRI")
    parser.add_argument("--datafolder", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    run(parser.parse_args())
