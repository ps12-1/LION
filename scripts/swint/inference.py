#!/usr/bin/env python3
"""
Simple Example: FDK + Hugging Face Model Inference

This example loads a real sample from LIDC-IDRI dataset, performs FDK reconstruction,
applies Hugging Face model enhancement, and compares the results with metrics.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# LION imports
from LION.classical_algorithms.fdk import fdk
from LION.models.CNNs.huggingface import Swin2SR
import LION.experiments.ct_experiments as ct_experiments
from torch.utils.data import DataLoader


def real_dataset_inference_example():
    """Load real LIDC-IDRI sample and compare FDK vs HF-enhanced reconstruction"""

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize experiment for geometry
    experiment = ct_experiments.LowDoseCTRecon(dataset="LIDC-IDRI")
    geometry = experiment.geometry

    # Load test dataset
    test_dataset = experiment.get_testing_dataset()
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Get one sample from the dataset
    data_batch = next(iter(test_dataloader))
    sinogram, ground_truth = data_batch
    sinogram = sinogram.to(device)
    ground_truth = ground_truth.to(device)

    # Load Hugging Face model
    model_params = Swin2SR.default_parameters()
    model_params.hf_model_name = "caidas/swin2SR-classical-sr-x2-64"
    model_params.train_hf_backbone = False  # Inference only

    model = Swin2SR(geometry, model_params)
    model.to(device)
    model.eval()

    # Perform FDK reconstruction
    with torch.no_grad():
        fdk_recon = fdk(sinogram, geometry, clip=True)

        # Ensure correct dimensions for Swin2SR model: [batch, channels, height, width]
        if fdk_recon.dim() == 3:
            fdk_recon = fdk_recon.unsqueeze(1)  # Add channel dimension

    # Apply Hugging Face model enhancement
    with torch.no_grad():
        enhanced_recon = model(fdk_recon)

    # Calculate metrics
    metrics = calculate_metrics(ground_truth, fdk_recon, enhanced_recon)

    # Visualize and save results
    save_comparison_results(ground_truth, fdk_recon, enhanced_recon, metrics)

    return fdk_recon, enhanced_recon, ground_truth, metrics


def calculate_metrics(ground_truth, fdk_recon, enhanced_recon):
    """Calculate SSIM and PSNR metrics for comparison"""

    # Convert to numpy arrays for metric calculation
    gt_np = ground_truth.squeeze().cpu().numpy()
    fdk_np = fdk_recon.squeeze().cpu().numpy()
    enhanced_np = enhanced_recon.squeeze().cpu().numpy()

    # Normalize to [0, 1] for metric calculation
    def normalize_image(img):
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return img

    gt_norm = normalize_image(gt_np)
    fdk_norm = normalize_image(fdk_np)
    enhanced_norm = normalize_image(enhanced_np)

    # Calculate SSIM
    ssim_fdk = ssim(gt_norm, fdk_norm, data_range=1.0)
    ssim_enhanced = ssim(gt_norm, enhanced_norm, data_range=1.0)

    # Calculate PSNR
    psnr_fdk = psnr(gt_norm, fdk_norm, data_range=1.0)
    psnr_enhanced = psnr(gt_norm, enhanced_norm, data_range=1.0)

    metrics = {
        "ssim_fdk": ssim_fdk,
        "ssim_enhanced": ssim_enhanced,
        "psnr_fdk": psnr_fdk,
        "psnr_enhanced": psnr_enhanced,
        "ssim_improvement": ssim_enhanced - ssim_fdk,
        "psnr_improvement": psnr_enhanced - psnr_fdk,
    }

    return metrics


def create_synthetic_sinogram(geometry):
    """Create a synthetic sinogram for demonstration"""

    # Get geometry parameters
    angles = len(geometry.angles)
    detectors = geometry.detector_shape[1]

    # Create a simple phantom sinogram
    sinogram = torch.zeros(1, angles, detectors)

    # Add some simple patterns
    for i in range(angles):
        # Simple line pattern
        center = detectors // 2
        width = 20
        sinogram[0, i, center - width : center + width] = 1.0

        # Add some noise
        sinogram[0, i, :] += 0.1 * torch.randn(detectors)

    return sinogram


def synthetic_inference_example():
    """Fallback synthetic example if LIDC-IDRI is not available"""

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize experiment for geometry
    experiment = ct_experiments.LowDoseCTRecon(dataset="LIDC-IDRI")
    geometry = experiment.geometry

    # Create synthetic ground truth (simple phantom)
    ground_truth = torch.zeros(1, 1, 512, 512)
    # Add some simple structures
    ground_truth[0, 0, 200:300, 200:300] = 1.0  # Square
    ground_truth[0, 0, 100:150, 100:150] = 0.5  # Smaller square

    # Create synthetic sinogram
    sinogram = create_synthetic_sinogram(geometry)
    sinogram = sinogram.to(device)
    ground_truth = ground_truth.to(device)

    # Load Hugging Face model
    model_params = Swin2SR.default_parameters()
    model_params.hf_model_name = "caidas/swin2SR-classical-sr-x2-64"
    model_params.train_hf_backbone = False  # Inference only

    model = Swin2SR(geometry, model_params)
    model.to(device)
    model.eval()

    # Perform FDK reconstruction
    with torch.no_grad():
        fdk_recon = fdk(sinogram, geometry, clip=True)

        # Ensure correct dimensions for Swin2SR model: [batch, channels, height, width]
        if fdk_recon.dim() == 3:
            fdk_recon = fdk_recon.unsqueeze(1)  # Add channel dimension

    # Apply Hugging Face model enhancement
    with torch.no_grad():
        enhanced_recon = model(fdk_recon)

    # Calculate metrics
    metrics = calculate_metrics(ground_truth, fdk_recon, enhanced_recon)

    # Visualize and save results
    save_comparison_results(ground_truth, fdk_recon, enhanced_recon, metrics)

    return fdk_recon, enhanced_recon, ground_truth, metrics


def save_comparison_results(ground_truth, fdk_recon, enhanced_recon, metrics):
    """Save comprehensive comparison results with metrics"""

    # Create output directory
    output_dir = Path("inference_results")
    output_dir.mkdir(exist_ok=True)

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Convert to numpy for visualization
    gt_np = ground_truth.squeeze().cpu().numpy()
    fdk_np = fdk_recon.squeeze().cpu().numpy()
    enhanced_np = enhanced_recon.squeeze().cpu().numpy()

    # Debug: Print actual value ranges
    print(f"Ground Truth range: {gt_np.min():.3f} to {gt_np.max():.3f}")
    print(f"FDK range: {fdk_np.min():.3f} to {fdk_np.max():.3f}")
    print(f"Enhanced range: {enhanced_np.min():.3f} to {enhanced_np.max():.3f}")

    # Calculate common range for consistent visualization
    all_values = np.concatenate(
        [gt_np.flatten(), fdk_np.flatten(), enhanced_np.flatten()]
    )
    common_min = np.percentile(all_values, 1)  # Use 1st percentile to avoid outliers
    common_max = np.percentile(all_values, 99)  # Use 99th percentile to avoid outliers

    print(f"Using common range: {common_min:.3f} to {common_max:.3f}")

    # Top row: Original images
    im1 = axes[0, 0].imshow(gt_np, cmap="gray")
    # plt.clim(0,2.2)
    axes[0, 0].set_title("Ground Truth", fontsize=14, fontweight="bold")
    axes[0, 0].axis("off")
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(fdk_np, cmap="gray")
    # plt.clim(0,2.2)
    axes[0, 1].set_title(
        f'FDK Reconstruction\nSSIM: {metrics["ssim_fdk"]:.4f}, PSNR: {metrics["psnr_fdk"]:.2f}dB',
        fontsize=14,
        fontweight="bold",
    )
    axes[0, 1].axis("off")
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[0, 2].imshow(enhanced_np, cmap="gray")
    # plt.clim(0,2.2)
    axes[0, 2].set_title(
        f'SwinIR\nSSIM: {metrics["ssim_enhanced"]:.4f}, PSNR: {metrics["psnr_enhanced"]:.2f}dB',
        fontsize=14,
        fontweight="bold",
    )
    axes[0, 2].axis("off")
    plt.colorbar(im3, ax=axes[0, 2])

    # Bottom row: Differences and error maps
    fdk_diff = np.abs(gt_np - fdk_np)
    enhanced_diff = np.abs(gt_np - enhanced_np)
    improvement = enhanced_np - fdk_np

    im4 = axes[1, 0].imshow(fdk_diff, cmap="hot")
    axes[1, 0].set_title("FDK Error Map", fontsize=14, fontweight="bold")
    axes[1, 0].axis("off")
    plt.colorbar(im4, ax=axes[1, 0])

    im5 = axes[1, 1].imshow(enhanced_diff, cmap="hot")
    axes[1, 1].set_title("SwinIR Error Map", fontsize=14, fontweight="bold")
    axes[1, 1].axis("off")
    plt.colorbar(im5, ax=axes[1, 1])

    im6 = axes[1, 2].imshow(improvement, cmap="RdBu_r")
    axes[1, 2].set_title("Improvement (Enhanced - FDK)", fontsize=14, fontweight="bold")
    axes[1, 2].axis("off")
    plt.colorbar(im6, ax=axes[1, 2])

    plt.tight_layout()

    # Save the comprehensive comparison
    comparison_path = output_dir / "fdk_vs_hf_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Save individual images
    np.save(output_dir / "ground_truth.npy", gt_np)
    np.save(output_dir / "fdk_reconstruction.npy", fdk_np)
    np.save(output_dir / "enhanced_reconstruction.npy", enhanced_np)

    # Save metrics to file
    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("FDK vs Hugging Face Enhanced Reconstruction Metrics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"FDK Reconstruction:\n")
        f.write(f"  SSIM: {metrics['ssim_fdk']:.6f}\n")
        f.write(f"  PSNR: {metrics['psnr_fdk']:.2f} dB\n\n")
        f.write(f"HF Enhanced Reconstruction:\n")
        f.write(f"  SSIM: {metrics['ssim_enhanced']:.6f}\n")
        f.write(f"  PSNR: {metrics['psnr_enhanced']:.2f} dB\n\n")
        f.write(f"Improvements:\n")
        f.write(f"  SSIM improvement: {metrics['ssim_improvement']:+.6f}\n")
        f.write(f"  PSNR improvement: {metrics['psnr_improvement']:+.2f} dB\n")

    # Print metrics to console
    print("RECONSTRUCTION COMPARISON METRICS")
    print("=" * 50)
    print(f"FDK Reconstruction:")
    print(f"  SSIM: {metrics['ssim_fdk']:.6f}")
    print(f"  PSNR: {metrics['psnr_fdk']:.2f} dB")
    print(f"\nHF Enhanced Reconstruction:")
    print(f"  SSIM: {metrics['ssim_enhanced']:.6f}")
    print(f"  PSNR: {metrics['psnr_enhanced']:.2f} dB")
    print(f"\nImprovements:")
    print(f"  SSIM improvement: {metrics['ssim_improvement']:+.6f}")
    print(f"  PSNR improvement: {metrics['psnr_improvement']:+.2f} dB")
    print("=" * 50)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - Comparison image: {comparison_path}")
    print(f"  - Metrics: {metrics_path}")
    print(
        f"  - Numpy arrays: ground_truth.npy, fdk_reconstruction.npy, enhanced_reconstruction.npy"
    )


def batch_inference_example(sinogram_files, output_dir):
    """Example of batch processing multiple sinograms"""

    from scripts.inference.fdk_hf_inference import HFInferencePipeline, SinogramDataset

    # Initialize pipeline
    pipeline = HFInferencePipeline(
        model_name="caidas/swin2SR-classical-sr-x2-64",
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=1,
    )

    # Create dataset
    experiment = ct_experiments.LowDoseCTRecon(dataset="LIDC-IDRI")
    dataset = SinogramDataset(sinogram_files, experiment.geometry)

    # Process dataset
    results = pipeline.process_dataset(
        dataset, output_dir, save_images=True, save_metrics=True
    )

    return results


if __name__ == "__main__":
    print("FDK + Hugging Face Model Inference with Real LIDC-IDRI Data")
    print("=" * 60)

    try:
        # Try real dataset example first
        (
            fdk_result,
            enhanced_result,
            ground_truth,
            metrics,
        ) = real_dataset_inference_example()

        print("\nReal dataset example completed successfully!")
        print("\nTo process your own sinogram files, use:")
        print(
            "python scripts/inference/fdk_hf_inference.py --input_dir /path/to/sinograms --output_dir /path/to/output"
        )

    except Exception as e:
        print(f"\nReal dataset not available: {e}")
        print("Falling back to synthetic example...")

        try:
            # Fallback to synthetic example
            (
                fdk_result,
                enhanced_result,
                ground_truth,
                metrics,
            ) = synthetic_inference_example()

            print("\nSynthetic example completed successfully!")
            print(
                "Note: This used synthetic data. For real results, ensure LIDC-IDRI dataset is available."
            )

        except Exception as e2:
            print(f"\nBoth real and synthetic examples failed:")
            print(f"  Real dataset error: {e}")
            print(f"  Synthetic example error: {e2}")
            print("Please check your LION installation and dependencies.")
