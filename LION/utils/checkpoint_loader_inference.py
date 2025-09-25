"""
Noise2Inverse Checkpoint Loader and Visualization Script

This script demonstrates how to:
1. Load trained Noise2Inverse model checkpoints
2. Perform reconstruction on test data
3. Visualize and compare results

Requirements:
- Must have trained a Noise2Inverse model using the original Noise2Inverse.py script
- Checkpoint files should be available in the specified folder

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import warnings

# LION imports
from LION.classical_algorithms.fdk import fdk
from LION.optimizers.Noise2InverseSolver import Noise2InverseSolver
from LION.models.CNNs.UNets.Unet import UNet
import LION.experiments.ct_experiments as ct_experiments
import LION.CTtools.ct_utils as ct_utils
from LION.utils.parameter import LIONParameter


def load_noise2inverse_model(checkpoint_path, device=None):
    """
    Load a trained Noise2Inverse model from checkpoint

    Args:
        checkpoint_path (str or Path): Path to the checkpoint file
        device (torch.device, optional): Device to load model on

    Returns:
        tuple: (solver, model, experiment) - Ready to use solver and components
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Recreate the same experiment setup as training
    experiment = ct_experiments.LowDoseCTRecon()

    # Recreate the same model architecture
    model = UNet()

    # Create optimizer (needed for solver, but not used in inference)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()

    # Setup Noise2Inverse parameters (must match training configuration)
    n2i_params = Noise2InverseSolver.default_parameters()
    n2i_params.splits = 4  # Must match training
    n2i_params.algo = fdk
    n2i_params.strategy = Noise2InverseSolver.X_one_strategy(n2i_params.splits)

    # Create solver
    solver = Noise2InverseSolver(
        model,
        optimizer,
        loss_fn,
        n2i_params,
        experiment.geometry,
        verbose=True,
        device=device,
    )

    # Load the checkpoint
    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.suffix == ".pt":
        # Load final model
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint_data:
            model.load_state_dict(checkpoint_data["model_state_dict"])
        else:
            # Assume the entire file is the model state dict
            model.load_state_dict(checkpoint_data)
        print(f"Loaded final model from: {checkpoint_path}")
    else:
        # Try to load checkpoint using LION's checkpoint loading mechanism
        try:
            model, optimizer, epoch, train_loss, _ = model.load_checkpoint_if_exists(
                checkpoint_path, model, optimizer, np.array([])
            )
            print(f"Loaded checkpoint from epoch {epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise

    # Set model to evaluation mode
    model.eval()
    solver.model = model

    return solver, model, experiment


def visualize_reconstruction_comparison(
    sinogram,
    reconstruction,
    target=None,
    title_prefix="",
    save_path=None,
    show_diff=True,
):
    """
    Visualize reconstruction results with comparisons

    Args:
        sinogram (torch.Tensor): Input sinogram
        reconstruction (torch.Tensor): Model reconstruction
        target (torch.Tensor, optional): Ground truth target
        title_prefix (str): Prefix for plot titles
        save_path (str, optional): Path to save the figure
        show_diff (bool): Whether to show difference images
    """
    # Convert to numpy and move to CPU
    sino_np = sinogram.detach().cpu().numpy()
    recon_np = reconstruction.detach().cpu().numpy()

    # Determine number of subplots
    n_plots = 2  # sinogram + reconstruction
    if target is not None:
        target_np = target.detach().cpu().numpy()
        n_plots += 1
        if show_diff:
            n_plots += 1  # difference image

    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    # Plot sinogram
    im1 = axes[0].imshow(sino_np[0].T, cmap="gray", aspect="auto")
    axes[0].set_title(f"{title_prefix}Input Sinogram")
    axes[0].set_xlabel("Projection Angle")
    axes[0].set_ylabel("Detector Element")
    plt.colorbar(im1, ax=axes[0])

    # Plot reconstruction
    print(f"{recon_np.shape},recon")
    im2 = axes[1].imshow(recon_np[0][0], cmap="gray")
    axes[1].set_title(f"{title_prefix}N2I Reconstruction")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    plt.colorbar(im2, ax=axes[1])

    plot_idx = 2

    # Plot target if available
    if target is not None:
        # Use same color scale for reconstruction and target
        vmin = min(recon_np[0].min(), target_np[0].min())
        vmax = max(recon_np[0].max(), target_np[0].max())
        print(target_np.shape)

        axes[1].images[0].set_clim(vmin, vmax)

        im3 = axes[plot_idx].imshow(target_np[0][0], cmap="gray", vmin=vmin, vmax=vmax)
        axes[plot_idx].set_title(f"{title_prefix}Ground Truth")
        axes[plot_idx].set_xlabel("X")
        axes[plot_idx].set_ylabel("Y")
        plt.colorbar(im3, ax=axes[plot_idx])
        plot_idx += 1

        # Plot difference if requested
        if show_diff:
            diff = recon_np[0][0] - target_np[0][0]
            im4 = axes[plot_idx].imshow(diff, cmap="RdBu_r")
            axes[plot_idx].set_title(f"{title_prefix}Difference (Recon - GT)")
            axes[plot_idx].set_xlabel("X")
            axes[plot_idx].set_ylabel("Y")
            plt.colorbar(im4, ax=axes[plot_idx])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")

    plt.show()


def compute_metrics(reconstruction, target):
    """
    Compute reconstruction quality metrics

    Args:
        reconstruction (torch.Tensor): Model reconstruction
        target (torch.Tensor): Ground truth target

    Returns:
        dict: Dictionary of computed metrics
    """
    # Convert to numpy and flatten
    recon_np = reconstruction.detach().cpu().numpy().flatten()
    target_np = target.detach().cpu().numpy().flatten()

    # Mean Squared Error
    mse = np.mean((recon_np - target_np) ** 2)

    # Peak Signal-to-Noise Ratio
    data_range = target_np.max() - target_np.min()
    psnr = 20 * np.log10(data_range / np.sqrt(mse))

    # Structural Similarity Index (simplified version)
    # For full SSIM, you'd want to use skimage.metrics.structural_similarity
    mean_recon = np.mean(recon_np)
    mean_target = np.mean(target_np)
    std_recon = np.std(recon_np)
    std_target = np.std(target_np)

    # Correlation coefficient as a proxy for SSIM structure component
    correlation = np.corrcoef(recon_np, target_np)[0, 1]

    metrics = {
        "MSE": mse,
        "PSNR": psnr,
        "Mean_Reconstruction": mean_recon,
        "Mean_Target": mean_target,
        "Std_Reconstruction": std_recon,
        "Std_Target": std_target,
        "Correlation": correlation,
    }

    return metrics


def test_noise2inverse_reconstruction(
    checkpoint_path, save_folder=None, n_test_samples=5, device=None
):
    """
    Complete testing pipeline for Noise2Inverse model

    Args:
        checkpoint_path (str): Path to the model checkpoint
        save_folder (str, optional): Folder to save results
        n_test_samples (int): Number of test samples to process
        device (torch.device, optional): Device to run on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    print("Loading Noise2Inverse model...")
    solver, model, experiment = load_noise2inverse_model(checkpoint_path, device)

    # Get test data
    print("Loading test dataset...")
    test_dataset = experiment.get_testing_dataset()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Create save folder if specified
    if save_folder:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

    # Process test samples
    all_metrics = []

    print(f"Processing {n_test_samples} test samples...")
    with torch.no_grad():
        for i, (sinogram, target) in enumerate(test_loader):
            if i >= n_test_samples:
                break

            print(f"Processing sample {i+1}/{n_test_samples}")

            # Move to device
            sinogram = sinogram.to(device)
            target = target.to(device)

            # Perform reconstruction using Noise2Inverse
            reconstruction = solver.reconstruct(sinogram)

            # Compute metrics
            metrics = compute_metrics(reconstruction, target)
            metrics["sample_id"] = i
            all_metrics.append(metrics)

            # Print metrics for this sample
            print(
                f"  Sample {i+1} - PSNR: {metrics['PSNR']:.2f} dB, "
                f"MSE: {metrics['MSE']:.6f}, Correlation: {metrics['Correlation']:.4f}"
            )

            # Visualize and save
            save_path = None
            if save_folder:
                save_path = save_folder / f"reconstruction_sample_{i+1}.png"

            visualize_reconstruction_comparison(
                sinogram,
                reconstruction,
                target,
                title_prefix=f"Sample {i+1} - ",
                save_path=save_path,
            )

    # Print summary statistics
    print("\n" + "=" * 50)
    print("RECONSTRUCTION QUALITY SUMMARY")
    print("=" * 50)

    psnr_values = [m["PSNR"] for m in all_metrics]
    mse_values = [m["MSE"] for m in all_metrics]
    corr_values = [m["Correlation"] for m in all_metrics]

    print(f"PSNR: {np.mean(psnr_values):.2f} ± {np.std(psnr_values):.2f} dB")
    print(f"MSE:  {np.mean(mse_values):.6f} ± {np.std(mse_values):.6f}")
    print(f"Correlation: {np.mean(corr_values):.4f} ± {np.std(corr_values):.4f}")

    return all_metrics


def compare_with_classical_methods(checkpoint_path, save_folder=None, device=None):
    """
    Compare Noise2Inverse reconstruction with classical FDK

    Args:
        checkpoint_path (str): Path to the model checkpoint
        save_folder (str, optional): Folder to save results
        device (torch.device, optional): Device to run on
    """
    if device is None:
        device = torch.device("cuda :1" if torch.cuda.is_available() else "cpu")

    # Load the model
    print("Loading Noise2Inverse model...")
    solver, model, experiment = load_noise2inverse_model(checkpoint_path, device)

    # Get test data
    test_dataset = experiment.get_testing_dataset()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Get one sample for comparison
    sinogram, target = next(iter(test_loader))
    sinogram = sinogram.to(device)
    target = target.to(device)

    # Noise2Inverse reconstruction
    print("Performing Noise2Inverse reconstruction...")
    with torch.no_grad():
        n2i_reconstruction = solver.reconstruct(sinogram)

    # Classical FDK reconstruction
    print("Performing classical FDK reconstruction...")
    fdk_reconstruction = fdk(sinogram, solver.geometry)

    # Compute metrics for both methods
    n2i_metrics = compute_metrics(n2i_reconstruction, target)
    fdk_metrics = compute_metrics(fdk_reconstruction, target)

    print("\n" + "=" * 50)
    print("COMPARISON: Noise2Inverse vs Classical FDK")
    print("=" * 50)
    print(f"Noise2Inverse PSNR: {n2i_metrics['PSNR']:.2f} dB")
    print(f"Classical FDK PSNR: {fdk_metrics['PSNR']:.2f} dB")
    print(f"PSNR Improvement: {n2i_metrics['PSNR'] - fdk_metrics['PSNR']:.2f} dB")
    print()
    print(f"Noise2Inverse MSE: {n2i_metrics['MSE']:.6f}")
    print(f"Classical FDK MSE: {fdk_metrics['MSE']:.6f}")
    print(f"MSE Reduction: {fdk_metrics['MSE'] / n2i_metrics['MSE']:.2f}x")

    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Convert to numpy
    sino_np = sinogram.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    n2i_np = n2i_reconstruction.detach().cpu().numpy()
    fdk_np = fdk_reconstruction.detach().cpu().numpy()

    # Common intensity range for fair comparison
    vmin = min(target_np[0].min(), n2i_np[0].min(), fdk_np[0].min())
    vmax = max(target_np[0].max(), n2i_np[0].max(), fdk_np[0].max())

    # Top row: reconstructions
    axes[0, 0].imshow(target_np[0][0], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Ground Truth")
    axes[0, 0].axis("off")

    print(fdk_np.shape)
    axes[0, 1].imshow(fdk_np[0][0], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'Classical FDK\nPSNR: {fdk_metrics["PSNR"]:.2f} dB')
    axes[0, 1].axis("off")

    axes[0, 2].imshow(n2i_np[0][0], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f'Noise2Inverse\nPSNR: {n2i_metrics["PSNR"]:.2f} dB')
    axes[0, 2].axis("off")

    # Bottom row: difference images
    axes[1, 0].imshow(sino_np[0].T, cmap="gray", aspect="auto")
    axes[1, 0].set_title("Input Sinogram")
    axes[1, 0].set_xlabel("Projection Angle")
    axes[1, 0].set_ylabel("Detector Element")

    fdk_diff = fdk_np[0][0] - target_np[0][0]
    axes[1, 1].imshow(fdk_diff, cmap="RdBu_r")
    axes[1, 1].set_title("FDK Error")
    axes[1, 1].axis("off")

    n2i_diff = n2i_np[0][0] - target_np[0][0]
    axes[1, 2].imshow(n2i_diff, cmap="RdBu_r")
    axes[1, 2].set_title("N2I Error")
    axes[1, 2].axis("off")

    plt.tight_layout()

    if save_folder:
        save_path = Path(save_folder) / "method_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison to: {save_path}")

    plt.show()

    return n2i_metrics, fdk_metrics


if __name__ == "__main__":
    # Example usage

    # Update these paths according to your setup
    CHECKPOINT_PATH = "/store/LION/ps2050/trained_models/test_debbuging/N2I.pt"
    SAVE_FOLDER = "/store/LION/ps2050/trained_models/test_debbuging"

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Test the model on multiple samples
        print("Testing Noise2Inverse reconstruction...")
        metrics = test_noise2inverse_reconstruction(
            CHECKPOINT_PATH, save_folder=SAVE_FOLDER, n_test_samples=3, device=device
        )

        # Compare with classical methods
        print("\nComparing with classical FDK...")
        n2i_metrics, fdk_metrics = compare_with_classical_methods(
            CHECKPOINT_PATH, save_folder=SAVE_FOLDER, device=device
        )

    except FileNotFoundError:
        print(f"Checkpoint file not found at: {CHECKPOINT_PATH}")
        print(
            "Please update the CHECKPOINT_PATH variable to point to your trained model."
        )
        print("Make sure you have run the Noise2Inverse.py training script first.")
    except Exception as e:
        print(f"Error during execution: {e}")
        print("Please check your LION installation and data paths.")
