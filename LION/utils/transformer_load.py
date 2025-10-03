"""
Swin Transformer Checkpoint Loader and Visualization Script

This script demonstrates how to:
1. Load trained Swin Transformer model checkpoints
2. Perform reconstruction on test data
3. Visualize and compare results

Requirements:
- Must have trained a Swin Transformer model using the swint.py script
- Checkpoint files should be available in the specified folder

"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import warnings

# LION imports
from LION.classical_algorithms.fdk import fdk
from LION.optimizers.SupervisedSolver import SupervisedSolver
from LION.models.CNNs.SwinTransformer import SwinTransformer
import LION.experiments.ct_experiments as ct_experiments
import LION.CTtools.ct_utils as ct_utils
from LION.utils.parameter import LIONParameter


def load_transformer_model(checkpoint_path, device=None):
    """
    Load a trained SwinTransformer model from checkpoint

    Args:
        checkpoint_path (str or Path): Path to the checkpoint file
        device (torch.device, optional): Device to load model on

    Returns:
        tuple: (model, experiment) - Ready to use model and experiment
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Recreate the same experiment setup as training
    experiment = ct_experiments.LowDoseCTRecon(dataset="LIDC-IDRI")

    print(f"Experiment image size: {experiment.geometry.image_shape}")

    # Recreate the same model architecture with matching parameters
    # IMPORTANT: These must match the training configuration in swint.py
    model_params = SwinTransformer.default_parameters()

    # Must match training configuration from swint.py
    model_params.img_size = 512  # This was used during training
    model_params.patch_size = 4
    model_params.embed_dim = 96
    model_params.depths = [2, 2, 6, 2]
    model_params.num_heads = [3, 6, 12, 24]
    model_params.window_size = 8  # Must be divisible into 128 (512/4)
    model_params.mlp_ratio = 4.0
    model_params.drop_rate = 0.0
    model_params.attn_drop_rate = 0.0
    model_params.drop_path_rate = 0.1

    print(f"Model configured for: {model_params.img_size}x{model_params.img_size}")

    # Create model
    model = SwinTransformer(experiment.geometry, model_params)
    model.to(device)

    # Load the checkpoint
    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.exists():
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint_data = torch.load(checkpoint_path, map_location=device)

        if "model_state_dict" in checkpoint_data:
            model.load_state_dict(checkpoint_data["model_state_dict"])
            print(f" Loaded model weights from checkpoint")
            if "epoch" in checkpoint_data:
                print(f"  Trained for {checkpoint_data['epoch']} epochs")
        else:
            # Assume the entire file is the model state dict
            model.load_state_dict(checkpoint_data)
            print(f"Loaded model weights")
    else:
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Set model to evaluation mode
    model.eval()
    print(" Model set to evaluation mode")

    return model, experiment


def resize_to_model_size(image, target_size):
    """
    Resize image to match model's expected input size

    Args:
        image: Input tensor of shape (B, C, H, W)
        target_size: Target size (single int for square images)

    Returns:
        Resized tensor of shape (B, C, target_size, target_size)
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    return F.interpolate(image, size=target_size, mode="bilinear", align_corners=False)


def visualize_reconstruction_comparison(
    fdk_input,
    reconstruction,
    target=None,
    title_prefix="",
    save_path=None,
    show_diff=True,
):
    """
    Visualize reconstruction results with comparisons

    Args:
        fdk_input (torch.Tensor): Input FDK reconstruction
        reconstruction (torch.Tensor): Swin Transformer enhanced reconstruction
        target (torch.Tensor, optional): Ground truth target
        title_prefix (str): Prefix for plot titles
        save_path (str, optional): Path to save the figure
        show_diff (bool): Whether to show difference images
    """
    # Convert to numpy and move to CPU
    fdk_np = fdk_input.detach().cpu().numpy()
    recon_np = reconstruction.detach().cpu().numpy()

    # Determine number of subplots
    n_plots = 2  # FDK input + Swin reconstruction
    if target is not None:
        target_np = target.detach().cpu().numpy()
        n_plots += 1
        if show_diff:
            n_plots += 1  # difference image

    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    # Plot FDK input
    im1 = axes[0].imshow(fdk_np[0][0], cmap="gray")
    axes[0].set_title(f"{title_prefix}FDK Input")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    plt.colorbar(im1, ax=axes[0])

    # Plot reconstruction
    im2 = axes[1].imshow(recon_np[0][0], cmap="gray")
    axes[1].set_title(f"{title_prefix}Swin Transformer Reconstruction")
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


def test_swin_transformer_reconstruction(
    checkpoint_path, save_folder=None, n_test_samples=5, device=None
):
    """
    Complete testing pipeline for Swin Transformer model

    Args:
        checkpoint_path (str): Path to the model checkpoint
        save_folder (str, optional): Folder to save results
        n_test_samples (int): Number of test samples to process
        device (torch.device, optional): Device to run on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    print("Loading Swin Transformer model...")
    model, experiment = load_transformer_model(checkpoint_path, device)

    # Get model's expected input size
    model_input_size = model.img_size

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

            # Step 1: FDK reconstruction from sinogram (use original geometry)
            fdk_recon = fdk(sinogram, experiment.geometry)

            # Step 2: Resize FDK reconstruction to match model input size if needed
            original_size = fdk_recon.shape[2:]
            if (
                fdk_recon.shape[2] != model_input_size
                or fdk_recon.shape[3] != model_input_size
            ):
                print(
                    f"  Resizing from {original_size} to {model_input_size}x{model_input_size}"
                )
                fdk_resized = resize_to_model_size(fdk_recon, model_input_size)
                target_resized = resize_to_model_size(target, model_input_size)
            else:
                fdk_resized = fdk_recon
                target_resized = target

            # Step 3: Enhance FDK reconstruction using Swin Transformer
            reconstruction = model(fdk_resized)

            # Use resized target for metrics
            target_for_metrics = target_resized

            # Compute metrics
            metrics = compute_metrics(reconstruction, target_for_metrics)
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
                fdk_resized,
                reconstruction,
                target_resized,
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
    Compare Swin Transformer reconstruction with classical FDK

    Args:
        checkpoint_path (str): Path to the model checkpoint
        save_folder (str, optional): Folder to save results
        device (torch.device, optional): Device to run on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    print("Loading Swin Transformer model...")
    model, experiment = load_transformer_model(checkpoint_path, device)

    # Get model's expected input size
    model_input_size = model.img_size

    # Get test data
    test_dataset = experiment.get_testing_dataset()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Get one sample for comparison
    sinogram, target = next(iter(test_loader))
    sinogram = sinogram.to(device)
    target = target.to(device)

    # Step 1: Classical FDK reconstruction from sinogram (use original geometry)
    print("Performing classical FDK reconstruction...")
    fdk_reconstruction = fdk(sinogram, experiment.geometry)

    # Step 2: Resize FDK and target if necessary to match model input size
    original_size = fdk_reconstruction.shape[2:]
    if (
        fdk_reconstruction.shape[2] != model_input_size
        or fdk_reconstruction.shape[3] != model_input_size
    ):
        print(f"Resizing from {original_size} to {model_input_size}x{model_input_size}")
        fdk_resized = resize_to_model_size(fdk_reconstruction, model_input_size)
        target_resized = resize_to_model_size(target, model_input_size)
    else:
        fdk_resized = fdk_reconstruction
        target_resized = target

    # Step 3: Enhance FDK reconstruction using Swin Transformer
    print("Performing Swin Transformer enhancement...")
    with torch.no_grad():
        swin_reconstruction = model(fdk_resized)

    # Compute metrics for both methods (using resized versions for fair comparison)
    swin_metrics = compute_metrics(swin_reconstruction, target_resized)
    fdk_metrics = compute_metrics(fdk_resized, target_resized)

    print("\n" + "=" * 50)
    print("COMPARISON: Swin Transformer Enhancement vs FDK")
    print("=" * 50)
    print("Note: Swin Transformer enhances the FDK reconstruction")
    print()
    print(f"Swin Enhanced PSNR: {swin_metrics['PSNR']:.2f} dB")
    print(f"FDK Baseline PSNR: {fdk_metrics['PSNR']:.2f} dB")
    print(f"PSNR Improvement: {swin_metrics['PSNR'] - fdk_metrics['PSNR']:.2f} dB")
    print()
    print(f"Swin Enhanced MSE: {swin_metrics['MSE']:.6f}")
    print(f"FDK Baseline MSE: {fdk_metrics['MSE']:.6f}")
    if swin_metrics["MSE"] > 0:
        print(f"MSE Reduction: {fdk_metrics['MSE'] / swin_metrics['MSE']:.2f}x")
    else:
        print("MSE Reduction: Perfect reconstruction!")

    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Convert to numpy (using resized versions for fair comparison)
    target_np = target_resized.detach().cpu().numpy()
    swin_np = swin_reconstruction.detach().cpu().numpy()
    fdk_np = fdk_resized.detach().cpu().numpy()

    # Common intensity range for fair comparison
    vmin = min(target_np[0].min(), swin_np[0].min(), fdk_np[0].min())
    vmax = max(target_np[0].max(), swin_np[0].max(), fdk_np[0].max())

    # Top row: reconstructions
    axes[0, 0].imshow(target_np[0][0], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Ground Truth")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(fdk_np[0][0], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'FDK Reconstruction\nPSNR: {fdk_metrics["PSNR"]:.2f} dB')
    axes[0, 1].axis("off")

    axes[0, 2].imshow(swin_np[0][0], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f'Swin \nPSNR: {swin_metrics["PSNR"]:.2f} dB')
    axes[0, 2].axis("off")

    # Bottom row: zoomed regions or difference images
    # Show the  input
    axes[1, 0].imshow(fdk_np[0][0], cmap="gray", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(" Input (to Swin)")
    axes[1, 0].axis("off")

    fdk_diff = fdk_np[0][0] - target_np[0][0]
    axes[1, 1].imshow(fdk_diff, cmap="RdBu_r")
    axes[1, 1].set_title("FDK Error")
    axes[1, 1].axis("off")

    swin_diff = swin_np[0][0] - target_np[0][0]
    axes[1, 2].imshow(swin_diff, cmap="RdBu_r")
    axes[1, 2].set_title("Swin Error")
    axes[1, 2].axis("off")

    plt.tight_layout()

    if save_folder:
        save_path = Path(save_folder) / "method_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison to: {save_path}")

    plt.show()

    return swin_metrics, fdk_metrics


if __name__ == "__main__":
    # Example usage

    # Update these paths according to your setup
    CHECKPOINT_PATH = (
        "/store/LION/ps2050/trained_models/swin_transformer/SwinTransformer.pt"
    )
    SAVE_FOLDER = "/store/LION/ps2050/trained_models/swin_transformer/test_results"

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Test the model on multiple samples
        print("Testing Swin Transformer reconstruction...")
        metrics = test_swin_transformer_reconstruction(
            CHECKPOINT_PATH, save_folder=SAVE_FOLDER, n_test_samples=3, device=device
        )

        # Compare with classical methods
        print("\nComparing with classical FDK...")
        swin_metrics, fdk_metrics = compare_with_classical_methods(
            CHECKPOINT_PATH, save_folder=SAVE_FOLDER, device=device
        )

    except FileNotFoundError:
        print(f"Checkpoint file not found at: {CHECKPOINT_PATH}")
        print(
            "Please update the CHECKPOINT_PATH variable to point to your trained model."
        )
        print("Make sure you have run the swint.py training script first.")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback

        traceback.print_exc()
        print("Please check your LION installation and data paths.")
