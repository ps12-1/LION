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
from pathlib import Path
from torch.utils.data import DataLoader

# LION imports
from LION.classical_algorithms.fdk import fdk
from LION.models.CNNs.SwinTransformer import SwinTransformer
import LION.experiments.ct_experiments as ct_experiments


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
            print(f"Loaded model weights from checkpoint")
            if "epoch" in checkpoint_data:
                print(f"  Trained for {checkpoint_data['epoch']} epochs")
        else:
            # Assume the entire file is the model state dict
            model.load_state_dict(checkpoint_data)
            print(f" Loaded model weights")
    else:
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Set model to evaluation mode
    model.eval()
    print("✓ Model set to evaluation mode")

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


def reconstruct_with_swin_only(
    checkpoint_path,
    output_dir=None,
    n_samples=5,
    device=None,
    save_numpy=True,
    save_png=False,
):
    """
    Load Swin Transformer checkpoint and run reconstructions only.

    - Loads the model trained via scripts/swint/swint.py
    - Runs FDK to produce the network input, then applies Swin Transformer
    - Optionally saves reconstructions (numpy arrays and/or PNGs)

    Args:
        checkpoint_path (str | Path): Path to model checkpoint produced by swint.py
        output_dir (str | Path | None): Where to save outputs; if None, doesn't save
        n_samples (int): Number of test samples to process
        device (torch.device | None): Device to use
        save_numpy (bool): Save outputs as .npy files
        save_png (bool): Save outputs as .png images

    Returns:
        list[torch.Tensor]: List of reconstructed tensors on CPU
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, experiment = load_transformer_model(checkpoint_path, device)
    model_input_size = model.img_size

    test_dataset = experiment.get_testing_dataset()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    save_path = None
    if output_dir is not None:
        save_path = Path(output_dir)
        save_path.mkdir(parents=True, exist_ok=True)

    reconstructions = []

    with torch.no_grad():
        for i, (sinogram, _) in enumerate(test_loader):
            if i >= n_samples:
                break

            sinogram = sinogram.to(device)

            # 1) Generate FDK input using original geometry
            fdk_recon = fdk(sinogram, experiment.geometry)

            # 2) Resize to model input size if needed
            if (
                fdk_recon.shape[2] != model_input_size
                or fdk_recon.shape[3] != model_input_size
            ):
                fdk_input = resize_to_model_size(fdk_recon, model_input_size)
            else:
                fdk_input = fdk_recon

            # 3) Swin Transformer reconstruction
            recon = model(fdk_input).detach().cpu()
            reconstructions.append(recon)

            if save_path is not None:
                if save_numpy:
                    np.save(save_path / f"reconstruction_{i+1}.npy", recon.numpy())
                if save_png:
                    try:
                        import matplotlib.pyplot as plt  # optional import only if requested

                        arr = recon.numpy()[0, 0]
                        plt.imsave(
                            save_path / f"reconstruction_{i+1}.png", arr, cmap="gray"
                        )
                    except Exception:
                        pass

            print(f"Reconstructed sample {i+1}/{n_samples}")

    return reconstructions


if __name__ == "__main__":
    # Minimal example: load checkpoint and reconstruct with Swin only
    CHECKPOINT_PATH = (
        "/store/LION/ps2050/trained_models/swin_transformer/SwinTransformer.pt"
    )
    OUTPUT_DIR = "/store/LION/ps2050/trained_models/swin_transformer/inference_only"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        reconstruct_with_swin_only(
            CHECKPOINT_PATH,
            output_dir=OUTPUT_DIR,
            n_samples=3,
            device=device,
            save_numpy=True,
            save_png=False,
        )
    except FileNotFoundError:
        print(f"Checkpoint file not found at: {CHECKPOINT_PATH}")
        print("Please update CHECKPOINT_PATH to your trained model from swint.py.")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback

        traceback.print_exc()
