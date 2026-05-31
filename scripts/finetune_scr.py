#!/usr/bin/env python3
"""
Finetune Swin2SR for Physically Interpretable CT Reconstruction

This script demonstrates how to finetune the Hugging Face Swin2SR model
to produce physically meaningful CT values in Hounsfield Units (HU).

Key features:
- Preserves CT units during training
- Uses supervised learning with clean CT targets
- Implements proper normalization strategies
- Saves checkpoints and metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
from contextlib import nullcontext

# LION imports
from LION.models.CNNs.huggingface import Swin2SR
from LION.optimizers.SupervisedSolver import SupervisedSolver
from LION.experiments.ct_experiments import LowDoseCTRecon
from LION.utils.parameter import LIONParameter
from LION.metrics.psnr import PSNR
from LION.metrics.ssim import SSIM


class PhysicallyAwareSwin2SR(Swin2SR):
    """
    Swin2SR with modifications for physically interpretable outputs
    """

    def __init__(self, geometry, model_parameters=None):
        super().__init__(geometry, model_parameters)

        # Add output scaling layer to ensure proper HU range
        self.output_scaling = nn.Parameter(torch.tensor(1.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        """Forward pass with physical constraints"""
        # Get base Swin2SR output
        base_output = super().forward(x)

        # Apply learnable scaling and bias to maintain HU range
        # This helps the model learn to output values in the correct HU range
        scaled_output = base_output * self.output_scaling + self.output_bias

        return scaled_output


class CTNormalizer:
    """
    Custom normalizer that preserves CT units while enabling stable training
    """

    def __init__(self, hu_min=-1000, hu_max=2000):
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.range = hu_max - hu_min

    def normalize(self, x):
        """Normalize to [0, 1] while preserving relative HU relationships"""
        return (x - self.hu_min) / self.range

    def denormalize(self, x):
        """Convert back to HU units"""
        return x * self.range + self.hu_min


def create_finetuning_dataset(experiment, normalizer):
    """
    Create training dataset with proper normalization
    """
    print("Creating finetuning dataset...")

    # Get training data
    train_dataset = experiment.get_training_dataset()
    val_dataset = experiment.get_validation_dataset()

    # Create custom dataset that handles normalization
    class CTFinetuningDataset:
        def __init__(self, base_dataset, normalizer):
            self.base_dataset = base_dataset
            self.normalizer = normalizer

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            sinogram, ground_truth = self.base_dataset[idx]

            # Ensure CPU tensors for DataLoader pinning
            if torch.is_tensor(sinogram):
                sinogram = sinogram.detach().cpu()
            if torch.is_tensor(ground_truth):
                ground_truth = ground_truth.detach().cpu()

            # Create FDK reconstruction (input to model)
            from LION.classical_algorithms.fdk import fdk

            fdk_recon = fdk(sinogram, experiment.geometry, clip=True)

            if torch.is_tensor(fdk_recon):
                fdk_recon = fdk_recon.detach().cpu()

            # Add channel dimension if needed
            if fdk_recon.dim() == 2:
                fdk_recon = fdk_recon.unsqueeze(0)
            elif fdk_recon.dim() == 3 and fdk_recon.size(0) != 1:
                fdk_recon = fdk_recon.unsqueeze(1)

            if ground_truth.dim() == 2:
                ground_truth = ground_truth.unsqueeze(0)
            elif ground_truth.dim() == 3 and ground_truth.size(0) != 1:
                ground_truth = ground_truth.unsqueeze(1)

            # Normalize both input and target for stable training
            fdk_normalized = normalizer.normalize(fdk_recon)
            target_normalized = normalizer.normalize(ground_truth)

            return fdk_normalized, target_normalized

    train_finetuning = CTFinetuningDataset(train_dataset, normalizer)
    val_finetuning = CTFinetuningDataset(val_dataset, normalizer)

    return train_finetuning, val_finetuning


def _iter_tiles(h, w, tile_size, tile_overlap):
    stride = max(tile_size - tile_overlap, 1)
    y = 0
    while y < h:
        x = 0
        y1 = min(y + tile_size, h)
        y0 = max(y1 - tile_size, 0)
        while x < w:
            x1 = min(x + tile_size, w)
            x0 = max(x1 - tile_size, 0)
            yield y0, y1, x0, x1
            x += stride
        y += stride


def compute_tiled_loss(
    model, fdk_input, target, loss_fn, tile_size, tile_overlap, use_amp
):
    b, c, h, w = fdk_input.shape
    tile_loss = 0.0
    tile_count = 0

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", enabled=use_amp)
        if use_amp
        else nullcontext()
    )

    for y0, y1, x0, x1 in _iter_tiles(h, w, tile_size, tile_overlap):
        fdk_tile = fdk_input[..., y0:y1, x0:x1]
        target_tile = target[..., y0:y1, x0:x1]
        with autocast_ctx:
            output_tile = model(fdk_tile)
            loss = loss_fn(output_tile, target_tile)
        tile_loss = tile_loss + loss
        tile_count += 1

    if tile_count == 0:
        return loss_fn(model(fdk_input), target)

    return tile_loss / tile_count


def tiled_forward(model, x, tile_size, tile_overlap, use_amp):
    b, c, h, w = x.shape
    output = torch.zeros((b, 1, h, w), device=x.device, dtype=x.dtype)
    weight = torch.zeros_like(output)

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", enabled=use_amp)
        if use_amp
        else nullcontext()
    )

    for y0, y1, x0, x1 in _iter_tiles(h, w, tile_size, tile_overlap):
        tile = x[..., y0:y1, x0:x1]
        with autocast_ctx:
            out_tile = model(tile)
        output[..., y0:y1, x0:x1] += out_tile
        weight[..., y0:y1, x0:x1] += 1.0

    return output / weight.clamp_min(1.0)


def finetune_swin2sr(
    model_name="caidas/swin2SR-classical-sr-x2-64",
    dataset="LIDC-IDRI",
    epochs=100,
    batch_size=4,
    learning_rate=1e-5,
    save_dir="finetuned_swin2sr",
    device=None,
    use_amp=True,
    tile_size=256,
    tile_overlap=32,
    use_tiling=True,
    device_ids=None,
    num_workers=0,
    use_data_parallel=True,
    max_train_steps=None,
    max_val_steps=None,
    preview_recons=0,
):
    """
    Main finetuning function
    """

    # Setup
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    print(f"Finetuning Swin2SR on device: {device}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Epochs: {epochs}")

    # Initialize experiment
    experiment = LowDoseCTRecon(dataset=dataset)
    geometry = experiment.geometry

    # Create normalizer
    normalizer = CTNormalizer(hu_min=-1000, hu_max=2000)

    # Create model with physical constraints
    model_params = Swin2SR.default_parameters()
    model_params.hf_model_name = model_name
    model_params.train_hf_backbone = True  # Enable training
    model_params.do_rescale = False  # Disable rescaling to preserve HU units

    model = PhysicallyAwareSwin2SR(geometry, model_params)
    model.to(device)

    if model_params.train_hf_backbone and hasattr(
        model.model, "gradient_checkpointing_enable"
    ):
        model.model.gradient_checkpointing_enable()

    use_data_parallel = (
        use_data_parallel
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    )
    if use_data_parallel:
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        print(f"Using DataParallel on GPUs: {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)

    # Create datasets
    train_dataset, val_dataset = create_finetuning_dataset(experiment, normalizer)

    # Create dataloaders
    pin_memory = device.type == "cuda" and num_workers == 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=False,
    )

    # Setup optimizer - use different learning rates for different parts
    hf_params = []
    new_params = []

    for name, param in model.named_parameters():
        if "model." in name:  # Hugging Face model parameters
            hf_params.append(param)
        else:  # New parameters (output scaling, etc.)
            new_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": hf_params, "lr": learning_rate * 0.1},  # Lower LR for pretrained
            {"params": new_params, "lr": learning_rate},  # Higher LR for new parameters
        ]
    )

    # Loss function - MSE for regression
    loss_fn = nn.MSELoss()

    # Metrics
    psnr_metric = PSNR()
    ssim_metric = SSIM()

    # Mixed precision setup
    use_amp = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Training loop
    train_losses = []
    val_losses = []
    val_psnr = []
    val_ssim = []

    best_val_loss = float("inf")

    print("Starting finetuning...")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0

        for batch_idx, (fdk_input, target) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        ):
            optimizer.zero_grad(set_to_none=True)

            fdk_input = fdk_input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Forward pass
            if use_tiling:
                loss = compute_tiled_loss(
                    model, fdk_input, target, loss_fn, tile_size, tile_overlap, use_amp
                )
            else:
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    output = model(fdk_input)
                    loss = loss_fn(output, target)

            # Backward pass
            scaler.scale(loss).backward()

            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_steps += 1

            if max_train_steps is not None and train_steps >= max_train_steps:
                break

        train_loss /= max(train_steps, 1)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_psnr_score = 0.0
        val_ssim_score = 0.0

        val_steps = 0
        with torch.no_grad():
            for fdk_input, target in val_loader:
                fdk_input = fdk_input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                if use_tiling:
                    output = tiled_forward(
                        model, fdk_input, tile_size, tile_overlap, use_amp
                    )
                    loss = loss_fn(output, target)
                else:
                    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                        output = model(fdk_input)
                        loss = loss_fn(output, target)
                val_loss += loss.item()
                val_steps += 1

                # Convert back to HU for metrics
                output_hu = normalizer.denormalize(output.float())
                target_hu = normalizer.denormalize(target.float())

                # Calculate metrics
                val_psnr_score += psnr_metric(output_hu, target_hu, reduce="mean")
                val_ssim_score += ssim_metric(
                    output_hu, target_hu, reduce="mean", channel_axis=None
                )

                if max_val_steps is not None and val_steps >= max_val_steps:
                    break

        val_loss /= max(val_steps, 1)
        val_psnr_score /= max(val_steps, 1)
        val_ssim_score /= max(val_steps, 1)

        val_losses.append(float(val_loss))
        val_psnr.append(float(val_psnr_score))
        val_ssim.append(float(val_ssim_score))

        if preview_recons and (epoch == 0):
            _save_preview_reconstructions(
                model,
                experiment,
                normalizer,
                device,
                save_dir,
                use_tiling,
                tile_size,
                tile_overlap,
                use_amp,
                max_samples=preview_recons,
            )

        # Print progress
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Val PSNR: {val_psnr_score:.2f} dB")
        print(f"  Val SSIM: {val_ssim_score:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state_dict = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_psnr": val_psnr_score,
                    "val_ssim": val_ssim_score,
                },
                save_dir / "best_model.pt",
            )
            print(f"  New best model saved! (Val Loss: {val_loss:.6f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            state_dict = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                },
                save_dir / f"checkpoint_epoch_{epoch+1}.pt",
            )

    # Save final results
    state_dict = (
        model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict()
    )
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
        },
        save_dir / "final_model.pt",
    )

    # Save training history
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_psnr": val_psnr,
        "val_ssim": val_ssim,
        "best_val_loss": best_val_loss,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "model_name": model_name,
        "dataset": dataset,
    }

    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    plt.subplot(1, 3, 2)
    plt.plot(val_psnr)
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("Validation PSNR")

    plt.subplot(1, 3, 3)
    plt.plot(val_ssim)
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("Validation SSIM")

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nFinetuning completed!")
    print(f"Results saved to: {save_dir}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final validation PSNR: {val_psnr[-1]:.2f} dB")
    print(f"Final validation SSIM: {val_ssim[-1]:.4f}")

    return model, history


def _save_preview_reconstructions(
    model,
    experiment,
    normalizer,
    device,
    save_dir,
    use_tiling,
    tile_size,
    tile_overlap,
    use_amp,
    max_samples=1,
):
    model.eval()
    val_dataset = experiment.get_validation_dataset()
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, (sinogram, ground_truth) in enumerate(test_loader):
            if i >= max_samples:
                break

            sinogram = sinogram.to(device)
            ground_truth = ground_truth.to(device)

            from LION.classical_algorithms.fdk import fdk

            fdk_recon = fdk(sinogram, experiment.geometry, clip=True)
            if fdk_recon.dim() == 3:
                fdk_recon = fdk_recon.unsqueeze(1)

            fdk_normalized = normalizer.normalize(fdk_recon)

            if use_tiling:
                output_normalized = tiled_forward(
                    model, fdk_normalized, tile_size, tile_overlap, use_amp
                )
            else:
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    output_normalized = model(fdk_normalized)

            fdk_hu = fdk_recon.squeeze().cpu().numpy()
            enhanced_hu = (
                normalizer.denormalize(output_normalized.float())
                .squeeze()
                .cpu()
                .numpy()
            )
            ground_truth_hu = ground_truth.squeeze().cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            im1 = axes[0].imshow(ground_truth_hu, cmap="gray", vmin=-1000, vmax=2000)
            axes[0].set_title("Ground Truth")
            axes[0].axis("off")
            plt.colorbar(im1, ax=axes[0])

            im2 = axes[1].imshow(fdk_hu, cmap="gray", vmin=-1000, vmax=2000)
            axes[1].set_title("FDK Reconstruction")
            axes[1].axis("off")
            plt.colorbar(im2, ax=axes[1])

            im3 = axes[2].imshow(enhanced_hu, cmap="gray", vmin=-1000, vmax=2000)
            axes[2].set_title("Preview Swin2SR")
            axes[2].axis("off")
            plt.colorbar(im3, ax=axes[2])

            plt.tight_layout()
            out_path = Path(save_dir) / f"preview_epoch0_sample_{i+1}.png"
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)


def test_finetuned_model(model_path, experiment, device=None):
    """
    Test the finetuned model and show results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)

    # Create model
    model_params = Swin2SR.default_parameters()
    model_params.train_hf_backbone = False  # Inference only
    model_params.do_rescale = False

    model = PhysicallyAwareSwin2SR(experiment.geometry, model_params)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Create normalizer
    normalizer = CTNormalizer(hu_min=-1000, hu_max=2000)

    # Test on validation data
    val_dataset = experiment.get_validation_dataset()
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print("Testing finetuned model...")

    with torch.no_grad():
        for i, (sinogram, ground_truth) in enumerate(test_loader):
            if i >= 3:  # Test on 3 samples
                break

            sinogram = sinogram.to(device)
            ground_truth = ground_truth.to(device)

            # FDK reconstruction
            from LION.classical_algorithms.fdk import fdk

            fdk_recon = fdk(sinogram, experiment.geometry, clip=True)
            if fdk_recon.dim() == 3:
                fdk_recon = fdk_recon.unsqueeze(1)

            # Normalize for model
            fdk_normalized = normalizer.normalize(fdk_recon)

            # Model prediction
            output_normalized = model(fdk_normalized)

            # Convert back to HU
            fdk_hu = fdk_recon.squeeze().cpu().numpy()
            enhanced_hu = (
                normalizer.denormalize(output_normalized).squeeze().cpu().numpy()
            )
            ground_truth_hu = ground_truth.squeeze().cpu().numpy()

            # Print value ranges
            print(f"\nSample {i+1}:")
            print(
                f"  Ground Truth HU range: {ground_truth_hu.min():.1f} to {ground_truth_hu.max():.1f}"
            )
            print(f"  FDK HU range: {fdk_hu.min():.1f} to {fdk_hu.max():.1f}")
            print(
                f"  Enhanced HU range: {enhanced_hu.min():.1f} to {enhanced_hu.max():.1f}"
            )

            # Visualize
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            im1 = axes[0].imshow(ground_truth_hu, cmap="gray", vmin=-1000, vmax=2000)
            axes[0].set_title("Ground Truth")
            axes[0].axis("off")
            plt.colorbar(im1, ax=axes[0])

            im2 = axes[1].imshow(fdk_hu, cmap="gray", vmin=-1000, vmax=2000)
            axes[1].set_title("FDK Reconstruction")
            axes[1].axis("off")
            plt.colorbar(im2, ax=axes[1])

            im3 = axes[2].imshow(enhanced_hu, cmap="gray", vmin=-1000, vmax=2000)
            axes[2].set_title("Finetuned Swin2SR")
            axes[2].axis("off")
            plt.colorbar(im3, ax=axes[2])

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    # Example usage
    print("Swin2SR Finetuning for Physically Interpretable CT Reconstruction")
    print("=" * 70)

    # Finetune the model
    model, history = finetune_swin2sr(
        model_name="caidas/swin2SR-classical-sr-x2-64",
        dataset="LIDC-IDRI",
        epochs=1,
        batch_size=1,
        learning_rate=1e-5,
        save_dir="finetuned_swin2sr_results",
        device=torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu"),
        use_data_parallel=True,
        device_ids=[0, 1, 2, 3],
        max_train_steps=25,
        max_val_steps=5,
        preview_recons=2,
        tile_size=128,
        tile_overlap=16,
    )

    # Test the finetuned model
    experiment = LowDoseCTRecon(dataset="LIDC-IDRI")
    test_finetuned_model("finetuned_swin2sr_results/best_model.pt", experiment)
