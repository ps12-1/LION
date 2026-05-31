#!/usr/bin/env python3
"""
Angle experiment: Noise2Inverse+UNet vs HuggingFace ViT reconstructor.

Creates a 3x2 reconstruction figure:
rows   = [sparse angle, limited angle, full angle]
cols   = [UNet (Noise2Inverse), HuggingFace ViT]
"""

from __future__ import annotations

import argparse
import math
import random
import os
import gc
from pathlib import Path
from multiprocessing import Process, Queue, set_start_method

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Subset

import LION.experiments.ct_experiments as ct_experiments
from LION.classical_algorithms.fdk import fdk
from LION.models.CNNs.UNets.Unet import UNet
from LION.models.CNNs.huggingface import Swin2SR
from LION.models.LIONmodel import LIONModelParameter, LIONmodel, ModelInputType
from LION.optimizers.Noise2InverseSolver import Noise2InverseSolver
from LION.optimizers.SupervisedSolver import SupervisedSolver
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


def build_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepare_loaders(experiment, batch_size: int, train_n: int, val_n: int):
    train_ds = subset_if_needed(experiment.get_training_dataset(), train_n)
    val_ds = subset_if_needed(experiment.get_validation_dataset(), val_n)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_n2i_unet(
    experiment,
    train_loader,
    device,
    output_dir: Path,
    tag: str,
    epochs: int,
    sino_splits: int,
):
    model = UNet(UNet.default_parameters())
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
        checkpoint_freq=max(epochs + 1, 99999),
        load_checkpoint_if_exists=False,
        save_folder=output_dir,
    )
    solver.train(epochs)
    return solver


def train_hf_vit(
    experiment,
    train_loader,
    val_loader,
    device,
    output_dir: Path,
    tag: str,
    epochs: int,
    hf_model_name: str,
):
    model_params = Swin2SR.default_parameters()
    model_params.hf_model_name = hf_model_name
    model_params.train_hf_backbone = (
        False  # Frozen backbone for inference only (saves ~80% memory)
    )
    model = Swin2SR(experiment.geometry, model_params)
    model.to(device)
    model.eval()

    # Create a minimal wrapper to match solver interface
    class InferenceWrapper:
        def __init__(self, model):
            self.model = model

    return InferenceWrapper(model)


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
    """Compute SSIM and PSNR metrics between GT and prediction."""
    gt_np = to_np(ground_truth)
    pred_np = to_np(prediction)

    # Normalize to [0, 1] for metric calculation
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


def train_single_setting(
    setting_idx, setting_name, exp_ctor_name, args_dict, gpu_id, result_queue
):
    """Train a single angle setting on a specific GPU in a separate process."""
    try:
        # Enable memory optimization at process start
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        set_seed(args_dict["seed"] + setting_idx)
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)

        print(f"\n[GPU {gpu_id}] ===== {setting_name} =====")

        # Reconstruct experiment from name
        exp_ctor = getattr(ct_experiments, exp_ctor_name)
        experiment = exp_ctor(
            dataset=args_dict["dataset"], datafolder=args_dict["datafolder"]
        )

        train_loader, val_loader = prepare_loaders(
            experiment,
            batch_size=args_dict["batch_size"],
            train_n=args_dict["train_samples"],
            val_n=args_dict["val_samples"],
        )

        tag = setting_name.lower().replace(" ", "_")
        output_dir = Path(args_dict["output_dir"])

        print(f"[GPU {gpu_id}] Training UNet+N2I for {setting_name}...")
        n2i_solver = train_n2i_unet(
            experiment=experiment,
            train_loader=train_loader,
            device=device,
            output_dir=output_dir,
            tag=tag,
            epochs=args_dict["epochs_n2i"],
            sino_splits=args_dict["sino_splits"],
        )

        # Aggressive cleanup after N2I
        del train_loader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print(
            f"[GPU {gpu_id}] Loading Swin2SR (frozen/inference-only) for {setting_name}..."
        )
        swin_solver = train_hf_vit(
            experiment=experiment,
            train_loader=None,
            val_loader=None,
            device=device,
            output_dir=output_dir,
            tag=tag,
            epochs=args_dict["epochs_vit"],
            hf_model_name=args_dict["hf_model_name"],
        )

        # Memory already minimal since model is frozen
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print(f"[GPU {gpu_id}] Running inference for {setting_name}...")
        sinogram, target = get_test_sample(experiment, args_dict["test_index"])
        sinogram = sinogram.to(device)
        target = target.to(device)

        with torch.no_grad():
            n2i_recon = n2i_solver.reconstruct(sinogram)
            fdk_recon = fdk(sinogram, experiment.geometry, clip=True)
            swin_recon = swin_solver.model(fdk_recon)

        result = {
            "setting": setting_name,
            "target": target.detach().cpu(),
            "unet_n2i": n2i_recon.detach().cpu(),
            "swin2sr": swin_recon.detach().cpu(),
        }

        # Save result to disk instead of passing through Queue (avoids large tensor transfer issues)
        result_path = output_dir / f"result_{setting_idx}_{tag}.pt"
        torch.save(result, result_path)

        # Clear everything from GPU
        del (
            n2i_solver,
            swin_solver,
            sinogram,
            target,
            n2i_recon,
            fdk_recon,
            swin_recon,
            experiment,
            result,
        )
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print(f"[GPU {gpu_id}] Completed {setting_name}")
        result_queue.put((setting_idx, str(result_path), None))
    except Exception as e:
        import traceback

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[GPU {gpu_id}] Error in {setting_name}: {error_msg}")
        result_queue.put((setting_idx, None, error_msg))


def run(args):
    # Set multiprocessing start method
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = [
        ("Sparse angle", "SparseAngleLowDoseCTRecon"),
        ("Limited angle", "LimitedAngleLowDoseCTRecon"),
        ("Full angle", "LowDoseCTRecon"),
    ]

    # Convert args to dict for multiprocessing
    args_dict = {
        "dataset": args.dataset,
        "datafolder": args.datafolder,
        "batch_size": args.batch_size,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "test_index": args.test_index,
        "epochs_n2i": args.epochs_n2i,
        "epochs_vit": args.epochs_vit,
        "sino_splits": args.sino_splits,
        "hf_model_name": args.hf_model_name,
        "output_dir": args.output_dir,
        "seed": args.seed,
    }

    # Parallelize across GPUs using multiprocessing
    gpu_ids = args.gpus.split(",") if args.gpus else ["0", "1", "2"]
    result_queue = Queue()
    processes = []

    print(f"\nParallelizing {len(settings)} settings across GPUs: {gpu_ids}")
    print("=" * 70)

    for idx, (setting_name, exp_ctor_name) in enumerate(settings):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        p = Process(
            target=train_single_setting,
            args=(idx, setting_name, exp_ctor_name, args_dict, gpu_id, result_queue),
        )
        p.start()
        processes.append(p)

    # Collect results
    results_dict = {}
    for _ in range(len(settings)):
        setting_idx, result_path, error = result_queue.get()
        if error:
            print(f"\nError in setting {setting_idx}: {error}")
            # Kill all processes on error
            for p in processes:
                p.terminate()
            raise RuntimeError(f"Training failed: {error}")
        # Load result from disk
        result = torch.load(result_path)
        results_dict[setting_idx] = result

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("\nAll GPU processes completed successfully!")

    # Sort results by original order
    results = [results_dict[i] for i in sorted(results_dict.keys())]

    # Build 3x2 reconstruction panel with metrics
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    for r, row in enumerate(results):
        setting_name = row["setting"]
        target = row["target"]
        unet_recon = row["unet_n2i"]
        swin_recon = row["swin2sr"]

        # Compute metrics
        unet_metrics = compute_metrics(target, unet_recon)
        swin_metrics = compute_metrics(target, swin_recon)

        # UNet column
        axes[r, 0].imshow(to_np(unet_recon), cmap="gray")
        axes[r, 0].set_title(
            f"{setting_name} | UNet+N2I\nSSIM: {unet_metrics['ssim']:.4f}, PSNR: {unet_metrics['psnr']:.2f}dB",
            fontsize=12,
            fontweight="bold",
        )
        axes[r, 0].axis("off")

        # Swin2SR column
        axes[r, 1].imshow(to_np(swin_recon), cmap="gray")
        axes[r, 1].set_title(
            f"{setting_name} | Swin2SR\nSSIM: {swin_metrics['ssim']:.4f}, PSNR: {swin_metrics['psnr']:.2f}dB",
            fontsize=12,
            fontweight="bold",
        )
        axes[r, 1].axis("off")

        # Store metrics in results
        row["unet_metrics"] = unet_metrics
        row["swin_metrics"] = swin_metrics

    plt.tight_layout()
    fig_path = output_dir / "angle_3x2_reconstructions.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {fig_path}")

    tensor_path = output_dir / "angle_3x2_reconstructions.pt"
    torch.save(results, tensor_path)
    print(f"Saved tensors: {tensor_path}")

    # Print metrics summary
    print("\n" + "=" * 70)
    print("RECONSTRUCTION METRICS SUMMARY")
    print("=" * 70)
    for row in results:
        print(f"\n{row['setting'].upper()}:")
        print(
            f"  UNet+N2I  - SSIM: {row['unet_metrics']['ssim']:.6f}, PSNR: {row['unet_metrics']['psnr']:.2f} dB"
        )
        print(
            f"  Swin2SR   - SSIM: {row['swin_metrics']['ssim']:.6f}, PSNR: {row['swin_metrics']['psnr']:.2f} dB"
        )
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3-angle x 2-model comparison experiment"
    )
    parser.add_argument("--dataset", type=str, default="LIDC-IDRI")
    parser.add_argument("--datafolder", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="scripts/angleexp/results")
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2",
        help="Comma-separated GPU IDs (e.g., '0,1,2,3')",
    )
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--test-index", type=int, default=0)

    parser.add_argument("--epochs-n2i", type=int, default=100)
    parser.add_argument("--epochs-vit", type=int, default=1)
    parser.add_argument("--sino-splits", type=int, default=5)

    parser.add_argument(
        "--hf-model-name", type=str, default="caidas/swin2SR-classical-sr-x2-64"
    )

    run(parser.parse_args())
