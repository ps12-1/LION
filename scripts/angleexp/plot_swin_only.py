"""
Plot Swin Scratch reconstructions only with individual pixel ranges per image.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt


def to_np(tensor):
    """Convert tensor to numpy, handling various input types."""
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


def run(args):
    swin_path = Path(args.swin_pt)
    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not swin_path.exists():
        raise FileNotFoundError(f"Swin file not found: {swin_path}")

    print(f"Loading Swin results: {swin_path}")
    swin_results = torch.load(swin_path, map_location="cpu")

    if len(swin_results) < 3:
        raise ValueError(
            "Expected at least 3 entries in .pt file (sparse/limited/full)."
        )

    # Organize by setting name
    swin_by_setting = {str(d.get("setting", i)): d for i, d in enumerate(swin_results)}
    preferred_order = ["Sparse angle", "Limited angle", "Full angle"]

    rows = []
    for s in preferred_order:
        if s in swin_by_setting:
            rows.append((s, swin_by_setting[s]))

    if len(rows) < 3:
        # Fallback to index-based
        rows = [
            (str(d.get("setting", f"Setting {i+1}")), d)
            for i, d in enumerate(swin_results[:3])
        ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    print("\nSwin Scratch Reconstruction Pixel Ranges:")
    print("=" * 60)

    for col, (setting, swin_row) in enumerate(rows):
        swin_img = to_np(swin_row["reconstruction"]).squeeze()
        swin_metrics = swin_row.get("metrics", {})

        vmin = float(np.min(swin_img))
        vmax = float(np.max(swin_img))

        print(f"\n{setting}:")
        print(f"  Min: {vmin:.6f}")
        print(f"  Max: {vmax:.6f}")
        print(f"  SSIM: {swin_metrics.get('ssim', float('nan')):.4f}")
        print(f"  PSNR: {swin_metrics.get('psnr', float('nan')):.2f} dB")

        axes[col].imshow(swin_img, cmap="gray", vmin=vmin, vmax=vmax)
        axes[col].set_title(
            f"{setting}\nSSIM: {swin_metrics.get('ssim', float('nan')):.4f}, "
            f"PSNR: {swin_metrics.get('psnr', float('nan')):.2f}dB\n"
            f"Range: [{vmin:.4f}, {vmax:.4f}]",
            fontsize=11,
            fontweight="bold",
        )
        axes[col].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 60)
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Plot Swin Scratch reconstructions with individual pixel ranges"
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
        help="Path to saved Swin results .pt file",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default=str(
            script_dir
            / "scripts"
            / "angleexp"
            / "results_swin"
            / "swin_scratch_individual_ranges.png"
        ),
        help="Output PNG path",
    )
    args = parser.parse_args()
    run(args)
