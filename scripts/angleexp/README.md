# angleexp

Scripts for angle-based experiment comparison:
- model 1: `UNet` trained with `Noise2InverseSolver` (self-supervised denoising)
- model 2: Hugging Face `Swin2SR` denoising transformer (frozen/inference-only)
- settings: sparse / limited / full angle
- output: 3x2 reconstruction panel with SSIM/PSNR metrics

## Run (Best Quality - ~30 mins with 3 GPUs)

Default settings (100 epochs, 64 training samples following Noise2Inverse paper):

```bash
python scripts/angleexp/run_angle_comparison.py --gpus 0,1,2
```

## Quick Test (5 mins)

For quick testing with fewer samples/epochs:

```bash
python scripts/angleexp/run_angle_comparison.py \
  --train-samples 16 \
  --val-samples 8 \
  --epochs-n2i 20 \
  --gpus 0,1,2
```

## Multi-GPU Parallelization

The script automatically parallelizes training across multiple GPUs:
- Each angle setting (sparse/limited/full) trains on a separate GPU
- Use `--gpus` to specify which GPUs to use (default: `0,1,2`)
- Example with 4 GPUs: `--gpus 0,1,2,3`
- Memory is automatically cleared after each training phase
- Results saved to disk to avoid multiprocessing issues

## Implementation Details

**Noise2Inverse (UNet):**
- Self-supervised denoising using split sinograms
- **100 epochs** (following original Noise2Inverse paper)
- **64 training samples** for better convergence and denoising
- Sino splits: 5 (divides all geometries evenly: 50, 60, 360)
- Batch size: 1 (per original paper)
- Learning rate: 1e-4 (Adam optimizer)

**Swin2SR (ViT):**
- Pretrained denoising transformer from HuggingFace
- Frozen/inference-only (no fine-tuning to save memory)
- Applied post-FDK reconstruction enhancement

## Output

- `angle_3x2_reconstructions.png` - Main comparison figure with SSIM/PSNR metrics
- `angle_3x2_reconstructions.pt` - Full tensor data and metrics
- Individual result files: `result_*.pt` (cleaned up after visualization)

## Notes

- Requires `transformers` for Swin2SR model
- Defaults follow Noise2Inverse original paper recommendations
- GPU memory ~22GB per process (fits on 24GB GPUs)
- Typical runtime: 30 mins with 3 GPUs, 100 epochs, 64 samples
- For even better quality: `--epochs-n2i 200 --train-samples 128`
