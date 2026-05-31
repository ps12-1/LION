## run_swin_noise2inverse.py

Trains a **Swin Transformer denoiser** using the **Noise2Inverse** self-supervised approach.
Creates a **3x1 reconstruction panel** showing results across sparse/limited/full angle settings.

### Features

- **Swin Transformer backbone** (microsoft/swin-base-patch4-window7-224)
- **Noise2Inverse training** (self-supervised, no clean labels needed)
- **Single GPU execution** (simpler than multi-GPU version)
- **3x1 output** (3 angle settings, 1 model)

### Run

```bash
# Default settings (100 epochs, 64 samples, single GPU)
python scripts/angleexp/run_swin_noise2inverse.py --gpu 0

# Quick test
python scripts/angleexp/run_swin_noise2inverse.py --gpu 0 --epochs 20 --train-samples 16

# High quality (200 epochs)
python scripts/angleexp/run_swin_noise2inverse.py --gpu 0 --epochs 200 --train-samples 128
```

### Output

- `swin_3x1_reconstructions.png` - 3x1 panel with SSIM/PSNR metrics
- `swin_3x1_reconstructions.pt` - Full tensor data and metrics

### Parameters

- `--gpu` (int): GPU ID to use (default: 0)
- `--epochs` (int): Training epochs (default: 100)
- `--train-samples` (int): Training samples per setting (default: 64)
- `--sino-splits` (int): Sinogram splits for N2I (default: 5)
- `--batch-size` (int): Batch size (default: 1, per N2I paper)

### Notes

- Uses Swin Transformer as denoising backbone
- Self-supervised training with split sinograms
- Same Noise2Inverse approach as UNet version
- Single GPU runs sequentially through angle settings
- Results shown in 3x1 format for clean comparison
