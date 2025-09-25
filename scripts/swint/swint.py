# %% This example shows how to train Swin Transformer for CT reconstruction in a supervised setting.

# %% Imports

# Standard imports
import matplotlib.pyplot as plt
import pathlib
from skimage.metrics import structural_similarity as ssim

# Torch imports
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

# Lion imports
from LION.models.CNNs.SwinTransformer import SwinTransformer
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
from LION.optimizers.SupervisedSolver import SupervisedSolver


def my_ssim(x, y):
    """Custom SSIM function for evaluation."""
    x = x.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    return ssim(x, y, data_range=x.max() - x.min())


# %%
# % Chose device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

# Define your data paths
savefolder = pathlib.Path("/store/LION/ps2050/trained_models/swin_transformer/")
savefolder.mkdir(parents=True, exist_ok=True)
final_result_fname = "SwinTransformer.pt"
checkpoint_fname = "SwinTransformer_check_*.pt"
validation_fname = "SwinTransformer_min_val.pt"

# %% Define experiment
experiment = ct_experiments.LowDoseCTRecon(dataset="LIDC-IDRI")
# experiment = ct_experiments.ExtremeLowDoseCTRecon(dataset="LIDC-IDRI")

# %% Dataset
lidc_dataset = experiment.get_training_dataset()
lidc_dataset_val = experiment.get_validation_dataset()
lidc_dataset_test = experiment.get_testing_dataset()

# smaller dataset for example. Remove this for full dataset
indices = torch.arange(1)  # Use 10 samples for demonstration ##use 1 sample
lidc_dataset = data_utils.Subset(lidc_dataset, indices)
lidc_dataset_val = data_utils.Subset(lidc_dataset_val, indices)

# %% Define DataLoader
batch_size = 1  # Smaller batch size for Swin Transformer
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
lidc_validation = DataLoader(lidc_dataset_val, batch_size, shuffle=False)
lidc_test = DataLoader(lidc_dataset_test, batch_size, shuffle=False)

# %% Model
# Create Swin Transformer model with default parameters
default_parameters = SwinTransformer.default_parameters()

# Adjust parameters for CT reconstruction
default_parameters.img_size = 512  # Standard CT image size  ##changed from 256 to 512
default_parameters.patch_size = 4  # Smaller patches for better detail
default_parameters.embed_dim = 96  # Base embedding dimension
default_parameters.depths = [2, 2, 6, 2]  # Number of blocks in each stage
default_parameters.num_heads = [3, 6, 12, 24]  # Number of attention heads
default_parameters.window_size = 7  # Window size for attention
default_parameters.mlp_ratio = 4.0  # MLP expansion ratio
default_parameters.drop_rate = 0.0  # Dropout rate
default_parameters.attn_drop_rate = 0.0  # Attention dropout rate
default_parameters.drop_path_rate = 0.1  # Stochastic depth rate

model = SwinTransformer(experiment.geometry, default_parameters)
model.to(device)

# Print model citation
print("Model Citation:")
model.cite()
print("\nBibTeX:")
model.cite("bib")

# Print model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# %% Optimizer
train_param = LIONParameter()

# Loss function
loss_fcn = torch.nn.MSELoss()
train_param.optimiser = "adam"

# Optimizer parameters
train_param.epochs = 50  # Fewer epochs for demonstration
train_param.learning_rate = 1e-4
train_param.betas = (0.9, 0.999)
train_param.loss = "MSELoss"
train_param.weight_decay = 0.01  # L2 regularization

optimiser = torch.optim.Adam(
    model.parameters(),
    lr=train_param.learning_rate,
    betas=train_param.betas,
    weight_decay=train_param.weight_decay,
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimiser, T_max=train_param.epochs, eta_min=1e-6
)

# %% Train
# Create solver
solver = SupervisedSolver(
    model, optimiser, loss_fcn, verbose=True, save_folder=savefolder
)

# Set data
solver.set_training(lidc_dataloader)
solver.set_validation(lidc_validation, 5, validation_fname=validation_fname)
solver.set_testing(lidc_test, my_ssim)

# Set checkpointing procedure
solver.set_checkpointing(
    checkpoint_fname, 5, load_checkpoint_if_exists=False, save_folder=savefolder
)

# Add learning rate scheduler to solver
solver.scheduler = scheduler

# Train
print("Starting training...")
solver.train(train_param.epochs)

# Delete checkpoints if finished
solver.clean_checkpoints()

# Save final result
solver.save_final_results(final_result_fname, savefolder)

# Test
print("Testing model...")
test_results = solver.test()
print(f"Test SSIM: {test_results:.4f}")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.semilogy(solver.train_loss[1:])
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (log scale)")
plt.grid(True)
plt.savefig(savefolder / "training_loss.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot validation loss if available
if hasattr(solver, "val_loss") and solver.val_loss:
    plt.figure(figsize=(10, 6))
    plt.semilogy(solver.val_loss[1:])
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.grid(True)
    plt.savefig(savefolder / "validation_loss.png", dpi=300, bbox_inches="tight")
    plt.show()

print("Training completed successfully!")
print(f"Model saved to: {savefolder / final_result_fname}")
print(f"Training plots saved to: {savefolder}")
