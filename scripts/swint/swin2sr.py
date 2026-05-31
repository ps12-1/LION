# %% Training example for Swin2SR wrapped in LION

# Standard imports
import matplotlib.pyplot as plt
import pathlib
from skimage.metrics import structural_similarity as ssim
import os

# Torch imports
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

# LION imports
from LION.models.CNNs.huggingface import Swin2SR
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
from LION.optimizers.SupervisedSolver import SupervisedSolver

os.environ["HF_HOME"] = "/store/LION/ps2050/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/store/LION/ps2050/.cache/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/store/LION/ps2050/.cache/huggingface/datasets"


def my_ssim(x, y):
    x = x.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    return ssim(x, y, data_range=x.max() - x.min())


# %% Choose device
# Set Hugging Face cache before any HF calls
os.environ.setdefault("HF_HOME", "/store/LION/ps2050/.cache/huggingface")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

# Paths
savefolder = pathlib.Path("/store/LION/ps2050/trained_models/swin2sr/")
savefolder.mkdir(parents=True, exist_ok=True)
final_result_fname = "Swin2SR.pt"
checkpoint_fname = "Swin2SR_check_*.pt"
validation_fname = "Swin2SR_min_val.pt"

# %% Experiment
experiment = ct_experiments.LowDoseCTRecon(dataset="LIDC-IDRI")

# %% Dataset
lidc_dataset = experiment.get_training_dataset()
lidc_dataset_val = experiment.get_validation_dataset()
lidc_dataset_test = experiment.get_testing_dataset()

# Small subset for example. Remove to use full dataset
indices = torch.arange(1)
lidc_dataset = data_utils.Subset(lidc_dataset, indices)
lidc_dataset_val = data_utils.Subset(lidc_dataset_val, indices)

# %% DataLoaders
batch_size = 1
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
lidc_validation = DataLoader(lidc_dataset_val, batch_size, shuffle=False)
lidc_test = DataLoader(lidc_dataset_test, batch_size, shuffle=False)

# %% Model
default_parameters = Swin2SR.default_parameters()

# Example: switch to another HF repo if needed
# default_parameters.hf_model_name = "caidas/swin2SR-classical-sr-x4-48"
default_parameters.train_hf_backbone = True  # finetune to enable gradients
default_parameters.hf_cache_dir = "/store/LION/ps2050/.cache/huggingface"

model = Swin2SR(experiment.geometry, default_parameters)
model.to(device)

print("Model Citation:")
model.cite()
print("\nBibTeX:")
model.cite("bib")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Fail fast if no trainable parameters are detected (prevents silent no-grad losses)
if trainable_params == 0:
    raise RuntimeError(
        "No trainable parameters found. Ensure default_parameters.train_hf_backbone=True before model init."
    )

# %% Optimizer
train_param = LIONParameter()
loss_fcn = torch.nn.MSELoss()
train_param.optimiser = "adam"
train_param.epochs = 5
train_param.learning_rate = 1e-4
train_param.betas = (0.9, 0.999)
train_param.loss = "MSELoss"
train_param.weight_decay = 0.0

optimiser = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=train_param.learning_rate,
    betas=train_param.betas,
    weight_decay=train_param.weight_decay,
)

# Optional scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimiser, T_max=train_param.epochs, eta_min=1e-6
)

# %% Train
solver = SupervisedSolver(
    model, optimiser, loss_fcn, verbose=True, save_folder=savefolder
)

solver.set_training(lidc_dataloader)
solver.set_validation(lidc_validation, 2, validation_fname=validation_fname)
solver.set_testing(lidc_test, my_ssim)

solver.set_checkpointing(
    checkpoint_fname, 2, load_checkpoint_if_exists=False, save_folder=savefolder
)

solver.scheduler = scheduler

print("Starting training...")
solver.train(train_param.epochs)

solver.clean_checkpoints()

solver.save_final_results(final_result_fname, savefolder)

print("Testing model...")
test_results = solver.test()
if hasattr(test_results, "__iter__") and not isinstance(test_results, str):
    print(f"Test SSIM (mean): {test_results.mean():.4f}")
    print(f"Test SSIM (all): {test_results}")
else:
    print(f"Test SSIM: {test_results:.4f}")

# %% Plots
plt.figure(figsize=(10, 6))
plt.semilogy(solver.train_loss[1:])
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (log scale)")
plt.grid(True)
plt.savefig(savefolder / "training_loss.png", dpi=300, bbox_inches="tight")
plt.show()

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
