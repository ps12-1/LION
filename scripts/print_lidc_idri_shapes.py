import torch
from LION.experiments.ct_experiments import LowDoseCTRecon


def print_lidc_idri_shapes(dataset_name="LIDC-IDRI", num_samples=5, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment = LowDoseCTRecon(dataset=dataset_name)
    train_dataset = experiment.get_training_dataset()
    val_dataset = experiment.get_validation_dataset()

    print(f"--- Training set ({dataset_name}) ---")
    for i in range(min(num_samples, len(train_dataset))):
        sinogram, ground_truth = train_dataset[i]
        print(
            f"[Train Sample {i}] Sinogram shape: {sinogram.shape}, Ground truth shape: {ground_truth.shape}"
        )

    print(f"\n--- Validation set ({dataset_name}) ---")
    for i in range(min(num_samples, len(val_dataset))):
        sinogram, ground_truth = val_dataset[i]
        print(
            f"[Val Sample {i}] Sinogram shape: {sinogram.shape}, Ground truth shape: {ground_truth.shape}"
        )


if __name__ == "__main__":
    print_lidc_idri_shapes()
