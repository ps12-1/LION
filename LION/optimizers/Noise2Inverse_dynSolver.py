from __future__ import annotations

from typing import Optional, List

import numpy as np
import torch

import LION.CTtools.ct_utils as ct
from LION.CTtools.ct_geometry import Geometry
from LION.classical_algorithms.fdk import fdk
from LION.models.LIONmodel import LIONmodel
from LION.utils.parameter import LIONParameter
from LION.optimizers.Noise2InverseSolver import Noise2InverseSolver
from LION.metrics.psnr import PSNR


class Noisier2InverseParams(LIONParameter):
    def __init__(self):
        super().__init__()
        # Inherit Noise2Inverse defaults conceptually
        self.sino_split_count: int = 4
        self.recon_fn = fdk
        self.cali_J: List[List[int]] = Noise2InverseSolver.X_one_strategy(
            self.sino_split_count
        )
        # Added-noise parameters (measurement domain)
        self.I0: int = 1000
        self.sigma: float = 5.0
        self.cross_talk: float = 0.05
        self.enable_gradients: bool = False
        # Loss space: 'measurement' | 'image'
        self.loss_space: str = "measurement"
        # Measurement-space weighting: 'none' | 'poisson' | 'gaussian' | 'inv_var'
        self.measurement_weighting: str = "none"
        # If True, target measurement is taken from ORIGINAL sino; otherwise from augmented sino
        self.use_original_for_target: bool = True
        # Probability to apply augmentation per batch (1.0 = always)
        self.augment_prob: float = 1.0
        # If True, compute loss over all splits; otherwise use held-out J only
        self.use_all_splits_for_loss: bool = False
        # Dynamic model selection parameters
        self.validation_interval: int = 1
        self.early_stopping_patience: int = 10
        self.select_by: str = "psnr"  # or "loss"


class Noisier2InverseSolver(Noise2InverseSolver):
    def __init__(
        self,
        model: LIONmodel,
        optimizer,
        loss_fn,
        solver_params: Optional[Noisier2InverseParams] = None,
        geometry: Geometry = None,
        verbose: bool = True,
        device: torch.device = None,
    ) -> None:
        if solver_params is None:
            solver_params = self.default_parameters()
        super().__init__(
            model,
            optimizer,
            loss_fn,
            solver_params,
            geometry,
            verbose,
            device,
        )

    @staticmethod
    def default_parameters() -> Noisier2InverseParams:
        return Noisier2InverseParams()

    def _add_noisier_measurements(self, sinos: torch.Tensor) -> torch.Tensor:
        params: Noisier2InverseParams = self.solver_params  # type: ignore
        return ct.sinogram_add_noise(
            sinos,
            I0=params.I0,
            sigma=params.sigma,
            cross_talk=params.cross_talk,
            flat_field=None,
            dark_field=None,
            enable_gradients=params.enable_gradients,
        )

    def _compute_measurement_weights(self, sino_subset: torch.Tensor) -> torch.Tensor:
        params: Noisier2InverseParams = self.solver_params  # type: ignore
        eps = 1e-8
        if params.measurement_weighting == "none":
            return torch.ones_like(sino_subset)
        elif params.measurement_weighting == "gaussian":
            var = params.sigma * params.sigma
            return torch.full_like(sino_subset, 1.0 / (var + eps))
        elif params.measurement_weighting == "poisson":
            # Approximate counts from log-domain measurement (heuristic consistent with ct_utils)
            # scale with I0 * exp(-proj/max_val)
            max_val = torch.amax(sino_subset.detach()) + eps
            counts = params.I0 * torch.exp(-sino_subset / max_val)
            var = torch.clamp(counts, min=eps)
            return 1.0 / var
        elif params.measurement_weighting == "inv_var":
            # If user pre-normalized to variance map, assume sino_subset already encodes variance in channel 1
            # For lack of explicit channeling in current pipeline, fall back to ones
            return torch.ones_like(sino_subset)
        else:
            return torch.ones_like(sino_subset)

    def mini_batch_step(self, sinos, targets):
        params: Noisier2InverseParams = self.solver_params  # type: ignore

        # Optionally create a noisier copy of the given measurements (measurement-space augmentation)
        if (
            params.augment_prob >= 1.0
            or torch.rand(1, device=self.device) < params.augment_prob
        ):
            sinos_noisier = self._add_noisier_measurements(sinos)
        else:
            sinos_noisier = sinos

        # Reconstruct sub-splits from the augmented measurements
        noisy_sub_recons = self._calculate_noisy_sub_recons(sinos_noisier)
        # b, split, c, w, h

        loss = torch.zeros(len(self.cali_J), device=self.device)
        for i, J in enumerate(self.cali_J):
            J_zero_indexing = [j - 1 for j in J]
            J_c = [
                n for n in np.arange(self.sino_split_count) if n not in J_zero_indexing
            ]

            # Mean reconstructions
            target_recons = torch.mean(
                noisy_sub_recons[:, J_zero_indexing, :, :, :], dim=1
            )
            input_recons = torch.mean(noisy_sub_recons[:, J_c, :, :, :], dim=1)

            # Model output in image-space
            output_images = self.model(input_recons)

            if params.loss_space == "measurement":
                # Compute measurement-space loss: forward project output and compare to measurement(s)
                # Select which splits to supervise
                supervise_splits = (
                    list(range(self.sino_split_count))
                    if params.use_all_splits_for_loss
                    else J_zero_indexing
                )
                proj_loss = 0.0
                denom = 0.0
                for j in supervise_splits:
                    Aj = self.sub_ops[j]
                    projected = Aj(output_images)
                    # Choose target measurement domain (original or augmented)
                    ref_sinos = (
                        sinos if params.use_original_for_target else sinos_noisier
                    )
                    sino_j = ref_sinos[:, :, j :: self.sino_split_count, :]
                    # Optional weighting
                    w = self._compute_measurement_weights(sino_j)
                    diff = projected - sino_j
                    # Weighted MSE
                    proj_loss = proj_loss + torch.mean(w * diff * diff)
                    denom = denom + 1.0
                loss[i] = proj_loss / max(denom, 1.0)
            else:
                # Fallback: image-space MSE between model output and target reconstructions
                loss[i] = self.loss_fn(output_images, target_recons)

        return loss.sum() / len(self.cali_J)

    def reconstruct(self, sinos):
        # One-step inference: reconstruct full FDK once and pass through model
        base_recon = self.recon_fn(sinos, self.op)
        return self.model(base_recon)

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format.lower() in ("mla",):
            print(
                "Boosting Noise2Inverse via Enhanced Model Selection for Denoising Computed Tomography Data"
            )
        elif cite_format.lower() in ("bib",):
            print(
                "@article{noisier2inverse2025, title={Boosting Noise2Inverse via Enhanced Model Selection for Denoising Computed Tomography Data}, author={...}, journal={arXiv}, year={2025}}"
            )
        else:
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )

    # --- Dynamic model selection / validation ---
    def set_validation(
        self,
        validation_loader,
        validation_freq=1,
        validation_fn=None,
        validation_fname=None,
        save_folder=None,
    ):
        super().set_validation(
            validation_loader,
            validation_freq,
            validation_fn,
            validation_fname,
            save_folder,
        )
        # Ensure we have PSNR for selection if requested
        if self.solver_params.select_by == "psnr":  # type: ignore
            self._psnr_metric = PSNR()
        else:
            self._psnr_metric = None

    def validate(self):
        # Override to compute either loss or PSNR, according to select_by
        assert self.validation_loader is not None
        select_by = getattr(self.solver_params, "select_by", "psnr")
        self.model.eval()
        vals = []
        with torch.no_grad():
            for sinos, targets in self.validation_loader:
                # One-step inference path
                outputs = self.reconstruct(sinos.to(self.device))
                if select_by == "psnr":
                    psnr_val = self._psnr_metric(outputs, targets.to(self.device), reduce="mean")  # type: ignore
                    vals.append(psnr_val.item())
                else:
                    vals.append(self.loss_fn(outputs, targets.to(self.device)).item())
        mean_val = float(np.mean(vals)) if len(vals) > 0 else 0.0
        return mean_val

    def train(self, n_epochs):
        # Extend parent training with dynamic best-model tracking and early stopping
        assert n_epochs > 0
        self.check_training_ready()

        # Setup tracking
        select_by = getattr(self.solver_params, "select_by", "psnr")
        patience = getattr(self.solver_params, "early_stopping_patience", 10)
        best_score = -float("inf") if select_by == "psnr" else float("inf")
        worse = lambda cur, best: cur <= best if select_by == "psnr" else cur >= best
        no_improve = 0

        # Initialize storage for losses as in base class
        if self.do_load_checkpoint:
            self.current_epoch = self.load_checkpoint()
            self.train_loss = np.append(self.train_loss, np.zeros((n_epochs)))
        else:
            self.train_loss = np.zeros(n_epochs)

        if self.check_validation_ready() == 0:
            self.validation_loss = np.zeros((n_epochs))
        if self.validation_loader is None:
            self.validation_loss = None

        self.model.train()
        while self.current_epoch < n_epochs:
            self.train_loss[self.current_epoch] = self.train_step()

            # Periodic validation and dynamic selection
            if (
                self.validation_loader is not None
                and (self.current_epoch + 1)
                % getattr(self.solver_params, "validation_interval", 1)
                == 0
            ):
                score = self.validate()
                if self.validation_loss is not None:
                    self.validation_loss[self.current_epoch] = score
                if select_by == "psnr":
                    improved = score > best_score
                else:
                    improved = score < best_score
                if improved:
                    best_score = score
                    no_improve = 0
                    # Save "best" checkpoint distinctly if user configured validation_fname
                    if (
                        self.validation_fname is not None
                        and self.validation_save_folder is not None
                    ):
                        self.model.save(
                            self.validation_save_folder.joinpath(self.validation_fname),
                            epoch=self.current_epoch,
                            training=self.metadata,
                            loss=score,
                            dataset=self.dataset_param,
                        )
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        if self.verbose:
                            print("Early stopping triggered.")
                        break

            if (self.current_epoch + 1) % self.checkpoint_freq == 0:
                self.save_checkpoint(self.current_epoch)

            if self.validation_loader is None and self.verbose:
                print(
                    f"Epoch {self.current_epoch+1} - Training loss: {self.train_loss[self.current_epoch]}"
                )

            self.current_epoch += 1
