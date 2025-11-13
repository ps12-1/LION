import torch
import torch.nn as nn
from typing import Optional
import os

from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType
from LION.utils.parameter import LIONParameter
import LION.CTtools.ct_geometry as ct

# Hugging Face Transformers
try:
    from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
except Exception as e:
    AutoImageProcessor = None
    Swin2SRForImageSuperResolution = None


class Swin2SR(LIONmodel):
    """Hugging Face Swin2SR wrapper for CT reconstruction within LION.

    This wraps a pretrained Swin2SR model and adapts inputs/outputs to
    LION's conventions for single-channel CT images.
    """

    def __init__(self, geometry: Optional[ct.Geometry], model_parameters: Optional[LIONModelParameter] = None):
        if geometry is None:
            raise ValueError("Geometry Parameters Required")

        super().__init__(model_parameters, geometry)

        if AutoImageProcessor is None or Swin2SRForImageSuperResolution is None:
            raise ImportError(
                " transformers is required for Swin2SR. Please install it: pip install transformers "
            )

        # Ensure HF cache directory is set
        cache_dir = self.model_parameters.hf_cache_dir if hasattr(self.model_parameters, "hf_cache_dir") else "/store/LION/ps2050/.cache/hf"
        if "HF_HOME" not in os.environ:
            os.environ["HF_HOME"] = cache_dir

        # HF artifacts
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_parameters.hf_model_name, cache_dir=cache_dir
        )
        self.model = Swin2SRForImageSuperResolution.from_pretrained(
            self.model_parameters.hf_model_name, cache_dir=cache_dir
        )
        
        # Set train/eval and parameter requires_grad according to flag
        if self.model_parameters.train_hf_backbone:
            self.model.train(True)
            for p in self.model.parameters():
                p.requires_grad = True
        else:
            self.model.train(False)
            for p in self.model.parameters():
                p.requires_grad = False

    @staticmethod
    def default_parameters() -> LIONModelParameter:
        params = LIONModelParameter()
        params.model_input_type = ModelInputType.IMAGE
        params.img_size = 256
        params.in_chans = 1
        params.out_chans = 1
        # HF repo id; change as needed (x2 upsampler variant)
        params.hf_model_name = "caidas/swin2SR-classical-sr-x2-64"
        # HF cache location
        params.hf_cache_dir = "/store/LION/ps2050/.cache/hf"
        # Pre/post-processing knobs
        params.do_rescale = False
        params.rescale_factor = 1.0 / 255.0
        params.do_pad = True
        params.size_divisor = 8
        # Training flag: set True to finetune HF model
        params.train_hf_backbone = False
        return params

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print("Conde, Marcos V., et al.")
            print('"Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration."')
            print("AIM 2022 Challenge on Super-Resolution of Compressed Image and Video (2022).")
        elif cite_format == "bib":
            string = """
            @inproceedings{conde2022swin2sr,
              title={Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration},
              author={Conde, Marcos V. and Choi, Ui-Jin and Burchi, Maxime and Timofte, Radu},
              booktitle={Proceedings of the AIM 2022 Challenge on Super-Resolution of Compressed Image and Video},
              year={2022}
            }
            """
            print(string)
        else:
            raise AttributeError('cite_format not understood, only "MLA" and "bib" supported')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, 1, H, W) with pixel intensities.

        Returns:
            Tensor of shape (B, 1, H, W), reconstructed/restored image.
        """
        assert x.dim() == 4 and x.size(1) == 1, "Swin2SR expects single-channel CT inputs"

        b, _, h, w = x.shape

        # Swin2SR pretrained models are typically RGB; replicate channel to 3
        x_rgb = x.repeat(1, 3, 1, 1)

        # Prepare inputs for HF model
        # Convert to [0,1] if requested; assume input already normalised otherwise
        pixel_values = x_rgb
        if self.model_parameters.do_rescale:
            pixel_values = pixel_values * 1.0  # keep tensor type

        # HF processor handles padding to size_divisor when do_pad=True
        inputs = self.processor(
            images=pixel_values,
            do_rescale=self.model_parameters.do_rescale,
            rescale_factor=self.model_parameters.rescale_factor,
            do_pad=self.model_parameters.do_pad,
            size_divisor=self.model_parameters.size_divisor,
            return_tensors="pt",
            data_format="channels_first",
        )

        # Ensure device alignment
        device = x.device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        self.model.to(device)

        # Enable gradient tracking unless explicitly disabled by user
        # Respect module.training, but also force-enable if any parameter requires grad
        requires_grad_any = any(p.requires_grad for p in self.model.parameters())
        with torch.set_grad_enabled(self.training or requires_grad_any):
            outputs = self.model(**inputs)

        recon = outputs.reconstruction  # (B, C=3, H*, W*)

        # Downmix back to single channel
        recon_gray = recon.mean(dim=1, keepdim=True)

        # If processor padded or upsampled, resize/crop back to original H, W
        if recon_gray.shape[-2] != h or recon_gray.shape[-1] != w:
            recon_gray = nn.functional.interpolate(recon_gray, size=(h, w), mode="bilinear", align_corners=False)

        # Ensure output participates in autograd when backbone is trainable
        if requires_grad_any and not recon_gray.requires_grad:
            recon_gray = recon_gray.requires_grad_()

        return recon_gray



