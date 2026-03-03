import torch
from PIL import Image
from transformers import AutoImageProcessor

from config.model_config import TinyAyaVisionConfig


class ImageProcessor:
    """Preprocesses images for the SigLIP2 vision encoder.

    Wraps SiglipImageProcessor to handle loading, resizing, and normalization.
    """

    def __init__(self, config: TinyAyaVisionConfig):
        self.config = config
        self.processor = AutoImageProcessor.from_pretrained(config.vision_model_name)

    def __call__(
        self,
        images: Image.Image | list[Image.Image],
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """Process one or more PIL images into pixel_values tensors.

        Args:
            images: Single PIL Image or list of PIL Images.
            return_tensors: Tensor format (default "pt" for PyTorch).

        Returns:
            Dict with "pixel_values" tensor of shape (B, C, H, W).
        """
        if isinstance(images, Image.Image):
            images = [images]

        return self.processor(images=images, return_tensors=return_tensors)
