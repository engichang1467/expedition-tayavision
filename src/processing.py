import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from config.model_config import TinyAyaVisionConfig


class TinyAyaVisionProcessor:
    """Combined processor for Tiny Aya Vision multimodal inputs.

    Handles both image preprocessing (via SiglipImageProcessor) and text
    tokenization (via CohereTokenizer), inserting the correct number of
    <image> placeholder tokens per image.
    """

    def __init__(self, config: TinyAyaVisionConfig):
        self.config = config
        self.image_processor = AutoImageProcessor.from_pretrained(
            config.vision_model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name)

        # Add the <image> special token
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [config.image_token]}
        )
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(config.image_token)

    @property
    def image_placeholder(self) -> str:
        """The string of <image> tokens to insert per image."""
        return self.config.image_token * self.config.num_tokens_after_shuffle

    def __call__(
        self,
        text: str | list[str],
        images: Image.Image | list[Image.Image] | None = None,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """Process text and optional images into model inputs.

        The text should contain `<image>` markers where images should be
        inserted. Each `<image>` marker is expanded to num_tokens_after_shuffle
        (196) placeholder tokens.

        Args:
            text: Input text or list of texts. Use "<image>" as image placeholder.
            images: Optional PIL Image(s) corresponding to <image> markers.
            padding: Padding strategy.
            truncation: Whether to truncate.
            max_length: Maximum sequence length.
            return_tensors: Output tensor format.

        Returns:
            Dict with "input_ids", "attention_mask", and optionally "pixel_values".
        """
        if isinstance(text, str):
            text = [text]

        # Expand each single <image> marker into the full placeholder sequence
        expanded_text = []
        for t in text:
            expanded = t.replace(self.config.image_token, self.image_placeholder)
            expanded_text.append(expanded)

        # Tokenize
        text_inputs = self.tokenizer(
            expanded_text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )

        result = dict(text_inputs)

        # Process images
        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]
            image_inputs = self.image_processor(
                images=images, return_tensors=return_tensors
            )
            result["pixel_values"] = image_inputs["pixel_values"]

        return result
