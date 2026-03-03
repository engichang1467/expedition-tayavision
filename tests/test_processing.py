import pytest

from src.processing import TinyAyaVisionProcessor


@pytest.fixture
def processor(config):
    return TinyAyaVisionProcessor(config)


class TestTokenizer:
    """Tests for tokenizer special token setup."""

    def test_image_token_added(self, processor, config):
        """<image> should be a single token after adding."""
        encoded = processor.tokenizer.encode(
            config.image_token, add_special_tokens=False
        )
        assert len(encoded) == 1
        assert encoded[0] == processor.image_token_id

    def test_image_token_id_valid(self, processor):
        """Image token ID should be a valid integer."""
        assert isinstance(processor.image_token_id, int)
        assert processor.image_token_id > 0

    def test_placeholder_length(self, processor, config):
        """Placeholder contains num_tokens_after_shuffle tokens."""
        placeholder = processor.image_placeholder
        encoded = processor.tokenizer.encode(placeholder, add_special_tokens=False)
        assert len(encoded) == config.num_tokens_after_shuffle


class TestTextProcessing:
    """Tests for text-only processing."""

    def test_text_only(self, processor):
        """Text without image markers produces valid input_ids."""
        result = processor("Hello world")
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "pixel_values" not in result

    def test_batch_text(self, processor):
        """Batch of texts produces batched tensors."""
        result = processor(["Hello", "World"], padding=True)
        assert result["input_ids"].shape[0] == 2


class TestMultimodalProcessing:
    """Tests for combined text + image processing."""

    def test_image_placeholder_expansion(self, processor, config):
        """Single <image> marker expands to correct number of tokens."""
        result = processor("<image> Describe this.")
        n_image_tokens = (result["input_ids"] == processor.image_token_id).sum().item()
        assert n_image_tokens == config.num_tokens_after_shuffle

    def test_with_image(self, processor, config, dummy_image):
        """Processing with image returns pixel_values."""
        result = processor("<image> What is this?", images=dummy_image)
        assert "pixel_values" in result
        assert result["pixel_values"].ndim == 4  # (B, C, H, W)
        assert result["pixel_values"].shape[0] == 1

    def test_multiple_images(self, processor, config, dummy_image_batch):
        """Multiple images produce correctly shaped pixel_values."""
        result = processor(
            "<image> First. <image> Second.",
            images=dummy_image_batch,
        )
        assert result["pixel_values"].shape[0] == 2

        n_image_tokens = (result["input_ids"] == processor.image_token_id).sum().item()
        assert n_image_tokens == config.num_tokens_after_shuffle * 2

    def test_no_image_marker_no_pixel_values(self, processor, dummy_image):
        """Text without <image> marker but with image still returns pixel_values."""
        result = processor("No marker here.", images=dummy_image)
        assert "pixel_values" in result
        # No image tokens in text since no markers
        n_image_tokens = (result["input_ids"] == processor.image_token_id).sum().item()
        assert n_image_tokens == 0
