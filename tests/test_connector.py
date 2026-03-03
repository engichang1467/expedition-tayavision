import pytest
import torch

from src.connector import MultiModalProjector


@pytest.fixture
def connector(config):
    return MultiModalProjector(config)


class TestPixelShuffle:
    """Tests for the Pixel Shuffle token compression."""

    def test_standard_input_shape(self, connector, config):
        """729 tokens (27x27 grid) -> 196 tokens (14x14 grid)."""
        x = torch.randn(1, 729, config.vision_hidden_size)
        out = connector.pixel_shuffle(x)
        expected = (1, config.num_tokens_after_shuffle, config.pixel_shuffle_embed_dim)
        assert out.shape == expected

    def test_even_grid_no_padding_needed(self, connector):
        """784 tokens (28x28 grid) should also work without padding."""
        x = torch.randn(1, 784, 1152)
        out = connector.pixel_shuffle(x)
        assert out.shape == (1, 196, 4608)

    def test_batch_dimension(self, connector, config):
        """Batch of 4 images processes correctly."""
        x = torch.randn(4, 729, config.vision_hidden_size)
        out = connector.pixel_shuffle(x)
        assert out.shape[0] == 4
        assert out.shape[1] == config.num_tokens_after_shuffle

    def test_compression_ratio(self, connector, config):
        """Token count reduces by ~4x (729 -> 196)."""
        x = torch.randn(1, 729, config.vision_hidden_size)
        out = connector.pixel_shuffle(x)
        ratio = 729 / out.shape[1]
        assert 3.5 < ratio < 4.5  # approximately 4x


class TestMultiModalProjector:
    """Tests for the full connector (Pixel Shuffle + SwiGLU MLP)."""

    def test_output_shape(self, connector, config):
        """Full connector maps to LLM hidden size."""
        x = torch.randn(1, 729, config.vision_hidden_size)
        out = connector(x)
        assert out.shape == (1, config.num_tokens_after_shuffle, config.llm_hidden_size)

    def test_output_dtype(self, connector, config):
        """Output dtype matches input dtype."""
        x = torch.randn(1, 729, config.vision_hidden_size)
        out = connector(x)
        assert out.dtype == x.dtype

    def test_parameter_count(self, connector):
        """Connector has ~11.5M parameters."""
        n_params = sum(p.numel() for p in connector.parameters())
        assert 10_000_000 < n_params < 13_000_000

    def test_all_parameters_trainable(self, connector):
        """All connector parameters should require gradients."""
        for name, param in connector.named_parameters():
            assert param.requires_grad, f"{name} should be trainable"

    def test_gradient_flow(self, connector, config):
        """Gradients flow through the connector."""
        x = torch.randn(1, 729, config.vision_hidden_size, requires_grad=True)
        out = connector(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        for param in connector.parameters():
            assert param.grad is not None

    def test_batch_processing(self, connector, config):
        """Batch of images processes correctly."""
        x = torch.randn(3, 729, config.vision_hidden_size)
        out = connector(x)
        assert out.shape == (3, config.num_tokens_after_shuffle, config.llm_hidden_size)
