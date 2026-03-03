import pytest
import torch

from config.model_config import TinyAyaVisionConfig
from src.vision_encoder import VisionEncoder
from tests.conftest import requires_gpu


@requires_gpu
class TestVisionEncoder:
    """Tests for the frozen SigLIP2 vision encoder.

    These tests require GPU and download the SigLIP2 model (~0.8GB).
    Run with: pytest tests/test_vision_encoder.py -v
    """

    @pytest.fixture(scope="class")
    def encoder(self):
        config = TinyAyaVisionConfig()
        encoder = VisionEncoder(config)
        return encoder.cuda()

    def test_loads_successfully(self, encoder):
        """Vision model loads without errors."""
        assert encoder.vision_model is not None

    def test_all_params_frozen(self, encoder):
        """All vision encoder parameters should be frozen."""
        for name, param in encoder.named_parameters():
            assert not param.requires_grad, f"{name} should be frozen"

    def test_output_shape_single(self, encoder):
        """Single image produces (1, 729, 1152) features."""
        x = torch.randn(1, 3, 384, 384, device="cuda", dtype=torch.bfloat16)
        out = encoder(x)
        assert out.shape == (1, 729, 1152)

    def test_output_shape_batch(self, encoder):
        """Batch of 2 images produces (2, 729, 1152) features."""
        x = torch.randn(2, 3, 384, 384, device="cuda", dtype=torch.bfloat16)
        out = encoder(x)
        assert out.shape == (2, 729, 1152)

    def test_deterministic(self, encoder):
        """Same input produces same output (frozen model, no dropout at eval)."""
        encoder.eval()
        x = torch.randn(1, 3, 384, 384, device="cuda", dtype=torch.bfloat16)
        out1 = encoder(x)
        out2 = encoder(x)
        assert torch.allclose(out1, out2)

    def test_no_gradient_computation(self, encoder):
        """Forward pass should not compute gradients."""
        x = torch.randn(1, 3, 384, 384, device="cuda", dtype=torch.bfloat16)
        out = encoder(x)
        assert not out.requires_grad
