from dataclasses import dataclass


@dataclass
class TinyAyaVisionConfig:
    """Central configuration for the Tiny Aya Vision model."""

    # Vision encoder (SigLIP2-so400m-patch14-384)
    vision_model_name: str = "google/siglip2-so400m-patch14-384"
    vision_hidden_size: int = 1152
    image_size: int = 384
    patch_size: int = 14
    vision_grid_size: int = 27  # 384 // 14
    num_vision_tokens: int = 729  # 27 * 27

    # Pixel Shuffle (4x token compression)
    downsample_factor: int = 2
    padded_grid_size: int = 28  # ceil_to_even(27)
    num_tokens_after_shuffle: int = 196  # (28 // 2) ** 2
    pixel_shuffle_embed_dim: int = 4608  # 1152 * (2 ** 2)

    # Vision-Language connector (2-layer MLP with SwiGLU)
    connector_intermediate_size: int = 2048  # matches LLM hidden_size
    adapter_layer_norm_eps: float = 1e-6

    # LLM backbone (Tiny Aya Base)
    llm_model_name: str = "CohereLabs/tiny-aya-base"
    llm_hidden_size: int = 2048
    llm_vocab_size: int = 262144

    # Special tokens
    image_token: str = "<image>"

    # Inference defaults
    torch_dtype: str = "bfloat16"

    # Vision feature extraction
    vision_feature_layer: int = -1
    # "full" = all patches, "default" = crop CLS
    vision_feature_select_strategy: str = "full"
