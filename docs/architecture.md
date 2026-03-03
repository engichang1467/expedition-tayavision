# Tiny Aya Vision — Architecture

## Overview

Tiny Aya Vision is a multilingual vision-language model (VLM) built by connecting a frozen SigLIP 2 vision encoder to the Tiny Aya Base language model via a learned Pixel Shuffle + SwiGLU MLP connector. The design targets strong multilingual VQA and captioning performance across 70+ languages, with emphasis on low-resource languages.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    TinyAyaVisionForConditionalGeneration             │
│                                                                      │
│  ┌─────────────┐    ┌────────────────────────┐    ┌───────────────┐  │
│  │ SigLIP2     │    │  MultiModalProjector   │    │ Tiny Aya Base │  │
│  │ so400m      │───>│  Pixel Shuffle + SwiGLU│───>│ Cohere2 LLM   │  │
│  │ (frozen)    │    │  (trainable, ~11.5M)   │    │ (3.35B)       │  │
│  └─────────────┘    └────────────────────────┘    └───────────────┘  │
│                                                           ↑          │
│                               Text tokens (with <image> placeholders)│
└──────────────────────────────────────────────────────────────────────┘
```

**Total trainable parameters (Phase 1):** ~11.5M (connector only — both the vision encoder and LLM are frozen during Phase 1 evaluation).

---

## Components

### 1. Vision Encoder — `src/vision_encoder.py`

**Model:** `google/siglip2-so400m-patch14-384`
**Class:** `SiglipVisionModel` (registered as `siglip`, not `siglip2`)
**Parameters:** ~400M (frozen)

SigLIP 2 is a contrastive vision encoder trained with sigmoid loss on image-text pairs. It uses a plain ViT architecture with no CLS token — the output is a flat sequence of patch embeddings.

| Property | Value |
|---|---|
| Input resolution | 384 × 384 |
| Patch size | 14 × 14 |
| Grid dimensions | 27 × 27 |
| Number of patches | 729 |
| Hidden size | 1152 |
| Output shape | `(B, 729, 1152)` |

All parameters are frozen (`requires_grad=False`) during Phase 1. Features are extracted from the last hidden state (`vision_feature_layer=-1`). Because SigLIP has no CLS token, the `"full"` strategy is used — all 729 patch embeddings are passed to the connector.

---

### 2. Connector — `src/connector.py`

**Class:** `MultiModalProjector`
**Parameters:** ~11.5M (trainable)
**Reference:** Adapted from `AyaVisionMultiModalProjector` in HuggingFace Transformers

The connector bridges the vision encoder's embedding space (dim 1152) to the LLM's embedding space (dim 2048) while reducing token count 4× via Pixel Shuffle.

#### 2a. Pixel Shuffle

Pixel Shuffle groups 2×2 spatial patch blocks and concatenates their embeddings along the feature dimension, trading spatial resolution for embedding depth.

**The grid padding problem:** SigLIP with a 384×384 image and patch size 14 produces a 27×27 grid. Since 27 is odd, a naïve 2×2 block grouping would leave a remainder. The connector handles this by zero-padding the spatial grid to 28×28 before applying the shuffle.

```
Input:  (B, 729, 1152)   # 27×27 grid, dim 1152
Reshape: (B, 27, 27, 1152)
Pad:     (B, 28, 28, 1152)  # pad 1 row and 1 col
Shuffle: (B, 196, 4608)     # 14×14 grid, dim 4608 (1152 × 4)
```

The shuffle itself follows the reshape-permute pattern from Aya Vision:

```python
# (B, H, W, D) → (B, H, W/dsf, D*dsf)  merge cols into channels
features = features.reshape(B, H, W//dsf, D*dsf)
features = features.permute(0, 2, 1, 3)             # swap H, W dims
# (B, W/dsf, H/dsf, D*dsf²)  merge rows into channels
features = features.reshape(B, W//dsf, H//dsf, -1)
features = features.permute(0, 2, 1, 3)             # restore H, W order
features = features.reshape(B, -1, features.shape[-1])  # flatten spatial
```

#### 2b. SwiGLU MLP

After Pixel Shuffle, a 2-layer MLP with SwiGLU activation projects the compressed features into the LLM's hidden dimension.

```
(B, 196, 4608)
→ LayerNorm(4608, eps=1e-6)
→ Linear(4608 → 2048)
→ SwiGLU: chunk into (x, gate) each of dim 1024; output = SiLU(gate) * x
→ Linear(1024 → 2048)
(B, 196, 2048)
```

`connector_intermediate_size=2048` is set equal to the LLM hidden size. After SwiGLU gating halves the dimension, the second linear maps 1024 → 2048.

**Full tensor trace through the connector:**

| Step | Shape | Notes |
|---|---|---|
| Vision encoder output | `(B, 729, 1152)` | 27×27 patches |
| After padding | `(B, 784, 1152)` | 28×28 patches |
| After pixel shuffle | `(B, 196, 4608)` | 14×14 patches |
| After LayerNorm | `(B, 196, 4608)` | |
| After Linear 1 | `(B, 196, 2048)` | |
| After SwiGLU | `(B, 196, 1024)` | chunk → gate → halved |
| After Linear 2 | `(B, 196, 2048)` | LLM embedding space |

---

### 3. Language Model — `models/tiny_aya_vision.py`

**Model:** `CohereLabs/tiny-aya-base`
**Class:** `Cohere2ForCausalLM`
**Parameters:** ~3.35B

Tiny Aya Base is a multilingual LLM supporting 70+ languages. It uses a hybrid attention scheme: groups of 4 layers repeat 9 times, where 3 layers use sliding window attention (window size 4096) and 1 layer uses global attention without positional embeddings.

| Property | Value |
|---|---|
| Hidden size | 2048 |
| Number of layers | 36 |
| Vocab size | 262144 (config) / 261000 (tokenizer) |
| Context length | 8192 |
| Attention | Sliding window (3 of 4) + global (1 of 4) |
| dtype | bfloat16 |

---

## Input Pipeline — `src/processing.py`

**Class:** `TinyAyaVisionProcessor`

The processor combines `SiglipImageProcessor` (for images) with `CohereTokenizer` (for text). Images are referenced in text with a single `<image>` marker, which the processor expands to 196 consecutive `<image>` placeholder tokens before tokenization.

```
User text: "What is in this image? <image>"
                         ↓
Expanded:  "What is in this image? <image><image>...<image>"
                                   └──────── 196 ──────────┘
                         ↓
Tokenized input_ids: [..., 261002, 261002, ..., 261002]
                                └─── image_token_id ───┘
```

The `<image>` token is added as a special token (`additional_special_tokens`) so it is always encoded as a single token ID (261002). After adding it, `resize_token_embeddings()` is called on the LLM to accommodate the expanded vocabulary.

---

## Feature Merging — `models/tiny_aya_vision.py`

The full forward pass merges image features into the text embedding sequence by replacing the `<image>` placeholder token positions with the projected connector output. This follows the `masked_scatter` pattern from Aya Vision.

```
input_ids:        [tok, tok, img, img, ..., img, tok, tok]
                                └─── 196 image tokens ───┘
                         ↓
inputs_embeds:    text_embedding_table[input_ids]
                         ↓
image_features:   connector(vision_encoder(pixel_values))  # (B, 196, 2048)
                         ↓
merged_embeds:    inputs_embeds.masked_scatter(image_mask, image_features)
                         ↓
LLM forward:      Cohere2ForCausalLM(inputs_embeds=merged_embeds)
```

During autoregressive generation, `pixel_values` are only processed on the first step (when there is no KV cache). On subsequent steps, image information is already encoded in the cached key-values.

---

## Memory Footprint

| Component | Size (bfloat16) |
|---|---|
| SigLIP2-so400m | ~0.8 GB |
| Tiny Aya Base | ~6.7 GB |
| Connector | ~0.05 GB |
| **Total** | **~7.5 GB** |

Fits within a single 24 GB GPU with headroom for activations and KV cache.

---

## Evaluation Harness

### CVQA — `evaluation/tasks/cvqa/`

Multilingual visual question answering across 31 languages. Configured as an `lm-evaluation-harness` task using `generate_until` output type with greedy decoding. Answers are matched to the closest option letter (A–D).

- Dataset: `afaji/cvqa`
- Split: `test`
- Label: integer 0–3, mapped to letters A–D
- Metric: exact match accuracy

### xMMMU — `evaluation/tasks/xmmmu/`

Multilingual MMMU (massive multidisciplinary understanding) benchmark across 7 languages: Arabic, English, French, Hindi, Indonesian, Japanese, Portuguese.

- Dataset: `neulab/PangeaBench-xmmmu`
- Splits: language codes (`ar`, `en`, `fr`, `hi`, `id`, `ja`, `pt`)
- Images: up to 7 images per question (fields `image_1`–`image_7`), only non-None images are used
- Options: stored as a string representation of a Python list, parsed with `ast.literal_eval`
- Metric: exact match accuracy

### m-ArenaHard — `evaluation/m_arena_hard.py`

Multilingual text-only benchmark with 500 prompts × 23 languages. Uses LLM-as-judge scoring (win-rates vs. a reference model). In Phase 1, only data loading and response generation are scaffolded — judge scoring is deferred to Phase 2.

- Dataset: `CohereLabs/m-ArenaHard`
- Split: `test`
- Languages: `ar cs de el en es fa fr he hi id it ja ko nl pl pt ro ru tr uk vi zh`
- Scoring: LLM-as-judge (Phase 2)

---

## Repository Layout

```
config/
  model_config.py          # TinyAyaVisionConfig — all hyperparameters

src/
  vision_encoder.py        # VisionEncoder — frozen SigLIP2 wrapper
  image_processor.py       # ImageProcessor — AutoImageProcessor wrapper
  connector.py             # MultiModalProjector — Pixel Shuffle + SwiGLU
  processing.py            # TinyAyaVisionProcessor — text + image inputs

models/
  tiny_aya_vision.py       # TinyAyaVisionForConditionalGeneration

evaluation/
  tasks/cvqa/
    cvqa.yaml              # lm-eval task config
    utils.py               # doc_to_text, doc_to_target, process_results
  tasks/xmmmu/
    xmmmu.yaml             # lm-eval task config
    utils.py               # doc_to_text, doc_to_image, process_results
  m_arena_hard.py          # m-ArenaHard data loading + response generation

tests/
  conftest.py              # Shared fixtures and markers
  test_connector.py        # Pixel Shuffle + MLP unit tests (CPU)
  test_processing.py       # Tokenizer + multimodal processing tests (CPU)
  test_vision_encoder.py   # VisionEncoder tests (GPU)
  test_vlm_assembly.py     # End-to-end VLM tests (GPU)
  test_evaluation.py       # Evaluation harness tests (CPU + network)
```
