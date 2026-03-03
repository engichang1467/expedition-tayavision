"""xMMMU evaluation task utilities for lm-evaluation-harness.

xMMMU (PangeaBench-xmmmu) is a cross-lingual multimodal understanding
benchmark with 7 languages: ar, en, fr, hi, id, ja, pt.
Dataset: neulab/PangeaBench-xmmmu

Fields:
    question: str (may contain <image N> placeholders)
    options: str (JSON-like list of option strings)
    answer: str (letter A-D)
    image_1 through image_7: PIL Image or None
"""

import ast


OPTION_LETTERS = ["A", "B", "C", "D"]


def xmmmu_doc_to_image(doc):
    """Extract all non-None images from the sample."""
    images = []
    for i in range(1, 8):
        img = doc.get(f"image_{i}")
        if img is not None:
            images.append(img)
    return images


def xmmmu_doc_to_text(doc):
    """Format the question with options as a prompt.

    The question may contain <image N> markers that reference
    specific images. We replace them with <image> for our model.
    """
    question = doc["question"]

    # Replace <image N> markers with generic <image>
    for i in range(1, 8):
        question = question.replace(f"<image {i}>", "<image>")

    # Parse options (stored as string repr of list)
    try:
        options = ast.literal_eval(doc["options"])
    except (ValueError, SyntaxError):
        options = doc["options"]

    if isinstance(options, list):
        options_str = "\n".join(
            f"{OPTION_LETTERS[i]}. {opt}"
            for i, opt in enumerate(options)
        )
    else:
        options_str = str(options)

    return (
        f"{question}\n{options_str}\n"
        "Answer with the option letter (A, B, C, or D)."
    )


def xmmmu_process_results(doc, results):
    """Check if the model's answer matches the correct option."""
    pred = results[0].strip().upper()
    gold = doc["answer"].strip().upper()

    # Extract just the first letter if the model outputs more
    if pred and pred[0] in OPTION_LETTERS:
        pred = pred[0]

    return {"exact_match": float(pred == gold)}
