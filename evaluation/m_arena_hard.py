"""m-ArenaHard evaluation scaffold.

m-ArenaHard is a multilingual text-only evaluation benchmark with 23
languages, using LLM-as-judge for scoring. This module provides the
data loading and response generation infrastructure. Full LLM judge
scoring is deferred to Phase 2.

Dataset: CohereLabs/m-ArenaHard
Languages: ar, cs, de, el, en, es, fr, he, hi, id, it, ja, ko, nl,
           fa, pl, pt, ro, ru, tr, uk, vi, zh
"""

from datasets import load_dataset


LANGUAGES = [
    "ar", "cs", "de", "el", "en", "es", "fr", "he", "hi",
    "id", "it", "ja", "ko", "nl", "fa", "pl", "pt", "ro",
    "ru", "tr", "uk", "vi", "zh",
]


def load_arena_hard(lang: str):
    """Load m-ArenaHard dataset for a specific language.

    Args:
        lang: ISO 639-1 language code (e.g. "en", "ar", "zh").

    Returns:
        HuggingFace Dataset with columns: question_id, cluster,
        category, prompt.
    """
    if lang not in LANGUAGES:
        raise ValueError(
            f"Language '{lang}' not in m-ArenaHard. "
            f"Available: {LANGUAGES}"
        )
    return load_dataset("CohereLabs/m-ArenaHard", lang, split="test")


def generate_responses(model, tokenizer, dataset, max_new_tokens=512):
    """Generate model responses for each prompt in the dataset.

    Args:
        model: Language model (or VLM in text-only mode).
        tokenizer: Tokenizer for the model.
        dataset: m-ArenaHard dataset for one language.
        max_new_tokens: Maximum tokens to generate per response.

    Returns:
        List of dicts with question_id and model_response.
    """
    results = []
    for sample in dataset:
        inputs = tokenizer(
            sample["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(model.device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # Decode only the generated tokens (skip prompt)
        response = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        results.append({
            "question_id": sample["question_id"],
            "prompt": sample["prompt"],
            "model_response": response,
        })

    return results


# TODO (Phase 2): Add LLM-as-judge scoring
# - Send (prompt, response) pairs to judge LLM (Claude / GPT-4)
# - Compute win-rates per language
# - Aggregate across languages
