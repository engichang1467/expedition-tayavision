"""Tests for the evaluation harness setup."""

import pytest

from evaluation.m_arena_hard import LANGUAGES, load_arena_hard


class TestMArenaHard:
    """Tests for the m-ArenaHard data loading scaffold."""

    def test_languages_list(self):
        """All 23 languages should be listed."""
        assert len(LANGUAGES) == 23
        assert "en" in LANGUAGES
        assert "ar" in LANGUAGES
        assert "zh" in LANGUAGES

    def test_load_english(self):
        """English subset loads correctly."""
        ds = load_arena_hard("en")
        assert len(ds) > 0
        sample = ds[0]
        assert "question_id" in sample
        assert "prompt" in sample

    def test_load_arabic(self):
        """Arabic subset loads correctly."""
        ds = load_arena_hard("ar")
        assert len(ds) > 0

    def test_invalid_language_raises(self):
        """Invalid language code raises ValueError."""
        with pytest.raises(ValueError, match="not in m-ArenaHard"):
            load_arena_hard("xx")


class TestCVQATask:
    """Tests for the CVQA lm-eval task configuration."""

    def test_utils_import(self):
        """CVQA utils module imports without errors."""
        from evaluation.tasks.cvqa.utils import (
            cvqa_doc_to_image,
            cvqa_doc_to_target,
            cvqa_doc_to_text,
            cvqa_process_results,
        )

        assert callable(cvqa_doc_to_image)
        assert callable(cvqa_doc_to_text)
        assert callable(cvqa_doc_to_target)
        assert callable(cvqa_process_results)

    def test_doc_to_text_format(self):
        """CVQA prompt formatting produces expected format."""
        from evaluation.tasks.cvqa.utils import cvqa_doc_to_text

        doc = {
            "Question": "What color is the sky?",
            "Options": ["Red", "Blue", "Green", "Yellow"],
            "Label": 1,
        }
        text = cvqa_doc_to_text(doc)
        assert "<image>" in text
        assert "A." in text
        assert "B." in text
        assert "Blue" in text

    def test_process_results_correct(self):
        """Correct answer scores 1.0."""
        from evaluation.tasks.cvqa.utils import cvqa_process_results

        doc = {"Label": 1}  # B
        results = ["B"]
        score = cvqa_process_results(doc, results)
        assert score["exact_match"] == 1.0

    def test_process_results_incorrect(self):
        """Incorrect answer scores 0.0."""
        from evaluation.tasks.cvqa.utils import cvqa_process_results

        doc = {"Label": 1}  # B
        results = ["A"]
        score = cvqa_process_results(doc, results)
        assert score["exact_match"] == 0.0


class TestXMMMUTask:
    """Tests for the xMMMU lm-eval task configuration."""

    def test_utils_import(self):
        """xMMMU utils module imports without errors."""
        from evaluation.tasks.xmmmu.utils import (
            xmmmu_doc_to_image,
            xmmmu_doc_to_text,
            xmmmu_process_results,
        )

        assert callable(xmmmu_doc_to_image)
        assert callable(xmmmu_doc_to_text)
        assert callable(xmmmu_process_results)

    def test_doc_to_image_extracts_non_none(self):
        """Only non-None images are extracted."""
        from evaluation.tasks.xmmmu.utils import xmmmu_doc_to_image

        doc = {
            "image_1": "img1",
            "image_2": None,
            "image_3": "img3",
            **{f"image_{i}": None for i in range(4, 8)},
        }
        images = xmmmu_doc_to_image(doc)
        assert len(images) == 2
        assert images[0] == "img1"
        assert images[1] == "img3"

    def test_doc_to_text_replaces_image_markers(self):
        """<image N> markers are replaced with <image>."""
        from evaluation.tasks.xmmmu.utils import xmmmu_doc_to_text

        doc = {
            "question": "<image 1> What is shown?",
            "options": "['A thing', 'B thing', 'C thing', 'D thing']",
        }
        text = xmmmu_doc_to_text(doc)
        assert "<image 1>" not in text
        assert "<image>" in text

    def test_process_results_correct(self):
        """Correct answer scores 1.0."""
        from evaluation.tasks.xmmmu.utils import xmmmu_process_results

        doc = {"answer": "B"}
        results = ["B"]
        score = xmmmu_process_results(doc, results)
        assert score["exact_match"] == 1.0
