"""Baselines and TF-IDF model on a small synthetic corpus (TC-8, TC-9)."""

from __future__ import annotations

import numpy as np
import pytest

from voc_arc.classifier import (
    LexiconBaseline,
    MajorityBaseline,
    build_model,
    evaluate,
    per_class_f1,
)
from voc_arc.data import EMOTION_TO_ID, EMOTIONS

# Synthetic corpus: each emotion has a distinctive vocabulary, plus a
# dominant neutral class, mimicking the DailyDialog imbalance in miniature.
_TEMPLATES = {
    "no_emotion": ["the meeting is at {} o'clock", "please pass the {} report", "it is {} today"],
    "anger": ["i am so angry about the {}", "this {} makes me furious", "i hate the {}"],
    "happiness": ["i am so happy about the {}", "what a wonderful {}", "i love this {}"],
    "sadness": ["i feel sad about the {}", "the {} makes me cry", "i miss the old {}"],
    "surprise": ["wow the {} is unbelievable", "no way , a {}", "i can't believe the {}"],
}
_FILLERS = ["budget", "garden", "movie", "train", "dinner", "letter", "game", "road"]


def _synthetic_corpus(n_per_class: int = 40, seed: int = 0) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    texts, labels = [], []
    for name, templates in _TEMPLATES.items():
        count = n_per_class * 4 if name == "no_emotion" else n_per_class
        for _ in range(count):
            template = templates[rng.integers(len(templates))]
            texts.append(template.format(_FILLERS[rng.integers(len(_FILLERS))]))
            labels.append(EMOTION_TO_ID[name])
    return texts, np.array(labels)


class TestMajorityBaseline:
    def test_predicts_most_frequent_class(self) -> None:
        model = MajorityBaseline().fit(["a", "b", "c"], np.array([0, 0, 4]))
        assert model.predict(["x", "y"]).tolist() == [0, 0]

    def test_unfitted_raises(self) -> None:
        with pytest.raises(ValueError, match="fit"):
            MajorityBaseline().predict(["x"])


class TestLexiconBaseline:
    def test_keywords_map_to_emotions(self) -> None:
        model = LexiconBaseline()
        preds = model.predict(
            ["I am so happy today .", "I hate this .", "The meeting starts at nine ."]
        )
        assert preds.tolist() == [
            EMOTION_TO_ID["happiness"],
            EMOTION_TO_ID["anger"],
            EMOTION_TO_ID["no_emotion"],
        ]

    def test_case_insensitive(self) -> None:
        model = LexiconBaseline()
        assert model.predict(["WONDERFUL !"]).tolist() == [EMOTION_TO_ID["happiness"]]


class TestTfidfModel:
    def test_beats_majority_baseline_on_synthetic_corpus(self) -> None:
        texts, labels = _synthetic_corpus()
        rng = np.random.default_rng(1)
        order = rng.permutation(len(texts))
        split = int(len(texts) * 0.7)
        train_idx, test_idx = order[:split], order[split:]
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]

        model = build_model(seed=0).fit(train_texts, labels[train_idx])
        majority = MajorityBaseline().fit(train_texts, labels[train_idx])

        model_f1 = evaluate(labels[test_idx], model.predict(test_texts))["macro_f1"]
        majority_f1 = evaluate(labels[test_idx], majority.predict(test_texts))["macro_f1"]
        assert model_f1 > majority_f1
        assert model_f1 > 0.8  # separable vocabulary: the model must be strong here

    def test_predict_proba_shape_and_simplex(self) -> None:
        texts, labels = _synthetic_corpus(n_per_class=15)
        model = build_model(seed=0).fit(texts, labels)
        proba = model.predict_proba(["i am so happy about the game"])
        assert proba.shape == (1, len(np.unique(labels)))
        assert np.isclose(proba.sum(), 1.0)


class TestMetrics:
    def test_evaluate_perfect_prediction(self) -> None:
        y = np.array([0, 1, 4, 4])
        scores = evaluate(y, y)
        assert scores["macro_f1"] == 1.0
        assert scores["accuracy"] == 1.0

    def test_per_class_f1_index_and_support(self) -> None:
        y_true = np.array([0, 0, 4, 5])
        y_pred = np.array([0, 0, 4, 0])
        table = per_class_f1(y_true, y_pred)
        assert list(table.index) == list(EMOTIONS)
        assert table.loc["happiness", "f1"] == 1.0
        assert table.loc["sadness", "f1"] == 0.0
        assert table["support"].sum() == 4
