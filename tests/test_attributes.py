"""Attribute builders and label-noise machinery on crafted fixtures (TC-14, TC-15)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voc_arc.attributes import (
    ATTRIBUTES,
    build_attributes,
    degrade_labels,
    persuadee_engagement,
    strategy_attributes,
    valence_arc_end,
)
from voc_arc.p4g import STRATEGIES


class OneHotStubModel:
    """predict_proba stub: a fixed emotion id per text, one-hot over 7 classes."""

    def __init__(self, emotion_by_text: dict[str, int]) -> None:
        self.emotion_by_text = emotion_by_text

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        proba = np.zeros((len(texts), 7))
        for i, text in enumerate(texts):
            proba[i, self.emotion_by_text[text]] = 1.0
        return proba


@pytest.fixture()
def crafted() -> pd.DataFrame:
    """Two dialogues with hand-checkable attribute values.

    d1 (6 sentences, 4 persuader): one logical appeal, one emotion appeal,
    one donation ask at sent_idx 3, one other; persuadee asks one question
    out of two sentences. d2 (3 sentences, 2 persuader): credibility appeal
    and personal story, no ask, no persuadee question.
    """
    rows = [
        ("d1", 0, 0, "persuader", "We can save lives.", "logical_appeal"),
        ("d1", 1, 0, "persuadee", "Why should I?", None),
        ("d1", 2, 1, "persuader", "Think of the children.", "emotion_appeal"),
        ("d1", 3, 1, "persuader", "Would you donate a dollar?", "donation_ask"),
        ("d1", 4, 2, "persuadee", "Okay.", None),
        ("d1", 5, 2, "persuader", "Thanks.", "other"),
        ("d2", 0, 0, "persuader", "The charity is rated four stars.", "credibility_appeal"),
        ("d2", 1, 0, "persuader", "I donated last year.", "personal_story"),
        ("d2", 2, 1, "persuadee", "Sure.", None),
    ]
    return pd.DataFrame(
        rows, columns=["dialogue_id", "sent_idx", "turn", "role", "text", "strategy"]
    )


class TestStrategyAttributes:
    def test_rates_are_per_persuader_sentence(self, crafted: pd.DataFrame) -> None:
        out = strategy_attributes(crafted)
        assert out.loc["d1", "logical_appeal_rate"] == 0.25
        assert out.loc["d1", "emotional_appeal_rate"] == 0.25
        assert out.loc["d1", "credibility_appeal_rate"] == 0.0
        assert out.loc["d2", "credibility_appeal_rate"] == 0.5
        assert out.loc["d2", "personal_story_rate"] == 0.5

    def test_ask_timing_normalized_position(self, crafted: pd.DataFrame) -> None:
        out = strategy_attributes(crafted)
        assert out.loc["d1", "ask_timing"] == pytest.approx(3 / 5)  # sent 3 of 6

    def test_ask_timing_censored_at_one_without_ask(self, crafted: pd.DataFrame) -> None:
        out = strategy_attributes(crafted)
        assert out.loc["d2", "ask_timing"] == 1.0

    def test_alternative_label_column(self, crafted: pd.DataFrame) -> None:
        relabeled = crafted.assign(predicted=crafted["strategy"])
        relabeled.loc[0, "predicted"] = "credibility_appeal"
        out = strategy_attributes(relabeled, label_col="predicted")
        assert out.loc["d1", "logical_appeal_rate"] == 0.0
        assert out.loc["d1", "credibility_appeal_rate"] == 0.25


class TestAnnotationFreeAttributes:
    def test_engagement_is_the_question_rate(self, crafted: pd.DataFrame) -> None:
        out = persuadee_engagement(crafted)
        assert out.loc["d1"] == 0.5  # one question out of two sentences
        assert out.loc["d2"] == 0.0

    def test_valence_arc_end_hand_computed(self, crafted: pd.DataFrame) -> None:
        # d1 persuadee: "Why should I?" -> no_emotion (0), "Okay." ->
        # happiness (+1); EWMA alpha 0.4: [0, 0.4] -> end 0.4.
        # d2 persuadee: "Sure." -> sadness (-1) -> end -1.
        model = OneHotStubModel({"Why should I?": 0, "Okay.": 4, "Sure.": 5})
        out = valence_arc_end(crafted, model, alpha=0.4)
        assert out.loc["d1"] == pytest.approx(0.4)
        assert out.loc["d2"] == pytest.approx(-1.0)

    def test_build_attributes_shape_and_order(self, crafted: pd.DataFrame) -> None:
        model = OneHotStubModel({"Why should I?": 0, "Okay.": 4, "Sure.": 5})
        out = build_attributes(crafted, model)
        assert list(out.columns) == list(ATTRIBUTES)
        assert len(out) == crafted["dialogue_id"].nunique()
        assert not out.isna().any().any()


class TestDegradeLabels:
    @pytest.fixture()
    def labels(self) -> pd.Series:
        rng = np.random.default_rng(7)
        values = rng.choice(STRATEGIES, size=20_000).astype(object)
        values[::100] = np.nan  # sprinkle missing (persuadee-like) rows
        return pd.Series(values)

    def test_epsilon_zero_is_identity(self, labels: pd.Series) -> None:
        out = degrade_labels(labels, 0.0, seed=1)
        present = labels.notna()
        assert (out[present] == labels[present]).all()

    def test_label_space_is_preserved(self, labels: pd.Series) -> None:
        out = degrade_labels(labels, 0.3, seed=1)
        present = labels.notna()
        assert set(out[present].unique()) <= set(STRATEGIES)
        assert out[~present].isna().all()  # missing rows stay missing

    def test_flip_rate_hits_the_target(self, labels: pd.Series) -> None:
        for epsilon in (0.1, 0.3):
            out = degrade_labels(labels, epsilon, seed=2)
            present = labels.notna()
            realized = float((out[present] != labels[present]).mean())
            assert abs(realized - epsilon) < 0.015

    def test_seeded_determinism(self, labels: pd.Series) -> None:
        a = degrade_labels(labels, 0.2, seed=3)
        b = degrade_labels(labels, 0.2, seed=3)
        c = degrade_labels(labels, 0.2, seed=4)
        present = labels.notna()
        assert (a[present] == b[present]).all()
        assert (a[present] != c[present]).any()

    def test_invalid_epsilon_raises(self, labels: pd.Series) -> None:
        with pytest.raises(ValueError, match="epsilon"):
            degrade_labels(labels, -0.1, seed=0)
        with pytest.raises(ValueError, match="epsilon"):
            degrade_labels(labels, 1.5, seed=0)

    def test_label_outside_space_raises(self) -> None:
        with pytest.raises(ValueError, match="label space"):
            degrade_labels(pd.Series(["logical_appeal", "not-a-strategy"]), 0.1, seed=0)
