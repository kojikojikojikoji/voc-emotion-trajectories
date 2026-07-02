"""Persuasion for Good loading, taxonomy mapping, outcome preparation (TC-12, TC-13)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voc_arc import p4g
from voc_arc.p4g import (
    STRATEGIES,
    STRATEGY_MAP,
    load_annotated_sample,
    load_info_sample,
    map_strategy,
    prepare_outcome_controls,
)


class TestStrategyMapping:
    def test_salient_labels(self) -> None:
        assert map_strategy("logical-appeal") == "logical_appeal"
        assert map_strategy("emotion-appeal") == "emotion_appeal"
        assert map_strategy("credibility-appeal") == "credibility_appeal"
        assert map_strategy("personal-story") == "personal_story"
        assert map_strategy("self-modeling") == "personal_story"
        assert map_strategy("proposition-of-donation") == "donation_ask"
        assert map_strategy("ask-donation-amount") == "donation_ask"
        assert map_strategy("ask-donate-more") == "donation_ask"

    def test_non_salient_labels_fall_into_other(self) -> None:
        for raw in ("greeting", "thank", "donation-information", "task-related-inquiry"):
            assert map_strategy(raw) == "other"

    def test_unknown_label_falls_into_other(self) -> None:
        assert map_strategy("some-future-label") == "other"

    def test_missing_label_maps_to_none(self) -> None:
        assert map_strategy(np.nan) is None
        assert map_strategy(None) is None

    def test_map_targets_are_inside_the_strategy_space(self) -> None:
        assert set(STRATEGY_MAP.values()) <= set(STRATEGIES)
        assert "other" in STRATEGIES


class TestAnnotatedSample:
    def test_schema_and_size(self) -> None:
        df = load_annotated_sample()
        assert list(df.columns) == [
            "dialogue_id",
            "sent_idx",
            "turn",
            "role",
            "text",
            "er_label",
            "ee_label",
            "strategy",
        ]
        assert df["dialogue_id"].nunique() == 300
        assert set(df["role"].unique()) == {"persuader", "persuadee"}

    def test_strategy_only_on_persuader_rows(self) -> None:
        df = load_annotated_sample()
        assert df.loc[df["role"] == "persuader", "strategy"].notna().all()
        assert df.loc[df["role"] == "persuadee", "strategy"].isna().all()
        labeled = df.loc[df["role"] == "persuader", "strategy"]
        assert set(labeled.unique()) <= set(STRATEGIES)

    def test_sent_idx_is_zero_based_and_ordered(self) -> None:
        df = load_annotated_sample()
        for _, group in df.groupby("dialogue_id"):
            assert group["sent_idx"].tolist() == list(range(len(group)))

    def test_info_sample_covers_the_same_dialogues(self) -> None:
        df = load_annotated_sample()
        info = load_info_sample()
        assert set(info["dialogue_id"]) == set(df["dialogue_id"])
        assert len(info) == 300


class TestOutcomeControls:
    @pytest.fixture()
    def info(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "dialogue_id": ["a", "b", "c", "d"],
                "donation_raw": [0.0, 0.5, 700.0, 2.0],
                "age": [30.0, np.nan, 50.0, 40.0],
                "sex": ["Male", "Female", None, "Other"],
                "extrovert": [1.0, 2.0, 3.0, np.nan],
                "agreeable": [1.0, 2.0, 3.0, 4.0],
                "conscientious": [1.0, 2.0, 3.0, 4.0],
                "neurotic": [1.0, 2.0, 3.0, 4.0],
                "open": [1.0, 2.0, 3.0, 4.0],
            }
        )

    def test_winsorization_at_the_task_cap(self, info: pd.DataFrame) -> None:
        out = prepare_outcome_controls(info)
        assert out.loc["c", "donation"] == 2.0  # $700 entry capped
        assert out.loc["b", "donation"] == 0.5  # in-cap value untouched

    def test_donated_flag(self, info: pd.DataFrame) -> None:
        out = prepare_outcome_controls(info)
        assert out["donated"].tolist() == [0, 1, 1, 1]

    def test_median_imputation_and_sex_dummy(self, info: pd.DataFrame) -> None:
        out = prepare_outcome_controls(info)
        assert out.loc["b", "age"] == 40.0  # median of 30, 50, 40
        assert out.loc["d", "extrovert"] == 2.0  # median of 1, 2, 3
        assert out["is_male"].tolist() == [1, 0, 0, 0]
        assert not out.isna().any().any()

    def test_duplicate_dialogue_raises(self, info: pd.DataFrame) -> None:
        doubled = pd.concat([info, info.iloc[[0]]])
        with pytest.raises(ValueError, match="duplicate"):
            prepare_outcome_controls(doubled)

    def test_real_sample_share_of_donors(self) -> None:
        out = prepare_outcome_controls(load_info_sample())
        assert len(out) == 300
        # About half of the annotated persuadees donated; guard the join
        # logic against silently dropping donors or non-donors.
        assert 0.4 < out["donated"].mean() < 0.6


class TestPaths:
    def test_raw_paths_are_under_the_data_dir(self) -> None:
        assert p4g.annotated_xlsx_path().name == "300_dialog.xlsx"
        assert p4g.full_dialog_path().parent == p4g.p4g_raw_dir()
        assert p4g.annotated_sample_path().parent.name == "sample"
