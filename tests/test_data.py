"""Parser, path resolution and no-leakage split (TC-1, TC-2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import EMOTION_LINES, TEXT_LINES

from voc_arc import data
from voc_arc.data import (
    EMOTION_TO_ID,
    EMOTIONS,
    load_sample,
    parse_dailydialog,
    split_dialogues,
)


class TestParser:
    def test_shape_and_columns(self, mini_df: pd.DataFrame) -> None:
        assert list(mini_df.columns) == ["dialogue_id", "turn", "text", "emotion", "emotion_name"]
        assert len(mini_df) == 7
        assert mini_df["dialogue_id"].nunique() == 3

    def test_alignment(self, mini_df: pd.DataFrame) -> None:
        first = mini_df[mini_df["dialogue_id"] == 0]
        assert first["text"].tolist() == ["The kitchen stinks .", "I'll throw out the garbage ."]
        assert first["emotion"].tolist() == [2, 0]
        assert first["emotion_name"].tolist() == ["disgust", "no_emotion"]

    def test_turn_index_is_zero_based_and_ordered(self, mini_df: pd.DataFrame) -> None:
        for _, group in mini_df.groupby("dialogue_id"):
            assert group["turn"].tolist() == list(range(len(group)))

    def test_misaligned_dialogue_is_dropped(self) -> None:
        df = parse_dailydialog(TEXT_LINES, ["2 0", "4 4", "5 5"])  # middle line too short
        assert df["dialogue_id"].nunique() == 2
        assert 1 not in df["dialogue_id"].to_numpy()

    def test_line_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="disagree"):
            parse_dailydialog(TEXT_LINES, EMOTION_LINES[:2])

    def test_label_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            parse_dailydialog(["Hi . __eou__"], ["7"])

    def test_emotion_constants(self) -> None:
        assert len(EMOTIONS) == 7
        assert EMOTIONS[0] == "no_emotion"
        assert EMOTION_TO_ID["happiness"] == 4


class TestDataDir:
    def test_default_is_repo_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VOC_DATA_DIR", raising=False)
        assert data.data_dir() == data._REPO_ROOT / "data"

    def test_relative_env_resolves_from_repo_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VOC_DATA_DIR", "./elsewhere")
        assert data.data_dir() == data._REPO_ROOT / "elsewhere"

    def test_absolute_env_is_used_as_is(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("VOC_DATA_DIR", str(tmp_path))
        assert data.data_dir() == tmp_path

    def test_missing_raw_error_names_download_script(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("VOC_DATA_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="download_data.py"):
            data.load_raw()


class TestSample:
    def test_sample_loads_with_schema(self) -> None:
        df = load_sample()
        assert list(df.columns) == ["dialogue_id", "turn", "text", "emotion", "emotion_name"]
        assert df["dialogue_id"].nunique() == 1000
        assert df["emotion"].between(0, 6).all()

    def test_sample_is_majority_no_emotion(self) -> None:
        df = load_sample()
        share = (df["emotion"] == 0).mean()
        assert 0.75 < share < 0.90  # the imbalance is real and preserved


class TestSplit:
    def test_no_dialogue_crosses_the_split(self, mini_df: pd.DataFrame) -> None:
        train, test = split_dialogues(mini_df, test_size=0.34, seed=0)
        assert set(train["dialogue_id"]).isdisjoint(set(test["dialogue_id"]))
        assert len(train) + len(test) == len(mini_df)

    def test_no_leakage_on_larger_frame(self) -> None:
        df = load_sample()
        train, test = split_dialogues(df, test_size=0.2, seed=42)
        assert set(train["dialogue_id"]).isdisjoint(set(test["dialogue_id"]))
        n_test = test["dialogue_id"].nunique()
        assert n_test == round(df["dialogue_id"].nunique() * 0.2)

    def test_same_seed_same_split(self, mini_df: pd.DataFrame) -> None:
        a = split_dialogues(mini_df, test_size=0.34, seed=7)[1]["dialogue_id"].unique()
        b = split_dialogues(mini_df, test_size=0.34, seed=7)[1]["dialogue_id"].unique()
        assert np.array_equal(a, b)

    def test_different_seed_can_differ(self) -> None:
        df = load_sample()
        a = set(split_dialogues(df, seed=1)[1]["dialogue_id"].unique())
        b = set(split_dialogues(df, seed=2)[1]["dialogue_id"].unique())
        assert a != b

    def test_invalid_test_size_raises(self, mini_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="test_size"):
            split_dialogues(mini_df, test_size=1.5)
