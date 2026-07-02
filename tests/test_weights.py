"""Weight estimation: planted-coefficient recovery and fidelity metric (TC-16, TC-17)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voc_arc.classifier import MajorityBaseline, build_model, evaluate
from voc_arc.p4g import load_annotated_sample
from voc_arc.weights import (
    bootstrap_linear_weights,
    bootstrap_logistic_weights,
    linear_weights,
    logistic_weights,
    standardize,
    weight_fidelity,
)

TRUE_BETA = np.array([1.2, -0.8, 0.5, 0.0, 0.3])
COLUMNS = ["a", "b", "c", "d", "e"]


@pytest.fixture(scope="module")
def planted() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic logistic data with known standardized coefficients."""
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(4000, 5)), columns=COLUMNS)
    logits = X.to_numpy() @ TRUE_BETA
    y = pd.Series((rng.random(4000) < 1 / (1 + np.exp(-logits))).astype(int))
    return X, y


class TestStandardize:
    def test_zero_mean_unit_variance(self) -> None:
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [5.0, 5.0, 5.0]})
        Z = standardize(X)
        assert Z["a"].mean() == pytest.approx(0.0)
        assert Z["a"].std(ddof=0) == pytest.approx(1.0)
        assert (Z["b"] == 0).all()  # constant column becomes zeros, not NaN


class TestLogisticWeights:
    def test_planted_signs_and_magnitudes_recovered(
        self, planted: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = planted
        coefs = logistic_weights(X, y, seed=0)
        for name, true in zip(COLUMNS, TRUE_BETA, strict=True):
            if true != 0:
                assert np.sign(coefs[name]) == np.sign(true)
            assert abs(coefs[name] - true) < 0.2

    def test_bootstrap_interval_brackets_truth(
        self, planted: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = planted
        out = bootstrap_logistic_weights(X, y, n_boot=200, seed=0)
        assert (out["lo"] <= out["coef"]).all() and (out["coef"] <= out["hi"]).all()
        # Non-zero planted coefficients are detected, the zero one is not.
        assert out.loc["a", "excludes_zero"]
        assert out.loc["b", "excludes_zero"]
        assert not out.loc["d", "excludes_zero"]
        assert out.loc["a", "lo"] <= TRUE_BETA[0] <= out.loc["a", "hi"]

    def test_single_class_outcome_raises(self) -> None:
        X = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
        with pytest.raises(ValueError, match="both classes"):
            logistic_weights(X, pd.Series([1, 1, 1]))

    def test_seeded_determinism(self, planted: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = planted
        a = bootstrap_logistic_weights(X, y, n_boot=50, seed=5)
        b = bootstrap_logistic_weights(X, y, n_boot=50, seed=5)
        pd.testing.assert_frame_equal(a, b)


class TestLinearWeights:
    def test_planted_recovery(self) -> None:
        rng = np.random.default_rng(3)
        X = pd.DataFrame(rng.normal(size=(2000, 3)), columns=["a", "b", "c"])
        y = pd.Series(X.to_numpy() @ np.array([2.0, -1.0, 0.0]) + rng.normal(0, 0.5, 2000))
        coefs = linear_weights(X, y)
        assert coefs["a"] == pytest.approx(2.0, abs=0.1)
        assert coefs["b"] == pytest.approx(-1.0, abs=0.1)
        assert coefs["c"] == pytest.approx(0.0, abs=0.1)
        boot = bootstrap_linear_weights(X, y, n_boot=100, seed=0)
        assert boot.loc["a", "excludes_zero"] and not boot.loc["c", "excludes_zero"]


class TestWeightFidelity:
    def test_identical_vectors_give_one(self) -> None:
        coefs = pd.Series([0.5, -0.2, 0.1], index=["x", "y", "z"])
        assert weight_fidelity(coefs, coefs, ["x", "y", "z"]) == pytest.approx(1.0)

    def test_reversed_ranks_give_minus_one(self) -> None:
        a = pd.Series([3.0, 2.0, 1.0], index=["x", "y", "z"])
        b = pd.Series([1.0, 2.0, 3.0], index=["x", "y", "z"])
        assert weight_fidelity(a, b, ["x", "y", "z"]) == pytest.approx(-1.0)

    def test_too_few_attributes_raises(self) -> None:
        coefs = pd.Series([1.0, 2.0], index=["x", "y"])
        with pytest.raises(ValueError, match="at least 3"):
            weight_fidelity(coefs, coefs, ["x", "y"])

    def test_missing_name_raises(self) -> None:
        coefs = pd.Series([1.0, 2.0, 3.0], index=["x", "y", "z"])
        with pytest.raises(ValueError, match="missing"):
            weight_fidelity(coefs, coefs, ["x", "y", "w"])


class TestStrategyClassifierOnGold:
    def test_beats_majority_baseline_on_held_out_dialogues(self) -> None:
        """TC-17: the classical annotator learns real signal (small, fast run)."""
        df = load_annotated_sample()
        persuader = df[df["role"] == "persuader"]
        ids = sorted(persuader["dialogue_id"].unique())[:120]
        train_ids, test_ids = set(ids[:80]), set(ids[80:])
        train = persuader[persuader["dialogue_id"].isin(train_ids)]
        test = persuader[persuader["dialogue_id"].isin(test_ids)]
        labels = {name: i for i, name in enumerate(sorted(train["strategy"].unique()))}
        y_train = train["strategy"].map(labels).to_numpy()
        y_test = test["strategy"].map(labels).to_numpy()

        model = build_model(seed=0, max_word_features=20_000)
        model.fit(train["text"].tolist(), y_train)
        scores = evaluate(y_test, model.predict(test["text"].tolist()))

        majority = MajorityBaseline().fit(train["text"].tolist(), y_train)
        base = evaluate(y_test, majority.predict(test["text"].tolist()))

        assert scores["macro_f1"] > base["macro_f1"]
        assert scores["accuracy"] > base["accuracy"]
        assert scores["macro_f1"] > 0.4  # well above chance on 6 classes
