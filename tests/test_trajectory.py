"""Trajectory layer: valence, EWMA, transitions, arcs, shift detector.

Covers TC-3 through TC-7, including the hand-computed transition matrix
and the planted-shift detection that serve as known-answer checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voc_arc.trajectory import (
    VALENCE,
    aggregate_arc,
    detect_shift,
    dialogue_valences,
    ewma,
    expected_valence,
    normalized_positions,
    transition_matrix,
    valence_series,
)


class TestValence:
    def test_mapping_values(self) -> None:
        # 0 no_emotion, 1 anger, 2 disgust, 3 fear, 4 happiness, 5 sadness, 6 surprise
        assert VALENCE == {0: 0.0, 1: -1.0, 2: -1.0, 3: -1.0, 4: 1.0, 5: -1.0, 6: 0.0}

    def test_series(self) -> None:
        out = valence_series(np.array([0, 4, 1, 6, 5]))
        assert out.tolist() == [0.0, 1.0, -1.0, 0.0, -1.0]

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="0..6"):
            valence_series(np.array([7]))

    def test_expected_valence_weights_probabilities(self) -> None:
        proba = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # certain no_emotion -> 0
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # certain happiness -> +1
                [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],  # half happy -> +0.5
                [0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0],  # anger and happiness cancel
            ]
        )
        assert expected_valence(proba).tolist() == [0.0, 1.0, 0.5, 0.0]

    def test_expected_valence_shape_check(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            expected_valence(np.ones((3, 5)))


class TestEwma:
    def test_hand_computed(self) -> None:
        out = ewma(np.array([0.0, 1.0, 1.0]), alpha=0.5)
        assert out.tolist() == [0.0, 0.5, 0.75]

    def test_alpha_one_is_identity(self) -> None:
        x = np.array([0.3, -0.7, 1.0])
        assert ewma(x, alpha=1.0).tolist() == x.tolist()

    def test_single_element(self) -> None:
        assert ewma(np.array([0.4]), alpha=0.3).tolist() == [0.4]

    def test_empty(self) -> None:
        assert ewma(np.array([]), alpha=0.3).size == 0

    def test_constant_series_stays_constant(self) -> None:
        out = ewma(np.full(10, -1.0), alpha=0.2)
        assert np.allclose(out, -1.0)

    @pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
    def test_invalid_alpha_raises(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            ewma(np.array([1.0]), alpha=alpha)


class TestNormalizedPositions:
    def test_endpoints(self) -> None:
        pos = normalized_positions(5)
        assert pos[0] == 0.0 and pos[-1] == 1.0 and len(pos) == 5

    def test_single_utterance_sits_at_center(self) -> None:
        assert normalized_positions(1).tolist() == [0.5]

    def test_nonpositive_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            normalized_positions(0)


class TestTransitionMatrix:
    def test_hand_computed_toy_sequence(self) -> None:
        # Sequence 0 -> 0 -> 4 -> 0 and a second dialogue 4 -> 4.
        # Transitions from 0: {0: 1, 4: 1}  -> row [0.5, ..., 0.5, ...]
        # Transitions from 4: {0: 1, 4: 1}  -> row [0.5, ..., 0.5, ...]
        matrix = transition_matrix([np.array([0, 0, 4, 0]), np.array([4, 4])])
        expected_row = np.array([0.5, 0, 0, 0, 0.5, 0, 0])
        assert np.allclose(matrix[0], expected_row)
        assert np.allclose(matrix[4], expected_row)

    def test_rows_sum_to_one_including_unseen_states(self) -> None:
        matrix = transition_matrix([np.array([0, 4, 0])])
        assert np.allclose(matrix.sum(axis=1), 1.0)
        # state 1 (anger) never occurs: uniform row
        assert np.allclose(matrix[1], 1.0 / 7.0)

    def test_no_transitions_across_dialogue_boundaries(self) -> None:
        # 0->4 within dialogues never happens; concatenation would create it.
        matrix = transition_matrix([np.array([0, 0]), np.array([4, 4])])
        assert matrix[0, 4] == 0.0
        assert matrix[0, 0] == 1.0

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="labels"):
            transition_matrix([np.array([0, 9])])


class TestAggregateArc:
    def test_columns_and_positions(self) -> None:
        arcs = [np.array([0.0, 1.0, 1.0]), np.array([0.0, 0.0, 1.0, 1.0])]
        arc = aggregate_arc(arcs, n_bins=4, n_boot=50, seed=0)
        assert list(arc.columns) == ["position", "mean", "lo", "hi"]
        assert np.allclose(arc["position"], [0.125, 0.375, 0.625, 0.875])

    def test_rising_signal_recovered(self) -> None:
        rng = np.random.default_rng(3)
        arcs = []
        for _ in range(200):
            n = rng.integers(6, 14)
            arcs.append(np.linspace(-0.5, 0.5, n) + rng.normal(0, 0.1, n))
        arc = aggregate_arc(arcs, n_bins=5, n_boot=100, seed=0)
        assert arc["mean"].iloc[-1] > arc["mean"].iloc[0]
        assert (arc["lo"] <= arc["mean"]).all() and (arc["mean"] <= arc["hi"]).all()

    def test_deterministic_given_seed(self) -> None:
        arcs = [np.array([0.0, 1.0]), np.array([1.0, 0.0, 1.0])]
        a = aggregate_arc(arcs, n_bins=3, n_boot=30, seed=5)
        b = aggregate_arc(arcs, n_bins=3, n_boot=30, seed=5)
        pd.testing.assert_frame_equal(a, b)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            aggregate_arc([])


class TestDetectShift:
    def test_planted_shift_found_at_the_right_position(self) -> None:
        # 10 neutral turns, then 6 negative turns: shift starts at index 10.
        valence = np.array([0.0] * 10 + [-1.0] * 6)
        index, magnitude = detect_shift(valence, alpha=1.0)  # no smoothing: exact
        assert index == 10
        assert magnitude == -1.0

    def test_positive_shift_with_smoothing_lands_near_the_plant(self) -> None:
        valence = np.array([-1.0] * 8 + [1.0] * 8)
        index, magnitude = detect_shift(valence, alpha=0.5)
        assert 8 <= index <= 10  # smoothing delays the detected onset slightly
        assert magnitude > 1.0

    def test_flat_series_has_near_zero_magnitude(self) -> None:
        index, magnitude = detect_shift(np.zeros(12), alpha=0.4)
        assert magnitude == 0.0
        assert 1 <= index < 12

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            detect_shift(np.array([0.5]))


class TestDialogueValences:
    def test_grouping_and_order(self, mini_df: pd.DataFrame) -> None:
        shuffled = mini_df.sample(frac=1.0, random_state=0)
        arcs = dialogue_valences(shuffled)
        assert len(arcs) == 3
        assert arcs[0].tolist() == [-1.0, 0.0]  # disgust, no_emotion
        assert arcs[1].tolist() == [1.0, 1.0, 1.0]  # happiness x3
        assert arcs[2].tolist() == [-1.0, -1.0]  # sadness x2
