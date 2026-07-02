"""Smoke tests for the plot helpers on the Agg backend (TC-10)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from voc_arc.plotting import (
    plot_arc_band,
    plot_class_distribution,
    plot_dialogue_trajectory,
    plot_transition_heatmap,
    plot_weight_forest,
    set_style,
)


@pytest.fixture(autouse=True)
def _style():
    set_style()
    yield
    plt.close("all")


def test_dialogue_trajectory_smoke() -> None:
    valence = np.array([0.0, 1.0, -1.0, -1.0])
    ax = plot_dialogue_trajectory(valence, valence, title="t", shift_at=2)
    assert ax.get_title() == "t"
    assert len(ax.lines) >= 3  # raw, smoothed, shift line (+ zero line)


def test_arc_band_smoke() -> None:
    arc = pd.DataFrame(
        {"position": [0.25, 0.75], "mean": [0.0, 0.1], "lo": [-0.1, 0.0], "hi": [0.1, 0.2]}
    )
    ax = plot_arc_band({"gold": arc, "predicted": arc}, title="arcs")
    assert ax.get_title() == "arcs"
    assert len(ax.collections) == 2  # one band per arc


def test_transition_heatmap_smoke() -> None:
    matrix = np.full((7, 7), 1.0 / 7.0)
    ax = plot_transition_heatmap(matrix, title="transitions")
    assert ax.get_title() == "transitions"
    assert len(ax.texts) == 49


def test_weight_forest_smoke() -> None:
    weights = pd.DataFrame(
        {"coef": [0.4, -0.2, 0.0], "lo": [0.1, -0.5, -0.2], "hi": [0.7, 0.1, 0.2]},
        index=["ask_timing", "age", "is_male"],
    )
    ax = plot_weight_forest(weights, title="weights", highlight=["ask_timing"])
    assert ax.get_title() == "weights"
    assert len(ax.lines) == 2 * 3 + 1  # whisker and marker per row plus the zero line
    assert [label.get_text() for label in ax.get_yticklabels()] == ["is_male", "age", "ask_timing"]


def test_class_distribution_smoke() -> None:
    counts = pd.Series([80, 10, 10], index=["no_emotion", "happiness", "anger"])
    ax = plot_class_distribution(counts, title="counts")
    assert ax.get_title() == "counts"
    assert len(ax.patches) == 3
