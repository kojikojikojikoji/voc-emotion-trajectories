"""Shared matplotlib style and plot helpers for the notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import EMOTIONS

#: Categorical palette (max six colors, no plain tab10, no red-green pair).
PALETTE = ["#3b6ea5", "#c66a2c", "#4b8a5a", "#7b5aa6", "#b3473f", "#6b6b6b"]


def set_style() -> None:
    """Apply the shared plot style (call once at the top of a notebook)."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 110,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
        }
    )


def plot_dialogue_trajectory(
    valence: np.ndarray,
    smoothed: np.ndarray,
    title: str,
    shift_at: int | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Raw and smoothed valence of one dialogue over its turns.

    Args:
        valence: raw per-utterance valence.
        smoothed: EWMA-smoothed valence of the same length.
        title: axes title; state the conclusion, not the plot type.
        shift_at: optional turn index where a detected shift starts (drawn
            as a dashed vertical line).
        ax: existing axes, or ``None`` to create a new figure.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.2))
    turns = np.arange(len(valence))
    ax.plot(turns, valence, "o", color=PALETTE[5], ms=5, alpha=0.7, label="utterance valence")
    ax.plot(turns, smoothed, "-", color=PALETTE[0], lw=2.0, label="EWMA-smoothed")
    if shift_at is not None:
        ax.axvline(shift_at - 0.5, ls="--", color=PALETTE[4], lw=1.4, label="detected shift")
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("turn index within the dialogue")
    ax.set_ylabel("valence (-1 to +1)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best", fontsize=9)
    return ax


def plot_arc_band(
    arcs: dict[str, pd.DataFrame],
    title: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """One or more aggregate valence arcs with bootstrap CI bands.

    Args:
        arcs: mapping from label to a DataFrame with columns ``position``,
            ``mean``, ``lo`` and ``hi`` (from ``trajectory.aggregate_arc``).
        title: axes title; state the conclusion, not the plot type.
        ax: existing axes, or ``None`` to create a new figure.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for i, (label, arc) in enumerate(arcs.items()):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(arc["position"], arc["mean"], "-o", color=color, lw=1.8, ms=4, label=label)
        ax.fill_between(arc["position"], arc["lo"], arc["hi"], color=color, alpha=0.18, lw=0)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xlabel("normalized position in the conversation (0 = start, 1 = end)")
    ax.set_ylabel("mean valence (95% bootstrap CI)")
    ax.set_title(title)
    ax.legend(frameon=False)
    return ax


def plot_transition_heatmap(
    matrix: np.ndarray,
    title: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Row-stochastic emotion transition matrix as an annotated heatmap.

    Args:
        matrix: (7, 7) transition matrix, rows = current emotion.
        title: axes title; state the conclusion, not the plot type.
        ax: existing axes, or ``None`` to create a new figure.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45, ha="right")
    ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value < 0.5 else "black",
            )
    ax.set_xlabel("next utterance emotion")
    ax.set_ylabel("current utterance emotion")
    ax.set_title(title)
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(next | current)")
    return ax


def plot_weight_forest(
    weights: pd.DataFrame,
    title: str,
    highlight: list[str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Standardized coefficients with CI whiskers, one row per predictor.

    Args:
        weights: frame indexed by predictor name with columns ``coef``,
            ``lo`` and ``hi`` (from the ``weights`` module).
        title: axes title; state the conclusion, not the plot type.
        highlight: predictor names drawn in the accent color (e.g. the
            conversation attributes, as opposed to controls).
        ax: existing axes, or ``None`` to create a new figure.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.42 * len(weights) + 1.2))
    highlight = highlight or []
    order = weights.iloc[::-1]
    positions = np.arange(len(order))
    for y, (name, row) in zip(positions, order.iterrows(), strict=True):
        color = PALETTE[0] if name in highlight else PALETTE[5]
        ax.plot([row["lo"], row["hi"]], [y, y], "-", color=color, lw=1.8)
        ax.plot(row["coef"], y, "o", color=color, ms=6)
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_yticks(positions, order.index)
    ax.set_xlabel("standardized coefficient (95% bootstrap CI)")
    ax.set_title(title)
    return ax


def plot_class_distribution(
    counts: pd.Series,
    title: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Bar chart of emotion label counts with percentage annotations."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.6))
    total = counts.sum()
    ax.bar(counts.index.astype(str), counts.to_numpy(), color=PALETTE[0], alpha=0.85)
    for x, value in enumerate(counts.to_numpy()):
        ax.text(x, value, f"{value / total:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("emotion label")
    ax.set_ylabel("utterance count")
    ax.set_title(title)
    return ax
