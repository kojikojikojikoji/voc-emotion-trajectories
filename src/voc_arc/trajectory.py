"""Turn per-utterance emotion sequences into conversation-level trajectories.

This is the layer that treats a conversation as a time series instead of a
bag of utterances. Core pieces:

- a valence mapping from the seven DailyDialog emotions to [-1, 0, +1]
  (a modeling choice, documented and kept deliberately coarse);
- EWMA smoothing of per-utterance valence, because single-utterance labels
  are noisy and an arc is a low-frequency object;
- position normalization to [0, 1] so dialogues of different lengths can be
  aggregated into one average arc with bootstrap confidence bands;
- a first-order Markov transition matrix between emotions (row-stochastic);
- a single change-point detector that finds the largest mean shift in a
  smoothed valence series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import EMOTIONS

#: Emotion id -> valence. A modeling choice, not a property of the data:
#: happiness is positive; anger, disgust, fear and sadness are negative;
#: surprise is ambiguous (can be either) and no_emotion carries no signal,
#: so both map to zero. Any downstream valence result depends on this table.
VALENCE: dict[int, float] = {
    0: 0.0,  # no_emotion
    1: -1.0,  # anger
    2: -1.0,  # disgust
    3: -1.0,  # fear
    4: 1.0,  # happiness
    5: -1.0,  # sadness
    6: 0.0,  # surprise
}

_VALENCE_VECTOR = np.array([VALENCE[i] for i in range(len(EMOTIONS))])


def valence_series(labels: np.ndarray) -> np.ndarray:
    """Map integer emotion labels to their valence values.

    Raises:
        ValueError: if a label is outside 0..6.
    """
    labels = np.asarray(labels, dtype=int)
    if labels.size and (labels.min() < 0 or labels.max() >= len(EMOTIONS)):
        raise ValueError("labels must be in 0..6")
    return _VALENCE_VECTOR[labels]


def expected_valence(proba: np.ndarray) -> np.ndarray:
    """Probability-weighted valence from a (n, 7) class-probability matrix.

    Using the full distribution instead of the argmax keeps low-confidence
    predictions close to zero rather than snapping them to +-1.
    """
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2 or proba.shape[1] != len(EMOTIONS):
        raise ValueError(f"proba must have shape (n, {len(EMOTIONS)})")
    return proba @ _VALENCE_VECTOR


def ewma(values: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Exponentially weighted moving average with explicit edge behavior.

    ``out[0] == values[0]``; ``alpha=1`` returns the input unchanged; an
    empty input returns an empty array.

    Raises:
        ValueError: if ``alpha`` is outside (0, 1].
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    if values.size == 0:
        return out
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def normalized_positions(n: int) -> np.ndarray:
    """Positions of ``n`` utterances on [0, 1]; a single utterance sits at 0.5.

    Raises:
        ValueError: if ``n`` is not positive.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    if n == 1:
        return np.array([0.5])
    return np.linspace(0.0, 1.0, n)


def transition_matrix(sequences: list[np.ndarray], n_states: int = len(EMOTIONS)) -> np.ndarray:
    """First-order Markov transition matrix from label sequences.

    Counts transitions within each sequence (never across sequence
    boundaries) and row-normalizes. States that never occur as a source get
    a uniform row so that every row sums to exactly 1 and the matrix stays
    a valid stochastic matrix.

    Args:
        sequences: list of integer label arrays, one per dialogue.
        n_states: number of states (default 7 emotions).

    Returns:
        (n_states, n_states) array; ``out[i, j]`` is P(next=j | current=i).
    """
    counts = np.zeros((n_states, n_states), dtype=float)
    for seq in sequences:
        seq = np.asarray(seq, dtype=int)
        if seq.size and (seq.min() < 0 or seq.max() >= n_states):
            raise ValueError(f"labels must be in 0..{n_states - 1}")
        if seq.size >= 2:
            np.add.at(counts, (seq[:-1], seq[1:]), 1.0)
    row_sums = counts.sum(axis=1, keepdims=True)
    out = np.where(row_sums > 0, counts / np.where(row_sums == 0, 1.0, row_sums), 1.0 / n_states)
    return out


def aggregate_arc(
    dialogues: list[np.ndarray],
    n_bins: int = 10,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Average valence arc over many dialogues with a bootstrap CI band.

    Each dialogue's utterances are placed at normalized positions on
    [0, 1] and assigned to ``n_bins`` equal-width bins; the arc is the mean
    valence per bin. Uncertainty comes from resampling whole dialogues
    (not utterances) with replacement, because utterances within a dialogue
    are dependent and resampling them would understate the variance.

    Args:
        dialogues: list of per-dialogue valence arrays.
        n_bins: number of position bins.
        n_boot: bootstrap resamples of the dialogue list.
        alpha: two-sided CI level (0.05 gives a 95 percent band).
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame with columns ``position`` (bin centers), ``mean``,
        ``lo`` and ``hi``.
    """
    if not dialogues:
        raise ValueError("dialogues must be non-empty")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0

    def _bin_sums(valence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos = normalized_positions(len(valence))
        idx = np.clip(np.digitize(pos, edges[1:-1]), 0, n_bins - 1)
        sums = np.bincount(idx, weights=valence, minlength=n_bins)
        counts = np.bincount(idx, minlength=n_bins).astype(float)
        return sums, counts

    per_dialogue = [_bin_sums(np.asarray(v, dtype=float)) for v in dialogues]
    all_sums = np.stack([s for s, _ in per_dialogue])
    all_counts = np.stack([c for _, c in per_dialogue])

    def _mean_arc(indices: np.ndarray) -> np.ndarray:
        sums = all_sums[indices].sum(axis=0)
        counts = all_counts[indices].sum(axis=0)
        with np.errstate(invalid="ignore"):
            return np.where(counts > 0, sums / np.where(counts == 0, 1.0, counts), np.nan)

    n = len(dialogues)
    point = _mean_arc(np.arange(n))
    rng = np.random.default_rng(seed)
    boot = np.stack([_mean_arc(rng.integers(0, n, size=n)) for _ in range(n_boot)])
    lo = np.nanpercentile(boot, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0)
    return pd.DataFrame({"position": centers, "mean": point, "lo": lo, "hi": hi})


def detect_shift(valence: np.ndarray, alpha: float = 0.4) -> tuple[int, float]:
    """Locate the single largest mean shift in a smoothed valence series.

    The series is EWMA-smoothed, then every split point ``k`` is scored by
    the absolute difference between the mean after and the mean before the
    split; the best split is returned. This is a deliberately simple
    single change-point detector: it assumes at most one dominant shift and
    reports the strongest one even in flat series (magnitude near zero).

    Args:
        valence: raw per-utterance valence array (length >= 2).
        alpha: EWMA smoothing factor passed to :func:`ewma`.

    Returns:
        ``(index, magnitude)``: the shift starts at ``index`` (first
        utterance of the new regime) and ``magnitude`` is the signed mean
        difference (after minus before) of the smoothed series.

    Raises:
        ValueError: if the series has fewer than 2 points.
    """
    valence = np.asarray(valence, dtype=float)
    if valence.size < 2:
        raise ValueError("need at least 2 points to detect a shift")
    smoothed = ewma(valence, alpha=alpha)
    best_k, best_delta = 1, 0.0
    for k in range(1, len(smoothed)):
        delta = smoothed[k:].mean() - smoothed[:k].mean()
        if abs(delta) > abs(best_delta):
            best_k, best_delta = k, delta
    return best_k, float(best_delta)


def dialogue_valences(df: pd.DataFrame, column: str = "emotion") -> list[np.ndarray]:
    """Per-dialogue valence arrays from an utterance-level frame.

    Groups by ``dialogue_id`` (preserving turn order) and maps the given
    integer label column through :func:`valence_series`.
    """
    ordered = df.sort_values(["dialogue_id", "turn"])
    return [
        valence_series(group[column].to_numpy())
        for _, group in ordered.groupby("dialogue_id", sort=True)
    ]
