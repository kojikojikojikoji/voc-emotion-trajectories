"""Standardized outcome weights with bootstrap uncertainty.

The estimand is deliberately plain: a logistic regression of the binary
outcome (donated or not) on z-scored conversation attributes plus persuadee
controls, so a coefficient reads as "log-odds change per one standard
deviation of the attribute, holding the controls fixed". Uncertainty comes
from resampling dialogues with replacement (percentile intervals). A linear
counterpart handles the secondary donation-amount model, and a Spearman
fidelity metric compares an estimated weight vector against the gold one
for the annotation-quality study.

None of this is causal machinery. Persuaders adapt their strategy to the
persuadee, so the weights are descriptive associations; the notebook says
so explicitly rather than hedging in code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression


def standardize(X: pd.DataFrame) -> pd.DataFrame:
    """Z-score every column; constant columns become all zeros.

    Standardizing once on the estimation sample (rather than inside every
    bootstrap resample) keeps the coefficient scale fixed across resamples,
    which is what a percentile interval for "the" standardized coefficient
    needs.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0).replace(0, 1)
    return (X - mean) / std


def _fit_logistic(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    # C=inf is the unpenalized MLE (sklearn 1.9 deprecated penalty=None).
    model = LogisticRegression(C=np.inf, max_iter=5000, random_state=seed)
    model.fit(X, y)
    return model.coef_.ravel()


def logistic_weights(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> pd.Series:
    """Point estimates: standardized logistic coefficients.

    Args:
        X: design matrix (attributes plus controls), one row per dialogue.
        y: binary outcome aligned with ``X``.
        seed: passed to the solver for determinism.

    Raises:
        ValueError: if ``y`` is not binary with both classes present.
    """
    y_arr = np.asarray(y, dtype=int)
    if set(np.unique(y_arr)) != {0, 1}:
        raise ValueError("y must contain both classes 0 and 1")
    Z = standardize(X)
    coefs = _fit_logistic(Z.to_numpy(), y_arr, seed)
    return pd.Series(coefs, index=X.columns, name="coef")


def bootstrap_logistic_weights(
    X: pd.DataFrame,
    y: pd.Series,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Standardized logistic coefficients with percentile bootstrap CIs.

    Rows (dialogues) are resampled with replacement; resamples that lose one
    outcome class entirely are redrawn (they carry no information about the
    coefficients). Standardization is fixed on the full sample, see
    :func:`standardize`.

    Returns:
        Frame indexed by column name with ``coef`` (full-sample point
        estimate), ``lo`` and ``hi`` (percentile interval at level
        ``1 - alpha``), and ``excludes_zero`` (bool).
    """
    point = logistic_weights(X, y, seed=seed)
    Z = standardize(X).to_numpy()
    y_arr = np.asarray(y, dtype=int)
    n = len(y_arr)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, Z.shape[1]))
    for b in range(n_boot):
        while True:
            idx = rng.integers(0, n, size=n)
            if 0 < y_arr[idx].sum() < n:
                break
        draws[b] = _fit_logistic(Z[idx], y_arr[idx], seed)
    lo = np.percentile(draws, 100 * alpha / 2, axis=0)
    hi = np.percentile(draws, 100 * (1 - alpha / 2), axis=0)
    out = pd.DataFrame({"coef": point, "lo": lo, "hi": hi}, index=X.columns)
    out["excludes_zero"] = (out["lo"] > 0) | (out["hi"] < 0)
    return out


def _fit_linear(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return beta[1:]


def linear_weights(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Standardized OLS coefficients (secondary donation-amount model)."""
    Z = standardize(X)
    coefs = _fit_linear(Z.to_numpy(), np.asarray(y, dtype=float))
    return pd.Series(coefs, index=X.columns, name="coef")


def bootstrap_linear_weights(
    X: pd.DataFrame,
    y: pd.Series,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Standardized OLS coefficients with percentile bootstrap CIs."""
    point = linear_weights(X, y)
    Z = standardize(X).to_numpy()
    y_arr = np.asarray(y, dtype=float)
    n = len(y_arr)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, Z.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = _fit_linear(Z[idx], y_arr[idx])
    lo = np.percentile(draws, 100 * alpha / 2, axis=0)
    hi = np.percentile(draws, 100 * (1 - alpha / 2), axis=0)
    out = pd.DataFrame({"coef": point, "lo": lo, "hi": hi}, index=X.columns)
    out["excludes_zero"] = (out["lo"] > 0) | (out["hi"] < 0)
    return out


def weight_fidelity(coefs: pd.Series, reference: pd.Series, subset: list[str]) -> float:
    """Spearman rank correlation between two weight vectors on ``subset``.

    The metric asks a decision question, not a calibration one: does the
    degraded annotation still rank the conversation attributes in the same
    order of importance as the gold annotation? Identical vectors give 1.0.

    Raises:
        ValueError: if ``subset`` has fewer than 3 names or a name is
            missing from either vector.
    """
    if len(subset) < 3:
        raise ValueError("need at least 3 attributes for a rank correlation")
    missing = [name for name in subset if name not in coefs.index or name not in reference.index]
    if missing:
        raise ValueError(f"missing coefficients: {missing}")
    result = spearmanr(coefs[subset].to_numpy(), reference[subset].to_numpy())
    return float(result.statistic)
