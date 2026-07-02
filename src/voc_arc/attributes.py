"""Conversation-level attributes for the annotation-to-outcome study.

Seven per-dialogue attributes summarize how a persuasion conversation was
conducted. Five depend on per-sentence strategy annotations (and therefore
degrade when the annotation degrades); two are annotation-free contrasts:

- ``logical_appeal_rate``, ``emotional_appeal_rate``,
  ``credibility_appeal_rate``, ``personal_story_rate``: share of persuader
  sentences carrying the respective strategy;
- ``ask_timing``: normalized position (0 = first sentence, 1 = last) of the
  first explicit donation ask; dialogues without an ask are censored at 1;
- ``persuadee_engagement``: share of persuadee sentences containing a
  question mark (raw text only);
- ``valence_arc_end``: final value of the persuadee's EWMA-smoothed expected
  valence, from an utterance emotion classifier trained elsewhere (a domain
  transfer; the caller owns that caveat).

The module also provides the seeded label-noise machinery that simulates an
imperfect annotator by corrupting gold labels at a controlled rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .p4g import STRATEGIES
from .trajectory import ewma, expected_valence

#: Attribute names in canonical order (annotation-dependent first).
ATTRIBUTES: tuple[str, ...] = (
    "logical_appeal_rate",
    "emotional_appeal_rate",
    "credibility_appeal_rate",
    "personal_story_rate",
    "ask_timing",
    "persuadee_engagement",
    "valence_arc_end",
)

#: strategy class -> rate attribute name.
RATE_ATTRIBUTES: dict[str, str] = {
    "logical_appeal": "logical_appeal_rate",
    "emotion_appeal": "emotional_appeal_rate",
    "credibility_appeal": "credibility_appeal_rate",
    "personal_story": "personal_story_rate",
}


def _normalized_position(sent_idx: int, n_sentences: int) -> float:
    """Position of a sentence on [0, 1] within its dialogue."""
    if n_sentences <= 1:
        return 0.0
    return sent_idx / (n_sentences - 1)


def strategy_attributes(df: pd.DataFrame, label_col: str = "strategy") -> pd.DataFrame:
    """Annotation-dependent attributes from a labeled sentence frame.

    Args:
        df: sentence-level frame with ``dialogue_id``, ``sent_idx``,
            ``role`` and ``label_col`` (a :data:`~voc_arc.p4g.STRATEGIES`
            value on persuader rows).
        label_col: column holding the strategy labels; passing a different
            column is how degraded or predicted labels enter the pipeline.

    Returns:
        Frame indexed by ``dialogue_id`` with the four strategy-rate columns
        (share of persuader sentences per strategy; 0 when a dialogue has no
        persuader sentences) and ``ask_timing`` (normalized position of the
        first ``donation_ask`` sentence, 1.0 when the dialogue contains
        none, a censoring choice documented in the notebook).
    """
    ordered = df.sort_values(["dialogue_id", "sent_idx"])
    n_sentences = ordered.groupby("dialogue_id")["sent_idx"].size()
    persuader = ordered[ordered["role"] == "persuader"]

    counts = (
        persuader.groupby("dialogue_id")[label_col]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=list(STRATEGIES), fill_value=0)
    )
    counts = counts.reindex(n_sentences.index, fill_value=0)
    denom = counts.sum(axis=1).replace(0, 1)

    out = pd.DataFrame(index=counts.index)
    for strategy, attribute in RATE_ATTRIBUTES.items():
        out[attribute] = counts[strategy] / denom

    asks = persuader[persuader[label_col] == "donation_ask"]
    first_ask = asks.groupby("dialogue_id")["sent_idx"].min()
    timing = pd.Series(1.0, index=counts.index, name="ask_timing")
    for dialogue_id, sent_idx in first_ask.items():
        timing.loc[dialogue_id] = _normalized_position(
            int(sent_idx), int(n_sentences.loc[dialogue_id])
        )
    out["ask_timing"] = timing
    out.index.name = "dialogue_id"
    return out


def persuadee_engagement(df: pd.DataFrame) -> pd.Series:
    """Share of persuadee sentences containing a question mark.

    Annotation-free: uses only ``role`` and ``text``. Dialogues without
    persuadee sentences get 0.
    """
    persuadee = df[df["role"] == "persuadee"]
    rate = persuadee.groupby("dialogue_id")["text"].apply(
        lambda s: float(s.str.contains("?", regex=False).mean())
    )
    all_ids = df["dialogue_id"].unique()
    return rate.reindex(all_ids, fill_value=0.0).rename("persuadee_engagement")


def valence_arc_end(df: pd.DataFrame, emotion_model: object, alpha: float = 0.4) -> pd.Series:
    """Final smoothed expected valence of the persuadee, per dialogue.

    The persuadee's sentences are scored with ``emotion_model`` (anything
    with a scikit-learn ``predict_proba`` over the seven DailyDialog emotion
    classes), converted to expected valence, EWMA-smoothed within the
    dialogue, and the last value is returned. Dialogues without persuadee
    sentences get 0 (no signal).

    Args:
        df: sentence-level frame with ``dialogue_id``, ``sent_idx``,
            ``role`` and ``text``.
        emotion_model: fitted classifier with ``predict_proba``.
        alpha: EWMA smoothing factor (same default as the trajectory layer).
    """
    persuadee = df[df["role"] == "persuadee"].sort_values(["dialogue_id", "sent_idx"])
    all_ids = df["dialogue_id"].unique()
    out = pd.Series(0.0, index=pd.Index(all_ids, name="dialogue_id"), name="valence_arc_end")
    if persuadee.empty:
        return out
    proba = emotion_model.predict_proba(persuadee["text"].tolist())
    valence = pd.Series(expected_valence(np.asarray(proba)), index=persuadee.index)
    for dialogue_id, values in valence.groupby(persuadee["dialogue_id"]):
        out.loc[dialogue_id] = float(ewma(values.to_numpy(), alpha=alpha)[-1])
    return out


def build_attributes(
    df: pd.DataFrame,
    emotion_model: object,
    label_col: str = "strategy",
    alpha: float = 0.4,
) -> pd.DataFrame:
    """All seven attributes for every dialogue in ``df``.

    Combines :func:`strategy_attributes` (annotation-dependent, driven by
    ``label_col``) with :func:`persuadee_engagement` and
    :func:`valence_arc_end` (annotation-free). Returns a frame indexed by
    ``dialogue_id`` with exactly the columns of :data:`ATTRIBUTES`.
    """
    out = strategy_attributes(df, label_col=label_col)
    out["persuadee_engagement"] = persuadee_engagement(df).reindex(out.index)
    out["valence_arc_end"] = valence_arc_end(df, emotion_model, alpha=alpha).reindex(out.index)
    return out[list(ATTRIBUTES)]


def degrade_labels(
    labels: pd.Series,
    epsilon: float,
    seed: int,
    label_space: tuple[str, ...] = STRATEGIES,
) -> pd.Series:
    """Corrupt gold labels at rate ``epsilon`` to simulate a noisy annotator.

    Each label is independently selected for corruption with probability
    ``epsilon``; a corrupted label is replaced by one of the other classes,
    chosen uniformly, so the result always stays inside ``label_space`` and
    the realized flip rate concentrates around ``epsilon``. ``epsilon=0``
    returns the input values unchanged. Missing values (persuadee rows) are
    passed through untouched.

    Args:
        labels: gold labels (may contain NaN / None on unlabeled rows).
        epsilon: corruption probability in [0, 1].
        seed: RNG seed; same seed, same corruption.
        label_space: allowed classes; non-missing labels must be inside it.

    Raises:
        ValueError: if ``epsilon`` is outside [0, 1] or a non-missing label
            is outside ``label_space``.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
    values = labels.to_numpy(dtype=object).copy()
    present = labels.notna().to_numpy()
    known = np.isin(values[present].astype(str), np.array(label_space))
    if not known.all():
        bad = sorted(set(values[present][~known].astype(str)))
        raise ValueError(f"labels outside the label space: {bad}")
    rng = np.random.default_rng(seed)
    flip = present & (rng.random(len(values)) < epsilon)
    if flip.any():
        space = np.array(label_space)
        current = np.searchsorted(np.sort(space), values[flip].astype(str))
        sorted_space = np.sort(space)
        offsets = rng.integers(1, len(space), size=int(flip.sum()))
        values[flip] = sorted_space[(current + offsets) % len(space)]
    return pd.Series(values, index=labels.index, name=labels.name)
