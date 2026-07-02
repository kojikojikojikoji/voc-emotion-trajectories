"""Load and split the DailyDialog dataset.

DailyDialog (Li et al., IJCNLP 2017) is a corpus of 13,118 scripted
multi-turn dialogues with one emotion label per utterance, released under
CC BY-NC-SA 4.0. The raw distribution is a zip archive containing
``dialogues_text.txt`` (one dialogue per line, utterances separated by
``__eou__``) and ``dialogues_emotion.txt`` (one line of space-separated
integer labels per dialogue). This module parses that format into a tidy
utterance-level DataFrame and provides a dialogue-level train/test split:
utterances of one dialogue never cross the split boundary, because
utterances within a dialogue are strongly dependent and splitting them
would leak conversational context between train and test.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

#: Emotion id -> name, from the readme of the DailyDialog distribution.
EMOTIONS: tuple[str, ...] = (
    "no_emotion",
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
)

EMOTION_TO_ID: dict[str, int] = {name: i for i, name in enumerate(EMOTIONS)}

RAW_ZIP_NAME = "ijcnlp_dailydialog.zip"
SAMPLE_CSV_NAME = "dailydialog_sample.csv"

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def repo_root() -> Path:
    """Repository root (parent of ``src/``); used to locate ``assets/``."""
    return _REPO_ROOT


def data_dir() -> Path:
    """Resolve the data directory from ``VOC_DATA_DIR`` (default ``./data``).

    The value is read from the environment after loading a ``.env`` file at
    the repository root, if present. Relative values are resolved from the
    repository root so the result does not depend on the working directory.
    """
    load_dotenv(_REPO_ROOT / ".env")
    raw = os.environ.get("VOC_DATA_DIR", "./data")
    path = Path(raw)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def raw_zip_path() -> Path:
    """Expected location of the raw DailyDialog archive."""
    return data_dir() / "raw" / RAW_ZIP_NAME


def sample_csv_path() -> Path:
    """Location of the committed utterance-level sample CSV."""
    return data_dir() / "sample" / SAMPLE_CSV_NAME


def raw_available() -> bool:
    """Whether the raw archive has been downloaded."""
    return raw_zip_path().is_file()


def parse_dailydialog(text_lines: list[str], emotion_lines: list[str]) -> pd.DataFrame:
    """Parse the raw DailyDialog line format into an utterance-level frame.

    Args:
        text_lines: one dialogue per line, utterances separated by ``__eou__``.
        emotion_lines: one line per dialogue with space-separated integer
            labels, aligned with ``text_lines``.

    Returns:
        DataFrame with columns ``dialogue_id`` (int, position in the input),
        ``turn`` (int, 0-based utterance index), ``text`` (str),
        ``emotion`` (int label id) and ``emotion_name`` (str). Dialogues
        whose utterance count and label count disagree are dropped (the
        original distribution contains one such dialogue).

    Raises:
        ValueError: if the two inputs have different numbers of lines or a
            label is outside the documented 0..6 range.
    """
    if len(text_lines) != len(emotion_lines):
        raise ValueError(
            f"text and emotion files disagree: {len(text_lines)} vs {len(emotion_lines)} lines"
        )
    rows: list[tuple[int, int, str, int]] = []
    for dialogue_id, (text_line, emotion_line) in enumerate(
        zip(text_lines, emotion_lines, strict=True)
    ):
        utterances = [u.strip() for u in text_line.split("__eou__") if u.strip()]
        labels = [int(tok) for tok in emotion_line.split()]
        if any(label not in range(len(EMOTIONS)) for label in labels):
            raise ValueError(f"emotion label out of range in dialogue {dialogue_id}")
        if len(utterances) != len(labels):
            continue  # known annotation misalignment; drop the dialogue
        rows.extend(
            (dialogue_id, turn, text, label)
            for turn, (text, label) in enumerate(zip(utterances, labels, strict=True))
        )
    df = pd.DataFrame(rows, columns=["dialogue_id", "turn", "text", "emotion"])
    df["emotion_name"] = df["emotion"].map(dict(enumerate(EMOTIONS)))
    return df


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the full dataset from the raw zip archive.

    Args:
        path: archive location; defaults to :func:`raw_zip_path`.

    Raises:
        FileNotFoundError: with a pointer to the download script when the
            archive is missing.
    """
    path = raw_zip_path() if path is None else path
    if not path.is_file():
        raise FileNotFoundError(
            f"DailyDialog archive not found ({path.name}). "
            "Run 'python scripts/download_data.py' first, or use load_sample()."
        )
    with zipfile.ZipFile(path) as zf:
        with zf.open("ijcnlp_dailydialog/dialogues_text.txt") as fh:
            text_lines = fh.read().decode("utf-8").splitlines()
        with zf.open("ijcnlp_dailydialog/dialogues_emotion.txt") as fh:
            emotion_lines = fh.read().decode("utf-8").splitlines()
    return parse_dailydialog(text_lines, emotion_lines)


def load_sample(path: Path | None = None) -> pd.DataFrame:
    """Load the committed 1,000-dialogue sample (same schema as ``load_raw``)."""
    path = sample_csv_path() if path is None else path
    if not path.is_file():
        raise FileNotFoundError(
            f"sample CSV not found ({path.name}); expected it under data/sample/ in the repository"
        )
    df = pd.read_csv(path, dtype={"dialogue_id": int, "turn": int, "text": str, "emotion": int})
    df["emotion_name"] = df["emotion"].map(dict(enumerate(EMOTIONS)))
    return df


def load_dataset() -> tuple[pd.DataFrame, str]:
    """Load the full dataset when available, else fall back to the sample.

    Returns:
        ``(df, source)`` where ``source`` is ``"raw"`` or ``"sample"``.
        The fallback keeps the notebooks runnable (and CI green) without
        any download.
    """
    if raw_available():
        return load_raw(), "raw"
    return load_sample(), "sample"


def split_dialogues(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic dialogue-level train/test split.

    Dialogue ids are shuffled with a seeded RNG and assigned to one side as
    whole conversations, so no dialogue contributes utterances to both
    sides (utterance-level splitting would leak conversational context).

    Args:
        df: utterance-level frame with a ``dialogue_id`` column.
        test_size: fraction of dialogues (not utterances) in the test set.
        seed: RNG seed; the same seed always yields the same split.

    Raises:
        ValueError: if ``test_size`` is outside (0, 1).
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    ids = np.array(sorted(df["dialogue_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_test = max(1, int(round(len(ids) * test_size)))
    test_ids = set(ids[:n_test].tolist())
    mask = df["dialogue_id"].isin(test_ids)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)
