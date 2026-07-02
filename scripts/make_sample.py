"""Regenerate the committed sample from the raw archive (maintainers only).

Usage:
    python scripts/download_data.py && python scripts/make_sample.py

Writes ``data/sample/dailydialog_sample.csv``: a seeded random sample of
1,000 whole dialogues (about 7,900 utterances) in the parsed utterance-level
schema. Whole dialogues are sampled, never individual utterances, so the
sample preserves conversational structure for the trajectory notebooks. The
label distribution is left as-is (about 83 percent ``no_emotion``); the
imbalance is part of what the notebooks demonstrate.

Redistribution basis: DailyDialog is CC BY-NC-SA 4.0, which allows sharing
adapted material with attribution under the same license. Attribution and
license pointers live in ``data/sample/README.md`` next to the CSV (a
comment header inside the CSV would break parsing of utterances that
contain a ``#`` character).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voc_arc.data import load_raw, sample_csv_path  # noqa: E402

N_DIALOGUES = 1000
SEED = 42


def main() -> int:
    df = load_raw()
    rng = np.random.default_rng(SEED)
    ids = np.sort(df["dialogue_id"].unique())
    chosen = set(rng.choice(ids, size=N_DIALOGUES, replace=False).tolist())
    sample = df[df["dialogue_id"].isin(chosen)].drop(columns=["emotion_name"])
    dest = sample_csv_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(dest, index=False)
    print(f"wrote {dest.name}: {N_DIALOGUES} dialogues, {len(sample)} utterances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
