"""Regenerate the committed samples from the raw data (maintainers only).

Usage:
    python scripts/download_data.py && python scripts/make_sample.py [dailydialog|p4g|all]

DailyDialog sample (``data/sample/dailydialog_sample.csv``): a seeded random
sample of 1,000 whole dialogues (about 7,900 utterances) in the parsed
utterance-level schema. Whole dialogues are sampled, never individual
utterances, so the sample preserves conversational structure for the
trajectory notebooks. The label distribution is left as-is (about 83
percent ``no_emotion``); the imbalance is part of what the notebooks
demonstrate. Redistribution basis: CC BY-NC-SA 4.0 allows sharing adapted
material with attribution under the same license.

Persuasion for Good sample (``data/sample/p4g_annotated_sample.csv`` and
``p4g_info_sample.csv``): the complete 300-dialogue annotated subset in the
tidy sentence schema (dropping only the sentence-level sentiment scores of
the raw workbook), plus the persuadee outcome and profile rows for those
dialogues. All 300 dialogues are kept because they are the gold annotation
set notebook 04 is about; a random subset would change every reported
number. Redistribution basis: Apache-2.0 permits redistribution with
attribution.

Attribution and license pointers live in ``data/sample/README.md`` next to
the CSVs (a comment header inside a CSV would break parsing of utterances
that contain a ``#`` character).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voc_arc import p4g  # noqa: E402
from voc_arc.data import load_raw, sample_csv_path  # noqa: E402

N_DIALOGUES = 1000
SEED = 42


def make_dailydialog_sample() -> None:
    df = load_raw()
    rng = np.random.default_rng(SEED)
    ids = np.sort(df["dialogue_id"].unique())
    chosen = set(rng.choice(ids, size=N_DIALOGUES, replace=False).tolist())
    sample = df[df["dialogue_id"].isin(chosen)].drop(columns=["emotion_name"])
    dest = sample_csv_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(dest, index=False)
    print(f"wrote {dest.name}: {N_DIALOGUES} dialogues, {len(sample)} utterances")


def make_p4g_sample() -> None:
    annotated = p4g.load_annotated_raw()
    dest = p4g.annotated_sample_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    # The mapped strategy column is derived, not source data: it is
    # recomputed at load time, so the committed file stays close to the
    # original workbook.
    annotated.drop(columns=["strategy"]).to_csv(dest, index=False)
    n_dialogues = annotated["dialogue_id"].nunique()
    print(f"wrote {dest.name}: {n_dialogues} dialogues, {len(annotated)} sentences")

    info = p4g.load_info_raw()
    subset = info[info["dialogue_id"].isin(set(annotated["dialogue_id"]))]
    subset.to_csv(p4g.info_sample_path(), index=False)
    print(f"wrote {p4g.info_sample_path().name}: {len(subset)} persuadee rows")


def main(argv: list[str]) -> int:
    choice = argv[1] if len(argv) > 1 else "all"
    if choice not in ("dailydialog", "p4g", "all"):
        print(__doc__)
        return 2
    if choice in ("dailydialog", "all"):
        make_dailydialog_sample()
    if choice in ("p4g", "all"):
        make_p4g_sample()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
