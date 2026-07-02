"""Load and normalize the Persuasion for Good dataset.

Persuasion for Good (Wang et al., ACL 2019) is a corpus of 1,017 dyadic
conversations in which one crowd worker (the persuader) tries to convince
another (the persuadee) to donate part of their task payment to the charity
Save the Children, released under Apache-2.0. Two distributions matter here:

- ``data/FullData``: all 1,017 dialogues as one-sentence-per-row CSVs plus a
  participant-info table with the actual donation (``B6``) and psychological
  profiles (Big-Five, demographics) for both roles;
- ``data/AnnotatedData``: a 300-dialogue subset in which every sentence
  carries a human dialogue-act / persuasion-strategy label
  (``er_label_1`` for the persuader side, ``ee_label_1`` for the persuadee).

This module parses both into tidy sentence-level frames and maps the raw
persuader taxonomy onto the compact strategy space used by the attribute
layer (:data:`STRATEGIES`). The persuadee's actual donation is winsorized at
the task's $2 cap: the payment scheme made donations above $2 impossible, so
the 35 larger values in the raw table (up to $700) are treated as data-entry
noise, as flagged in the dataset's own documentation.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from .data import data_dir

RAW_SUBDIR = "persuasionforgood"
ANNOTATED_XLSX_NAME = "300_dialog.xlsx"
FULL_DIALOG_CSV_NAME = "full_dialog.csv"
FULL_INFO_CSV_NAME = "full_info.csv"
ANNOTATED_SAMPLE_NAME = "p4g_annotated_sample.csv"
INFO_SAMPLE_NAME = "p4g_info_sample.csv"

#: Task cap on donations in USD; values above it are data-entry noise.
DONATION_CAP = 2.0

#: Compact strategy space for the attribute layer. The first five classes
#: drive attributes; ``other`` absorbs the remaining dialogue acts.
STRATEGIES: tuple[str, ...] = (
    "logical_appeal",
    "emotion_appeal",
    "credibility_appeal",
    "personal_story",
    "donation_ask",
    "other",
)

#: Raw persuader labels (``er_label_1``) that map to a non-``other`` class.
#: ``personal_story`` bundles the two self-disclosure strategies of the
#: original taxonomy; ``donation_ask`` bundles the three explicit asks. Any
#: raw label not listed here (greeting, thank, acknowledgement,
#: donation-information, the inquiry acts, and so on) maps to ``other``.
STRATEGY_MAP: dict[str, str] = {
    "logical-appeal": "logical_appeal",
    "emotion-appeal": "emotion_appeal",
    "credibility-appeal": "credibility_appeal",
    "personal-story": "personal_story",
    "self-modeling": "personal_story",
    "proposition-of-donation": "donation_ask",
    "ask-donation-amount": "donation_ask",
    "ask-donate-more": "donation_ask",
}

#: Big-Five columns in ``full_info.csv`` -> control names used downstream.
BIG_FIVE_MAP: dict[str, str] = {
    "extrovert.x": "extrovert",
    "agreeable.x": "agreeable",
    "conscientious.x": "conscientious",
    "neurotic.x": "neurotic",
    "open.x": "open",
}

CONTROL_COLUMNS: tuple[str, ...] = (*BIG_FIVE_MAP.values(), "age", "is_male")


def p4g_raw_dir() -> Path:
    """Directory holding the downloaded Persuasion for Good files."""
    return data_dir() / "raw" / RAW_SUBDIR


def annotated_xlsx_path() -> Path:
    """Expected location of the annotated 300-dialogue workbook."""
    return p4g_raw_dir() / ANNOTATED_XLSX_NAME


def full_dialog_path() -> Path:
    """Expected location of the full 1,017-dialogue sentence CSV."""
    return p4g_raw_dir() / FULL_DIALOG_CSV_NAME


def full_info_path() -> Path:
    """Expected location of the participant-info CSV."""
    return p4g_raw_dir() / FULL_INFO_CSV_NAME


def annotated_sample_path() -> Path:
    """Location of the committed annotated-subset sample CSV."""
    return data_dir() / "sample" / ANNOTATED_SAMPLE_NAME


def info_sample_path() -> Path:
    """Location of the committed persuadee-info sample CSV."""
    return data_dir() / "sample" / INFO_SAMPLE_NAME


def p4g_raw_available() -> bool:
    """Whether the full raw distribution has been downloaded."""
    return annotated_xlsx_path().is_file() and full_info_path().is_file()


def map_strategy(raw_label: object) -> str | None:
    """Map a raw ``er_label_1`` value onto :data:`STRATEGIES`.

    Unknown but present labels fall into ``other`` (the taxonomy has around
    30 dialogue acts and only the eight in :data:`STRATEGY_MAP` are salient
    here); missing labels (persuadee rows) return ``None``.
    """
    if pd.isna(raw_label):
        return None
    return STRATEGY_MAP.get(str(raw_label), "other")


def _tidy_dialog(raw: pd.DataFrame, with_labels: bool) -> pd.DataFrame:
    """Normalize a raw sentence-level frame into the tidy schema.

    Output columns: ``dialogue_id`` (str), ``sent_idx`` (0-based sentence
    position within the dialogue, in file order), ``turn`` (int),
    ``role`` (``"persuader"`` / ``"persuadee"``), ``text`` (str) and, when
    ``with_labels`` is set, ``er_label`` / ``ee_label`` (raw taxonomy) plus
    ``strategy`` (mapped, ``None`` on persuadee rows).
    """
    out = pd.DataFrame(
        {
            "dialogue_id": raw["B2"].astype(str),
            "turn": raw["Turn"].astype(int),
            "role": raw["B4"].astype(int).map({0: "persuader", 1: "persuadee"}),
            "text": raw["Unit"].astype(str),
        }
    )
    if with_labels:
        out["er_label"] = raw["er_label_1"]
        out["ee_label"] = raw["ee_label_1"]
        out["strategy"] = out["er_label"].map(map_strategy)
    out = out.reset_index(drop=True)
    out["sent_idx"] = out.groupby("dialogue_id", sort=False).cumcount()
    columns = ["dialogue_id", "sent_idx", "turn", "role", "text"]
    if with_labels:
        columns += ["er_label", "ee_label", "strategy"]
    return out[columns]


def load_annotated_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the annotated 300-dialogue subset from the raw workbook.

    Raises:
        FileNotFoundError: with a pointer to the download script when the
            workbook is missing.
    """
    path = annotated_xlsx_path() if path is None else path
    if not path.is_file():
        raise FileNotFoundError(
            f"Persuasion for Good annotated workbook not found ({path.name}). "
            "Run 'python scripts/download_data.py p4g' first, or use "
            "load_annotated_sample()."
        )
    raw = pd.read_excel(path)
    return _tidy_dialog(raw, with_labels=True)


def load_annotated_sample(path: Path | None = None) -> pd.DataFrame:
    """Load the committed annotated sample (same schema as the raw loader)."""
    path = annotated_sample_path() if path is None else path
    if not path.is_file():
        raise FileNotFoundError(
            f"sample CSV not found ({path.name}); expected it under data/sample/ in the repository"
        )
    raw = pd.read_csv(path)
    raw["strategy"] = raw["er_label"].map(map_strategy)
    return raw[
        ["dialogue_id", "sent_idx", "turn", "role", "text", "er_label", "ee_label", "strategy"]
    ]


def load_annotated() -> tuple[pd.DataFrame, str]:
    """Annotated subset from raw when available, else the committed sample.

    Returns:
        ``(df, source)`` where ``source`` is ``"raw"`` or ``"sample"``. Both
        paths contain the same 300 dialogues; the sample only drops the
        sentence-level sentiment columns.
    """
    if annotated_xlsx_path().is_file():
        return load_annotated_raw(), "raw"
    return load_annotated_sample(), "sample"


def load_full_dialog(path: Path | None = None) -> pd.DataFrame:
    """Load all 1,017 dialogues (sentence level, no labels) from the raw CSV.

    Raises:
        FileNotFoundError: when the raw CSV has not been downloaded; the
            full corpus has no committed sample, only the annotated subset.
    """
    path = full_dialog_path() if path is None else path
    if not path.is_file():
        raise FileNotFoundError(
            f"Persuasion for Good full corpus not found ({path.name}). "
            "Run 'python scripts/download_data.py p4g' first."
        )
    raw = pd.read_csv(path)
    return _tidy_dialog(raw, with_labels=False)


def _tidy_info(raw: pd.DataFrame) -> pd.DataFrame:
    """Persuadee rows of the info table in the tidy control schema."""
    ee = raw[raw["B4"].astype(int) == 1].copy()
    out = pd.DataFrame(
        {
            "dialogue_id": ee["B2"].astype(str),
            "donation_raw": ee["B6"].astype(float),
            "age": ee["age.x"].astype(float),
            "sex": ee["sex.x"],
        }
    )
    for src, dst in BIG_FIVE_MAP.items():
        out[dst] = ee[src].astype(float).to_numpy()
    return out.reset_index(drop=True)


def load_info_raw(path: Path | None = None) -> pd.DataFrame:
    """Persuadee outcome and profile rows from the raw ``full_info.csv``.

    Raises:
        FileNotFoundError: with a pointer to the download script.
    """
    path = full_info_path() if path is None else path
    if not path.is_file():
        raise FileNotFoundError(
            f"Persuasion for Good info table not found ({path.name}). "
            "Run 'python scripts/download_data.py p4g' first, or use load_info_sample()."
        )
    return _tidy_info(pd.read_csv(path))


def load_info_sample(path: Path | None = None) -> pd.DataFrame:
    """Committed persuadee info for the 300 annotated dialogues."""
    path = info_sample_path() if path is None else path
    if not path.is_file():
        raise FileNotFoundError(
            f"sample CSV not found ({path.name}); expected it under data/sample/ in the repository"
        )
    return pd.read_csv(path)


def load_info() -> tuple[pd.DataFrame, str]:
    """Persuadee info from raw when available, else the committed sample."""
    if full_info_path().is_file():
        return load_info_raw(), "raw"
    return load_info_sample(), "sample"


def prepare_outcome_controls(info: pd.DataFrame, cap: float = DONATION_CAP) -> pd.DataFrame:
    """Build the outcome and control matrix from tidy persuadee info.

    Args:
        info: frame from one of the info loaders (one row per dialogue).
        cap: winsorization cap for the donation (default the $2 task cap).

    Returns:
        Frame indexed by ``dialogue_id`` with ``donation`` (winsorized at
        ``cap``), ``donated`` (int, donation > 0) and the controls of
        :data:`CONTROL_COLUMNS`: the five Big-Five scores and age
        (median-imputed where missing, which affects a handful of rows) and
        an ``is_male`` indicator (1 for ``Male``, 0 otherwise, including the
        few missing / ``Other`` entries).

    Raises:
        ValueError: if a dialogue id appears more than once.
    """
    if info["dialogue_id"].duplicated().any():
        raise ValueError("info contains duplicate dialogue ids")
    out = pd.DataFrame(index=pd.Index(info["dialogue_id"], name="dialogue_id"))
    donation = info["donation_raw"].astype(float).clip(upper=cap).to_numpy()
    out["donation"] = donation
    out["donated"] = (donation > 0).astype(int)
    for column in (*BIG_FIVE_MAP.values(), "age"):
        values = info[column].astype(float)
        out[column] = values.fillna(values.median()).to_numpy()
    out["is_male"] = (info["sex"] == "Male").astype(int).to_numpy()
    return out


def looks_like_annotated_xlsx(path: Path) -> bool:
    """Cheap validity check used by the download script (xlsx is a zip)."""
    if not path.is_file() or path.stat().st_size < 100_000:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            return "xl/workbook.xml" in zf.namelist()
    except zipfile.BadZipFile:
        return False
