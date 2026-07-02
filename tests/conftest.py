"""Shared fixtures: a crafted mini-corpus in the raw DailyDialog format."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voc_arc.data import parse_dailydialog

TEXT_LINES = [
    "The kitchen stinks . __eou__ I'll throw out the garbage . __eou__",
    "Good news ! __eou__ Wonderful , I am so happy . __eou__ Me too . __eou__",
    "I lost my keys . __eou__ That is sad . __eou__",
]

EMOTION_LINES = [
    "2 0",
    "4 4 4",
    "5 5",
]


@pytest.fixture()
def mini_df() -> pd.DataFrame:
    """Parsed 3-dialogue, 7-utterance frame."""
    return parse_dailydialog(TEXT_LINES, EMOTION_LINES)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
