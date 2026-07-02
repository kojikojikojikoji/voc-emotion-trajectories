"""Utterance-level emotion classifiers and honest baselines.

The main model is deliberately simple: TF-IDF features (word unigrams and
bigrams plus character 3-5 grams) feeding a multinomial logistic regression
with balanced class weights. It is fast, fully inspectable, needs no GPU
and exposes calibrated-ish class probabilities via ``predict_proba``, which
the trajectory layer consumes. Two baselines frame the comparison: a
majority-class predictor (the floor any model must beat on a corpus that is
83 percent ``no_emotion``) and a small keyword lexicon (what a rule-based
VoC dashboard would do).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from .data import EMOTION_TO_ID, EMOTIONS

#: Minimal emotion keyword lexicon for the rule baseline. Deliberately
#: small and generic; the point is a credible rule-based reference, not a
#: tuned competitor.
LEXICON: dict[str, tuple[str, ...]] = {
    "anger": ("angry", "furious", "annoyed", "hate", "damn", "mad", "fed up", "outrageous"),
    "disgust": ("disgusting", "gross", "nasty", "awful", "yuck", "sickening"),
    "fear": ("afraid", "scared", "terrified", "frightened", "worried sick", "horrible"),
    "happiness": (
        "happy",
        "glad",
        "great",
        "wonderful",
        "love",
        "thank",
        "nice",
        "pleased",
        "excellent",
        "enjoy",
    ),
    "sadness": ("sad", "sorry to hear", "miss", "unhappy", "depressed", "cry", "lonely"),
    "surprise": ("wow", "really ?", "unbelievable", "amazing", "no way", "can't believe"),
}


class MajorityBaseline:
    """Predict the most frequent training label for every utterance."""

    def __init__(self) -> None:
        self.majority_: int | None = None

    def fit(self, texts: list[str], labels: np.ndarray) -> MajorityBaseline:
        values, counts = np.unique(np.asarray(labels), return_counts=True)
        self.majority_ = int(values[np.argmax(counts)])
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        if self.majority_ is None:
            raise ValueError("fit must be called before predict")
        return np.full(len(texts), self.majority_, dtype=int)


class LexiconBaseline:
    """Keyword-matching baseline: first emotion whose keyword appears wins.

    Utterances are lowercased; emotions are checked in the fixed order of
    :data:`LEXICON` and the fallback label is ``no_emotion``. ``fit`` is a
    no-op kept for interface symmetry.
    """

    def fit(self, texts: list[str], labels: np.ndarray) -> LexiconBaseline:
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        out = np.zeros(len(texts), dtype=int)
        for i, text in enumerate(texts):
            lowered = text.lower()
            for emotion, keywords in LEXICON.items():
                if any(keyword in lowered for keyword in keywords):
                    out[i] = EMOTION_TO_ID[emotion]
                    break
        return out


def build_model(seed: int = 42, max_word_features: int = 50_000) -> Pipeline:
    """TF-IDF (word 1-2 grams + char 3-5 grams) into logistic regression.

    Balanced class weights counter the 83 percent ``no_emotion`` skew:
    without them the model collapses onto the majority class and macro-F1
    barely moves. Character n-grams pick up morphology and punctuation
    (exclamation-like patterns, elongations) that word tokens miss.
    """
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=max_word_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=max_word_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("tfidf", features),
            (
                "logreg",
                LogisticRegression(
                    C=2.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Macro-F1, weighted-F1 and accuracy in one dict."""
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Per-class F1 with support, indexed by emotion name."""
    labels = list(range(len(EMOTIONS)))
    scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    support = pd.Series(y_true).value_counts().reindex(labels, fill_value=0)
    return pd.DataFrame(
        {"f1": scores, "support": support.to_numpy()},
        index=pd.Index(EMOTIONS, name="emotion"),
    )
