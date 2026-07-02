# voc-emotion-trajectories

Documentation in Japanese: [README.ja.md](./README.ja.md)

Emotion trajectories for Voice-of-Customer analysis: per-utterance emotion classification on DailyDialog, then the conversation treated as a time series, with an honest measurement of how much classifier error the trajectory layer can absorb.

![Aggregate valence arc with bootstrap confidence band](assets/valence_arc.png)

Most public emotion-classification projects stop at labeling single utterances. For Voice-of-Customer work that is the least interesting layer: what a reviewer wants to know is how a conversation moved, whether it recovered after a complaint, and whether it ended better than it started. This project builds that trajectory layer on the DailyDialog corpus: a documented emotion-to-valence mapping, EWMA-smoothed per-dialogue timelines, an aggregate valence arc over normalized conversation position with bootstrap confidence bands (dialogues, not utterances, are resampled), a Markov transition matrix between emotions, and a change-point detector validated on planted shifts. The classifier underneath is TF-IDF plus logistic regression on purpose: it trains in seconds, its errors are inspectable, and notebook 03 quantifies exactly what those errors do to the trajectories instead of assuming they average out.

## Notebooks

| # | Notebook | What it does |
|---|---|---|
| 01 | [Utterance emotion classifier](notebooks/01_utterance_emotion_classifier.ipynb) | Shows the 83 percent `no_emotion` imbalance, splits at the dialogue level to avoid leakage, and compares majority and keyword-lexicon baselines against TF-IDF plus logistic regression (macro-F1 0.390 vs 0.286 vs 0.129, asserted); per-class F1 shows fear and disgust are close to unlearnable at this scale |
| 02 | [Emotion trajectories](notebooks/02_emotion_trajectories.ipynb) | Gold-label trajectories: valence mapping (stated as a modeling choice), single-dialogue timelines, the aggregate arc (valence triples over the final fifth of a conversation), a row-stochastic transition matrix (asserted; only `no_emotion` and `happiness` are sticky), and a shift detector that must find a planted shift at the exact position (asserted) |
| 03 | [Predicted vs gold pipeline](notebooks/03_pipeline_predicted_vs_gold.ipynb) | The honest end-to-end run: trajectories from predicted labels vs gold on held-out dialogues; the aggregate arc survives (correlation 0.98, asserted above 0.8) with a documented positive level bias, while per-dialogue series reach only median correlation 0.73 and one in ten dialogues anti-correlates |

Headline result of the committed run: with an utterance classifier at macro-F1 0.39, corpus-level valence arcs are reproduced almost exactly (arc correlation 0.98) but inflated in level, and single-dialogue trajectories are right on median and badly wrong about 10 percent of the time. Aggregate VoC dashboards built this way are defensible; automated single-conversation diagnosis is not.

## Quickstart

Python 3.11.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
jupyter lab
```

All three notebooks run without any download by falling back to the committed 1,000-dialogue sample under `data/sample/` (this is how CI runs them). The download script fetches the full 13,118-dialogue archive (about 4.5 MB) for the results shown in the committed outputs; each notebook then runs in well under a minute.

## Data and license

| Dataset | Source | License | Committed here |
|---|---|---|---|
| DailyDialog (13,118 dialogues, 7 emotion labels per utterance) | Internet Archive capture (2022-05-16) of `yanran.li/files/ijcnlp_dailydialog.zip` via `scripts/download_data.py` | CC BY-NC-SA 4.0 | a 1,000-dialogue sample (`data/sample/`, about 0.6 MB), redistributed under the same license with attribution |

Notes:

- The original host `yanran.li` is a parked domain as of 2026-07 (the download URL returns an HTML placeholder), and the Hugging Face mirror `li2017dailydialog/daily_dialog` is a loading script that points at the same dead URL. The download script therefore uses the Internet Archive capture as its primary source and keeps the historical URL as a fallback.
- When using the dataset, cite Li, Su, Shen, Li, Cao, Niu, "DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset" (IJCNLP 2017).
- The CC BY-NC-SA 4.0 license is non-commercial; the code in this repository is MIT, but anything you do with the data inherits the data license.

## Repository layout

```
notebooks/            executed notebooks (outputs committed)
notebooks_src/        jupytext py:percent sources of the same notebooks
src/voc_arc/          data loading, classifier, trajectory layer, plotting
tests/                pytest suite, including hand-computed transition
                      matrix values and planted-shift detection
scripts/download_data.py
scripts/make_sample.py
data/sample/          committed 1,000-dialogue sample with attribution
assets/               representative figure used above
```

## Limitations

- DailyDialog is scripted English small talk written for language learners, not real customer conversations. The end-of-conversation positivity ritual, the non-sticky anger, and every absolute number should be re-estimated on in-domain data; what transfers is the method and the evaluation design.
- The valence mapping (happiness +1; anger, disgust, fear, sadness -1; surprise and no_emotion 0) is a modeling choice with a debatable surprise row, and every valence result depends on it. It is one small table in `src/voc_arc/trajectory.py`.
- Labels are single-annotator and about 83 percent `no_emotion`; three emotion classes have too little support to learn or to evaluate reliably at the utterance level.
- One dataset, one split, one classifier configuration. No transformer comparison is included, deliberately: this repository is the dependency-light reference point such a comparison would be measured against.
- English only; the character n-grams in the feature set do not transfer to other languages as-is.

## License

Code is MIT licensed (see [LICENSE](./LICENSE)). The DailyDialog data, including the committed sample, is CC BY-NC-SA 4.0 (table above).
