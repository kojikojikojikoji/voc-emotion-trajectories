# VoC Emotion Trajectories

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
| 04 | [Annotation quality vs outcome weights](notebooks/04_annotation_to_outcome_weights.ipynb) | A second corpus (Persuasion for Good, 1,017 donation-persuasion dialogues, 300 with gold strategy annotations and a real donation outcome): seven conversation attributes, standardized logistic weights with bootstrap CIs, and the full annotation-quality-to-weight-fidelity curve, with a classical TF-IDF annotator (accuracy 0.71, kappa 0.58) placed on it |

Headline result of the committed run: with an utterance classifier at macro-F1 0.39, corpus-level valence arcs are reproduced almost exactly (arc correlation 0.98) but inflated in level, and single-dialogue trajectories are right on median and badly wrong about 10 percent of the time. Aggregate VoC dashboards built this way are defensible; automated single-conversation diagnosis is not.

Notebook 04 asks the follow-up systematization question: if per-utterance annotation is automated, can conversation attributes and their weights on an outcome still be estimated? Gold human labels stand in for the upper bound an ideal AI annotator could reach, seeded random corruption maps the whole accuracy-to-distortion curve (weight rank correlation vs gold: 1.00 at full accuracy, 0.87 at 90 percent, 0.79 at 60 percent), and the real classical-NLP annotator lands at fidelity 0.82 with 71 percent label accuracy. Dialogue-level weights are much more robust to annotation error than the labels themselves; no LLM is called anywhere.

![Annotation accuracy vs downstream weight fidelity](assets/annotation_quality_vs_weights.png)

## Quickstart

Python 3.11.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
jupyter lab
```

All four notebooks run without any download by falling back to the committed samples under `data/sample/` (this is how CI runs them). The download script fetches both datasets (about 8.4 MB total; `python scripts/download_data.py dailydialog` or `p4g` fetches one) for the results shown in the committed outputs; notebooks 01-03 then run in well under a minute each and notebook 04 in about 40 seconds.

## Data and license

| Dataset | Source | License | Committed here |
|---|---|---|---|
| DailyDialog (13,118 dialogues, 7 emotion labels per utterance) | Internet Archive capture (2022-05-16) of `yanran.li/files/ijcnlp_dailydialog.zip` via `scripts/download_data.py` | CC BY-NC-SA 4.0 | a 1,000-dialogue sample (`data/sample/`, about 0.6 MB), redistributed under the same license with attribution |
| Persuasion for Good (1,017 persuasion dialogues with actual donations; 300 with per-sentence strategy annotations) | `gitlab.com/ucdavisnlp/persuasionforgood` (data/FullData and data/AnnotatedData) via `scripts/download_data.py p4g` | Apache-2.0 | the complete 300-dialogue annotated subset (`data/sample/`, about 1.2 MB), redistributed with attribution |

Notes:

- The original host `yanran.li` is a parked domain as of 2026-07 (the download URL returns an HTML placeholder), and the Hugging Face mirror `li2017dailydialog/daily_dialog` is a loading script that points at the same dead URL. The download script therefore uses the Internet Archive capture as its primary source and keeps the historical URL as a fallback.
- When using the datasets, cite Li, Su, Shen, Li, Cao, Niu, "DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset" (IJCNLP 2017) and Wang, Shi, Kim, Oh, Yang, Zhang, Yu, "Persuasion for Good: Towards a Personalized Persuasive Dialogue System for Social Good" (ACL 2019).
- The CC BY-NC-SA 4.0 license is non-commercial; the code in this repository is MIT, but anything you do with the DailyDialog data inherits the data license. The Persuasion for Good data is Apache-2.0.
- The Persuasion for Good donation outcome is winsorized at the $2 task cap; 35 raw entries above the cap (up to $700) are data-entry noise per the dataset's own documentation.

## Repository layout

```
notebooks/            executed notebooks (outputs committed)
notebooks_src/        jupytext py:percent sources of the same notebooks
src/voc_arc/          data loading, classifier, trajectory layer, plotting,
                      plus p4g / attributes / weights for notebook 04
tests/                pytest suite, including hand-computed transition
                      matrix values, planted-shift detection, and
                      planted-coefficient recovery for the weight models
scripts/download_data.py
scripts/make_sample.py
data/sample/          committed samples with attribution (DailyDialog
                      1,000 dialogues; Persuasion for Good annotated 300)
assets/               representative figures used above
```

## Limitations

- DailyDialog is scripted English small talk written for language learners, not real customer conversations. The end-of-conversation positivity ritual, the non-sticky anger, and every absolute number should be re-estimated on in-domain data; what transfers is the method and the evaluation design.
- The valence mapping (happiness +1; anger, disgust, fear, sadness -1; surprise and no_emotion 0) is a modeling choice with a debatable surprise row, and every valence result depends on it. It is one small table in `src/voc_arc/trajectory.py`.
- Labels are single-annotator and about 83 percent `no_emotion`; three emotion classes have too little support to learn or to evaluate reliably at the utterance level.
- One split and one classifier configuration per corpus. No transformer comparison is included, deliberately: this repository is the dependency-light reference point such a comparison would be measured against.
- English only; the character n-grams in the feature set do not transfer to other languages as-is.
- Notebook 04's noise model is uniform random label corruption; real annotators make systematic errors. The single classical-NLP point behaves similarly on this corpus, but that is one annotator, not a general law, and no actual LLM annotator was measured.
- Notebook 04's weights are associations, not causal effects: persuaders adapt their strategy to the persuadee, and the donation itself shapes late-conversation features such as the closing valence.

## License

Code is MIT licensed (see [LICENSE](./LICENSE)). The DailyDialog data, including the committed sample, is CC BY-NC-SA 4.0; the Persuasion for Good data, including the committed sample, is Apache-2.0 (table above).
