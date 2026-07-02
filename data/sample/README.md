# DailyDialog sample

`dailydialog_sample.csv` contains 1,000 whole dialogues (7,845 utterances)
drawn from the DailyDialog dataset with a seeded random sample of dialogue
ids (`scripts/make_sample.py`, seed 42). Whole conversations are sampled,
never individual utterances, so the file preserves conversational structure
for the trajectory analysis. The label distribution is left untouched;
about 83 percent of utterances are `no_emotion`, and that imbalance is part
of what the notebooks demonstrate.

Columns: `dialogue_id` (position in the original corpus), `turn` (0-based
utterance index), `text` (tokenized utterance), `emotion` (0=no_emotion,
1=anger, 2=disgust, 3=fear, 4=happiness, 5=sadness, 6=surprise).

Source and attribution: Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang
Cao, Shuzi Niu, "DailyDialog: A Manually Labelled Multi-turn Dialogue
Dataset", IJCNLP 2017. The dataset is distributed under CC BY-NC-SA 4.0;
this sample is redistributed under the same license with attribution, as
that license permits.
