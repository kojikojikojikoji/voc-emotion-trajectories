# Committed data samples

## DailyDialog sample

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

## Persuasion for Good annotated sample

`p4g_annotated_sample.csv` contains the complete 300-dialogue annotated
subset of the Persuasion for Good corpus (10,864 sentences) in the tidy
schema of `voc_arc.p4g` (`scripts/make_sample.py p4g`). All 300 dialogues
are kept, not a random subset, because they are the gold annotation set of
notebook 04 and any subsampling would change every reported number. The
only columns dropped relative to the original workbook are the per-sentence
sentiment scores, which the notebook does not use.

Columns: `dialogue_id`, `sent_idx` (0-based sentence position within the
dialogue), `turn`, `role` (`persuader` / `persuadee`), `text`, `er_label` /
`ee_label` (the original dialogue-act taxonomy for each side; the compact
strategy classes are derived at load time by `voc_arc.p4g.map_strategy`).

`p4g_info_sample.csv` holds the persuadee row of the participant info table
for the same 300 dialogues: `donation_raw` (actual donation in USD, before
winsorization at the $2 task cap), Big-Five scores (`extrovert`,
`agreeable`, `conscientious`, `neurotic`, `open`), `age` and `sex`.

Source and attribution: Xuewei Wang, Weiyan Shi, Richard Kim, Yoojung Oh,
Sijia Yang, Jingwen Zhang, Zhou Yu, "Persuasion for Good: Towards a
Personalized Persuasive Dialogue System for Social Good", ACL 2019.
Original data: gitlab.com/ucdavisnlp/persuasionforgood, distributed under
Apache-2.0, which permits this redistribution with attribution.
