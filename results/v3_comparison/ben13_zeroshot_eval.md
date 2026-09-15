# Ben13 zero-shot evaluation — Stage A on a voice never in training

The production Stage A verify-head (Milestone 5) trained and measured on R1/R2/R8. Ben13's segmented clips (`Data_Prep/segment_ben13_clips.py`, Milestone 1C) are a genuinely independent test: a fourth voice, studio-clean, that appears nowhere in training.

**Scope**: 104/106 segments have a true class inside the production model's vocabulary; the other 2 (Qarney_Para, Yerach_Ben_Yomo) have no slot in the verify-head's one-hot space and are excluded, not silently dropped. Siluk never appears (verse-final, doesn't occur mid-phrase in a combo clip). Same pairing convention as Milestone 3: every segment gives one positive pair (true class) and one random-wrong-class negative.

| Set | Pairs | Accuracy | Precision | Recall | F1 | Match conf. (mean/median) | No-match conf. (mean/median) |
|---|---|---|---|---|---|---|---|
| All 104 segments (208 pairs) | 208 | 0.745 | 0.870 | 0.577 | 0.694 | 0.545 / 0.634 | 0.149 / 0.046 |
| High-confidence only (102 segments, 204 pairs) | 204 | 0.745 | 0.879 | 0.569 | 0.690 | 0.537 / 0.595 | 0.146 / 0.045 |

For reference, Milestone 3's R1/R2/R8 LORO mean was **88.4%** (range 77.8-94.4%).