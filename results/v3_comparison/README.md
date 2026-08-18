# Window comparison — LORO, 3 readers (R1, R2, R6), deep_cnn, top-11

Same records, same labels, same folds. One variable: what slice of audio
goes into the spectrogram.

| Metric | word | stressed |
|---|---|---|
| accuracy | 0.3785 ± 0.1520 | 0.3199 ± 0.1292 |
| balanced_accuracy | 0.3878 ± 0.1612 | 0.3201 ± 0.1323 |
| mcc | 0.3136 ± 0.1625 | 0.2505 ± 0.1364 |
| f1_macro | 0.3615 ± 0.1598 | 0.2957 ± 0.1284 |
| auc_macro | 0.7857 ± 0.1117 | 0.7366 ± 0.1100 |

The whole-word window wins on every fold and every metric. Narrowing to the
stressed syllable removes information rather than noise — consistent with the
melisma study, where melodic activity showed no concentration in that
syllable for 17 of 21 ta'amim.
