# Data Audit Report — Cleaning v2 (precision-first)

Parameters: min_prob=0.2, min_run=2, duration=[0.1,3.0]s

## Per-reader summary (all readers cleaned so far)

| Reader | Recordings | Whisper words | Verified | Kept % | v1 cleaned (old) |
|--------|-----------|---------------|----------|--------|------------------|
| R1 | 52 | 81607 | 29993 | 36.8% | 0 |
| R2 | 54 | 81757 | 36974 | 45.2% | 0 |
| R3 | 1 | 1911 | 1172 | 61.3% | 0 |
| R4 | 1 | 2021 | 673 | 33.3% | 0 |
| R5 | 1 | 2022 | 986 | 48.8% | 0 |
| R6 | 1 | 0 | 67007 | 6700700.0% | 0 |
| R7 | 50 | 78022 | 7011 | 9.0% | 0 |
| R8 | 33 | 53002 | 29094 | 54.9% | 0 |

## Ta'am distribution (all readers)

| # | Ta'am | Count |
|---|-------|-------|
| 1 | Tipecha | 16355 |
| 2 | Munach | 13926 |
| 3 | Mercha | 13724 |
| 4 | Zaqef_Qatan | 10250 |
| 5 | Etnachta | 7924 |
| 6 | Pashta | 7811 |
| 7 | Siluk | 7690 |
| 8 | Mahapakh | 4125 |
| 9 | Tevir | 3971 |
| 10 | Revia | 3481 |
| 11 | Qadma | 2721 |
| 12 | Darga | 1500 |
| 13 | Geresh | 1303 |
| 14 | Zaqef_Gadol | 723 |
| 15 | Yetiv | 688 |
| 16 | Telisha_Qetana | 609 |
| 17 | Gershayim | 604 |
| 18 | Segol | 507 |
| 19 | Zarqa | 456 |
| 20 | Telisha_Gedola | 349 |
| 21 | Pazer | 166 |
| 22 | Mercha_Kefula | 6 |
| 23 | Yerach_Ben_Yomo | 2 |
| 24 | Shalshelet | 1 |

## Ta'am distribution per reader

| Ta'am | R8 |
|---|---|
| Tipecha | 4791 |
| Munach | 4002 |
| Mercha | 4034 |
| Zaqef_Qatan | 2973 |
| Etnachta | 2297 |
| Pashta | 2301 |
| Siluk | 2294 |
| Mahapakh | 1248 |
| Tevir | 1137 |
| Revia | 1052 |
| Qadma | 846 |
| Darga | 427 |
| Geresh | 409 |
| Zaqef_Gadol | 223 |
| Yetiv | 183 |
| Telisha_Qetana | 192 |
| Gershayim | 203 |
| Segol | 146 |
| Zarqa | 151 |
| Telisha_Gedola | 127 |
| Pazer | 55 |
| Mercha_Kefula | 2 |
| Yerach_Ben_Yomo | 1 |
| Shalshelet | 0 |

## Per-parasha detail


### R8

| Parasha | Whisper | Matched | Verified | Dropped: unmatched / short-run / low-prob / duration / no-taam |
|---------|---------|---------|----------|------|
| 02_noah | 1899 | 1350 | 1086 | 485 / 64 / 1 / 7 / 256 |
| 03_lech_lecha | 1723 | 1048 | 860 | 552 / 115 / 5 / 5 / 178 |
| 04_vayera | 2265 | 1382 | 1114 | 797 / 86 / 11 / 8 / 249 |
| 05_chayei_sarah | 1287 | 729 | 594 | 493 / 63 / 5 / 8 / 122 |
| 06_toledot | 1466 | 938 | 816 | 462 / 64 / 3 / 1 / 118 |
| 07_vayetze | 2148 | 1410 | 1179 | 654 / 80 / 3 / 12 / 216 |
| 08_vayishlach | 2045 | 1393 | 1161 | 583 / 69 / 3 / 5 / 224 |
| 09_vayeshev | 1614 | 1060 | 890 | 488 / 66 / 2 / 3 / 165 |
| 11_vayigash | 1522 | 1134 | 935 | 349 / 39 / 6 / 5 / 188 |
| 12_vayechi | 1217 | 785 | 668 | 382 / 50 / 4 / 2 / 111 |
| 15_bo | 1667 | 1132 | 915 | 475 / 58 / 3 / 7 / 207 |
| 16_beshalach | 1793 | 1158 | 960 | 562 / 69 / 4 / 6 / 188 |
| 19_terumah | 1225 | 791 | 712 | 382 / 44 / 0 / 3 / 76 |
| 22_vayakhel | 1640 | 882 | 722 | 647 / 103 / 0 / 12 / 148 |
| 23_pekudei | 1249 | 876 | 688 | 330 / 39 / 2 / 10 / 176 |
| 24_vayikra | 1744 | 1222 | 966 | 462 / 58 / 7 / 6 / 243 |
| 26_shemini | 1254 | 889 | 697 | 312 / 51 / 2 / 6 / 184 |
| 31_emor | 1644 | 1167 | 973 | 416 / 61 / 1 / 5 / 188 |
| 33_bechukotai | 1015 | 723 | 600 | 251 / 39 / 6 / 4 / 113 |
| 34_bamidbar | 1886 | 1342 | 1142 | 479 / 65 / 2 / 9 / 189 |
| 35_naso | 2143 | 1269 | 1084 | 779 / 89 / 4 / 10 / 171 |
| 37_shlach | 1577 | 1112 | 887 | 409 / 52 / 2 / 2 / 221 |
| 38_korach | 1482 | 1065 | 850 | 364 / 49 / 3 / 15 / 197 |
| 40_balak | 1473 | 889 | 746 | 514 / 68 / 3 / 2 / 138 |
| 42_matot | 1531 | 1055 | 852 | 401 / 75 / 5 / 4 / 194 |
| 43_masei | 1579 | 1015 | 865 | 500 / 62 / 3 / 10 / 137 |
| 44_devarim | 1610 | 1127 | 940 | 414 / 63 / 2 / 12 / 173 |
| 46_eikev | 1492 | 958 | 798 | 467 / 57 / 1 / 7 / 152 |
| 47_reeh | 2013 | 1333 | 1115 | 601 / 79 / 2 / 13 / 203 |
| 48_shoftim | 1592 | 1119 | 916 | 437 / 32 / 1 / 10 / 192 |
| 49_ki_teitzei | 1637 | 1143 | 933 | 436 / 56 / 2 / 10 / 198 |
| 50_ki_tavo | 1843 | 1263 | 1069 | 506 / 68 / 4 / 12 / 178 |
| 51_nitzavim | 727 | 439 | 361 | 261 / 27 / 3 / 6 / 69 |
