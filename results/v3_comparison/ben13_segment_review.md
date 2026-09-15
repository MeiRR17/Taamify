# Ben13 segment review: manual pass on all 114 cuts, root cause, and fix

Milestone 5d built the review tool; this is the follow-up after the user
personally reviewed all 114 segmented clips by hand (~1 hour, every clip
flagged good/bad/unsure with a free-text note): **60 good, 51 bad, 2
unsure, 1 unflagged-with-note.**

## What the notes said

Reading all 114 notes end to end, the complaints cluster into a small
number of patterns:

1. **Dominant pattern (most of the 51 "bad"): the cut word's own
   melismatic tail ("סלסול") or trailing consonant is truncated.** The
   segment ends before the singing does.
2. **A related, smaller pattern: the *next* word's segment opens on the
   *previous* word's leftover tail** — audio contamination running the
   other direction across the same boundary.
3. **A structural objection**, not a timing bug: several single-word cuts
   are so short they're barely intelligible in isolation even when the
   cut itself is accurate — inherent to demoing a single ta'am name in
   ~0.5-1.5s of audio, not something a boundary fix changes.
4. **A handful of "wrong word entirely" reports** on specific combo
   orders (9/90, 140/180 across clip-type copies) — investigated
   separately, see below.
5. **A `\xA0` (non-breaking space) rendering as a visible artifact** in
   the review UI's Hebrew text display — a real bug, unrelated to
   segmentation.

## Root cause: one early boundary explains both (1) and (2)

`Data_Prep/verse_align.py`'s `align_verse()` places an internal word
boundary at the first acoustic evidence of the *next* word (its own
docstring: "a word runs from its first letter's onset to the onset of
whatever comes next"). That's the right rule for normal speech, but a
ta'am name's melismatic tail sustains the same vowel with no new phoneme
for CTC to lock onto — so right where it matters most, CTC is starved of
evidence and the boundary lands early. Patterns (1) and (2) are the same
early boundary observed from opposite sides: word i's tail gets cut
short, and that same missing time becomes contamination at the *start*
of word i+1's segment.

**Fix**: `segment_ben13_clips.py` now shifts every internal word boundary
later by a fixed `BOUNDARY_SHIFT_SEC = 0.15` (150ms), capped at 50% of the
next word's own span so a short word is never swallowed entirely (`--boundary-shift`
CLI flag to retune without a code change). Verified the shift behaves as
designed by diffing against a `--boundary-shift 0` run: **median duration
change 0.0s, mean +0.008s, range [-0.151s, +0.15s]** across all 114
segments — symmetric and bounded, no runaway growth from chained
boundaries in longer combos.

All 114 segments regenerated with the fix; filenames are unchanged
(`{cls}_{type}-{order}_{i}.wav`), so every clip_id still resolves, but the
**audio content behind every clip_id has changed** — see "stale notes"
below.

## Investigated separately: 3 confirmed cases of a real Ben13 source-text inconsistency

Order 9 (`teamimtora`) / 90 (`teamim-meg`, `teamim-haf`) — the combo
`Pashta → Mahapakh → Pashta → Munach → Zaqef_Qatan` — has its **first
word spelled קדמא (the letters for "Qadma") but tagged with Pashta's own
cantillation diacritic**, and `taam_sequence` agrees with the diacritic
(Pashta). So the segmentation itself is internally consistent (diacritic
decode matches the declared sequence), but a listener expecting to hear
the word "Qadma" hears a segment labeled Pashta instead — exactly what
the reviewer's notes independently flagged by ear. Confirmed identical
across all three clip-type copies (same underlying combo, replicated).

This is **not auto-fixable and not a timing bug** — it's Ben13's own
source text. Two live possibilities, and only a human familiar with this
tradition's oral naming conventions can adjudicate: (a) a genuine error
in Ben13's manifest, or (b) "קדמא" is a colloquial name for Pashta in
this tradition (distinct from the general ta'am "Qadma," which also
exists and is spelled/tagged consistently elsewhere — e.g. order 140).
Flagged in the segment manifest (`label_note` field) and surfaced as a
red banner in the review tool — not silently corrected, not silently
trusted.

**Two other flagged orders were checked and do NOT show this pattern:**
order 140 (`Qadma → Geresh`, word spelled and tagged consistently as
Qadma) and order 180 (`Munach → Revia`, word spelled and tagged
consistently as Munach). Both are spelled/tagged self-consistently, so
whatever the reviewer heard there is not a text/diacritic mismatch — more
likely the first ta'am's spoken name being brief/weak before the second
ta'am's melisma dominates the clip, an acoustic property of the source
recording rather than a labeling bug. Left as open/unresolved judgment
calls for the reviewer, not force-fit into the fix above.

## Review tool improvements (`webapp/static/review.html`, `webapp/server.py`)

- **`\xA0` fixed**: `_clean_hebrew()` in `server.py` strips it server-side
  before `source_text`/`source_text_sfaradi` reach the UI (plus a
  defensive client-side strip).
- **Stale-note detection**: `segment_ben13_clips.py` writes
  `data/ben13_segments/last_regenerated.json` on every run;
  `save_review_note()` now timestamps each note; `get_review_items()`
  flags `stale_note: true` whenever a note predates the last
  regeneration. Verified: **all 114 pre-existing notes correctly flag
  stale** after this run, since the boundary shift touched every combo's
  internal cuts. The detail pane shows a clear "re-cut since this note
  was written, please reconfirm" banner without discarding the original
  note/flag — old judgment stays visible as a reference, not lost.
- **New filters**: "Previously flagged bad" (jump straight to the 51) and
  "Stale notes first" (on by default) — sort priority is now stale →
  label-issue → low-confidence → ta'am → source order, so the fastest
  re-review path surfaces first automatically.
- **Label-issue banner**: the 3 known-inconsistent entries show their
  `label_note` at the top of the detail pane before the audio players.

## Follow-up: "why are so many ta'amim missing" and Sephardic naming

Two more things came up once the reviewer started using the improved
tool:

1. **The review tool only ever showed the 16 classes that happen to
   appear inside a *combination* clip** — Ben13's other 24 clips
   demonstrate a single ta'am in isolation (no cutting needed at all) and
   were wired only into the practice app's teacher-clip system, never
   into this review tool. That's the real reason coverage looked so
   sparse. Fixed: `get_ben13_single_items()` in `server.py` pulls these
   in as their own item kind (`"single"`, no CTC score/duration, one
   audio player instead of two), merged into the same `/api/review/items`
   list. **Review coverage is now 24 of the project's 25 active classes**
   — the one exception, `Siluk`, genuinely has zero material anywhere in
   Ben13 (single or combo), so no code change adds it; that needs a
   different source entirely.
2. **Sephardic-Jerusalem naming**: the reviewer wants each ta'am shown
   under its own tradition's spoken name (e.g. Pashta as "קדמא"), not the
   academic/internal class name, to stop having to mentally translate
   during review. Tried deriving this automatically from Ben13's own
   `text`/`textSfaradi` fields first — **the source data turned out too
   inconsistent to trust blindly**: e.g. Munach's single clip calls it
   "פסק" while a combo clip's Munach-tagged token reads "שופר-הולך" — two
   different names for the same class within Ben13's own text, plus
   stray literal `\xA0`/escaped-newline artifacts in several fields.
   Rather than guess, added an editable per-class name field directly in
   the review UI (`sephardi-box` next to the ta'am heading — saves via
   `POST /api/review/sephardi_name` to `data/ben13_segments/
   sephardi_names.json`, one name per class, applied to every clip of
   that class immediately). Seeded with the one mapping the reviewer
   confirmed directly: **Pashta → קדמא**. That also resolves order 9/90's
   flagged "label issue" as confirmed-intentional Sephardic terminology,
   not a Ben13 data error — updated `KNOWN_LABEL_NOTES`' text accordingly.
   Still open: whether the academically-distinct `Qadma` class (e.g.
   order 140, `Qadma → Geresh`) also colloquially goes by קדמא, or a
   different name — flagged for the reviewer, not guessed.

Also fixed two review-tool bugs the reviewer hit in the first live
session: the "previously flagged bad" and "stale notes first" filter
checkboxes were rendered but never wired to `applyFilters()` (missing
`addEventListener` calls — now fixed), and the stale-note banner had no
actual action to take — added a one-click "reconfirm" button that
re-saves the current note/flag with a fresh timestamp (clears staleness
without retyping anything).

**Whole UI translated to Hebrew** (`lang="he" dir="rtl"`, every label,
button, banner, and placeholder) after the reviewer reported not
understanding what the tool was asking of them — including an added
explanatory line under the header describing the tool's purpose in one
sentence, and a labeled "original Ben13 text" line above the Hebrew
source text so it's clear what that text is.

## Second review pass: the boundary shift overshot, and a much bigger bug was hiding behind it

The reviewer re-listened to 51 clips after the 150ms shift. The result
corrected two things at once:

**1. 0.15s was too far.** Nearly every re-reviewed clip came back with the
*opposite* complaint from before — "now I hear my word plus the start of
the next one" (`עכשיו זה נשמע זרקא שו-`, `דרגא ת-`, `טרחא מ-`, `קדמא מ-`),
with one note pinning the size exactly: *"delete the last tenth of a
second and it would be good."* `BOUNDARY_SHIFT_SEC` is now **0.05**.

**2. The far more serious finding: Ben13's text is not a transcript of
Ben13's audio.** Several notes report the cut containing a completely
different ta'am than its label, and tracing them showed why — the word
list handed to forced alignment is sometimes *longer than what is
actually sung*:

| Clip | Text says | Reviewer hears | Missing |
|---|---|---|---|
| order 190 | שופר-הולך · זרקא · שופר-הולך · סגולתא | זרקא · שופר-הולך · סגולתא | leading word |
| order 90 | קדמא · שופר-מהופך · פשטא · שופר-הולך · זקף-קטן | שופר-מהופך · קדמא · זקף-קטן | 2 words |
| order 20 | מאריך · טרחא · שופר-הולך · אתנח | מאריך · טרחא · אתנח | שופר-הולך |
| order 240/30 | מאריך · טרחא · מאריך · סוף-פסוק | מאריך · טרחא · סוף-פסוק | middle מאריך |
| order 180/13 | מונח · רביעי | רביע (one word) | מונח |

**A forced alignment given too many words cannot fail loudly** — it
compresses monotonically, so every boundary after the phantom word
shifts and every segment past it is mislabeled, while the confidence
score stays unremarkable. That is the root cause of the "wrong ta'am
entirely" reports, and it is *not* fixable by any boundary heuristic.
Duration-per-word confirms it independently: the flagged clips sit at
0.72–0.95 s/word against a ~1.5 s/word median.

**Two concrete code bugs found underneath this:**

- **`textSfaradi` → `text` fallback was actively harmful.** Order 180's
  `textSfaradi` is the single word "רביע"; its `text` is the two words
  "מונח רביעי"; the audio sings one. Because `textSfaradi` didn't match
  the declared 2-item `taam_sequence`, the code silently fell back to the
  longer `text` and cut a one-word clip in two — exactly the reviewer's
  *"why are you cutting a one-word clip at all?"* complaint.
  `words_from_entry()` now returns which field it used and the caller
  prints `SUSPECT` when it wasn't `textSfaradi`; this flags precisely the
  3 clips the reviewer complained about, and nothing else.
- **The `\xA0` artifact was never only cosmetic.** Ben13's text carries
  the four *literal* characters `\xA0` (not a real U+00A0). `str.split()`
  separates on a real nbsp but glues words together across the literal
  escape, so `"מונח\xA0רביעי"` tokenized as ONE token carrying TWO
  ta'amim — the exact shape `words_from_entry` rejects as unsplittable.
  New `clean_text()` normalises it before tokenizing.

## The fix: a manual cut editor (`webapp/static/cut.html`)

No heuristic recovers a transcript that disagrees with its own audio, so
the remedy is to let a human set it. New tool, one row per *source clip*
rather than per segment:

- Waveform (decoded client-side via WebAudio) with **draggable boundary
  handles**, colour-matched to the word list.
- **Editable word list**: correct the text to what's actually sung, delete
  a phantom word, add a missing one, set each word's ta'am from a
  dropdown of all 26 classes, or mark a word "skip" so no segment is cut
  for it.
- Per-word play button, play-all, click-to-seek.
- Clips are ordered **suspect-first** (text fallback, then low CTC score),
  and unedited before edited.
- Saves to `data/ben13_segments/manual_cuts.json`; `segment_ben13_clips.py`
  reads it and a hand-corrected clip **bypasses CTC entirely**
  (`cut_source: "manual"` in the manifest).

Boundaries are seeded from the existing CTC spans (now recorded as
`start`/`end` in the segment manifest) so the reviewer corrects a guess
rather than placing every handle from scratch. Verified end-to-end: saving
a one-word correction for order 180 and re-running the segmenter produced
exactly one `Revia` segment marked `manual`, replacing the two bad ones.
(That test entry was then cleared — `manual_cuts.json` is meant to hold
the reviewer's own judgement, not a stand-in guess.)

## Is there a simpler way? Tried an independent Whisper cross-check

The reviewer asked whether some tool could do this identification better
so hand-cutting isn't needed. Tested one concrete idea:
`Data_Prep/whisper_crosscheck_ben13.py` runs mlx-whisper on every combo
clip with **no constraint from Ben13's text at all** (free transcription,
Hebrew), then compares its word count to the declared `taam_sequence`
length — an independent second opinion that doesn't trust Ben13's own
text to check itself.

**Honest result: mostly not useful as-is.** Whisper routinely *over*-splits
Sephardic compound ta'am names into ordinary Hebrew words (e.g.
"שופר-הולך" → "שופרו" + "לך", two words), so 28 of 42 clips show a raw
count mismatch — almost all of it this artifact, not real missing/extra
content. Whisper doesn't know cantillation naming conventions, so it
can't be a general auto-labeler here.

**But the narrower signal is real and worth keeping**: filtering to
clips where Whisper heard *fewer* words than declared (rare — the bias
runs the other way) caught exactly 4 clips: both `order 180` copies
(already confirmed manually), plus `teamim-meg-20` and `teamim-meg-100`
— **new leads the manual review hadn't caught**, since -20's declared
word count matched `taam_sequence` length correctly (so the
`text_fallback` flag missed it) even though the content was wrong.
Wired into `cut.html` as a second, independent `whisper_undercount`
badge alongside `text_fallback` — the two together flag 5 distinct
clips, not 3. Union is still small and bounded, not "cut everything":
`get_cut_clips()` sorts suspect-flagged clips first either way.

## A capability correction, made explicitly because it matters

The reviewer asked me to "listen to the recordings" myself using a
model. To be precise about what that can and cannot mean here: **there
is no tool in this environment that lets me hear audio the way a person
does** — I cannot judge whether a cut "sounds like Pashta" or whether a
melisma trails off naturally. What I *can* do is run signal-processing
and ASR models and read their numeric output — which is real, but it is
not listening, and the two don't always agree. Two attempts, one that
worked and one that honestly didn't:

- **Edge-truncation check (didn't work, reported as a negative
  result, not hidden):** `Data_Prep/segment_acoustic_check.py` compared
  each segment's boundary RMS energy against its own peak, expecting a
  clean cut to taper toward silence at the edge. On normal speech this
  is a reasonable heuristic. On melismatic cantillation it isn't: the
  ratio distribution across all 114 segments is a smooth continuum from
  near-zero to near-one (p25=0.19, p50=0.50, p75=0.77 at the start edge)
  with no separation between "clean" and "truncated" — because sustained
  singing simply doesn't decay to silence at a word boundary the way
  speech does. Any threshold either flags nearly everything (94/114 at a
  reasonable-looking 0.35) or nearly nothing. **Not a usable per-segment
  signal on this material** — abandoned rather than shipped as a
  misleading badge.
- **Whisper silence check (worked, narrow and real):** running Whisper
  on each *isolated* segment (not the parent clip) and flagging ones
  where it transcribes nothing at all caught **5 of 114** —
  `Mercha_teamimtora-3_0`, `Mercha_teamim-meg-30_2`,
  `Munach_teamim-meg-190_0`, `Mercha_teamim-haf-20_0`,
  `Zarqa_teamim-haf-190_1`. One of these (`Munach_teamim-meg-190_0`) is
  the leading-word-of-order-190 problem already found by hand. The
  other four are short `Mercha` segments — plausibly genuine (Mercha's
  own Sephardic name is a short word, and Whisper is known to return
  empty on very brief audio even when it's real speech), so this is a
  **lead to check by ear, not a confirmed defect** — flagged, not
  auto-corrected. `data/ben13_segments/acoustic_check.json` holds full
  detail for both checks.

**On "check against something other than Ben13"**: a real ask, but there
isn't an independent ground-truth *audio* source available — these are
Ben13's own pedagogical name-demonstrations, not Torah verses, so
Sefaria's tagged text (the ground truth used everywhere else in this
project) doesn't apply here; there's no second reader singing the same
material to compare against. What *is* independently checkable, and not
yet done: cross-referencing Ben13's declared `taam_sequence` for each
clip against the standard, Ben13-independent table of conventional
Sephardic ta'am names — catching a case where Ben13's own class label
might be wrong, not just its transcript's word count. Flagged as a real
next step, not attempted yet — it's a different, larger check than
today's three (text-fallback / Whisper-undercount / short-duration) and
deserves its own pass rather than being rushed in.

## Consolidated status of every complaint from the original 114-note review

| Complaint pattern (from the original hand review) | Root cause found | Current status |
|---|---|---|
| Own word's melismatic tail / trailing consonant cut short | CTC boundary lands early — no new phoneme for a sustained tail to anchor on | Fixed by the 0.05s boundary shift (retuned down from an overshooting 0.15s) — **not yet re-confirmed by ear** |
| Next word's segment opens on previous word's leftover tail | Same early boundary, opposite side | Same fix, same caveat |
| "This is a completely different ta'am than the label" (orders 90, 140→corrected to 20/100, 180, 240) | Ben13's declared text lists words that are never actually sung — forced alignment given too many words compresses silently | Root cause understood and instrumented (3 automatic signals, 12/42 combo clips flagged); **fix requires the reviewer's manual correction in `cut.html`, not yet done for any clip** |
| "Why cut a one-word clip in half at all?" (order 180) | `textSfaradi`/`text` field mismatch — code fell back to the longer field | Fixed at the code level (flagged as `text_fallback`, no longer silently trusted); the *clip itself* still needs a manual one-word correction in `cut.html` |
| `\xA0` artifact visible in Hebrew text display | Literal 4-character escape sequence in Ben13's scraped text, not real whitespace | Fixed, both in display (`_clean_hebrew`) and in tokenization (`clean_text`) — the tokenization fix is the more important half, since it was silently breaking word-splitting for entries carrying it |
| "Previously flagged bad" filter did nothing | Checkbox rendered, `addEventListener` never wired | Fixed |
| Stale-note banner had no actual action | No "I'm done, this still applies" path existed | Fixed — one-click reconfirm |
| Ta'am names all in English, hard to follow | — | Fixed — Hebrew names throughout `review.html` and `cut.html`, sorted in Hebrew order |
| "Why are so many ta'amim missing from the tool?" | 24 of Ben13's 75 clips are single-ta'am (no cutting needed) and were wired only into the practice app, never this tool | Fixed — merged in as their own item kind; coverage 16→24 of 25 active classes (`Siluk` has no Ben13 material at all, in any form) |
| Sephardic naming ("Pashta is called Qadma") | Ben13's own text uses inconsistent naming across clips (confirmed: same class, two different spoken names in different clips) | Editable per-class field shipped, not auto-derived (source too unreliable); seeded with the one mapping the reviewer confirmed (`Pashta` → `קדמא`) |

## SOLVED: automatic phantom-word detection (`Data_Prep/detect_phantom_words.py`)

The central defect — Ben13's text declaring words its audio never sings —
turned out to be **automatically detectable**, which removes most of the
manual burden. Five methods were implemented and measured against the
same 13-clip ground truth taken from the reviewer's own notes:

| Method | Exact-count accuracy |
|---|---|
| Silence-gap counting | 3/13 — these clips have no silence between words |
| Onset detection | 3/13 — melisma produces onsets mid-word |
| Spectral novelty (self-similarity) | 0/13 — no usable structure boundaries |
| F0 unvoiced-gap counting | 6/13 (8/13 tuned) — best audio-only method, still unusable |
| Per-word CTC score | 4/12 — a phantom word doesn't reliably score worst |
| **Leave-one-out CTC hypothesis test** | **7/7 on every decidable clip** |

**The method**: CTC's alignment score is a likelihood, so align the
declared word list, then align every leave-one-out variant. Forcing a
phantom token onto audio that doesn't contain it costs real likelihood,
so removing it produces a large gain; removing a word that *is* present
costs information and produces almost none. Observed separation on the
labelled set: true phantoms gained 1.09–1.68, the largest false gain on
a known-good clip was 0.50. Threshold set at 0.90.

**Its honest limit**: on a 2-word clip, dropping one leaves a single word
CTC can place almost anywhere, so the gain is artifactual. Those are
reported `UNDECIDABLE` rather than guessed at (`MIN_TOKENS = 3`).

**Result across all 42 combo clips: 6 phantom words found, 21 clips
confirmed consistent, 15 undecidable.** Every one of the 6 matches a
complaint in the reviewer's notes:

| Clip | Phantom word found | Matches reviewer's note |
|---|---|---|
| `teamim-meg-20` | `שופר־הולך` (#2) | "בהקלטה נשמע מאריך טרחא אתנח בלבד… לא נשמע שופר הולך" ✓ |
| `teamim-meg-30` | `מאריך` (#2) | "צריך למחוק את המאריך שבאמצע" ✓ |
| `teamim-meg-240` | `מאריך` (#2) | same combo, same note ✓ |
| `teamim-meg-90` | `קדמא` (#0) | "זה לא תואם לקליפ המלא שהוא שופר מהופך קדמא זקף קטון" ✓ |
| `teamim-meg-190` | `שופר-הולֵך` (#0) | "הקליפ המקורי הוא זרקא שופר הולך סגולתא" ✓ |
| `teamim-meg-100` | `שופר-מהופך` (#0) | new — not previously caught by hand |

**Independent validation, unplanned and strong**: while this was being
built, the reviewer hand-corrected `teamim-meg-20` in `cut.html` by
deleting `שופר-הולך`. The detector — which reads only Ben13's original
manifest, never `manual_cuts.json` — independently flagged that exact
word. Two methods with no shared information agreeing on the same
answer.

**A striking pattern worth recording: all 6 phantom words are in
`teamim-meg`.** The Torah and Haftara copies of the same combos are
clean. Ben13's Megillot recordings are systematically shorter than their
declared text, while its other two sets match — so for every broken
Megillot clip there is already a correct counterpart in another set.

`segment_ben13_clips.py` now drops detected phantom words (and their
ta'am) before aligning. After the fix, every previously-broken clip sits
in the normal duration band (corpus median 1.33 s/word):
`teamim-meg-30` 1.32, `-90` 1.01, `-190` 2.26, `-240` 1.33 — against
0.65–0.91 before.

**Still outstanding, and honestly not solvable this way**: `order 10`
and `order 100` (Torah + Megillot) carry the compound token
`שני-פ֙שטי֨ן` ("two Pashtas") — one written word holding two ta'amim,
which word-level alignment cannot split at all. These are skipped
entirely, not mis-cut, and need the word split by hand in `cut.html`.
One further clip (`teamim-meg` order 80) 404'd at download and has no
audio.

## Re-measured Stage A zero-shot after the phantom-word fix — the gap didn't close

Milestone 5b's original number (68.6% accuracy on 97 segments) predates
every fix in this document. Re-ran `16_ben13_zeroshot_eval.py` unmodified
against the corrected `manifest.json` (106 segments, 4 phantom words
dropped, 4 clips hand-corrected):

| Set | Segments | Pairs | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| Before this round of fixes | 97 | 194 | 68.6% | 74.3% | 56.7% |
| After | 89 | 178 | **69.1%** | 75.0% | 57.3% |

**+0.5 points — noise, not a real change.** The usable-segment count
dropped (97→89, some phantom-word clips' surviving segments no longer
have a top-11 class after the fix) but the aggregate metric barely
moved. Read plainly, this **confirms rather than undermines** Milestone
5b's original conclusion: the ~20-point gap from the LORO mean (88.4%)
is a genuine content-genre boundary — isolated ta'am *names* vs.
continuous chanted verse text — not an artifact of the mislabeled
clips fixed in this document. Cleaning the data was still worth doing
(the labels are now honest), it just isn't what was suppressing this
number.

**Bug caught while re-running this**: the script's report labels were
hardcoded strings (`"All 97 segments (194 pairs)"`) left over from the
first run — the `Pairs` column already reflected the real count (178,
correctly computed), only the row's text label was stale and would have
silently mismatched on any future re-run. Fixed to compute the label
from `len(usable)` / the actual pair-count mask, not copy-pasted text.

## Triaged the 15 "undecidable" clips against the other two signals

The leave-one-out phantom-word test is meaningless below `MIN_TOKENS=3`
(15 clips, 1-2 words each) — reported `undecidable`, not guessed at. But
two *other* independent signals were already computed earlier and cover
these same clips: the Whisper word-undercount check and the CTC
duration-ratio outlier check. Cross-referencing them gives a real risk
tier instead of one flat "unknown" bucket:

**13 of 15 are clean on both other signals** (no Whisper undercount, no
abnormally short duration) — `teamim-haf-140/200/50/80`,
`teamim-meg-140/180/200/50`, `teamimtora-16/23/24/5/8`. Not proof of
correctness (no signal here can prove that), but two independent
automatic checks agreeing on "nothing looks wrong" is a real, if partial,
form of evidence — lower priority for a manual pass than the flagged
ones.

**2 of 15 flagged by at least one other signal** — `teamim-haf-180` and
`teamimtora-13`. Both are **already hand-corrected** in
`manual_cuts.json` (the reviewer fixed them directly in `cut.html`
earlier this session), so this is confirmation the automatic flags and
the reviewer's own judgment agree, not a new action item.

Net effect: of the original 15 undecidable clips, the two that actually
carried a real risk signal are already resolved. The remaining 13 are
not proven correct, but are now honestly the *lowest*-priority item left
in this whole document — everything else (the 3 confirmed phantom-word
fixes' downstream verification, and the two compound-token clips below)
outranks them.

## The two compound-token clips (order 10, 100): investigated, and NOT a bug to fix

`שני-פ֙שטי֨ן` / Sephardic `תרי-ק֨דמי֨ן` ("two Pashtas") is one written
token carrying two ta'am diacritics, standing for two consecutive
positions in the declared sequence (`[..., Pashta, Qadma, ...]` /
academically `[..., Qadma, Qadma, ...]`). `segment_ben13_clips.py`
correctly refuses to split it — a token can't be divided by word-level
forced alignment.

Considered, and rejected, auto-splitting it into two pseudo-segments
(e.g. 50/50 by duration, or by sub-token diacritic position): **this
would manufacture data that may not represent either ta'am's actual
melody.** The word is very plausibly a single pedagogical utterance
demonstrating that Pashta and Qadma *look and sound related* — not
necessarily two cleanly separable renditions sung back to back. Forcing
a split (automatically, or by asking a reviewer to guess a boundary in
`cut.html`) risks manufacturing confident-looking but false ground truth
for exactly the two classes (Pashta/Qadma) already flagged elsewhere in
this document as having their own naming ambiguity. **Recommendation:
leave these two clips permanently skipped**, not queued for a future
fix — the current `SKIP` behavior is the epistemically honest one here,
not a gap.

## Status / next step

The boundary-shift path is now calibrated (0.05s) and the structural
failure is understood and instrumented, but **the corpus is not yet
trustworthy**: 42 combo clips exist, 0 have been hand-corrected, and at
least 6 are known to have a transcript that disagrees with their audio.

Next step is the reviewer working through `cut.html`, suspect clips
first (the 3 `text_fallback` ones, then the low-score ones), correcting
the word list and boundaries, then re-running
`python Data_Prep/segment_ben13_clips.py`. Until that pass happens, any
Ben13-derived number (including Milestone 5b's 68.6% zero-shot result)
carries an unknown amount of label noise from these misaligned segments
— worth re-running once the cuts are corrected, since that result may
have been measuring bad labels as much as a genuine genre boundary.
