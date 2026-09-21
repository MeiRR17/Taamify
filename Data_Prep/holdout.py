"""
The held-out parashot, defined once.
====================================
WHY THIS FILE EXISTS

Three places in this project drew "the eight held-out parashot" and three of
them drew a different set:

  00_validate_and_prepare_data.py   make_parasha_splits(), its own seed
  14_sequence_model.py              rng(42).shuffle(sorted(parashot))[:8]
  the evaluation scripts            rng(42).choice(unique(parashot), 8)

`shuffle` and `choice` do not produce the same eight names from the same
seed. So the CNN, the sequence tagger and the calibration were each honest
about a DIFFERENT holdout -- and a number produced by composing them (a
verdict that runs the CNN, the tagger and a threshold together, scored on
"held-out" parashot) is not honest about any of them. Each stage had seen
part of what the next stage was testing on.

Nothing fails when this happens. The numbers just come out too high.

So the holdout is defined here, once, as an explicit list, and every stage
imports it. An explicit list rather than a seeded draw, because a seeded draw
is only reproducible while nobody changes the draw -- which is exactly what
went wrong.

WHICH PARASHOT, AND WHY THESE

The eight below are the draw the evaluation scripts have been using, kept so
that already-measured numbers stay comparable. They span all five books, so
the holdout is not accidentally one book's recording conditions.

Usage:
  from holdout import HELD_OUT, is_held_out
"""

# Parashot are numbered 1-54 in the corpus metadata.
HELD_OUT = ("5", "6", "8", "13", "29", "37", "41", "42")

# Human names, for reports. Index = parasha number - 1.
NAMES = {
    "5": "חיי שרה", "6": "תולדות", "8": "וישלח", "13": "שמות",
    "29": "אמור", "37": "שלח לך", "41": "פינחס", "42": "מטות",
}


def is_held_out(parasha) -> bool:
    return str(parasha) in HELD_OUT


def split_indices(meta):
    """(train_idx, test_idx) over a corpus metadata list."""
    train, test = [], []
    for i, m in enumerate(meta):
        (test if is_held_out(m.get("parasha")) else train).append(i)
    return train, test


def describe():
    return "· ".join(f"{p} {NAMES.get(p, '')}".strip() for p in HELD_OUT)


if __name__ == "__main__":
    print(f"  {len(HELD_OUT)} פרשות מוחזקות:")
    print(f"  {describe()}")
