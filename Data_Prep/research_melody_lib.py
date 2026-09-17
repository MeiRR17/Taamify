"""
Shared primitives for the cross-reader melody research.
=======================================================
One place for the distance metric and the contour features, so every script
below measures the same thing and a change is a change everywhere.

DTW matches webapp/server.py's dtw_component semantics -- accumulated cost
divided by warping-path length -- but is compiled with numba, because the
variance decomposition needs millions of pairs and librosa.sequence.dtw costs
~3.5 ms each (measured).
"""
import numpy as np
from numba import njit

READERS = ["BEN13R", "R1", "R2", "R8"]
N_POINTS = 60


@njit(cache=True, fastmath=True)
def dtw_cost(a, b):
    """Normalised DTW cost, same convention as server.dtw_component()."""
    n, m = a.shape[0], b.shape[0]
    D = np.empty((n + 1, m + 1), dtype=np.float64)
    D[:, :] = 1e18
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            c = abs(ai - b[j - 1])
            best = D[i - 1, j - 1]
            if D[i - 1, j] < best:
                best = D[i - 1, j]
            if D[i, j - 1] < best:
                best = D[i, j - 1]
            D[i, j] = c + best
    # path length by backtrace
    i, j, steps = n, m, 0
    while i > 0 and j > 0:
        steps += 1
        d, u, l = D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]
        if d <= u and d <= l:
            i -= 1
            j -= 1
        elif u <= l:
            i -= 1
        else:
            j -= 1
    steps += i + j
    return D[n, m] / steps


@njit(cache=True, fastmath=True, parallel=False)
def dtw_pairs(A, B, ia, ib):
    out = np.empty(ia.shape[0], dtype=np.float64)
    for k in range(ia.shape[0]):
        out[k] = dtw_cost(A[ia[k]], B[ib[k]])
    return out


# ---------------------------------------------------------------- features
def features(curve, dur, stress=None, tonic_offset=0.0):
    """Interpretable scalars from one 60-point semitone contour.

    `tonic_offset` is (word median F0 - reader median F0) in semitones, so
    `end_vs_tonic` can ask the Ben-Shalom "finalis" question: does this ta'am
    land on the reader's own reference pitch?
    """
    c = np.asarray(curve, dtype=float)
    n = len(c)
    q = max(2, n // 8)                     # ~12.5% of the word
    head, tail = c[:q].mean(), c[-q:].mean()
    imax, imin = int(np.argmax(c)), int(np.argmin(c))
    # direction changes larger than 1 semitone -- an ornamentation count that
    # ignores tracker jitter
    turns, last_ext, direction = 0, c[0], 0
    for v in c:
        if direction >= 0 and v < last_ext - 1.0:
            turns += 1 if direction > 0 else 0
            direction, last_ext = -1, v
        elif direction <= 0 and v > last_ext + 1.0:
            turns += 1 if direction < 0 else 0
            direction, last_ext = 1, v
        elif direction > 0 and v > last_ext:
            last_ext = v
        elif direction < 0 and v < last_ext:
            last_ext = v
    half = n // 2
    f = {
        "net": tail - head,                       # overall direction
        "range": float(c.max() - c.min()),
        "peak_pos": imax / (n - 1),
        "trough_pos": imin / (n - 1),
        "final_slope": float(c[-q:].mean() - c[-2 * q:-q].mean()),
        "initial_slope": float(c[q:2 * q].mean() - c[:q].mean()),
        "rise_first": float(c[:half].max() - c[0]),
        "fall_last": float(c[-1] - c[half:].max()),
        "turns": float(turns),
        "sd": float(c.std()),
        "dur": float(dur),
        "end_vs_tonic": float(tail + tonic_offset),
        "head_vs_tonic": float(head + tonic_offset),
    }
    if stress and stress.get("n_syl"):
        srel = (stress["stress_idx"] + 0.5) / stress["n_syl"]
        f["stress_rel"] = float(srel)
        f["peak_minus_stress"] = float(f["peak_pos"] - srel)
    return f


FEATURE_NAMES = ["net", "range", "peak_pos", "trough_pos", "final_slope",
                 "initial_slope", "rise_first", "fall_last", "turns", "sd",
                 "dur", "end_vs_tonic", "head_vs_tonic"]
