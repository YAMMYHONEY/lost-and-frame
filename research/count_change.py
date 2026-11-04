# Count steps: how many distinct things the detector calls by the object's name on each frame (scores recovered below
# SAM's 0.4 threshold, de-duplicated), and where that count steps for good. "One of three people vanishes" or "a
# second egg appears" leaves every other track intact, so no track breaks, but the count still drops or rises.
import numpy as np
from evidence import candidates, iou

TAU, W = 0.3, 8  # a candidate counts from this score; frames compared on each side of a step


def counts(frames, n, tau=TAU):
    return np.array([sum(s >= tau for s, _ in candidates(frames.get(str(f), {}).get("det", []))) for f in range(n)])


def steps(frames, n, tau=TAU, w=W):
    """[(kind, frame, size)]: frames where the median count of the w frames after differs from the w before by >= 1,
    one per run (the largest), strongest first. Medians ignore single-frame flicker."""
    c = counts(frames, n, tau)
    raw = []
    for f in range(w, n - w + 1):
        d = float(np.median(c[f:f + w]) - np.median(c[f - w:f]))
        if abs(d) >= 1:
            raw.append((f, d))

    out, run = [], []
    for f, d in raw + [(None, 0)]:
        if run and (f is None or f - run[-1][0] > 1 or np.sign(d) != np.sign(run[-1][1])):
            top = max(abs(x[1]) for x in run)
            peak = [x for x in run if abs(x[1]) == top]  # the medians flag a step over a run: take its middle
            g, dd = peak[len(peak) // 2]
            out.append(("appear" if dd > 0 else "vanish", g, abs(dd)))
            run = []
        if f is not None:
            run.append((f, d))
    return sorted(out, key=lambda x: -x[2])


def changed_box(frames, i, kind, tau=TAU, w=W):
    """The box present on most of one side of step i and missing on most of the other (after for appear, before for
    vanish); None when no box changes side (a crowd that only reshuffles)."""
    side = lambda lo, hi: [[b for s, b in candidates(frames.get(str(f), {}).get("det", [])) if s >= tau]
                           for f in range(lo, hi)]
    before, after = side(i - w, i), side(i, i + w)
    new, old = (after, before) if kind == "appear" else (before, after)
    share = lambda b, fs: sum(any(iou(b, x) >= 0.3 for x in f) for f in fs) / max(len(fs), 1)
    best = max(((share(b, new) - share(b, old), b) for f in new for b in f), default=(0, None), key=lambda x: x[0])
    return best[1] if best[0] >= 0.5 else None


def detect_steps(d, tau=TAU, w=5):
    """Count steps of a detect.py file (every d["stride"]-th frame): [(kind, frame, size, box)], w in samples."""
    fr = sorted(map(int, d["det"]))
    frames = {str(i): {"det": d["det"][str(f)]} for i, f in enumerate(fr)}
    return [(k, fr[i], s, changed_box(frames, i, k, tau, w)) for k, i, s in steps(frames, len(fr), tau, w)]


if __name__ == "__main__":
    # Self-check: three things until frame 40, then two. Expect one vanish step near 40 and nothing else.
    box = lambda i: [0.1 + 0.3 * i, 0.4, 0.2 + 0.3 * i, 0.5]
    fr = {str(f): {"det": [[0.6, *box(i), 1] for i in range(3 if f < 40 else 2)]} for f in range(80)}
    s = steps(fr, 80)
    assert len(s) == 1 and s[0][0] == "vanish" and abs(s[0][1] - 40) <= 2, s
    # Strided (frames 0, 3, 6, ...): the third box leaves at frame 60, and it is the one changed_box names.
    d = dict(stride=3, det={str(f): [[0.6, *box(i), 1] for i in range(3 if f < 60 else 2)] for f in range(0, 120, 3)})
    k = detect_steps(d)
    assert len(k) == 1 and k[0][0] == "vanish" and abs(k[0][1] - 60) <= 3 and k[0][3] == box(2), k
    print("ok", s, k)
