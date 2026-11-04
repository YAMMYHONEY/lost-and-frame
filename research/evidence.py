# Evidence tracks: track-before-detect over SAM 3.1's detector field, below its 0.4 threshold and without the
# tracker's memory. Per frame, the detector's top candidates (scores recovered from NMS, trace.py) are de-duplicated
# and linked across frames by box overlap. Every track start or end inside the clip becomes a candidate appear or
# vanish event, scored by how much evidence the object had and how little is left at that place on the other side.
import numpy as np

FLOOR, NMS, LINK, GAP = 0.05, 0.5, 0.3, 3  # candidate floor, de-dup IoU, link IoU, frames a track may skip
W, K, EDGE = 5, 5, 0.02  # frames of evidence inside / after the break, frame-border margin (normalized)


def iou(a, b):  # normalized xyxy
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def candidates(dets):
    """Distinct detections on one frame: greedy NMS over [score, x1, y1, x2, y2, ...] rows above FLOOR."""
    keep = []
    for d in sorted(dets, key=lambda d: -d[0]):
        if d[0] >= FLOOR and all(iou(d[1:5], k[1:5]) < NMS for k in keep):
            keep.append(d)
    return [(d[0], d[1:5]) for d in keep]


def link(frames, n):
    """Tracks as lists of (frame, score, box), greedy by overlap with each track's last box."""
    done, active = [], []
    for f in range(n):
        cands = candidates(frames.get(str(f), {}).get("det", []))
        pairs = sorted(((iou(t[-1][2], c[1]), ti, ci) for ti, t in enumerate(active) for ci, c in enumerate(cands)),
                       reverse=True)
        used_t, used_c = set(), set()
        for o, ti, ci in pairs:
            if o < LINK:
                break
            if ti not in used_t and ci not in used_c:
                active[ti].append((f, *cands[ci]))
                used_t.add(ti)
                used_c.add(ci)
        active += [[(f, *c)] for ci, c in enumerate(cands) if ci not in used_c]
        done += [t for t in active if t[-1][0] < f - GAP]
        active = [t for t in active if t[-1][0] >= f - GAP]
    return done + active


def field_max(frames, f, box, grow=0.5):
    """Best detector score on frame f anywhere near box (box grown by half its size): is anything still there?"""
    w, h = box[2] - box[0], box[3] - box[1]
    g = [box[0] - grow * w, box[1] - grow * h, box[2] + grow * w, box[3] + grow * h]
    return max((d[0] for d in frames.get(str(f), {}).get("det", []) if iou(d[1:5], g) > 0), default=0.0)


def events(frames, n):
    """Candidate events: dict(kind, frame, box, strength, contrast, length, edge)."""
    out = []
    for t in link(frames, n):
        for kind, inside, f_out in (("appear", t[:W], range(t[0][0] - K, t[0][0])),
                                    ("vanish", t[-W:], range(t[-1][0] + 1, t[-1][0] + 1 + K))):
            f, s, box = (t[0] if kind == "appear" else t[-1])
            if (kind == "appear" and f <= GAP) or (kind == "vanish" and f >= n - 1 - GAP):
                continue  # starts with the clip or runs to its end
            strength = float(np.mean([s for _, s, _ in inside]))
            left = max((field_max(frames, g, box) for g in f_out if 0 <= g < n), default=0.0)
            out.append(dict(kind=kind, frame=f, box=box, strength=strength, contrast=strength - left,
                            length=len(t), edge=min(box[0], box[1], 1 - box[2], 1 - box[3]) < EDGE))
    return out


if __name__ == "__main__":
    # Self-check: an object at 0.25 (under SAM's 0.4) for 30 frames that then disappears is one vanish event.
    fr = {str(f): {"det": [[0.25, .4, .4, .5, .5, 0]] if f < 30 else [[0.01, .9, .9, .95, .95, 0]]} for f in range(60)}
    ev = events(fr, 60)
    assert [(e["kind"], e["frame"]) for e in ev] == [("vanish", 29)] and ev[0]["contrast"] > 0.2, ev
    print("ok")
