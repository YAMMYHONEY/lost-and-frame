# Do SAM 3.1's internal signals separate real permanence errors from tracker failures?
# For every break in a locally traced track (method A's alerts), label it by the expert annotations
# (same kind, within ±15 frames = real error) and measure how well each internal signal separates the
# two groups (AUC: 0.5 = no information, 1.0 = perfect).
# Usage: python research\analyze_signals.py clips\physion-labels.json clips\physion-tune.json ...
import json, sys
from pathlib import Path
import numpy as np
from paths import trace_path

GAP, TOL, K = 2, 15, 3  # flicker tolerance, label match window, frames inspected after a break


def iou(a, b):  # normalized xyxy
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def auc(pos, neg):  # probability a random real error scores higher than a random false alarm
    if not pos or not neg:
        return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def events(frames, n):
    tracks = {}
    for f in range(n):
        for i, (x, y, w, h, area) in frames.get(str(f), {}).get("out", {}).items():
            tracks.setdefault(i, []).append((f, [x, y, x + w, y + h], area))
    for i, pts in tracks.items():
        start = 0
        for j in range(1, len(pts) + 1):
            if j < len(pts) and pts[j][0] - pts[j - 1][0] <= GAP + 1:
                continue
            seg = pts[start:j]
            if seg[0][0] > 0:
                yield dict(id=i, kind="appear", frame=seg[0][0], box=seg[0][1], area=seg[0][2], seg=seg)
            if seg[-1][0] < n - 1:
                yield dict(id=i, kind="vanish", frame=seg[-1][0], box=seg[-1][1], area=seg[-1][2], seg=seg)
            start = j


def features(e, frames):
    side = range(e["frame"] + 1, e["frame"] + 1 + K) if e["kind"] == "vanish" else range(e["frame"] - K, e["frame"])
    det = [s for f in side for s, *box, _ in frames.get(str(f), {}).get("det", []) if iou(box, e["box"]) >= 0.3]
    trk = [frames.get(str(f), {}).get("trk", {}).get(e["id"]) for f in side]
    trk = [t for t in trk if t is not None]
    pres = [np.ravel(frames[str(f)]["presence"])[0] for f in side if "presence" in frames.get(str(f), {})]
    inside = e["seg"][-6:-1] if e["kind"] == "vanish" else e["seg"][1:6]
    peak = max((a for _, _, a in inside), default=0)
    return {
        "det_near": max(det, default=0.0),          # detector still sees something there (sub-threshold too)
        "trk_vis": max(trk, default=0.0),           # tracker still believes it is visible
        "shrink": e["area"] / peak if peak else 1.0,  # mask area at the break vs recent peak
        "presence": float(max(pres, default=0.0)),   # detector's "concept is in this frame" score
    }


rows = []
for path in sys.argv[1:]:  # development lists only; the sealed v2 set is kept for the final test
    report = json.loads(Path(path.replace(".json", "-eval.json")).read_text())["report"]
    for clip in json.loads(Path(path).read_text()):
        r = next(r for r in report if r["source"].endswith(Path(clip["file"]).name))
        for concept in clip["concepts"]:
            t = trace_path(clip["file"], r["samName"][concept])
            if not t.exists():
                continue
            frames = json.loads(t.read_text())["frames"]
            n, fps = len(frames), r["fps"]
            truth = [(x["kind"], round(x["time"] * fps)) for x in clip["events"] if x["concept"] == concept]
            for e in events(frames, n):
                real = any(k == e["kind"] and abs(fr - e["frame"]) <= TOL for k, fr in truth)
                rows.append(dict(clip=Path(clip["file"]).stem[:14], kind=e["kind"], frame=e["frame"], real=real,
                                 **features(e, frames)))

pos = [r for r in rows if r["real"]]
neg = [r for r in rows if not r["real"]]
print(f"breaks {len(rows)}: real errors {len(pos)}, false alarms {len(neg)}")
for k in ("det_near", "trk_vis", "shrink", "presence"):
    # Low values should mean "really gone", so the AUC of "is a real error" scores 1 - value.
    print(f"{k:9} AUC {auc([1 - r[k] for r in pos], [1 - r[k] for r in neg]):.3f}   "
          f"real median {np.median([r[k] for r in pos]) if pos else float('nan'):.3f}   "
          f"false median {np.median([r[k] for r in neg]) if neg else float('nan'):.3f}")
Path("research/signal-rows.json").write_text(json.dumps(rows, indent=1))
