# What kind of breaks are these? Compares the detector score while still tracked, the frame edge, re-tracking and
# track length.
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "research")
from paths import trace_path

GAP, TOL = 2, 15


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def det_at(frames, f, box):
    return max((s for s, *b, _ in frames.get(str(f), {}).get("det", []) if iou(b, box) >= 0.3), default=0.0)


rows = []
for path in sys.argv[1:]:
    report = json.loads(Path(path.replace(".json", "-eval.json")).read_text())["report"]
    for clip in json.loads(Path(path).read_text()):
        r = next(r for r in report if r["source"].endswith(Path(clip["file"]).name))
        for concept in clip["concepts"]:
            t = trace_path(clip["file"], r["samName"][concept])
            if not t.exists():
                continue
            frames = json.loads(t.read_text())["frames"]
            n = len(frames)
            fps = r["fps"]
            truth = [(x["kind"], round(x["time"] * fps)) for x in clip["events"] if x["concept"] == concept]
            tracks = {}
            for f in range(n):
                for i, (x, y, w, h, a) in frames.get(str(f), {}).get("out", {}).items():
                    tracks.setdefault(i, []).append((f, [x, y, x + w, y + h], a))
            allboxes = [(f, i, b) for i, pts in tracks.items() for f, b, _ in pts]
            for i, pts in tracks.items():
                start = 0
                for j in range(1, len(pts) + 1):
                    if j < len(pts) and pts[j][0] - pts[j - 1][0] <= GAP + 1:
                        continue
                    seg = pts[start:j]
                    start = j
                    for kind, (f, b, a) in (("appear", seg[0]), ("vanish", seg[-1])):
                        if (kind == "appear" and f == 0) or (kind == "vanish" and f == n - 1):
                            continue
                        real = any(k == kind and abs(fr - f) <= TOL for k, fr in truth)
                        edge = min(b[0], b[1], 1 - b[2], 1 - b[3]) < 0.02
                        other = any(i2 != i and abs(f2 - f) <= 5 and iou(b2, b) >= 0.3 for f2, i2, b2 in allboxes)
                        rows.append(dict(real=real, kind=kind, len=len(seg), edge=edge, handoff=other,
                                         det_self=det_at(frames, f, b), nobj=len(tracks), any_truth=bool(truth),
                                         clip=Path(clip["file"]).stem[:20], f=f))

R = lambda c: [r for r in rows if c(r)]
for name, grp in (("real", R(lambda r: r["real"])), ("false", R(lambda r: not r["real"]))):
    m = lambda k: np.mean([r[k] for r in grp])
    print(f"{name:5} n={len(grp):3}  edge {m('edge'):.2f}  handoff {m('handoff'):.2f}  short(<=5f) {np.mean([r['len']<=5 for r in grp]):.2f}"
          f"  det_self>0.5 {np.mean([r['det_self']>0.5 for r in grp]):.2f}  det_self median {np.median([r['det_self'] for r in grp]):.3f}"
          f"  clip has truth {m('any_truth'):.2f}  objs/clip median {np.median([r['nobj'] for r in grp])}")
print("false alarms by kind:", {k: len(R(lambda r: not r['real'] and r['kind'] == k)) for k in ('appear', 'vanish')})
print("false, not edge, not handoff, not short:",
      len(R(lambda r: not r['real'] and not r['edge'] and not r['handoff'] and r['len'] > 5)))
from collections import Counter
print("false alarms per clip (top):", Counter(r["clip"] for r in R(lambda r: not r["real"])).most_common(6))
