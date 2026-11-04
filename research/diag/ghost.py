# Missed events: does the tracker keep reporting an object (memory) while the memory-free detector no longer sees it?
# Ghost onset: the first of >= RUN consecutive tracked frames whose box has no detector candidate >= LOW (IoU >= 0.3).
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "research")
from paths import trace_path

GAP, TOL, RUN, LOW = 2, 15, 3, 0.1


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def det_at(frames, f, box):
    return max((s for s, *b, _ in frames.get(str(f), {}).get("det", []) if iou(b, box) >= 0.3), default=0.0)


ev_rows, ghost_onsets, clips_n = [], [], 0
for path in sys.argv[1:]:
    report = json.loads(Path(path.replace(".json", "-eval.json")).read_text())["report"]
    for clip in json.loads(Path(path).read_text()):
        r = next(r for r in report if r["source"].endswith(Path(clip["file"]).name))
        for concept in clip["concepts"]:
            t = trace_path(clip["file"], r["samName"][concept])
            if not t.exists():
                continue
            clips_n += 1
            frames = json.loads(t.read_text())["frames"]
            n = len(frames)
            fps = r["fps"]
            truth = [(x["kind"], round(x["time"] * fps)) for x in clip["events"] if x["concept"] == concept]
            tracks = {}
            for f in range(n):
                for i, (x, y, w, h, a) in frames.get(str(f), {}).get("out", {}).items():
                    tracks.setdefault(i, []).append((f, [x, y, x + w, y + h]))
            breaks, ghosts = [], []
            for i, pts in tracks.items():
                start = 0
                for j in range(1, len(pts) + 1):
                    if j < len(pts) and pts[j][0] - pts[j - 1][0] <= GAP + 1:
                        continue
                    seg = pts[start:j]
                    start = j
                    if seg[0][0] > 0:
                        breaks.append(("appear", seg[0][0]))
                    if seg[-1][0] < n - 1:
                        breaks.append(("vanish", seg[-1][0]))
                low = [det_at(frames, f, b) < LOW for f, b in pts]
                for k in range(len(pts) - RUN + 1):
                    if all(low[k:k + RUN]) and (k == 0 or not low[k - 1]) and k > 0:
                        ghosts.append(pts[k][0])
            for g in ghosts:
                ghost_onsets.append(dict(real=any(abs(fr - g) <= TOL for _, fr in truth),
                                         clip=Path(clip["file"]).stem[:20], f=g, issue=bool(truth)))
            for kind, fr in truth:
                ev_rows.append(dict(kind=kind, brk=any(k == kind and abs(b - fr) <= TOL for k, b in breaks),
                                    ghost=any(abs(g - fr) <= TOL for g in ghosts),
                                    clip=Path(clip["file"]).stem[:20], f=fr, nobj=len(tracks)))

E = ev_rows
print(f"traced clip-concepts {clips_n}, labeled events {len(E)}")
for kind in ("vanish", "appear"):
    g = [e for e in E if e["kind"] == kind]
    print(f"  {kind:6} n={len(g):2}  caught by break {sum(e['brk'] for e in g):2}  by ghost {sum(e['ghost'] for e in g):2}"
          f"  ghost-only {sum(e['ghost'] and not e['brk'] for e in g):2}  neither {sum(not e['ghost'] and not e['brk'] for e in g):2}  no SAM track at all {sum(e['nobj']==0 for e in g)}")
G = ghost_onsets
print(f"ghost onsets {len(G)}: near a labeled event {sum(g['real'] for g in G)}, in clips with no event for the concept {sum(not g['issue'] for g in G)}")
