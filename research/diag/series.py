# Memory-free concept signal: per frame, the detector's best (pre-NMS) score s(f) and presence p(f).
# Does a labeled event show up as a step in s or p (vanish: before > after; appear: after > before)?
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "research")
from paths import trace_path

W, TOL = 8, 15


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    p, n = np.array(pos)[:, None], np.array(neg)[None, :]
    return float(((p > n) + 0.5 * (p == n)).mean())


pos, neg, agree, untracked = {"s": [], "p": []}, {"s": [], "p": []}, {"s": [], "p": []}, []
for path in sys.argv[1:]:
    report = json.loads(Path(path.replace(".json", "-eval.json")).read_text())["report"]
    for clip in json.loads(Path(path).read_text()):
        r = next(r for r in report if r["source"].endswith(Path(clip["file"]).name))
        for concept in clip["concepts"]:
            frames = json.loads(trace_path(clip["file"], r["samName"][concept]).read_text())["frames"]
            n, fps = len(frames), r["fps"]
            s = np.array([frames[str(f)]["det"][0][0] if frames[str(f)].get("det") else 0 for f in range(n)])
            p = np.array([np.ravel(frames[str(f)].get("presence", [0]))[0] for f in range(n)])
            truth = [(x["kind"], round(x["time"] * fps)) for x in clip["events"] if x["concept"] == concept]
            tracked = any(frames[str(f)].get("out") for f in range(n))
            if not tracked:
                untracked.append((concept, len(truth), s.max(), np.mean(s >= 0.2), np.median(p)))
            for name, x in (("s", s), ("p", p)):
                for f in range(W, n - W):
                    step = x[f - W:f].mean() - x[f:f + W].mean()  # > 0: dropped at f
                    near = [k for k, fr in truth if abs(fr - f) <= 2]
                    if near:
                        pos[name].append(abs(step))
                        agree[name].append((step > 0) == (near[0] == "vanish"))
                    elif not any(abs(fr - f) <= TOL for _, fr in truth):
                        neg[name].append(abs(step))

print("never tracked by SAM (concept, events, best det score, frames with det>=0.2, presence median):")
for u in untracked:
    print(f"  {u[0]!r:15} ev={u[1]} best={u[2]:.2f} share>=0.2={u[3]:.2f} presence={u[4]:.2f}")
for name in ("s", "p"):
    print(f"step in {name}: event frames {len(pos[name])}, other frames {len(neg[name])}, AUC |step| {auc(pos[name], neg[name]):.3f}, direction matches kind {np.mean(agree[name]):.2f}")
