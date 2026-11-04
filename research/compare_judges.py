# Compare MLLM judges on the same candidate moments: how well each one's p(violation) separates moments near a
# labelled event from the rest (AUC), and how many of each it calls a violation.
# Usage: python research\compare_judges.py <manifest.json> <list.json>... -- <verdict file>...
import json, sys
from pathlib import Path
import numpy as np
from eval_evidence import TOL

if __name__ == "__main__":
    args = sys.argv[1:]
    cut = args.index("--")
    manifest, lists, files = json.loads(Path(args[0]).read_text()), args[1:cut], args[cut + 1:]
    truth = []
    for path in lists:  # same order as eval_evidence.load, so the manifest's clip index finds its labels
        report = json.loads(Path(path.replace(".json", "-eval.json")).read_text())["report"]
        for clip in json.loads(Path(path).read_text()):
            r = next(r for r in report if r["source"].endswith(Path(clip["file"]).name))
            for c in clip["concepts"]:
                truth.append([round(x["time"] * r["fps"]) for x in clip["events"] if x["concept"] == c])

    real = {m["id"]: any(abs(fr - m["frame"]) <= TOL for fr in truth[m["clip"]]) for m in manifest}
    p = lambda v: v["confidence"] / 100 if v["violation"] else 1 - v["confidence"] / 100
    for f in files:
        V = {v["id"]: v for v in json.loads(Path(f).read_text("utf-8")) if v["id"] in real}
        pos = np.array([p(v) for i, v in V.items() if real[i]])
        neg = np.array([p(v) for i, v in V.items() if not real[i]])
        auc = (float(((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean())
               if len(pos) and len(neg) else float("nan"))
        tp = sum(v["violation"] for i, v in V.items() if real[i])
        fp = sum(v["violation"] for i, v in V.items() if not real[i])
        cost = sum(v.get("cost", 0) for v in V.values())
        print(f"{Path(f).name:38} n={len(V):3} AUC {auc:.3f}  violation near events {tp}/{len(pos)}, elsewhere {fp}/{len(neg)}, ${cost:.3f}")
