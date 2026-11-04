# Judge-only test: windows centred on labelled events (positives) vs random moments of objects with no labelled
# event (negatives), from the oracle manifest. Sensitivity, false-positive rate and AUC per verdict file, so judges
# and prompts are compared without any candidate-finding error.
# Usage: python research\oracle_eval.py <oracle manifest.json> <verdict file>...
import json, sys
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    man = {m["id"]: m for m in json.loads(Path(sys.argv[1]).read_text())}
    for f in sys.argv[2:]:
        V = [v for v in json.loads(Path(f).read_text("utf-8")) if v["id"] in man]
        pos = [v for v in V if man[v["id"]]["positive"]]
        neg = [v for v in V if not man[v["id"]]["positive"]]
        p = lambda v: v["confidence"] / 100 if v["violation"] else 1 - v["confidence"] / 100
        a, b = np.array([p(v) for v in pos])[:, None], np.array([p(v) for v in neg])[None, :]
        auc = float(((a > b) + 0.5 * (a == b)).mean())
        print(f"{Path(f).name:42} sensitivity {sum(v['violation'] for v in pos)}/{len(pos)} ({np.mean([v['violation'] for v in pos]):.0%})"
              f"  false positives {sum(v['violation'] for v in neg)}/{len(neg)} ({np.mean([v['violation'] for v in neg]):.0%})"
              f"  AUC {auc:.3f}  ${sum(v['cost'] for v in V):.3f}")
