# How much does the MLLM verifier (verify.mjs) add on top of the candidate pool (verify_prep.py)?
# Each verified moment gets p(violation) = confidence if Spark says violation, else 1 - confidence. Kept moments
# become alerts of Spark's kind (transform: both kinds; none: the candidate's own kind, or both for in-place ones).
# They are scored like eval_evidence.py (same kind within ±15 frames; one moment per MERGE frames).
# Usage: python research\eval_verify.py <manifest.json> [--chosen] [--window] <list.json>...
import json, sys
from pathlib import Path
import numpy as np
from eval_evidence import load, score, chosen_names, TOL

BOTH = ("appear", "vanish")


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    p, n = np.array(pos)[:, None], np.array(neg)[None, :]
    return float(((p > n) + 0.5 * (p == n)).mean())


def alerts(m, v):
    kinds = {"vanish": ("vanish",), "appear": ("appear",)}.get(v["kind"]) or ((m["kind"],) if m["kind"] else BOTH)
    # A scan window (dws_prep.py) is not centred on a candidate: place it at the pair of moments Spark named.
    f = m["frame"] + ({"A-B": -5, "C-D": 5}.get(v.get("between"), 0) if m["source"] == "W" else 0)
    return [(k, f) for k in kinds]


if __name__ == "__main__":
    manifest_path, args = sys.argv[1], sys.argv[2:]
    clips = load([a for a in args if not a.startswith("--")], chosen_names() if "--chosen" in args else {})
    # Node writes UTF-8 (Spark's reasons have dashes); Windows Python would read it as cp949.
    suffix = "-verdicts-window.json" if "--window" in args else "-verdicts.json"
    verdicts = {v["id"]: v for v in json.loads(Path(manifest_path.replace(".json", suffix)).read_text("utf-8"))}
    moments = [m for m in json.loads(Path(manifest_path).read_text()) if m["id"] in verdicts]
    for m in moments:
        v = verdicts[m["id"]]
        m["p"] = v["confidence"] / 100 if v["violation"] else 1 - v["confidence"] / 100
        m["real"] = any(abs(fr - m["frame"]) <= TOL for _, fr in clips[m["clip"]]["truth"])

    pos, neg = [m["p"] for m in moments if m["real"]], [m["p"] for m in moments if not m["real"]]
    print(f"{len(moments)} verified moments ({len(pos)} near a labeled event): verifier AUC {auc(pos, neg):.3f}")
    for s in "APE":
        sp = [m["p"] for m in moments if m["real"] and m["source"] == s]
        sn = [m["p"] for m in moments if not m["real"] and m["source"] == s]
        print(f"  source {s}: {len(sp)} real / {len(sn)} other, AUC {auc(sp, sn):.3f}")
    cost = sum(verdicts[m["id"]]["cost"] for m in moments)
    print(f"Spark cost ${cost:.3f} (${cost / max(1, len(clips)):.4f} per object)")

    by_clip = {}
    for m in moments:
        by_clip.setdefault(m["clip"], []).append(m)
    ca, tot, fa = score(clips, lambda c: [a for m in by_clip.get(clips.index(c), [])
                                          for a in alerts(m, {"kind": "none"})])
    print(f"pool, no verifier:            caught {ca}/{tot} ({ca / tot:.0%}), false alarms {fa}")
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        ca, tot, fa = score(clips, lambda c: [a for m in by_clip.get(clips.index(c), []) if m["p"] >= t
                                              for a in alerts(m, verdicts[m["id"]])])
        print(f"pool, verifier p >= {t:.1f}:      caught {ca}/{tot} ({ca / tot:.0%}), false alarms {fa}")
