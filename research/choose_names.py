# Pick one name per object from name-probe.json, without labels ("anchored renaming").
# On each probed frame a candidate scores only where it fires on a region the given name or its head noun already
# weakly points to (one of their top-3 boxes with score >= ANCHOR, IoU >= 0.5), so SAM cannot swap in a different
# object ("egg" on some other round thing). The best mean score wins if SAM could start tracking it (>= TRACK on
# some frame); otherwise the name SAM was already given stays. A name SAM already grounds (mean >= TRACK) is never
# replaced: near-ties like "pizza" -> "pizza pie" only change which track SAM follows.
# Usage: python research\choose_names.py   -> research/names-chosen.json {"<file>|<concept>": {...}}
import json
from pathlib import Path
import numpy as np

ANCHOR, IOU, TRACK = 0.05, 0.5, 0.4  # TRACK: SAM 3.1's new-detection threshold (score_threshold_detection)


def iou(a, b):  # normalized xyxy
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def anchored(scores, base, name):
    """Per probed frame, the best score `name` gets on a region a base name points to."""
    per = []
    for f, dets in scores[name].items():
        anchors = [d[1:] for b in base for d in scores[b][f] if d[0] >= ANCHOR]
        per.append(max((d[0] for d in dets if any(iou(d[1:], a) >= IOU for a in anchors)), default=0.0))
    return per


def choose(concept, sam_name, scores):
    base = list(dict.fromkeys([concept.lower(), concept.lower().split()[-1]]))
    per = {n: anchored(scores, base, n) for n in scores}
    best = max(per, key=lambda n: (np.mean(per[n]), n == sam_name))
    grounded = sam_name in per and np.mean(per[sam_name]) >= TRACK
    name = best if max(per[best]) >= TRACK and not grounded else sam_name
    return {"name": name, "samName": sam_name, "mean": round(float(np.mean(per[best])), 3),
            "top": sorted(((round(float(np.mean(v)), 3), n) for n, v in per.items()), reverse=True)[:4]}


if __name__ == "__main__":
    # Self-check: a high score elsewhere ("egg") does not count, and a higher score on the anchored region wins.
    s = {"ball": {"0": [[0.2, 0, 0, .1, .1]]}, "egg": {"0": [[0.9, .5, .5, .6, .6]]},
         "sphere": {"0": [[0.6, 0, 0, .1, .1]]}}
    assert choose("ball", "ball", s)["name"] == "sphere", choose("ball", "ball", s)
    s["ball"]["0"][0][0] = 0.5  # already grounded: keep it even though "sphere" scores higher
    assert choose("ball", "ball", s)["name"] == "ball", choose("ball", "ball", s)

    probe = json.loads(Path("research/name-probe.json").read_text())
    chosen = {k: choose(k.split("|")[1], v["samName"], v["scores"]) for k, v in probe.items()}
    Path("research/names-chosen.json").write_text(json.dumps(chosen, indent=1))
    changed = {k: c for k, c in chosen.items() if c["name"] != c["samName"]}
    print(f"{len(chosen)} objects, {len(changed)} renamed")
    for k, c in changed.items():
        print(f"  {c['samName']!r} -> {c['name']!r}  (mean {c['mean']}, top {c['top'][:3]})")
