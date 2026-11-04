# Break-based detection (method A) with the names SAM was given vs the chosen names (names-chosen.json).
# Event caught = a same-kind break within ±15 frames; false alarm = a break with no same-kind event within ±15.
# Usage (project folder): python research\diag\compare_names.py clips\physion-labels.json clips\physion-tune.json ...
import json, sys
from pathlib import Path
sys.path.insert(0, "research")
from paths import trace_path

GAP, TOL = 2, 15
chosen = json.loads(Path("research/names-chosen.json").read_text())


def breaks(frames):
    n, tracks = len(frames), {}
    for f in range(n):
        for i in frames[str(f)].get("out", {}):
            tracks.setdefault(i, []).append(f)
    out = []
    for fs in tracks.values():
        cuts = [0] + [j for j in range(1, len(fs)) if fs[j] - fs[j - 1] > GAP + 1] + [len(fs)]
        for a, b in zip(cuts, cuts[1:]):
            if fs[a] > 0:
                out.append(("appear", fs[a]))
            if fs[b - 1] < n - 1:
                out.append(("vanish", fs[b - 1]))
    return out, bool(tracks)


stats = {m: dict(events=0, caught=0, alarms=0, tracked=0) for m in ("given", "chosen")}
for path in sys.argv[1:]:
    report = json.loads(Path(path.replace(".json", "-eval.json")).read_text())["report"]
    for clip in json.loads(Path(path).read_text()):
        r = next(r for r in report if r["source"].endswith(Path(clip["file"]).name))
        for concept in clip["concepts"]:
            c = chosen[f"{clip['file']}|{concept}"]
            truth = [(x["kind"], round(x["time"] * r["fps"])) for x in clip["events"] if x["concept"] == concept]
            for m, name in (("given", c["samName"]), ("chosen", c["name"])):
                bs, tracked = breaks(json.loads(trace_path(clip["file"], name).read_text())["frames"])
                s = stats[m]
                s["events"] += len(truth)
                s["tracked"] += tracked
                s["caught"] += sum(any(k == kind and abs(b - fr) <= TOL for k, b in bs) for kind, fr in truth)
                s["alarms"] += sum(not any(k == kind and abs(b - fr) <= TOL for kind, fr in truth) for k, b in bs)

for m, s in stats.items():
    print(f"{m:6}: objects tracked {s['tracked']}, events caught {s['caught']}/{s['events']}, false alarms {s['alarms']}")
