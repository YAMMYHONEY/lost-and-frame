# For each labeled event whose concept SAM tracked: signed frame offset to the nearest same-kind break (any distance).
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "research")
from paths import trace_path

GAP = 2
offs, far = [], []
for path in sys.argv[1:]:
    report = json.loads(Path(path.replace(".json", "-eval.json")).read_text())["report"]
    for clip in json.loads(Path(path).read_text()):
        r = next(r for r in report if r["source"].endswith(Path(clip["file"]).name))
        for concept in clip["concepts"]:
            frames = json.loads(trace_path(clip["file"], r["samName"][concept]).read_text())["frames"]
            n, fps = len(frames), r["fps"]
            tracks = {}
            for f in range(n):
                for i in frames[str(f)].get("out", {}):
                    tracks.setdefault(i, []).append(f)
            if not tracks:
                continue
            breaks = []
            for fs in tracks.values():
                cuts = [0] + [j for j in range(1, len(fs)) if fs[j] - fs[j - 1] > GAP + 1] + [len(fs)]
                for a, b in zip(cuts, cuts[1:]):
                    if fs[a] > 0:
                        breaks.append(("appear", fs[a]))
                    if fs[b - 1] < n - 1:
                        breaks.append(("vanish", fs[b - 1]))
            for e in clip["events"]:
                if e["concept"] != concept:
                    continue
                fr = round(e["time"] * fps)
                same = [b - fr for k, b in breaks if k == e["kind"]]
                o = min(same, key=abs) if same else None
                (offs if o is not None and abs(o) <= 45 else far).append((e["kind"], o, len(tracks), n))

print("events with a same-kind break within 45 frames:", len(offs), " offsets:", sorted(o for _, o, _, _ in offs))
print("events with none (kind, nearest offset, tracks, frames):", far)
