# Dense window scan (DWS) for objects SAM barely sees. Where SAM tracks the object on under COVER of the frames
# (after renaming and Spark exemplars), its breaks, in-place changes and evidence tracks say little. So the whole clip
# is tiled with box-free windows for the MLLM verifier instead. Windows sit every STEP frames and show f-8, f-3, f+3,
# f+8, so each window's last frame is the next one's first: every frame-to-frame change falls inside some window.
# Usage: python research\dws_prep.py <out name> [--chosen] <list.json>...   (adds source "W" moments to <out name>.json)
import json, sys
from pathlib import Path
import numpy as np
from paths import trace_path
from eval_evidence import chosen_names
from verify_prep import OUT, render_window

COVER, STEP = 0.1, 16

if __name__ == "__main__":
    name, args = sys.argv[1], sys.argv[2:]
    names = chosen_names() if "--chosen" in args else {}
    manifest_path = OUT / f"{name}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    manifest = [m for m in manifest if m["source"] != "W"]
    ci, added = -1, 0
    for path in [a for a in args if not a.startswith("--")]:  # same order as eval_evidence.load: clip index ci
        report = json.loads(Path(path.replace(".json", "-eval.json")).read_text())["report"]
        for clip in json.loads(Path(path).read_text()):
            r = next(r for r in report if r["source"].endswith(Path(clip["file"]).name))
            for concept in clip["concepts"]:
                ci += 1
                obj = names.get(f"{clip['file']}|{concept}", {}).get("name", r["samName"][concept])
                frames = json.loads(trace_path(clip["file"], obj).read_text())["frames"]
                n = len(frames)
                if np.mean([bool(f.get("out")) for f in frames.values()]) >= COVER:
                    continue
                for f in range(8, n - 8, STEP):
                    mid = f"{name}-{ci:03d}-w{f:03d}"
                    window = OUT / name / f"{mid}.window.jpg"
                    if not window.exists():
                        render_window(f"clips/{clip['file']}", n, f, window)
                    manifest.append(dict(id=mid, clip=ci, file=clip["file"], concept=concept,
                                         name=obj.removesuffix("+box"), frame=f, kind=None, source="W", box=None,
                                         image=None, window=str(window)))
                    added += 1
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"{added} scan windows added; manifest now {len(manifest)} moments")
