# Full tiling: every object's whole clip as eight-frame windows (frames8_prep.py spans ±15 frames) centred every STEP
# frames, so every moment of the clip is inside some window; no candidate finder can lose an event before the judge.
# Needs no SAM run, only the clip, its object names and its frame rate.
# --phase k moves every centre k frames later (a second, interleaved tiling; the judge is unstable to window placement).
# Usage: python research\tile_prep.py <out name> [--phase k] <list.json>...   -> $VC_WORK/verify/<out name>.json
import json, sys
from pathlib import Path
import cv2
from paths import WORK

OUT, STEP, HALF = WORK / "verify", 16, 15

if __name__ == "__main__":
    name, lists = sys.argv[1], sys.argv[2:]
    phase = int(lists[1]) if lists[0] == "--phase" else 0
    lists = lists[2:] if phase or lists[0] == "--phase" else lists
    (OUT / name).mkdir(parents=True, exist_ok=True)
    manifest, ci = [], -1
    for path in lists:  # same order as eval_evidence.load: clip index ci
        for clip in json.loads(Path(path).read_text()):
            n = int(cv2.VideoCapture(f"clips/{clip['file']}").get(cv2.CAP_PROP_FRAME_COUNT))
            centres = sorted({min(max(f, HALF), n - 1 - HALF) for f in range(HALF + phase, n + STEP, STEP)})
            for concept in clip["concepts"]:
                ci += 1
                for f in centres:
                    mid = f"{name}-{ci:03d}-t{f:03d}"
                    manifest.append(dict(id=mid, clip=ci, file=clip["file"], concept=concept, name=concept, frame=f,
                                         kind=None, source="T", box=None, image=None,
                                         window=str(OUT / name / f"{mid}.window.jpg")))
    (OUT / f"{name}.json").write_text(json.dumps(manifest, indent=1))
    print(len(manifest), "windows over", ci + 1, "objects")
