# Objects SAM never tracks under any name we tried (no label used): SAM cannot see them by name at all, so ask
# Muse Spark to point at them instead (localize.mjs), then give SAM that box as a visual exemplar (trace.py --boxes).
# Writes 4 frames per object (10%, 37%, 63%, 90% of the clip) and a manifest.
# Usage: python research\localize_prep.py <out name> <list.json>...
import json, sys
from pathlib import Path
import cv2
from paths import WORK, trace_path

OUT, AT, WIDTH = WORK / "localize", (0.1, 0.37, 0.63, 0.9), 960

if __name__ == "__main__":
    name, lists = sys.argv[1], sys.argv[2:]
    chosen_path = Path("research/names-chosen.json")
    chosen = json.loads(chosen_path.read_text()) if chosen_path.exists() else {}
    (OUT / name).mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in lists:
        report = json.loads(Path(path.replace(".json", "-eval.json")).read_text())["report"]
        for clip in json.loads(Path(path).read_text()):
            r = next(r for r in report if r["source"].endswith(Path(clip["file"]).name))
            for concept in clip["concepts"]:
                key = f"{clip['file']}|{concept}"
                sam_name = chosen.get(key, {}).get("name", r["samName"][concept])
                if sam_name != r["samName"][concept]:
                    continue  # renamed: choose_names.py only renames to a name SAM grounds (>= 0.4)
                frames = json.loads(trace_path(clip["file"], sam_name).read_text())["frames"]
                if any(f.get("out") for f in frames.values()):
                    continue  # SAM tracks something under its name: nothing to repair
                n, cap, images = len(frames), cv2.VideoCapture(f"clips/{clip['file']}"), []
                for a in AT:
                    f = int(a * (n - 1))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                    ok, img = cap.read()
                    out = OUT / name / f"{len(manifest):03d}-{f:03d}.jpg"
                    cv2.imwrite(str(out), cv2.resize(img, (WIDTH, WIDTH * img.shape[0] // img.shape[1])))
                    images.append(dict(frame=f, image=str(out)))
                manifest.append(dict(id=f"{name}-{len(manifest):03d}", key=key, file=clip["file"], concept=concept,
                                     name=sam_name, images=images))
    (OUT / f"{name}.json").write_text(json.dumps(manifest, indent=1))
    print(len(manifest), "objects SAM never tracks:", [m["concept"] for m in manifest])
