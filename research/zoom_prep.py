# Verifier v3 images for an existing manifest (verify_prep.py, dws_prep.py): the box-free 2x2 window plus a bottom
# row zoomed on the concept-conditioned change (change_zoom.py). That change is the motion-compensated residual
# change between the four moments, weighted by where SAM's detector sees the object's name on those frames (scores
# below 0.4 included).
# Usage: python research\zoom_prep.py <manifest.json> [--chosen] <list.json>...   (adds a "zoom" path per moment)
import json, sys
from pathlib import Path
import cv2
import numpy as np
from paths import trace_path
from eval_evidence import chosen_names
from verify_prep import OFFS
from change_zoom import change_box

if __name__ == "__main__":
    manifest_path, args = Path(sys.argv[1]), sys.argv[2:]
    names = chosen_names() if "--chosen" in args else {}
    sam = {}
    for path in [a for a in args if not a.startswith("--")]:
        report = json.loads(Path(path.replace(".json", "-eval.json")).read_text())["report"]
        for clip in json.loads(Path(path).read_text()):
            r = next(r for r in report if r["source"].endswith(Path(clip["file"]).name))
            for c in clip["concepts"]:
                sam[(clip["file"], c)] = names.get(f"{clip['file']}|{c}", {}).get("name", r["samName"][c])

    manifest, traces = json.loads(manifest_path.read_text()), {}
    for m in manifest:
        out = Path(m["window"].replace(".window.jpg", ".zoom.jpg"))
        m["zoom"] = str(out)
        if out.exists():
            continue
        key = (m["file"], m["concept"])
        if key not in traces:
            traces[key] = json.loads(trace_path(m["file"], sam[key]).read_text())["frames"]
        frames = traces[key]
        n, cap, shots, dets = len(frames), cv2.VideoCapture(f"clips/{m['file']}"), [], []
        for o in OFFS:
            f = min(n - 1, max(0, m["frame"] + o))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            shots.append(cv2.resize(img, (640, 360)))
            dets.append(frames.get(str(f), {}).get("det", []))
        x1, y1, x2, y2 = change_box(shots, dets)
        tiles = [s.copy() for s in shots]
        for t, letter in zip(tiles, "ABCD"):
            cv2.rectangle(t, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(t, letter, (12, 44), 0, 1.4, (255, 255, 255), 4)
        zoom = [cv2.resize(s[y1:y2, x1:x2], (320, 180)) for s in shots]
        cv2.imwrite(str(out), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:]), np.hstack(zoom)]))
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(len(manifest), "moments with zoom images")
