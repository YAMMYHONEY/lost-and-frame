# Eight separate frames per moment for the judge, spanning the whole ±15-frame window in which events are scored.
# The 2x2 window (f-8..f+8) left the labelled event outside what the judge saw for 18 of 51 dev moments near an
# event: the judge then answers "no change" truthfully and the candidate is dropped. Adds a "frames8" list per moment.
# Usage: python research\frames8_prep.py <manifest.json>
import json, sys
from pathlib import Path
import cv2

OFF8 = (-15, -11, -7, -3, 3, 7, 11, 15)

if __name__ == "__main__":
    path, full = Path(sys.argv[1]), "--full" in sys.argv  # --full: original resolution (field "frames8full")
    manifest = json.loads(path.read_text())
    for m in manifest:
        stem = m["window"].replace(".window.jpg", "")
        outs = [Path(f"{stem}.f8{'full' if full else ''}-{k}.jpg") for k in range(len(OFF8))]
        if not all(o.exists() for o in outs):
            cap = cv2.VideoCapture(f"clips/{m['file']}")
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for o, out in zip(OFF8, outs):
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(n - 1, max(0, m["frame"] + o)))
                ok, img = cap.read()
                if not full:
                    img = cv2.resize(img, (640, 360))
                cv2.putText(img, "ABCDEFGH"[OFF8.index(o)], (12, 44), 0, 1.4 * img.shape[0] / 360, (255, 255, 255), 4)
                cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        m["frames8full" if full else "frames8"] = [str(o) for o in outs]
    path.write_text(json.dumps(manifest, indent=1))
    print(len(manifest), "moments with 8 frames")
