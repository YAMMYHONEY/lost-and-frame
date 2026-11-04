# Slow-motion clips for the video judge: the 31 frames of the ±15-frame scoring window, re-encoded at FPS frames per
# second. A video judge that samples a second-long clip sparsely (Muse Spark Open said "violation" on 1 of 51 real
# moments that way) then sees every frame, in motion. Adds a "slowmo" path per moment.
# Usage: python research\slowmo_prep.py <manifest.json>
import json, subprocess, sys, tempfile
from pathlib import Path
import cv2

HALF, FPS = 15, 2
FFMPEG = dict(l.strip().split("=", 1) for l in Path("ffmpeg.env").read_text().splitlines() if "=" in l)["FFMPEG"]

if __name__ == "__main__":
    path = Path(sys.argv[1])
    manifest = json.loads(path.read_text())
    for m in manifest:
        out = Path(m["window"].replace(".window.jpg", ".slowmo.mp4"))
        if not out.exists():
            cap = cv2.VideoCapture(f"clips/{m['file']}")
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            lo, hi = max(0, m["frame"] - HALF), min(n - 1, m["frame"] + HALF)
            cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
            with tempfile.TemporaryDirectory() as tmp:
                for k in range(hi - lo + 1):
                    ok, img = cap.read()
                    if not ok:
                        break
                    cv2.imwrite(f"{tmp}/{k:03d}.jpg", cv2.resize(img, (854, 480)))
                subprocess.run([FFMPEG, "-v", "error", "-y", "-framerate", str(FPS), "-i", f"{tmp}/%03d.jpg",
                                "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p",
                                str(out)], check=True)
        m["slowmo"] = str(out)
    path.write_text(json.dumps(manifest, indent=1))
    print(len(manifest), "moments with slow-motion clips")
