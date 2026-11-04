# Short clips for the video verifier (verify.mjs open-clip): HALF frames on each side of every manifest moment,
# 480p H.264 without audio. Muse Spark Open takes H.264/H.265/VP9/AV1 and samples up to 96 frames, so a 25-frame
# clip is seen whole. Adds a "video" path per moment ("clip" is the moment's object index, used by the scorers).
# Usage: python research\clip_prep.py <manifest.json>
import json, subprocess, sys
from pathlib import Path
import cv2

HALF = 12
FFMPEG = dict(l.strip().split("=", 1) for l in Path("ffmpeg.env").read_text().splitlines() if "=" in l)["FFMPEG"]


def cut(video, f, out):
    cap = cv2.VideoCapture(video)
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, f - HALF)
    frames = min(n, f + HALF + 1) - start
    subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{start / fps:.4f}", "-i", video, "-frames:v", str(frames),
                    "-vf", "scale=-2:480", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                    "-pix_fmt", "yuv420p", "-an", str(out)], check=True)


if __name__ == "__main__":
    path = Path(sys.argv[1])
    manifest = json.loads(path.read_text())
    for m in manifest:
        out = Path(m["window"].replace(".window.jpg", ".clip.mp4"))
        if not out.exists():
            cut(f"clips/{m['file']}", m["frame"], out)
        m["video"] = str(out)
    path.write_text(json.dumps(manifest, indent=1))
    print(len(manifest), "moments with clips")
