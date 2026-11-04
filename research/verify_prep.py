# Candidate moments for the MLLM verifier: SAM breaks off the frame edge (A), each object's top in-place changes
# (deep, top 3), and its 3 strongest evidence-track events lasting >= 10 frames (E). One moment per object within
# MERGE frames.
# Each moment is rendered as one image: four frames around it (f-8, f-3, f+3, f+8) with the box in yellow on top,
# the same four frames zoomed on the box below. Writes the images and a manifest for verify.mjs.
# Usage: python research\verify_prep.py <out name> [--chosen] <list.json>...
import json, sys
from pathlib import Path
import cv2
import numpy as np
from eval_evidence import load, chosen_names, MERGE
from paths import WORK

OFFS, TILE = (-8, -3, 3, 8), (480, 270)
OUT = WORK / "verify"  # short path: Windows MAX_PATH


def pool(c):
    """[(source, kind or None, frame, box)] with earlier sources winning within MERGE frames."""
    strong = sorted((e for e in c["E"] if not e["edge"] and e["length"] >= 10), key=lambda e: -e["strength"])[:3]
    cands = [("A", k, f, b) for k, f, b in c["A"]] + [("P", None, f, b) for f, z, b in c["deep"][:3]] + \
            [("E", e["kind"], e["frame"], e["box"]) for e in strong]  # E: 442 moments on dev, mostly noise; top 3
    out = []
    for cand in cands:
        if not any(abs(cand[2] - o[2]) <= MERGE for o in out):
            out.append(cand)
    return out


def render_window(video, n, f, path):
    """Box-free layout: the same four moments as full frames in a 2x2 grid (A B / C D)."""
    cap, tiles = cv2.VideoCapture(video), []
    for o in OFFS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(n - 1, max(0, f + o)))
        ok, img = cap.read()
        img = cv2.resize(img, (640, 360))
        cv2.putText(img, "ABCD"[OFFS.index(o)], (12, 44), 0, 1.4, (255, 255, 255), 4)
        tiles.append(img)
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))


def render(video, n, f, box, path):
    cap, top, bottom = cv2.VideoCapture(video), [], []
    for o in OFFS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(n - 1, max(0, f + o)))
        ok, img = cap.read()
        h, w = img.shape[:2]
        x1, y1, x2, y2 = (np.array(box) * [w, h, w, h]).astype(int)
        bw, bh = max(x2 - x1, 24), max(y2 - y1, 24)
        cx, cy, half = (x1 + x2) // 2, (y1 + y2) // 2, int(max(bw, bh * 16 / 9) * 0.9)  # zoom: box plus margin
        crop = img[max(0, cy - half * 9 // 16):cy + half * 9 // 16, max(0, cx - half):cx + half].copy()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(img, f"{'ABCD'[OFFS.index(o)]}", (12, 52), 0, 1.8, (255, 255, 255), 4)
        top.append(cv2.resize(img, TILE))
        bottom.append(cv2.resize(crop, TILE) if crop.size else np.zeros((TILE[1], TILE[0], 3), np.uint8))
    cv2.imwrite(str(path), np.vstack([np.hstack(top), np.hstack(bottom)]))


if __name__ == "__main__":
    name, args = sys.argv[1], sys.argv[2:]
    names = chosen_names() if "--chosen" in args else {}
    clips = load([a for a in args if not a.startswith("--")], names)
    (OUT / name).mkdir(parents=True, exist_ok=True)
    manifest = []
    for ci, c in enumerate(clips):
        for source, kind, f, box in pool(c):
            # A repaired name means other boxes at the same frame: it gets its own id, so no cached verdict is reused.
            mid = f"{name}-{ci:03d}-{f:03d}" + ("" if c["name"] == c["sam_name"]
                                                else "-" + c["name"].replace(" ", "_").replace("+", "_"))
            path = OUT / name / f"{mid}.jpg"  # JPEG: a quarter of the PNG's bytes in the API request
            window = OUT / name / f"{mid}.window.jpg"
            if not path.exists():
                render(f"clips/{c['file']}", c["n"], f, box, path)
            if not window.exists():
                render_window(f"clips/{c['file']}", c["n"], f, window)
            manifest.append(dict(id=mid, clip=ci, file=c["file"], concept=c["concept"],
                                 name=c["name"].removesuffix("+box"), frame=f, kind=kind, source=source,
                                 box=[round(v, 4) for v in box], image=str(path), window=str(window)))
    (OUT / f"{name}.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} moments from {len(clips)} objects:",
          {s: sum(m["source"] == s for m in manifest) for s in "APE"})
