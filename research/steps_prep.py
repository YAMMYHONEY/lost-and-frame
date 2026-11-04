# Third witness, the spotter: SAM 3.1's detector on every third frame (detect.py), with no tracker memory. It proposes
# two kinds of moments:
# - where the count of the object steps for good (count_change.py);
# - where one detected instance's evidence track starts or ends away from the frame edge, with nothing left at its
#   place on the other side (evidence.py). A copy among many logs or rocks moves no count but leaves such a track.
# Each proposal's box is cropped with room around it from the eight frames of a window centred there, so the judge
# sees a small look-alike up close (field "zoom8"). Proposals where a given judge already alerts (same kind, ±TOL)
# get no zoom: the spotter speaks only where the judges are silent. Appends the windows (source "S") to a copy of the
# tile manifest for eval_tile.py --union. Also writes every proposal as an unverified verdict, to see what the zoomed
# judge adds.
# Usage: python research\steps_prep.py <tile manifest.json> [<judge verdicts.json>...]
#   -> <manifest>-steps.json, <manifest>-steps-verdicts-raw.json
import json, sys
from pathlib import Path
import cv2
from count_change import detect_steps
from detect import out_path
from evidence import events
from eval_tile import SHIFT, p
from eval_verify import BOTH
from eval_evidence import TOL
from frames8_prep import OFF8

# Proposals of each kind per object; window half-width; same-moment frames; judge alert threshold.
K, HALF, MERGE, T = 3, 15, 5, 0.5


def spots(d, k=K):
    """[(kind, frame, box)]: the k strongest evidence-track ends (contrast x length) and the k largest count steps."""
    fr = sorted(map(int, d["det"]))
    frames = {str(i): {"det": d["det"][str(f)]} for i, f in enumerate(fr)}
    ev = [e for e in events(frames, len(fr)) if not e["edge"] and e["length"] >= 5 and e["contrast"] > 0]
    out = []
    for c in ([(e["kind"], fr[e["frame"]], e["box"])
               for e in sorted(ev, key=lambda e: -e["contrast"] * e["length"])[:k]]
              + [(kind, f, box) for kind, f, _, box in detect_steps(d)[:k]]):
        if not any(c[0] == o[0] and abs(c[1] - o[1]) <= MERGE for o in out):  # a step at a track end adds nothing
            out.append(c)
    return out


def region(box, w, h):
    """Pixel crop (x0, y0, x1, y1) around a normalized xyxy box: three times its size, at least a third of the frame
    wide, the frame's aspect, inside the frame. No box: the whole frame."""
    if box is None:
        return 0, 0, w, h
    bw, bh = (box[2] - box[0]) * w, (box[3] - box[1]) * h
    rw = max(3 * bw, w / 3, 3 * bh * w / h)
    rw, rh = min(rw, w), min(rw * h / w, h)
    cx, cy = (box[0] + box[2]) / 2 * w, (box[1] + box[3]) / 2 * h
    x0, y0 = min(max(cx - rw / 2, 0), w - rw), min(max(cy - rh / 2, 0), h - rh)
    return int(x0), int(y0), int(x0 + rw), int(y0 + rh)


assert region(None, 640, 360) == (0, 0, 640, 360) and region([0.0, 0.0, 0.05, 0.05], 640, 360) == (0, 0, 213, 120)
assert region([0.9, 0.9, 1.0, 1.0], 640, 360)[2:] == (640, 360)
_d = dict(stride=3, det={str(f): [[0.6, .1, .4, .2, .5, 1], [0.6, .4, .4, .5, .5, 1]]
                                 + ([[0.6, .7, .4, .8, .5, 1]] if 30 <= f < 90 else [])
                         for f in range(0, 120, 3)})  # a third thing from frame 30 to 90: one appear, one vanish
assert sorted((k, f) for k, f, _ in spots(_d)) == [("appear", 30), ("vanish", 87)], spots(_d)

if __name__ == "__main__":
    path, judged = Path(sys.argv[1]), [json.loads(Path(f).read_text("utf-8")) for f in sys.argv[2:]]
    manifest, out, raw = json.loads(path.read_text()), [], []
    said = {}  # moment id -> the given judges' verdicts on it
    for v in (v for vs in judged for v in vs):
        said.setdefault(v["id"], []).append(v)
    folder = path.parent / f"{path.stem}-steps"
    folder.mkdir(exist_ok=True)

    for ci in sorted({m["clip"] for m in manifest}):
        mine = [m for m in manifest if m["clip"] == ci]
        alerts = [(k, m["frame"] + SHIFT.get(v["between"], 0))
                  for m in mine for v in said.get(m["id"], []) if p(v) >= T
                  for k in ({"vanish": ("vanish",), "appear": ("appear",)}.get(v["kind"]) or BOTH)]
        d = out_path(mine[0]["file"], mine[0]["concept"])
        if not d.exists():
            continue
        cap = cv2.VideoCapture(f"clips/{mine[0]['file']}")
        n, w, h = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        for kind, f, box in spots(json.loads(d.read_text())):
            c, (x0, y0, x1, y1) = min(max(f, HALF), n - 1 - HALF), region(box, w, h)
            mid = f"{path.stem}-{ci:03d}-s{c:03d}{kind[0]}"
            m = dict(id=mid, clip=ci, file=mine[0]["file"], concept=mine[0]["concept"], name=mine[0]["concept"],
                     frame=c, kind=kind, source="S", box=box, image=None, window=None)
            raw.append(dict(id=mid, violation=True, kind=kind, between="none", confidence=100, reason="spotter",
                            cost=0))
            if not any(k == kind and abs(a - f) <= TOL for k, a in alerts):
                files = [folder / f"{mid}-{j}.jpg" for j in range(len(OFF8))]
                for o, file in zip(OFF8, files):
                    if not file.exists():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, c + o)
                        img = cv2.resize(cap.read()[1][y0:y1, x0:x1], (640, 360))
                        cv2.putText(img, "ABCDEFGH"[OFF8.index(o)], (12, 44), 0, 1.4, (255, 255, 255), 4)
                        # imwrite returns False instead of raising (e.g. a path over Windows' 260 characters).
                        assert cv2.imwrite(str(file), img, [cv2.IMWRITE_JPEG_QUALITY, 92]), f"could not write {file}"
                m["zoom8"] = [str(x) for x in files]
            out.append(m)

    path.with_name(f"{path.stem}-steps.json").write_text(json.dumps(manifest + out, indent=1))
    path.with_name(f"{path.stem}-steps-verdicts-raw.json").write_text(json.dumps(raw, indent=1))
    print(len(out), "spotter proposals over", len({m["clip"] for m in out}), "objects;",
          sum("zoom8" in m for m in out), "where the judges are silent")
