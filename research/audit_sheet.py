# Visibility-audit sheets, as made by hand for v3. For every labelled event, the sheet shows the 8 frames at
# frames8_prep.py's offsets around its labelled frame (what the judge sees there), two events per image. An index
# keeps the expert note for each event.
# Made and viewed before any method runs on the set; the classes are in clips/physion-labels-v3-visibility.json.
# Usage: python research\audit_sheet.py <labels.json> <selection.json> <out dir>   -> sheet###.jpg, index.json
import json, sys
from pathlib import Path
import cv2, numpy as np
from frames8_prep import OFF8

TW, TH, HEAD = 400, 225, 40


def tile(img):  # letterboxed, so portrait clips keep their shape
    s = min(TW / img.shape[1], TH / img.shape[0])
    img, out = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s))), np.zeros((TH, TW, 3), np.uint8)
    y, x = (TH - img.shape[0]) // 2, (TW - img.shape[1]) // 2
    out[y:y + img.shape[0], x:x + img.shape[1]] = img
    return out


if __name__ == "__main__":
    labels, sel, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    notes = {}  # Physion-Eval: one expert note per clip; DeepTraceReward (select_dtr.py): one explanation per trace
    for a in json.loads(Path(sel).read_text("utf-8")):
        f, r = (a.get("file") or f"physion/videos/{a['filename']}.mp4",
                a.get("reasoning", a.get("faketrace_explanations", "")))
        notes[f] = " ".join(filter(None, [notes.get(f), r if isinstance(r, str) else " ".join(map(str, r))]))

    index, blocks = [], []
    for c in json.loads(Path(labels).read_text()):
        cap = cv2.VideoCapture(f"clips/{c['file']}")
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for e in c["events"]:
            f = round(e["time"] * fps)
            index.append(dict(i=len(index), file=c["file"], concept=e["concept"], kind=e["kind"], time=e["time"],
                              frame=f, n=n, note=notes.get(c["file"], "")))  # inject.py clips have no expert note
            tiles = []
            for k, o in enumerate(OFF8):
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(n - 1, max(0, f + o)))
                t = tile(cap.read()[1])
                cv2.putText(t, "ABCDEFGH"[k], (8, 30), 0, 0.9, (255, 255, 255), 3)
                tiles.append(t)
            head = np.zeros((HEAD, 4 * TW, 3), np.uint8)
            cv2.putText(head, f"#{index[-1]['i']:03d} {e['concept']} {e['kind']} {e['time']}s (f{f}/{n})",
                        (8, 29), 0, 0.9, (0, 255, 255), 2)
            blocks.append(np.vstack([head, np.hstack(tiles[:4]), np.hstack(tiles[4:])]))

    for s in range(0, len(blocks), 2):
        cv2.imwrite(str(out / f"sheet{s // 2:03d}.jpg"), np.vstack(blocks[s:s + 2]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(len(index), "events on", (len(blocks) + 1) // 2, "sheets")
