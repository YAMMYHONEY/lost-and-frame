# Adopt a Spark-pointed exemplar trace (trace.py --boxes) only where SAM confirms it: on the exemplar's frame one of
# SAM's tracks overlaps Spark's box (IoU >= MATCH). Otherwise the object keeps its chosen name (still untracked).
# Writes research/names-final.json = names-chosen.json + confirmed exemplars.
# Usage: python research\exemplar_names.py <localize boxes.json>...
import json, sys
from pathlib import Path
from paths import trace_path
from choose_names import iou

MATCH = 0.3

if __name__ == "__main__":
    final = json.loads(Path("research/names-chosen.json").read_text())
    for path in sys.argv[1:]:
        for o in json.loads(Path(path).read_text("utf-8")):  # written by Node
            t = trace_path(o["file"], f"{o['name']}+box")
            if not o["boxes"] or not t.exists():
                print(f"  {o['name']!r:16} no box from Spark" if not o["boxes"]
                      else f"  {o['name']!r:16} not traced yet")
                continue
            d = json.loads(t.read_text())
            out = d["frames"].get(str(d["box_frame"]), {}).get("out", {})
            best = max((iou([x, y, x + w, y + h], d["box"]) for x, y, w, h, _ in out.values()), default=0.0)
            tracked = sum(1 for f in d["frames"].values() if f.get("out"))
            ok = best >= MATCH
            print(f"  {o['name']!r:16} SAM overlap with Spark's box {best:.2f}, frames with a track {tracked}: {'adopted' if ok else 'rejected'}")
            if ok:
                final[o["key"]] = {**final.get(o["key"], {}), "name": f"{o['name']}+box"}
    Path("research/names-final.json").write_text(json.dumps(final, indent=1))
