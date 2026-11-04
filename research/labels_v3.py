# Sealed v3 labels, transcribed by hand from the Physion-Eval expert notes of selection-v3.json (select_v3.py) on
# 10-30, before any v3 video was downloaded or run. Rule: one short noun a user would type per object; only
# vanish/appear events with a point time ("at 2.5s"); interval-only or non-object notes (sky colour) are left out.
# No-issue clips track the main object of their caption. The transcribed events stay with the dataset, outside this
# repository (Physion-Eval is for research use only): clips/physion/labels-v3-transcribed.json.
# Writes clips/physion-labels-v3.json.
import json
from pathlib import Path

if __name__ == "__main__":
    sel = json.loads(Path("clips/physion/selection-v3.json").read_text())
    lab = json.loads(Path("clips/physion/labels-v3-transcribed.json").read_text("utf-8"))  # selection index: events
    EVENTS, CLEAN = ({int(i): v for i, v in lab[k].items()} for k in ("events", "clean"))
    out = []
    for i, a in enumerate(sel):
        if i in EVENTS:
            ev = EVENTS[i]
            out.append(dict(file=f"physion/videos/{a['filename']}.mp4", model=a["model"],
                            concepts=sorted({c for c, _, _ in ev}),
                            events=[dict(concept=c, kind=k, time=t) for c, k, t in ev]))
        elif i in CLEAN:
            out.append(dict(file=f"physion/videos/{a['filename']}.mp4", model=a["model"], concepts=[CLEAN[i]],
                            events=[]))
    assert all(sel[i]["has_glitches"] for i in EVENTS) and not any(sel[i]["has_glitches"] for i in CLEAN)
    Path("clips/physion-labels-v3.json").write_text(json.dumps(out, indent=1))
    print(len(out), "clips,", sum(len(c["events"]) for c in out), "events,", sum(not c["events"] for c in out),
          "no-issue")
