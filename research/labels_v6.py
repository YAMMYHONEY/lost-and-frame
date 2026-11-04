# Fresh real labels, written before any of their videos is looked at or run (10-31 evening).
# v6: Physion-Eval expert notes of selection-v6.json (other glitch categories that state a timed vanish/appear), turned
# into events by six helpers with the v4 rule given up front (clips/physion/v6-drafts/perm_*.json): one short noun per
# object, vanish/appear at a point time, a split or extra copy is an appear, a merge or count drop a vanish.
# DTR: DeepTraceReward traces whose explanation says something vanishes or appears (select_dtr.py), turned into events
# at the trace's onset (or offset) frame by one helper (clips/deeptrace/drafts.json).
# Writes clips/physion-labels-v6.json and clips/deeptrace-labels.json (only clips with an event).
import json
from pathlib import Path
from labels_v4 import DROP


def clips(sel, drafts, time):
    """[{file, model, concepts, events}] from per-row drafts; several rows of one video become one clip."""
    out = {}
    for i, a in enumerate(sel):
        ev = [(c.strip().lower(), k, time(a, t)) for c, k, t in drafts[i] if c.strip().lower() not in DROP]
        if ev:
            c = out.setdefault(a["file"], dict(file=a["file"], model=a.get("model") or a.get("video_source"),
                                               events=set()))
            c["events"] |= set(ev)
    for c in out.values():
        ev = sorted(c.pop("events"), key=lambda e: (e[2], e[0], e[1]))
        c["concepts"], c["events"] = sorted({e[0] for e in ev}), [dict(concept=n, kind=k, time=t) for n, k, t in ev]
    return list(out.values())


if __name__ == "__main__":
    sel = json.loads(Path("clips/physion/selection-v6.json").read_text())
    drafts = {int(i): ev for k in range(6)
              for i, ev in json.loads(Path(f"clips/physion/v6-drafts/perm_{k}.json").read_text("utf-8")).items()}
    assert set(drafts) == set(range(len(sel)))
    v6 = clips(sel, drafts, lambda a, t: float(t))

    dtr_sel = json.loads(Path("clips/deeptrace/selection.json").read_text("utf-8"))
    dtr_drafts = {int(i): ev for i, ev in json.loads(Path("clips/deeptrace/drafts.json").read_text("utf-8")).items()}
    assert set(dtr_drafts) == set(range(len(dtr_sel)))
    assert all(a["faketrace_first_frame"] <= f <= a["faketrace_last_frame"]
               for i, a in enumerate(dtr_sel) for _, _, f in dtr_drafts[i])
    dtr = clips(dtr_sel, dtr_drafts, lambda a, f: round(int(f) / a["video_frame_rate"], 4))

    for name, out in (("physion-labels-v6", v6), ("deeptrace-labels", dtr)):
        assert all(e["kind"] in ("appear", "vanish") for c in out for e in c["events"])
        Path(f"clips/{name}.json").write_text(json.dumps(out, indent=1))
        print(name, len(out), "clips,", sum(len(c["events"]) for c in out), "events,",
              sum(len(c["concepts"]) for c in out), "objects")
