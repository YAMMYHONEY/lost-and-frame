# v4 labels from the Physion-Eval expert notes of selection-v4.json, written on 10-31 before any v4 video was looked
# at or run. Six helpers transcribed the notes (clips/physion/v4-drafts/perm_*.json) with the labels_v3.py rule: one
# short noun per object, only vanish/appear at a point time ("at 2.5s"). The helpers disagreed on count changes, so
# one rule is applied here by hand:
# - a split, an extra copy or a count rise of discrete objects is an appear;
# - a merge or a count drop is a vanish;
# - amount changes of stuff (liquid, powder), clouds, light and flames are left out.
# No-issue clips watch the object their caption names (v4-drafts/clean.json). Events the drafts left out under the
# rule above are in v4-drafts/add.json. Like the drafts, they stay with the dataset, outside this repository.
# Writes clips/physion-labels-v4.json and, for inject.py, the no-issue clips whose object is a thing rather than
# scenery (clips/physion-clean-v4-things.json).
import json
from pathlib import Path

DROP = {"cloud"}  # sky, not an object
SCENERY = {"cloud", "firework", "waterfall", "sun", "mountain", "flame", "smoke", "sunbeam", "paint", "aurora", "wave",
           "campfire"}

if __name__ == "__main__":
    sel = json.loads(Path("clips/physion/selection-v4.json").read_text())
    drafts = {int(i): ev for k in range(6)
              for i, ev in json.loads(Path(f"clips/physion/v4-drafts/perm_{k}.json").read_text("utf-8")).items()}
    names = {int(i): n for i, n in json.loads(Path("clips/physion/v4-drafts/clean.json").read_text("utf-8")).items()}
    add = {int(i): ev for i, ev in json.loads(Path("clips/physion/v4-drafts/add.json").read_text("utf-8")).items()}
    assert (set(drafts) == {i for i, a in enumerate(sel) if a["has_glitches"]}
            and set(names) == {i for i, a in enumerate(sel) if not a["has_glitches"]})

    out = []
    for i, a in enumerate(sel):
        if i in names:
            out.append(dict(file=a["file"], model=a["model"], concepts=[names[i]], events=[]))
            continue
        ev = sorted({(c, k, float(t)) for c, k, t in drafts[i] + add.get(i, []) if c not in DROP},
                    key=lambda e: (e[2], e[0], e[1]))
        if ev:  # a permanence clip with no usable event is left out, as in v3
            out.append(dict(file=a["file"], model=a["model"], concepts=sorted({c for c, _, _ in ev}),
                            events=[dict(concept=c, kind=k, time=t) for c, k, t in ev]))
    assert all(e["kind"] in ("appear", "vanish") for c in out for e in c["events"])
    Path("clips/physion-labels-v4.json").write_text(json.dumps(out, indent=1))
    things = [c for c in out if not c["events"] and c["concepts"][0] not in SCENERY]
    Path("clips/physion-clean-v4-things.json").write_text(json.dumps(things, indent=1))
    print(len(out), "clips,", sum(len(c["events"]) for c in out), "events,", sum(not c["events"] for c in out),
          "no-issue,", len(things), "no-issue with a thing to copy")
