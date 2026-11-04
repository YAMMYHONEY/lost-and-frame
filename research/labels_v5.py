# v5 bases: the no-issue clips of select_v5.py with the object their caption names, minus scenery as in v4. One
# Claude agent named the objects from the captions only, never the videos (clips/physion/v5-drafts/clean.json).
# inject.py turns the bases into sealed set v5. Once clips/synv5.json exists, the no-issue clips that are not its
# bases (scenery, or nothing to copy) become the fresh clean control, each with its named object as in the v4 clean
# test.
# Usage: python research\labels_v5.py   -> clips/physion-clean-v5-things.json [, clips/physion-v5-clean.json]
import json
from pathlib import Path
from labels_v4 import SCENERY

if __name__ == "__main__":
    sel = json.loads(Path("clips/physion/selection-v5.json").read_text())
    names = {int(i): n.strip().lower()
             for i, n in json.loads(Path("clips/physion/v5-drafts/clean.json").read_text("utf-8")).items()}
    assert set(names) == set(range(len(sel)))
    things = [dict(file=a["file"], model=a["model"], concepts=[names[i]], events=[]) for i, a in enumerate(sel)
              if names[i] not in SCENERY | {"liquid", "geyser"}]  # stuff, not a thing to copy
    Path("clips/physion-clean-v5-things.json").write_text(json.dumps(things, indent=1))
    print(len(sel), "no-issue clips,", len(things), "with a thing to copy")
    if Path("clips/synv5.json").exists():
        bases = {c["base"] for c in json.loads(Path("clips/synv5.json").read_text())}
        control = [dict(file=a["file"], model=a["model"], concepts=[names[i]], events=[]) for i, a in enumerate(sel)
                   if a["file"] not in bases]
        assert len(control) + len(bases) == len(sel)
        Path("clips/physion-v5-clean.json").write_text(json.dumps(control, indent=1))
        print(len(control), "clean control clips (not injected)")
