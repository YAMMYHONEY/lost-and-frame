# Pick a fresh sealed set (v3) from Physion-Eval: per generator, PER_MODEL permanence clips whose expert note names a
# time and a vanish/appear, plus CLEAN_PER_MODEL no-issue clips; nothing already on disk. Fixed seed, written before
# any v3 video is downloaded or looked at. Labels are then drafted from the notes (clips/physion-labels-v3.draft.json).
# Usage: python research\select_v3.py
import json, random, re, glob
from pathlib import Path

PER_MODEL, CLEAN_PER_MODEL, SEED = 8, 4, 20261030
ann = json.loads(Path("clips/physion/Physion_Eval_20260322.json").read_text())
used = {Path(f).stem for f in glob.glob("clips/physion/videos/*.mp4")}
# Some notes are lists.
note = lambda a: a["reasoning"] if isinstance(a["reasoning"], str) else " ".join(map(str, a["reasoning"]))
# A point in time ("at 2.5s", "at 1.4 seconds"), not only an interval: events are scored within ±0.5 s.
clear = lambda a: (re.search(r"\bat\s+\d+(\.\d+)?\s*(s\b|sec)", note(a), re.I)
                   and re.search(r"disappear|vanish|appear", note(a), re.I))

rng = random.Random(SEED)
pick = []
for model in sorted({a["model"] for a in ann}):
    perm = [a for a in ann if a["model"] == model and a["Glitch_Category"] == "Object Permanence Violation"
            and a["filename"] not in used and clear(a)]
    clean = [a for a in ann if a["model"] == model and not a["has_glitches"] and a["filename"] not in used]
    pick += rng.sample(perm, PER_MODEL) + rng.sample(clean, CLEAN_PER_MODEL)
Path("clips/physion/selection-v3.json").write_text(json.dumps(pick, indent=1))
print(len(pick), "clips:", sum(a["has_glitches"] for a in pick), "permanence,",
      sum(not a["has_glitches"] for a in pick), "no-issue")
