# Pick v4 from Physion-Eval: every remaining permanence clip whose expert note names a time and a vanish/appear (the
# real-glitch test), plus CLEAN_PER_MODEL no-issue clips per generator (synthetic-injection bases and the clean
# control). Nothing already on disk (dev, sealed v3) is picked. Fixed seed, written before any v4 video is
# downloaded or looked at.
# Usage: python research\select_v4.py   -> clips/physion/selection-v4.json (also the zipfetch.mjs list: "file")
import json, random, re, glob
from pathlib import Path

CLEAN_PER_MODEL, SEED = 30, 20261031
ann = json.loads(Path("clips/physion/Physion_Eval_20260322.json").read_text())
used = {Path(f).stem for f in glob.glob("clips/physion/videos/*.mp4")}
note = lambda a: a["reasoning"] if isinstance(a["reasoning"], str) else " ".join(map(str, a["reasoning"]))
clear = lambda a: (re.search(r"\bat\s+\d+(\.\d+)?\s*(s\b|sec)", note(a), re.I)
                   and re.search(r"disappear|vanish|appear", note(a), re.I))

rng = random.Random(SEED)
pick = [a for a in ann
        if a["Glitch_Category"] == "Object Permanence Violation" and a["filename"] not in used and clear(a)]
for model in sorted({a["model"] for a in ann}):
    pick += rng.sample([a for a in ann if a["model"] == model and not a["has_glitches"] and a["filename"] not in used],
                       CLEAN_PER_MODEL)
pick = [dict(a, file=f"physion/videos/{a['filename']}.mp4") for a in pick]
assert len({a["filename"] for a in pick}) == len(pick) and not used & {a["filename"] for a in pick}
Path("clips/physion/selection-v4.json").write_text(json.dumps(pick, indent=1))
print(len(pick), "clips:", sum(a["has_glitches"] > 0 for a in pick), "permanence,",
      sum(not a["has_glitches"] for a in pick), "no-issue")
