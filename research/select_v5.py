# Pick v5 from Physion-Eval: CLEAN_PER_MODEL no-issue clips per generator that are not on disk yet, the bases of a fresh
# synthetic-injection test (sealed v5). Fixed seed, written before any v5 video is downloaded or looked at.
# Usage: python research\select_v5.py   -> clips/physion/selection-v5.json (also the zipfetch.mjs list: "file")
import json, random, glob
from pathlib import Path

CLEAN_PER_MODEL, SEED = 30, 20261101
ann = json.loads(Path("clips/physion/Physion_Eval_20260322.json").read_text())
used = {Path(f).stem for f in glob.glob("clips/physion/videos/*.mp4")}
rng = random.Random(SEED)
pick = []
for model in sorted({a["model"] for a in ann}):
    pick += rng.sample([a for a in ann if a["model"] == model and not a["has_glitches"] and a["filename"] not in used],
                       CLEAN_PER_MODEL)
pick = [dict(a, file=f"physion/videos/{a['filename']}.mp4") for a in pick]
assert len({a["filename"] for a in pick}) == len(pick) and not used & {a["filename"] for a in pick}
Path("clips/physion/selection-v5.json").write_text(json.dumps(pick, indent=1))
print(len(pick), "no-issue clips")
