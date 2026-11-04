# Pick v6 from Physion-Eval: every clip not on disk, in any glitch category other than object permanence (v3/v4
# used those), whose expert note names a time and an appearing or disappearing thing ("a black object suddenly
# appears on the left hand at 0.7s"). These are fresh real generator glitches of the permanence kind. Written before
# any v6 video is downloaded.
# Usage: python research\select_v6.py   -> clips/physion/selection-v6.json (also the zipfetch.mjs list: "file")
import json, glob, re
from pathlib import Path

# The v4 rule: a time and (dis)appearing. It is copied, not imported, because select_v4.py rewrites its own
# selection when run.
note = lambda a: a["reasoning"] if isinstance(a["reasoning"], str) else " ".join(map(str, a["reasoning"]))
clear = lambda a: (re.search(r"\bat\s+\d+(\.\d+)?\s*(s\b|sec)", note(a), re.I)
                   and re.search(r"disappear|vanish|appear", note(a), re.I))

ann = json.loads(Path("clips/physion/Physion_Eval_20260322.json").read_text())
used = {Path(f).stem for f in glob.glob("clips/physion/videos/*.mp4")}
pick = [dict(a, file=f"physion/videos/{a['filename']}.mp4") for a in ann
        if a["has_glitches"] and a["Glitch_Category"] != "Object Permanence Violation"
        and a["filename"] not in used and clear(a)]
assert len({a["filename"] for a in pick}) == len(pick)
Path("clips/physion/selection-v6.json").write_text(json.dumps(pick, indent=1))
print(len(pick), "clips")
