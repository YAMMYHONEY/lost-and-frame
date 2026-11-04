# Are the judge's alarms on "no issue" clips false? Physion-Eval's no-issue label is one expert's pass, and many
# alarm reasons read like real glitches ("towel held in both hands disappears").
# Blind audit: every window the judge flagged (p >= T) on the clean clips, plus a seeded random sample of as many
# windows it did not flag. They are shuffled into audit_sheet.py sheets with a neutral kind, so the auditor cannot
# tell which is which. Score, per group: the share the auditor judged a visible uncaused change.
# Usage: python research\alarm_audit.py prep <manifest.json> <verdicts.json> <out dir>
#        python research\alarm_audit.py score <out dir>          (after the auditor wrote <out dir>/out_0.json)
import json, random, subprocess, sys
from pathlib import Path
import cv2
from eval_tile import p

T, SEED = 0.5, 20261031

if __name__ == "__main__":
    mode, out = sys.argv[1], Path(sys.argv[-1])
    if mode == "prep":
        man = json.loads(Path(sys.argv[2]).read_text())
        ver = {v["id"]: v for v in json.loads(Path(sys.argv[3]).read_text("utf-8"))}
        flagged = [m for m in man if m["id"] in ver and p(ver[m["id"]]) >= T]
        rng = random.Random(SEED)
        rest = rng.sample([m for m in man if m["id"] in ver and p(ver[m["id"]]) < T], len(flagged))
        both = flagged + rest
        rng.shuffle(both)

        out.mkdir(parents=True, exist_ok=True)
        fps = {f: cv2.VideoCapture(f"clips/{f}").get(cv2.CAP_PROP_FPS) for f in {m["file"] for m in both}}
        labels = [dict(file=m["file"], concepts=[m["concept"]],
                       events=[dict(concept=m["concept"], kind="change", time=round(m["frame"] / fps[m["file"]], 3))])
                  for m in both]
        (out / "labels.json").write_text(json.dumps(labels, indent=1))
        (out / "key.json").write_text(json.dumps([dict(id=m["id"], flagged=m in flagged) for m in both], indent=1))
        subprocess.run([sys.executable, "research/audit_sheet.py", str(out / "labels.json"),
                        "clips/physion/selection-v4.json", str(out)], check=True)
    else:
        key, calls = json.loads((out / "key.json").read_text()), json.loads((out / "out_0.json").read_text("utf-8"))
        for flagged in (True, False):
            cls = [calls[str(i)][0] for i, k in enumerate(key) if k["flagged"] == flagged]
            print("judge flagged" if flagged else "not flagged ", len(cls),
                  {c: cls.count(c) for c in ("V", "V-lean", "A", "N")})
