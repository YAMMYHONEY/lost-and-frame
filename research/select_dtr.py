# Real-glitch test from DeepTraceReward (Apache-2.0, huggingface.co/datasets/DeepTraceReward/RewardData): every
# trace on a generated video whose expert explanation says something vanishes or appears (onset/offset frames and a
# box per trace). The rule is fixed from the annotation text alone, before any of these videos is downloaded or
# looked at.
# Usage: python -I research\select_dtr.py <all_data.json>
#        -> clips/deeptrace/selection.json (also the zipfetch.mjs list)
import json, re, sys
from pathlib import Path

PERM = re.compile(
    r"disappear|vanish|appears? (out of|from) (nowhere|thin air)|suddenly appear|pops? (in|up|out)|appear(s|ed)? "
    r"(suddenly|abruptly|out of nowhere)|emerg\w* (out of|from) nowhere|out of thin air|materializ|reappear", re.I)

if __name__ == "__main__":
    rows = json.loads(Path(sys.argv[1]).read_text("utf-8"))
    keep = [dict(a, file=f"deeptrace/videos/{Path(a['video_path']).name}") for a in rows
            if a["video_type"] == "fake_video" and PERM.search(a["faketrace_explanations"])]
    out = Path("clips/deeptrace/selection.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(keep, indent=1))
    print(len(keep), "traces in", len({a["file"] for a in keep}), "videos")
