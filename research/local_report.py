# The "<list>-eval.json" stub the research scripts read (fps, frame count, samName per object) for a list that never
# went through the paid API evaluation (eval.mjs), e.g. sealed v3. samName is the given name: the API's name check
# (given name, else its last word) is covered by anchored renaming, whose candidates include the head noun.
# Usage: python research\local_report.py clips\physion-labels-v3.json   -> clips\physion-labels-v3-eval.json
import json, sys
from pathlib import Path
import cv2

if __name__ == "__main__":
    path = Path(sys.argv[1])
    report = []
    for clip in json.loads(path.read_text()):
        cap = cv2.VideoCapture(f"clips/{clip['file']}")
        assert cap.isOpened(), clip["file"]
        report.append(dict(source=f"clips/{clip['file']}", fps=cap.get(cv2.CAP_PROP_FPS),
                           frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), samName={c: c for c in clip["concepts"]}))
    out = path.with_name(path.stem + "-eval.json")
    out.write_text(json.dumps({"list": str(path), "report": report}, indent=1))
    print(len(report), "clips ->", out)
