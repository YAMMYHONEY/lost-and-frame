# Detector-only pass: SAM 3.1's detector runs on every STRIDE-th frame of a clip for one name. There is no tracking,
# so no memory can carry a vanished object. Scores below the 0.4 threshold are recovered (trace.py's recorder).
# count_change.py turns the result into a count trace. Saves <DETECT_DIR>/<clip>.<name>.json; existing files are
# skipped.
# Usage: python research\detect.py <list.json>...   (every clip, every concept)
import json, os, sys, time, warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
warnings.simplefilter("ignore")
import cv2
from sam3.model_builder import build_sam3_predictor
from trace import recorder

# DETECT_DIR: the sealed runs used $VC_WORK/detects, a short path, because 64-character Physion names under a deep
# project folder passed Windows' 260-character limit. The app keeps clips/detects.
STRIDE, OUT = int(os.environ.get("DETECT_STRIDE", 3)), Path(os.environ.get("DETECT_DIR", "clips/detects"))
MAX_NEW = int(os.environ.get("DETECT_MAX_NEW", 0))  # exit 3 after that many clips: one long process pages (trace.py)
out_path = lambda file, name: OUT / f"{Path(file).stem}.{name}.json"

if __name__ == "__main__":
    predictor = build_sam3_predictor(version="sam3.1", use_fa3=False, compile=False, max_num_objects=8)
    predictor.model.batched_grounding_batch_size = 1  # 8 GB GPU, see sam_local.py
    frames, ask = recorder(predictor.model), lambda **r: predictor.handle_request(r)

    OUT.mkdir(exist_ok=True)
    new = 0
    for path in sys.argv[1:]:
        for clip in json.loads(Path(path).read_text()):
            todo = [c for c in clip["concepts"] if not out_path(clip["file"], c).exists()]
            if not todo:
                continue
            n = int(cv2.VideoCapture(f"clips/{clip['file']}").get(cv2.CAP_PROP_FRAME_COUNT))
            sid = ask(type="start_session", resource_path=f"clips/{clip['file']}",
                      offload_video_to_cpu=True)["session_id"]
            for concept in todo:
                t, det = time.time(), {}
                for f in range(0, n, STRIDE):
                    # Detector output depends only on frame and text (name_probe.py).
                    ask(type="reset_session", session_id=sid)
                    frames.clear()
                    ask(type="add_prompt", session_id=sid, frame_index=f, text=concept)
                    det[f] = frames[f]["det"]
                out_path(clip["file"], concept).write_text(
                    json.dumps(dict(file=clip["file"], concept=concept, n=n, stride=STRIDE, det=det)))
                print(f"{clip['file']} {concept}: {len(det)} frames in {time.time() - t:.0f} s", flush=True)
            ask(type="close_session", session_id=sid)
            new += 1
            if MAX_NEW and new >= MAX_NEW:
                sys.exit(3)
