# Score every candidate name (names.py) with SAM 3.1's detector alone on K evenly spaced frames, no tracking.
# Saves each candidate's top-3 detections (pre-NMS score, normalized xyxy box) per frame; choose_names.py picks one.
# Usage: python research\name_probe.py <dev list.json>...   (resumes research/name-probe.json)
import json, os, sys, time, warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
warnings.simplefilter("ignore")
import numpy as np
from sam3.model_builder import build_sam3_predictor
from trace import recorder, MAX_OBJECTS
from names import candidates

K, OUT = 8, Path("research/name-probe.json")
captions = {a["filename"]: a.get("caption", "")
            for a in json.loads(Path("clips/physion/all_metadata_with_caption.json").read_text())}
done = json.loads(OUT.read_text()) if OUT.exists() else {}
predictor = build_sam3_predictor(version="sam3.1", use_fa3=False, compile=False, max_num_objects=MAX_OBJECTS)
predictor.model.batched_grounding_batch_size = 1  # 8 GB GPU, see sam_local.py
frames = recorder(predictor.model)
ask = lambda **r: predictor.handle_request(r)
MAX_NEW, probed_clips = int(os.environ.get("PROBE_MAX_NEW", 0)), 0

for path in sys.argv[1:]:
    report = json.loads(Path(path.replace(".json", "-eval.json")).read_text())["report"]
    for clip in json.loads(Path(path).read_text()):
        r = next(r for r in report if r["source"].endswith(Path(clip["file"]).name))
        todo = [c for c in clip["concepts"] if f"{clip['file']}|{c}" not in done]
        if not todo:
            continue
        sample = sorted({int(f) for f in np.linspace(0, r["frames"] - 1, K).round()})
        sid = ask(type="start_session", resource_path=f"clips/{clip['file']}", offload_video_to_cpu=True)["session_id"]
        for concept in todo:
            names = candidates(concept, captions.get(Path(clip["file"]).stem, ""))
            scores = {name: {} for name in names}
            # The detector output depends only on frame and text (checked: identical scores with or without a
            # reset). But without resets, every prompt's objects pile up in the session until the 8 GB GPU spills.
            t = time.time()
            for f in sample:
                for name in names:
                    ask(type="reset_session", session_id=sid)
                    frames.clear()
                    ask(type="add_prompt", session_id=sid, frame_index=f, text=name)
                    scores[name][f] = [d[:5] for d in frames[f]["det"][:3]]
            done[f"{clip['file']}|{concept}"] = {"samName": r["samName"][concept], "scores": scores}
            OUT.write_text(json.dumps(done))
            print("probed", concept, len(scores), f"names in {time.time() - t:.0f} s", flush=True)
        ask(type="close_session", session_id=sid)
        probed_clips += 1
        if MAX_NEW and probed_clips >= MAX_NEW:
            sys.exit(3)  # the caller restarts a fresh process, see TRACE_MAX_NEW in trace.py
