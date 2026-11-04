# Run SAM 3.1 locally on one clip with one text prompt, report what it tracked, and compare the frames
# where the object is seen with the API's cached reply for the same clip and prompt (if any).
# Usage: python research\sam_local.py clips\syn-vanish.mp4 "red square"
import os, re, sys, time, warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the checkpoint is already in the HF cache
warnings.simplefilter("ignore")
import torch
from sam3.model_builder import build_sam3_predictor

video, text = sys.argv[1], sys.argv[2]
t0 = time.time()
predictor = build_sam3_predictor(version="sam3.1", use_fa3=False, compile=False)  # FA3 needs a Hopper GPU
# The builder batches 16 frames per detector pass (sized for an 80 GB H100). On an 8 GB GPU that needs
# > 12 GiB and Windows spills into shared memory, so tracking crawls; 1 frame peaks at 5.4 GiB (mem_probe.py).
predictor.model.batched_grounding_batch_size = 1
t1 = time.time()
sid = predictor.handle_request(dict(type="start_session", resource_path=video,
                                    offload_video_to_cpu=True))["session_id"]  # 8 GB GPU
predictor.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=text))
frames = {r["frame_index"]: r["outputs"]
          for r in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=sid))}
t2 = time.time()

first = frames[min(frames)]
print("output keys:", {k: (type(v).__name__, getattr(v, "shape", None)) for k, v in first.items()})
ids_key = next(k for k in first if "obj_id" in k)
seen = sorted(f for f, o in frames.items() if len(o[ids_key]))
print(f"frames returned {len(frames)}, frames with '{text}': {len(seen)} ({seen[:1]}..{seen[-1:]})")
print(f"build {t1 - t0:.1f}s, track {t2 - t1:.1f}s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

# The API cache names replies "<file>.<size>.<prompt>.txt" and lists only frames with detections.
api = Path("clips/raw") / f"{Path(video).name}.{Path(video).stat().st_size}.{text}.txt"
if api.exists():
    api_seen = sorted({int(m) for m in re.findall(r"<(\d+)f>\d", api.read_text())})
    both = set(seen) & set(api_seen)
    print(f"API frames with '{text}': {len(api_seen)}; agree on {len(both)}; local only {len(set(seen) - set(api_seen))}; API only {len(set(api_seen) - set(seen))}")
