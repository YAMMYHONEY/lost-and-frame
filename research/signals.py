# Print SAM 3.1's per-frame signals around a frame range: tracked objects with their score and box,
# and whatever frame_stats carries, to see what the tracker and detector report when an object goes away.
# Usage: python research\signals.py <video> "<prompt>" <from_frame> <to_frame>
import os, sys, warnings

os.environ.setdefault("HF_HUB_OFFLINE", "1")
warnings.simplefilter("ignore")
import numpy as np
from sam3.model_builder import build_sam3_predictor

video, text, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
predictor = build_sam3_predictor(version="sam3.1", use_fa3=False, compile=False)
predictor.model.batched_grounding_batch_size = 1  # 8 GB GPU, see sam_local.py
sid = predictor.handle_request(dict(type="start_session", resource_path=video, offload_video_to_cpu=True))["session_id"]
predictor.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=text))
for r in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
    f, o = r["frame_index"], r["outputs"]
    if f == lo:
        print("frame_stats keys:", {k: type(v).__name__ for k, v in (o.get("frame_stats") or {}).items()})
    if lo <= f <= hi:
        h, w = o["out_binary_masks"].shape[1:]
        boxes = o["out_boxes_xywh"] * np.array([w, h, w, h])  # normalized xywh -> pixels
        objs = [f"id{i} p={p:.3f} box={np.round(b).astype(int).tolist()} px={int(m.sum())}"
                for i, p, b, m in zip(o["out_obj_ids"], o["out_probs"], boxes, o["out_binary_masks"])]
        stats = {k: (np.round(v, 3).tolist() if isinstance(v, (np.ndarray, float)) else v)
                 for k, v in (o.get("frame_stats") or {}).items()}
        print(f"f{f}: {objs or 'none'} | {stats}")
