# GPU memory and time at each stage of a SAM 3.1 video run, stopping after a few frames.
# On Windows the driver spills past dedicated VRAM into shared system memory instead of raising
# out-of-memory, which makes every step crawl; allocated > total VRAM is the sign of that.
# Usage: python research\mem_probe.py <video> "<prompt>" [frames=3] [weights=fp32|bf16] [max_objects=16] [det_batch=16]
# det_batch: frames the detector processes at once during tracking. The builder hard-codes 16
# (sized for an 80 GB H100); on an 8 GB GPU that alone needs > 12 GiB.
import os, sys, time, warnings

os.environ.setdefault("HF_HUB_OFFLINE", "1")
warnings.simplefilter("ignore")
import torch
from sam3.model_builder import build_sam3_predictor

video, text = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
weights = sys.argv[4] if len(sys.argv) > 4 else "fp32"
max_obj = int(sys.argv[5]) if len(sys.argv) > 5 else 16
det_batch = int(sys.argv[6]) if len(sys.argv) > 6 else 16
total = torch.cuda.get_device_properties(0).total_memory


def mark(stage, t):
    torch.cuda.synchronize()
    a, r, p = torch.cuda.memory_allocated(), torch.cuda.memory_reserved(), torch.cuda.max_memory_allocated()
    flag = "  <-- past dedicated VRAM" if r > total else ""
    print(f"{stage:<22} {time.time() - t:6.1f}s  allocated {a / 2**30:5.2f}  reserved {r / 2**30:5.2f}  peak {p / 2**30:5.2f} GiB{flag}", flush=True)


t = time.time()
predictor = build_sam3_predictor(version="sam3.1", use_fa3=False, compile=False, max_num_objects=max_obj)
if weights == "bf16":
    predictor.model.to(torch.bfloat16)
predictor.model.batched_grounding_batch_size = det_batch
print(f"VRAM {total / 2**30:.2f} GiB, weights {weights}, max_num_objects {max_obj}, det_batch {det_batch}, "
      f"params {sum(p.numel() for p in predictor.model.parameters()) / 1e6:.0f}M")
mark("model built", t)

t = time.time()
sid = predictor.handle_request(dict(type="start_session", resource_path=video, offload_video_to_cpu=True))["session_id"]
mark("session started", t)

t = time.time()
out = predictor.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=text))["outputs"]
mark("prompt on frame 0", t)

t = time.time()
for i, r in enumerate(predictor.handle_stream_request(dict(type="propagate_in_video", session_id=sid))):
    mark(f"frame {r['frame_index']} tracked", t)
    t = time.time()
    if i + 1 >= n:
        break
