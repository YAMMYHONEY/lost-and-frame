# Record SAM 3.1's two internal witnesses on every frame, without changing the model:
#   detector (no memory): the top candidates' score and box for the prompt on this frame (before NMS), and its
#     presence score (whether the concept is in the frame at all);
#   tracker (memory): each tracked object's frame-level "is it visible" score (sigmoid).
# Wraps the model instance's per-frame methods, runs a clip, and saves JSON to $VC_WORK/traces (default work/traces,
# see paths.py).
# Usage: python research\trace.py <video> "<prompt>"
import json, os, sys, warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
warnings.simplefilter("ignore")
import torch
from sam3.model_builder import build_sam3_predictor
from paths import trace_path

# Objects tracked at once (builder default 16). Crowded clips (a pan of mushrooms) at 16 overflow the 8 GB GPU
# into system RAM and crawl (27 min without finishing a 116-frame clip), so such clips are re-run with fewer.
MAX_OBJECTS = int(os.environ.get("SAM_MAX_OBJECTS", 16))
# TRACE_MAX_NEW > 0: exit with code 3 after that many new traces, so the caller restarts a fresh process. One long
# --list run grew to 14 GB private memory over ~25 clips and paged (47 s per frame). A restart skips every trace that
# already exists.
MAX_NEW, _new = int(os.environ.get("TRACE_MAX_NEW", 0)), [0]


def tolist(t):
    return t.detach().float().cpu().numpy().round(4).tolist()


def recorder(model):
    """Install the per-frame recorders once per model; returns the dict they fill (frame index -> signals)."""
    if hasattr(model, "trace_frames"):
        return model.trace_frames
    frames, last = {}, {}
    det_fn, step_fn = model.run_backbone_and_detection, model._det_track_one_frame_impl

    def grounding_wrapper(fn):  # the prompt frame uses the plain path, propagation the batched one
        def wrapped(*a, **k):
            out = fn(*a, **k)
            last["out"] = out[0]
            return out
        return wrapped

    def det_wrapper(*a, **k):
        last.clear()
        det_out, keep = det_fn(*a, **k)
        rec, out = frames.setdefault(k["frame_idx"], {}), last["out"]
        # SAM 3.1 scores are logit(object x presence) clamped to [-10, 10], and NMS then subtracts 1e4 from every
        # candidate under the 0.4 threshold or suppressed, which makes det_out["scores"] 0 there: undo that.
        logits = out["pred_logits"][0, :, 0].float()
        scores, boxes = torch.where(logits < -1000, logits + 1e4, logits).sigmoid(), det_out["bbox"][0].float()
        top = scores.argsort(descending=True)[:20]
        rec["det"] = [[round(float(scores[i]), 4), *tolist(boxes[i]), int(keep[0][i])] for i in top]
        if "presence_logit_dec" in out:  # "is the concept in this frame"; the prompt frame's output lacks it
            rec["presence"] = tolist(out["presence_logit_dec"].float().sigmoid())
        return det_out, keep

    def step_wrapper(*a, **k):
        ids = list(k["tracker_metadata_prev"].get("obj_ids_all_gpu", []))
        out = step_fn(*a, **k)
        scores = out[5]  # tracker frame-level scores (sigmoid), one per previously tracked object
        frames.setdefault(k["frame_idx"], {})["trk"] = {int(i): round(float(s), 4) for i, s in zip(ids, scores)}
        return out

    model.run_backbone_and_detection, model._det_track_one_frame_impl = det_wrapper, step_wrapper
    for name in ("forward_video_grounding_multigpu", "forward_video_grounding_batched_multigpu"):
        setattr(model.detector, name, grounding_wrapper(getattr(model.detector, name)))
    model.trace_frames = frames
    return frames


def trace(predictor, video, text, box=None, frame=0):
    """box: optional visual exemplar (normalized xyxy) on `frame`, given with the text; tracking runs both ways."""
    frames = recorder(predictor.model)
    frames.clear()
    sid = predictor.handle_request(dict(type="start_session", resource_path=video,
                                        offload_video_to_cpu=True))["session_id"]
    exemplar = dict(bounding_boxes=[[box[0], box[1], box[2] - box[0], box[3] - box[1]]],
                    bounding_box_labels=[1]) if box else {}
    predictor.handle_request(dict(type="add_prompt", session_id=sid, frame_index=frame, text=text, **exemplar))
    for r in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
        o = r["outputs"]  # final output per object: normalized xywh box and mask area in pixels
        frames.setdefault(r["frame_index"], {})["out"] = {
            int(i): [*tolist(torch.as_tensor(b)), int(m.sum())]
            for i, b, m in zip(o["out_obj_ids"], o["out_boxes_xywh"], o["out_binary_masks"])}
    predictor.handle_request(dict(type="close_session", session_id=sid))
    return dict(frames)


def save(predictor, video, text, box=None, frame=0):
    out = trace_path(video, f"{text}+box" if box else text)
    if out.exists():
        return out
    frames = trace(predictor, video, text, box, frame)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"video": video, "prompt": text, "box": box, "box_frame": frame,
                               "max_objects": MAX_OBJECTS, "frames": frames}))
    _new[0] += 1
    if MAX_NEW and _new[0] >= MAX_NEW:
        print("saved", out, flush=True)
        sys.exit(3)
    return out


# Usage: trace.py <video> "<prompt>"   or   trace.py --list <eval list.json> <its -eval.json>
# The list form traces every clip with the name SAM actually tracked in that evaluation (samName).
if __name__ == "__main__":
    predictor = build_sam3_predictor(version="sam3.1", use_fa3=False, compile=False, max_num_objects=MAX_OBJECTS)
    predictor.model.batched_grounding_batch_size = 1  # 8 GB GPU, see sam_local.py
    if sys.argv[1] == "--list":
        clips = json.loads(Path(sys.argv[2]).read_text())
        report = json.loads(Path(sys.argv[3]).read_text())["report"]
        for c, r in zip(clips, report):
            for concept in c["concepts"]:
                try:  # one failed clip should not end a two-hour batch
                    print("saved", save(predictor, f"clips/{c['file']}", r["samName"][concept]), flush=True)
                except Exception as e:
                    print("FAILED", c["file"], concept, type(e).__name__, str(e)[:200], flush=True)
    elif sys.argv[1] == "--names":  # trace.py --names research/names-chosen.json: every chosen name
        for key, c in json.loads(Path(sys.argv[2]).read_text()).items():
            print("saved", save(predictor, f"clips/{key.split('|')[0]}", c["name"]), flush=True)
    elif sys.argv[1] == "--boxes":  # trace.py --boxes <localize.mjs output>: the surest box per object as exemplar
        for o in json.loads(Path(sys.argv[2]).read_text("utf-8")):  # written by Node
            if o["boxes"]:
                b = max(o["boxes"], key=lambda b: b["confidence"])
                print("saved", save(predictor, f"clips/{o['file']}", o["name"], b["box"], b["frame"]), flush=True)
    else:
        print("saved", save(predictor, sys.argv[1], sys.argv[2]))
