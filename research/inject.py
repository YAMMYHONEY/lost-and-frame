# Synthetic permanence glitches with exact truth: a copy of one real object (SAM 3.1's surest mask on frame a, prompted
# with the clip's concept) is pasted beside it from frame a to frame b, so a second one appears at a and vanishes at b.
# Every other clip fades in and out over RAMP frames instead of switching, closer to the melting glitches of real
# generators (truth is the middle of the fade). Only no-issue clips are used, so the injected events are the only ones.
# Usage: python research\inject.py <out name> <clean list.json>...
#        -> clips/syn/<out name>-###.mp4, clips/<out name>.json
import json, sys, tempfile
from pathlib import Path
import cv2, numpy as np, torch
from sam3.model_builder import build_sam3_predictor
from paths import WORK

RAMP, MIN_AREA, MAX_COPY = 8, 0.005, 0.08  # mask area as a share of the frame; bigger objects get a shrunken copy


def read(path):
    cap, frames = cv2.VideoCapture(path), []
    while (r := cap.read())[0]:
        frames.append(r[1])
    return cap.get(cv2.CAP_PROP_FPS), frames


def surest_mask(predictor, frame, text):
    (WORK / "scratch").mkdir(parents=True, exist_ok=True)  # short path: Windows MAX_PATH, see paths.py
    with tempfile.TemporaryDirectory(dir=WORK / "scratch") as d:
        cv2.imwrite(f"{d}/00000.jpg", frame)
        sid = predictor.handle_request(dict(type="start_session", resource_path=d))["session_id"]
        o = predictor.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=text))["outputs"]
        predictor.handle_request(dict(type="close_session", session_id=sid))
    ok = [(p, m) for p, m in zip(o["out_probs"], o["out_binary_masks"]) if m.mean() >= MIN_AREA]
    return max(ok, key=lambda pm: pm[0])[1] if ok else None


def place(m, w, h):
    """Top-left corner for a w x h copy that covers the least of the object's mask m, nearest the object on ties;
    also the share of the copy that lies on the object."""
    H, W = m.shape
    ii = cv2.integral(m.astype(np.uint8))
    ys, xs = np.mgrid[0:H - h + 1:8, 0:W - w + 1:8]
    cover = ii[ys + h, xs + w] - ii[ys, xs + w] - ii[ys + h, xs] + ii[ys, xs]
    oy, ox = np.argwhere(m).mean(0)
    near = np.hypot(xs + w / 2 - ox, ys + h / 2 - oy) / np.hypot(W, H)  # 0..1, as much as 1% of the copy's area
    i = np.argmin(cover + 0.01 * w * h * near)
    return int(xs.flat[i]), int(ys.flat[i]), cover.flat[i] / (w * h)


def alpha(t, a, b, fade):
    if not fade:
        return float(a <= t < b)
    return float(np.clip(min((t - a) / RAMP + 0.5, (b - t) / RAMP + 0.5), 0, 1))  # 0.5 exactly at a and at b


assert alpha(10, 10, 50, False) == 1 and alpha(50, 10, 50, False) == 0
assert alpha(10, 10, 50, True) == 0.5 and alpha(14, 10, 50, True) == 1 and alpha(54, 10, 50, True) == 0
_m = np.zeros((100, 200), bool)
_m[:, :20] = True  # an object along the left edge: the copy goes right beside it, level with its middle, on nothing
assert place(_m, 20, 20) == (24, 40, 0.0)

if __name__ == "__main__":
    name, lists = sys.argv[1], sys.argv[2:]
    predictor = build_sam3_predictor(version="sam3.1", use_fa3=False, compile=False, max_num_objects=8)
    predictor.model.batched_grounding_batch_size = 1  # 8 GB GPU, see trace.py
    Path("clips/syn").mkdir(exist_ok=True)

    out, skipped = [], []
    clean = [c for path in lists for c in json.loads(Path(path).read_text()) if not c["events"]]
    for c in clean:
        fps, frames = read(f"clips/{c['file']}")
        n, concept = len(frames), c["concepts"][0]
        a, b = round(0.3 * n), round(0.7 * n)
        m = None  # the copy is cut from the first of these frames where SAM finds the object
        for src in (a, round(0.5 * n), b) if b - a >= 32 else ():  # events further apart than 2 x TOL
            if (m := surest_mask(predictor, frames[src], concept)) is not None:
                break
        ys, xs = np.nonzero(m) if m is not None else ((), ())
        box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else None
        if box:
            x0, y0, x1, y1 = box
            s = min(1.0, max(0.4, (MAX_COPY / m.mean()) ** 0.5))  # shrink big objects, never below 40% (recognisable)
            w, h = round((x1 - x0) * s), round((y1 - y0) * s)
            X, Y, on = place(m, w, h)
        if not box or on > 0.5:  # nothing found, or the copy would mostly cover the object itself
            skipped.append(c["file"])
            continue

        fade = len(out) % 2 == 1
        patch = cv2.resize(frames[src][y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        soft = cv2.GaussianBlur(cv2.resize(m[y0:y1, x0:x1].astype(np.float32), (w, h), interpolation=cv2.INTER_AREA),
                                (5, 5), 0)[..., None]
        file = f"syn/{name}-{len(out):03d}.mp4"
        vw = cv2.VideoWriter(f"clips/{file}", cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (frames[0].shape[1], frames[0].shape[0]))
        for t, f in enumerate(frames):
            k = alpha(t, a, b, fade) * soft
            f[Y:Y + h, X:X + w] = (f[Y:Y + h, X:X + w] * (1 - k) + patch * k).astype(np.uint8)
            vw.write(f)
        vw.release()
        out.append(dict(file=file, base=c["file"], model=c.get("model"), concepts=[concept], fade=fade,
                        box=[X, Y, X + w, Y + h],
                        events=[dict(concept=concept, kind="appear", time=a / fps),
                                dict(concept=concept, kind="vanish", time=b / fps)]))
        print(file, concept, "fade" if fade else "cut", flush=True)
    Path(f"clips/{name}.json").write_text(json.dumps(out, indent=1, default=int))
    print(len(out), "clips,", 2 * len(out), "events; skipped", len(skipped), skipped)
