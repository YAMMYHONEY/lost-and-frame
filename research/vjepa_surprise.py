# World-model surprise: V-JEPA 2 (Meta, open weights, MIT) predicts the next half-second of a clip in its feature
# space from the half-second before. A thing that vanishes or appears without cause should make that prediction fail,
# just as infants look longer at violated expectations. Context features come from encoding the past frames alone (the
# whole-window encoder would let them see the future); targets come from encoding the whole window.
# Usage: python research\vjepa_surprise.py <oracle manifest.json>   (judge-only test: real events vs clean moments)
import json, os, sys, warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
warnings.simplefilter("ignore")
import cv2
import numpy as np
import torch
from transformers import VJEPA2Model

MODEL, SIDE, HALF, STEP = "facebook/vjepa2-vitl-fpc64-256", 256, 15, 2  # 16 frames: t-15..t+15, every 2nd
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
TOP = 0.02  # share of tokens with the largest error: a small object's vanish is a few tokens of 2048
_m = {}


def model():
    if not _m:
        _m["m"] = VJEPA2Model.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    return _m["m"]


def frames(video):
    """Every frame of the clip, whole frame squeezed to SIDE x SIDE (a centre crop would cut off corner objects)."""
    cap, out = cv2.VideoCapture(video), []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, (SIDE, SIDE), interpolation=cv2.INTER_AREA)[..., ::-1] / 255.0
        out.append(((img - MEAN) / STD).transpose(2, 0, 1))
    return torch.tensor(np.array(out), dtype=torch.float16)


@torch.no_grad()
def surprise(clip, t):
    """(mean, top-share) L1 error of layer-normed features for the 8 frames after t, predicted from the 8 before."""
    lo = min(max(0, t - HALF), max(0, len(clip) - 2 * HALF - 1))
    idx = [min(len(clip) - 1, i) for i in range(lo, lo + 2 * HALF + 1, STEP)][:16]
    x = clip[idx].unsqueeze(0).cuda()
    m = model()
    full = m.encoder(pixel_values_videos=x).last_hidden_state
    past = m.encoder(pixel_values_videos=x[:, :8]).last_hidden_state
    nc, n = past.shape[1], full.shape[1]
    ctx = torch.cat([past, torch.zeros_like(full[:, nc:])], 1)
    pred = m.predictor(encoder_hidden_states=ctx, context_mask=[torch.arange(nc, device="cuda")[None]],
                       target_mask=[torch.arange(nc, n, device="cuda")[None]]).last_hidden_state
    ln = lambda z: torch.nn.functional.layer_norm(z.float(), z.shape[-1:])
    err = (ln(pred) - ln(full[:, nc:])).abs().mean(-1)[0]
    k = max(1, int(TOP * len(err)))
    return float(err.mean()), float(err.topk(k).values.mean())


def zscore(value, refs):
    refs = np.array(refs)
    return (value - np.median(refs)) / (np.median(np.abs(refs - np.median(refs))) + 1e-6)


if __name__ == "__main__":
    man = json.loads(Path(sys.argv[1]).read_text())
    rows, cache = [], {}
    for m in man:
        if m["file"] not in cache:
            clip = frames(f"clips/{m['file']}")
            refs = [surprise(clip, t) for t in range(HALF, len(clip) - HALF, 8)]  # the same clip elsewhere
            cache[m["file"]] = (clip, refs)
        clip, refs = cache[m["file"]]
        s = surprise(clip, m["frame"])
        rows.append(dict(id=m["id"], positive=m["positive"], mean=s[0], top=s[1],
                         z_mean=float(zscore(s[0], [r[0] for r in refs])),
                         z_top=float(zscore(s[1], [r[1] for r in refs]))))
        print(m["id"], m["positive"], round(s[0], 4), round(s[1], 4), flush=True)

    Path(sys.argv[1].replace(".json", "-vjepa.json")).write_text(json.dumps(rows, indent=1))
    pos = [r for r in rows if r["positive"]]
    neg = [r for r in rows if not r["positive"]]
    for key in ("mean", "top", "z_mean", "z_top"):
        a, b = np.array([r[key] for r in pos])[:, None], np.array([r[key] for r in neg])[None, :]
        print(f"{key:7} AUC {float(((a > b) + 0.5 * (a == b)).mean()):.3f}")
