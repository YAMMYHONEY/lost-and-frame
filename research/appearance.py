# In-place changes: objects often vanish or appear where SAM keeps a track (a slice leaves a pizza, a lid leaves a
# drain, a banana appears inside its peel). The track never breaks, so break-based methods cannot see them.
# Per SAM track, describe the tracked box's pixels on every frame and find steps: how far the mean description of
# the frames before t is from that of the frames after t, compared with the track's own usual steps.
# Descriptions: "lab" = 16x16 Lab thumbnail (color and layout), "deep" = ImageNet ResNet-18 features (what it is).
from pathlib import Path
import cv2
import numpy as np
from paths import WORK

W, MIN_W, SIDE, CROP = 8, 3, 16, 112  # window, shortest window near the clip's ends, Lab side, deep crop side
CACHE = WORK / "thumbs"  # short path: Windows MAX_PATH, see paths.py
_net = {}


def deep(crops):
    """512-d L2-normalized ResNet-18 features of BGR uint8 crops (CPU: the GPU is left to SAM)."""
    import torch, torchvision
    if not _net:
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()
        _net["m"] = m.eval()
    x = torch.from_numpy(np.stack(crops)[..., ::-1].copy()).permute(0, 3, 1, 2).float() / 255
    x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    with torch.no_grad():
        f = torch.cat([_net["m"](x[i:i + 64]) for i in range(0, len(x), 64)])
    return torch.nn.functional.normalize(f, dim=1).numpy()


def thumbs(video, frames, cache_name, kind="lab"):
    """{track id: (frame indices, descriptions [T, D])} from the trace's tracked boxes."""
    cache = CACHE / f"{cache_name}.{kind}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k[2:]: (z["f_" + k[2:]], z[k]) for k in z.files if k.startswith("t_")}

    cap, seq, f = cv2.VideoCapture(str(video)), {}, 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        h, w = img.shape[:2]
        for i, (x, y, bw, bh, _) in frames.get(str(f), {}).get("out", {}).items():
            x1, y1 = max(0, int(x * w)), max(0, int(y * h))
            x2, y2 = min(w, max(x1 + 2, int((x + bw) * w))), min(h, max(y1 + 2, int((y + bh) * h)))
            crop = img[y1:y2, x1:x2]
            if kind == "lab":
                crop = cv2.cvtColor(cv2.resize(crop, (SIDE, SIDE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB)
                crop = crop.reshape(-1).astype(np.float32) / 255
            else:
                crop = cv2.resize(crop, (CROP, CROP), interpolation=cv2.INTER_AREA)
            seq.setdefault(i, ([], []))[0].append(f)
            seq[i][1].append(crop)
        f += 1

    out = {i: (np.array(fs), np.array(ts) if kind == "lab" else deep(ts)) for i, (fs, ts) in seq.items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **{f"f_{i}": v[0] for i, v in out.items()}, **{f"t_{i}": v[1] for i, v in out.items()})
    return out


def steps(fs, ts):
    """[(frame, z)]: robust z of the before/after distance, windows shrinking to MIN_W near the clip's ends."""
    if len(fs) < 2 * MIN_W + 1:
        return []
    c = np.cumsum(np.vstack([np.zeros(ts.shape[1]), ts]), axis=0)
    t_all = range(MIN_W, len(fs) - MIN_W + 1)
    w = [min(W, t, len(fs) - t) for t in t_all]
    d = np.array([np.linalg.norm((c[t] - c[t - k]) / k - (c[t + k] - c[t]) / k) for t, k in zip(t_all, w)])
    med, mad = np.median(d), np.median(np.abs(d - np.median(d))) + 1e-6
    return [(int(fs[t]), float((x - med) / mad)) for t, x in zip(t_all, d)]


def changes(video, frames, cache_name, min_z, kind="lab"):
    """Peaks of the step score per track, strongest first: [(frame, z, xyxy box)]. Step sizes of different clips are
    not comparable (3.8 to 19 at real events), so callers rank within a clip."""
    out = []
    for i, (fs, ts) in thumbs(video, frames, cache_name, kind).items():
        s = steps(fs, ts)
        for j, (f, z) in enumerate(s):
            if z >= min_z and all(z >= s[k][1] for k in range(max(0, j - W), min(len(s), j + W + 1))):
                x, y, w, h, _ = frames[str(f)]["out"][i]
                out.append((f, z, [x, y, x + w, y + h]))
    return sorted(out, key=lambda x: -x[1])


if __name__ == "__main__":
    # Self-check: a track whose look changes once at frame 30 peaks there, and only there. The same holds near the
    # clip's start.
    rng = np.random.default_rng(0)
    ts = np.vstack([rng.normal(0.3, 0.01, (30, 12)), rng.normal(0.7, 0.01, (30, 12))]).astype(np.float32)
    best = max(steps(np.arange(60), ts), key=lambda x: x[1])
    assert best[0] == 30 and best[1] > 10, best
    ts = np.vstack([rng.normal(0.3, 0.01, (4, 12)), rng.normal(0.7, 0.01, (56, 12))]).astype(np.float32)
    assert max(steps(np.arange(60), ts), key=lambda x: x[1])[0] == 4
    print("ok")
