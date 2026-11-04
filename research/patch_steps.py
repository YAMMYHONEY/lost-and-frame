# Name-free, recognition-free change detector: people notice a small thing vanish because the spot changes suddenly
# and stays changed. The frames around t are registered to frame t (ORB + RANSAC homography, so camera motion
# cancels) and cut into a grid of patches. Each patch's mean colour over the W frames before t is compared with the
# W frames after t, against the patch's own frame-to-frame jitter. A hand passing through returns the patch to where
# it was (no step); a vanished or appeared object leaves a step. Score = the largest patch step.
import cv2
import numpy as np
from change_zoom import register

W, SIZE, GRID = 12, (320, 180), 16  # frames on each side, working resolution, patch side in pixels


def window(video, t, n):
    cap = cv2.VideoCapture(video)
    lo, hi = max(0, t - W), min(n - 1, t + W)
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    frames = []
    for _ in range(lo, hi + 1):
        ok, img = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA))
    return frames, t - lo


def step_map(frames, c):
    """Per-patch step score at frame index c of the window (registered to frame c)."""
    ref = frames[c]
    lab = [cv2.cvtColor(register(ref, f) if i != c else f, cv2.COLOR_BGR2LAB).astype(np.float32)
           for i, f in enumerate(frames)]
    h, w = SIZE[1] // GRID, SIZE[0] // GRID
    pm = np.array([f[:h * GRID, :w * GRID].reshape(h, GRID, w, GRID, 3).mean(axis=(1, 3))
                   for f in lab])  # T x h x w x 3
    before, after = pm[:c], pm[c + 1:]
    if len(before) < 3 or len(after) < 3:
        return np.zeros((h, w))
    jitter = np.median(np.abs(np.diff(pm, axis=0)), axis=0).sum(-1) + 1.0  # typical frame-to-frame change per patch
    spread = (before.std(0) + after.std(0)).sum(-1) + 1.0                   # unsteady sides are not a clean step
    return np.linalg.norm(before.mean(0) - after.mean(0), axis=-1) / np.sqrt(jitter * spread)


def score(video, t, n):
    frames, c = window(video, t, n)
    m = step_map(frames, c)
    return float(m.max()), np.unravel_index(int(m.argmax()), m.shape)


if __name__ == "__main__":
    # Self-check: a pan with a small square that vanishes at t gives a high step where the square was. With no
    # vanish, the step stays low.
    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur((rng.random((420, 760, 3)) * 255).astype(np.uint8), (0, 0), 3)

    def shots(vanish):
        out = []
        for i in range(25):
            f = cv2.resize(base[20:380, 20 + i:660 + i].copy(), SIZE)
            if not (vanish and i > 12):
                cv2.rectangle(f, (200, 100), (210, 110), (0, 0, 255), -1)
            out.append(f)
        return out

    with_vanish, without = step_map(shots(True), 12).max(), step_map(shots(False), 12).max()
    assert with_vanish > 3 * without, (with_vanish, without)
    print("ok", round(float(with_vanish), 2), round(float(without), 2))
