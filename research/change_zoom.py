# Change-guided zoom: point the verifier at what actually changed, without trusting any candidate box. The four
# moments are registered to the first one (ORB features + RANSAC homography), so camera motion cancels. Then the
# region with the most residual change between consecutive moments is cropped from all four.
import cv2
import numpy as np

SIDE = 3  # crop is a third of the frame in each direction


def register(ref, img):
    """img warped onto ref's view; img unchanged if too few features match (flat or blurred frames)."""
    orb = cv2.ORB_create(1500)
    g1, g2 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return img
    matches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d2, d1), key=lambda m: m.distance)[:300]
    if len(matches) < 20:
        return img
    h, _ = cv2.findHomography(np.float32([k2[m.queryIdx].pt for m in matches]),
                              np.float32([k1[m.trainIdx].pt for m in matches]), cv2.RANSAC, 4.0)
    return img if h is None else cv2.warpPerspective(img, h, (ref.shape[1], ref.shape[0]),
                                                     borderMode=cv2.BORDER_REPLICATE)


FLOOR, BASE = 0.05, 0.1  # detector candidates counted from this score; weight left where the detector sees nothing


def concept_weight(dets_per_frame, shape):
    """Where the detector's candidates for the object's name sit on these frames (scores recovered below SAM's 0.4
    threshold, trace.py), as a [0, 1] map: hands and arms move the most in egocentric clips, the object does not."""
    h, w = shape
    m = np.zeros((h, w), np.float32)
    for dets in dets_per_frame:
        for s, x1, y1, x2, y2, *_ in dets:
            if s >= FLOOR:
                a, b = int(max(0, x1) * w), int(max(0, y1) * h)
                c, d = int(min(1, x2) * w), int(min(1, y2) * h)
                m[b:d, a:c] = np.maximum(m[b:d, a:c], s)
    return m / m.max() if m.max() > 0 else m


def change_box(frames, dets_per_frame=None):
    """(x1, y1, x2, y2) in pixels of the SIDE-th-of-the-frame window with the most residual change, weighted towards
    the named object when its detector candidates are given."""
    al = [frames[0]] + [register(frames[0], f) for f in frames[1:]]
    gray = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (9, 9), 0).astype(np.float32) for f in al]
    diff = sum(np.abs(a - b) for a, b in zip(gray, gray[1:]))
    if dets_per_frame is not None:
        diff *= BASE + concept_weight(dets_per_frame, diff.shape)
    h, w = diff.shape
    bh, bw = h // SIDE, w // SIDE
    pad = 8  # registration leaves seams at the borders: ignore them
    diff[:pad], diff[-pad:], diff[:, :pad], diff[:, -pad:] = 0, 0, 0, 0
    score = cv2.boxFilter(diff, -1, (bw, bh), normalize=False)
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    x1, y1 = min(max(0, cx - bw // 2), w - bw), min(max(0, cy - bh // 2), h - bh)
    return x1, y1, x1 + bw, y1 + bh


if __name__ == "__main__":
    # Self-check: a camera pan plus one square that vanishes. The crop must contain the square, not the panned edges.
    rng = np.random.default_rng(0)
    base = (rng.random((420, 760, 3)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 3)
    shots = []
    for i, dx in enumerate((0, 6, 12, 18)):
        f = base[10:370, 10 + dx:650 + dx].copy()
        if i < 2:
            cv2.rectangle(f, (450, 250), (500, 300), (0, 0, 255), -1)
        shots.append(f)
    x1, y1, x2, y2 = change_box(shots)
    assert x1 <= 450 - 18 and x2 >= 500 - 18 and y1 <= 250 and y2 >= 300, (x1, y1, x2, y2)
    print("ok", (x1, y1, x2, y2))
