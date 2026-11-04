# Full tiling (tile_prep.py): every eight-frame window of every object is judged, so no candidate step can lose an
# event. A window becomes an alert at the pair of frames the judge named (OFF8 in frames8_prep.py; the centre when it
# named none), of the judge's kind (transform/none: both kinds). Several verdict files are averaged. Scored like
# eval_evidence.py, from the label lists alone (no SAM trace needed). With --vis, recall is also given over the
# events a visibility audit marked V, and V or V-lean; false alarms are always counted against every labelled event.
# Every recall comes with its chance level: the same alerts per object moved to random times (mean of SHUFFLES). This
# matters because a ±TOL window over a short clip catches a fair share of events by timing alone (dev: 12 of 41).
# --union: a window alerts through each verdict file that passes on its own (judges OR-ed), not through their mean.
# Usage: python research\eval_tile.py <manifest.json> <verdicts.json>... [--vis <visibility.json>]... [--union]
#            -- <list.json>...
import json, random, sys
from pathlib import Path
import cv2
from eval_evidence import score, moments, TOL
from eval_verify import BOTH

OFF8, SHUFFLES = (-15, -11, -7, -3, 3, 7, 11, 15), 200
SHIFT = {f"{a}-{b}": (OFF8[i] + OFF8[i + 1]) // 2 for i, (a, b) in enumerate(zip("ABCDEFG", "BCDEFGH"))}
assert SHIFT["A-B"] == -13 and SHIFT["D-E"] == 0 and SHIFT["G-H"] == 13
p = lambda v: v["confidence"] / 100 if v["violation"] else 1 - v["confidence"] / 100


def load(lists):
    """One record per (clip, object), in tile_prep.py's order (its clip index): the labelled events as (kind, frame)."""
    clips = []
    for path in lists:
        for clip in json.loads(Path(path).read_text()):
            cap = cv2.VideoCapture(f"clips/{clip['file']}")
            fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for concept in clip["concepts"]:
                clips.append(dict(file=clip["file"], concept=concept, n=n,
                                  truth=[(e["kind"], round(e["time"] * fps))
                                         for e in clip["events"] if e["concept"] == concept]))
    return clips


def recall(clips, pick, keep):
    """Caught and total over the labelled events keep() accepts."""
    hits = [any(k == kind and abs(f - fr) <= TOL for k, f in moments(pick(c)))
            for c in clips for kind, fr in c["truth"] if keep(c, kind, fr)]
    return sum(hits), len(hits)


def chance(clips, pick, keep):
    """When each object's alerts keep their kinds but land at random frames of its clip: mean caught, and the share
    of shuffles catching at least as many as the real timing (a one-sided permutation p-value)."""
    rng, alerts = random.Random(0), {id(c): pick(c) for c in clips}
    real = recall(clips, pick, keep)[0]
    s = [recall(clips, lambda c: [(k, rng.randrange(c["n"])) for k, _ in alerts[id(c)]], keep)[0]
         for _ in range(SHUFFLES)]
    return f"chance {sum(s) / SHUFFLES:.1f}, p {sum(x >= real for x in s) / SHUFFLES:.3f}"


if __name__ == "__main__":
    cut = sys.argv.index("--")
    args, clips = sys.argv[2:cut], load(sys.argv[cut + 1:])
    vis = {}
    while "--vis" in args:  # one per labelled set (v3, v4, ...)
        i = args.index("--vis")
        vis |= {(e["file"], e["concept"], e["kind"], e["frame"]): e["visibility"]
                for e in json.loads(Path(args[i + 1]).read_text("utf-8"))["events"]}
        args = args[:i] + args[i + 2:]
    union, anykind = "--union" in args, "--anykind" in args  # anykind: every alert of both kinds (judges mix them up)
    runs = [{v["id"]: v for v in json.loads(Path(f).read_text("utf-8"))} for f in args if not f.startswith("--")]

    # Averaged judges need every window judged by all of them; OR-ed ones take each window from whichever judged it.
    windows = [m for m in json.loads(Path(sys.argv[1]).read_text())
               if (any if union else all)(m["id"] in r for r in runs)]
    for m in windows:
        vs = [r[m["id"]] for r in runs if m["id"] in r]
        m["p"] = sum(map(p, vs)) / len(vs)
        m["alerts"] = [(p(v), k, m["frame"] + SHIFT.get(v["between"], 0)) for v in vs
                       for k in (None if anykind
                                 else {"vanish": ("vanish",), "appear": ("appear",)}.get(v["kind"])) or BOTH]
    by = {}
    for m in windows:
        by.setdefault(m["clip"], []).append(m)
    print(f"{len(windows)} windows judged, ${sum(r[m['id']]['cost'] for r in runs for m in windows if m['id'] in r):.3f}")

    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        pick = lambda c: [(k, f) for m in by.get(clips.index(c), []) if union or m["p"] >= t
                          for q, k, f in m["alerts"] if not union or q >= t]
        ca, tot, fa = score(clips, pick)
        line = f"p >= {t:.1f}: caught {ca}/{tot} ({ca / max(tot, 1):.0%}, {chance(clips, pick, lambda *_: True)})"
        for name, ok in (("V", {"V"}), ("V+lean", {"V", "V-lean"}), ("A", {"A"}), ("N", {"N"})) if vis else ():
            keep = lambda c, k, f: vis.get((c["file"], c["concept"], k, f)) in ok
            cv, tv = recall(clips, pick, keep)
            line += f", {name} {cv}/{tv} ({chance(clips, pick, keep)})"
        vids = {}  # video level: any alert on any of its objects
        for c in clips:
            had, hit = vids.get(c["file"], (False, False))
            vids[c["file"]] = (had or bool(c["truth"]), hit or bool(pick(c)))
        bad, clean = [h for e, h in vids.values() if e], [h for e, h in vids.values() if not e]
        print(f"{line}; false alarms {fa}; videos flagged: with events {sum(bad)}/{len(bad)}, clean {sum(clean)}/{len(clean)}")
