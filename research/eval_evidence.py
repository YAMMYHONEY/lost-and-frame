# Recall vs false alarms on the development lists: SAM track breaks (A), evidence-track events (E), in-place changes
# (lab / deep) and their unions. Alerts of one kind within MERGE frames count as one moment. Caught = a same-kind alert
# within ±TOL of a labeled event; false alarm = a moment with no same-kind event within ±TOL.
# Usage: python research\eval_evidence.py [--chosen] [--frozen] clips\physion-labels.json clips\physion-tune.json ...
#   --chosen: each object's repaired name (renamed, or a confirmed Spark exemplar); --frozen: pre-registered settings
import itertools, json, sys
from pathlib import Path
from paths import trace_path
from evidence import events, EDGE
from appearance import changes

GAP, TOL, MERGE = 2, 15, 5
FROZEN = {}  # filled from the development sweep before the sealed test, see RESEARCH.md


def chosen_names():  # exemplar_names.py output when it exists, else choose_names.py output
    p = Path("research/names-final.json")
    return json.loads((p if p.exists() else Path("research/names-chosen.json")).read_text())


def breaks(frames, n):  # method A: every break in SAM's own tracks, (kind, frame, xyxy box)
    tracks = {}
    for f in range(n):
        for i, (x, y, w, h, _) in frames.get(str(f), {}).get("out", {}).items():
            tracks.setdefault(i, []).append((f, [x, y, x + w, y + h]))
    out = []
    for pts in tracks.values():
        cuts = [0] + [j for j in range(1, len(pts)) if pts[j][0] - pts[j - 1][0] > GAP + 1] + [len(pts)]
        for a, b in zip(cuts, cuts[1:]):
            if pts[a][0] > 0:
                out.append(("appear", pts[a][0], pts[a][1]))
            if pts[b - 1][0] < n - 1:
                out.append(("vanish", pts[b - 1][0], pts[b - 1][1]))
    return out


def moments(alerts):
    out = []
    for kind, f in sorted(alerts):
        if not any(k == kind and abs(g - f) <= MERGE for k, g in out):
            out.append((kind, f))
    return out


def score(clips, pick):
    caught = alarms = total = 0
    for c in clips:
        m = moments(pick(c))
        total += len(c["truth"])
        caught += sum(any(k == kind and abs(f - fr) <= TOL for k, f in m) for kind, fr in c["truth"])
        alarms += sum(not any(k == kind and abs(f - fr) <= TOL for kind, fr in c["truth"]) for k, f in m)
    return caught, total, alarms


def load(lists, names=None):
    """One record per (clip, object): its labels and every candidate source, with boxes."""
    clips = []
    for path in lists:
        report = json.loads(Path(path.replace(".json", "-eval.json")).read_text())["report"]
        for clip in json.loads(Path(path).read_text()):
            r = next(r for r in report if r["source"].endswith(Path(clip["file"]).name))
            for concept in clip["concepts"]:
                name = (names or {}).get(f"{clip['file']}|{concept}", {}).get("name", r["samName"][concept])
                frames = json.loads(trace_path(clip["file"], name).read_text())["frames"]
                n, video, stem = len(frames), f"clips/{clip['file']}", trace_path(clip["file"], name).stem
                A_all = breaks(frames, n)
                clips.append(dict(
                    file=clip["file"], concept=concept, name=name, sam_name=r["samName"][concept], n=n, fps=r["fps"],
                    truth=[(x["kind"], round(x["time"] * r["fps"])) for x in clip["events"] if x["concept"] == concept],
                    A_all=A_all, A=[a for a in A_all if min(a[2][0], a[2][1], 1 - a[2][2], 1 - a[2][3]) >= EDGE],
                    E=events(frames, n), **{d: changes(video, frames, stem, 3, d) for d in ("lab", "deep")}))
    return clips


# The clip's k strongest in-place changes. Whether a thing left or arrived is unknown, so both kinds are alerted
# (counted twice).
top = lambda c, d, k, z: [(kind, f) for f, s, _ in c[d][:k] if s >= z for kind in ("appear", "vanish")]


def combined(c, s_min, c_min, l_min, d, k, z):
    ok = lambda e: not e["edge"] and e["strength"] >= s_min and e["contrast"] >= c_min and e["length"] >= l_min
    return [(a, f) for a, f, _ in c["A"]] + top(c, d, k, z) + [(e["kind"], e["frame"]) for e in c["E"] if ok(e)]


if __name__ == "__main__":
    names = chosen_names() if "--chosen" in sys.argv else {}
    clips = load([a for a in sys.argv[1:] if not a.startswith("--")], names)
    for label, pick in (("A (every SAM break)", "A_all"), ("A without frame edge", "A")):
        ca, tot, fa = score(clips, lambda c: [(a, f) for a, f, _ in c[pick]])
        print(f"{label:24} caught {ca}/{tot} ({ca / tot:.0%}), false alarms {fa}")
    if "--frozen" in sys.argv:
        for label, cfg in FROZEN.items():
            ca, tot, fa = score(clips, lambda c: combined(c, *cfg))
            print(f"{label:12} {cfg}: caught {ca}/{tot} ({ca / tot:.0%}), false alarms {fa}")
        sys.exit()

    for d, k, z in itertools.product(("lab", "deep"), (1, 2, 3), (3, 4, 6)):
        ca, tot, fa = score(clips, lambda c: [(a, f) for a, f, _ in c["A"]] + top(c, d, k, z))
        print(f"A (no edge) + top-{k} in-place changes ({d:4}) z>={z}: caught {ca}/{tot} ({ca / tot:.0%}), false alarms {fa}")

    rows = []
    for cfg in itertools.product((0.05, 0.1, 0.2, 0.3), (0.0, 0.05, 0.1, 0.2), (3, 5, 10, 20), ("lab", "deep"),
                                 (0, 1, 2, 3), (3, 4, 6)):
        rows.append((cfg, score(clips, lambda c: combined(c, *cfg))))
    print("best recall of A + E + in-place changes at each false-alarm budget (strength, contrast, length, descriptor, top-k, z):")
    for budget in (50, 75, 100, 150, 250, 400):
        b = max((r for r in rows if r[1][2] <= budget), key=lambda r: (r[1][0], -r[1][2]), default=None)
        print(f"  <= {budget:3} alarms: "
              + (f"{b[1][0]}/{b[1][1]} ({b[1][0] / b[1][1]:.0%}) fa {b[1][2]} {b[0]}" if b else "-"))
