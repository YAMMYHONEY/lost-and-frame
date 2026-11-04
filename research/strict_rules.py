# Exploratory and free: re-score cached judge answers under stricter alert rules. The answers are Spark 1.3 on the
# sealed R6 (real glitches) and C5 (clean) sets, and Spark Open on R6 and the v4 clean set. Every set was already
# scored, so a rule picked from this needs a fresh set to count.
# Usage (project folder): PYTHONPATH=research python research/strict_rules.py
import json
from collections import Counter
from pathlib import Path
from eval_tile import load, recall, SHIFT, p
from eval_evidence import moments, score, TOL
from eval_verify import BOTH
from paths import WORK

V = WORK / "verify"
SETS = {
    "R6": (f"{V}/r6tile-m13-steps.json", ["clips/physion-v6-judged.json", "clips/deeptrace-judged.json"],
           ["clips/physion-labels-v6-visibility.json", "clips/deeptrace-visibility.json"],
           {"J": ["v6tile-m13-verdicts-frames8", "dtrtile-m13-verdicts-frames8"],
            "C": ["v6tile-m13-verdicts-frames8-count", "dtrtile-m13-verdicts-frames8-count"],
            "S": ["v6tile-m13-steps-verdicts-zoom8-count", "dtrtile-m13-steps-verdicts-zoom8-count"]}),
    "C5": (f"{V}/v5cleantile-m13-steps.json", ["clips/physion-v5-clean.json"], [],
           {"J": ["v5cleantile-m13-verdicts-frames8"], "C": ["v5cleantile-m13-verdicts-frames8-count"],
            "S": ["v5cleantile-m13-steps-verdicts-zoom8-count"]}),
    # Muse Spark Open (Baseten): R6 with all three witnesses; the older v4 clean control has the scene judge only.
    "R6-open": (f"{V}/r6tile-steps.json", ["clips/physion-v6-judged.json", "clips/deeptrace-judged.json"],
                ["clips/physion-labels-v6-visibility.json", "clips/deeptrace-visibility.json"],
                {"J": ["v6tile-verdicts-open-frames8", "dtrtile-verdicts-open-frames8"],
                 "C": ["v6tile-verdicts-open-frames8-count", "dtrtile-verdicts-open-frames8-count"],
                 "S": ["v6tile-steps-verdicts-open-zoom8-count", "dtrtile-steps-verdicts-open-zoom8-count"]}),
    "C4-open-J": (f"{V}/v4cleantile.json", ["clips/physion-v4-clean.json"], [],
                  {"J": ["v4cleantile-verdicts-open-frames8"]}),
}
# Each rule: (threshold, number of witnesses that must agree).
RULES = [(t, 1) for t in (0.5, 0.9, 0.95, 0.99)] + [(t, 2) for t in (0.5, 0.9)]
out = {}
for name, (manifest, lists, vises, runs) in SETS.items():
    clips, vis = load(lists), {}
    for f in vises:
        vis |= {(e["file"], e["concept"], e["kind"], e["frame"]): e["visibility"]
                for e in json.loads(Path(f).read_text("utf-8"))["events"]}
    said = {w: {v["id"]: v for f in fs for v in json.loads(Path(f"{V}/{f}.json").read_text("utf-8"))}
            for w, fs in runs.items()}
    alerts, ps = {}, []  # clip index -> [(p, kind, frame, witness)]; p of every judge call
    for m in json.loads(Path(manifest).read_text()):
        for w, s in said.items():
            if m["id"] in s:
                v = s[m["id"]]
                ps.append(p(v))
                for k in {"vanish": ("vanish",), "appear": ("appear",)}.get(v["kind"]) or BOTH:
                    alerts.setdefault(m["clip"], []).append((p(v), k, m["frame"] + SHIFT.get(v["between"], 0), w))
    index = {id(c): i for i, c in enumerate(clips)}
    print(f"== {name}: {len(ps)} judge calls over {len({c['file'] for c in clips})} videos; calls by p:",
          dict(sorted(Counter(round(x, 2) for x in ps if x >= 0.5).items())))
    for t, need in RULES:
        def pick(c):
            a = [x for x in alerts.get(index[id(c)], []) if x[0] >= t]
            return [(k, f) for k, f in moments([(k, f) for _, k, f, _ in a])
                    if len({w for _, kk, ff, w in a if kk == k and abs(ff - f) <= TOL}) >= need]

        ca, tot, fa = score(clips, pick)
        n = sum(len(moments(pick(c))) for c in clips)
        vids = {}
        for c in clips:
            had, hit = vids.get(c["file"], (False, False))
            vids[c["file"]] = (had or bool(c["truth"]), hit or bool(pick(c)))
        clean = [h for e, h in vids.values() if not e]
        row = dict(calls_alerting=sum(x >= t for x in ps), moments=n, false_alarms=fa, caught=ca, events=tot,
                   clean_flagged=f"{sum(clean)}/{len(clean)}")
        for label, ok in (("V", {"V"}), ("V+lean", {"V", "V-lean"})) if vis else ():
            row[label] = "%d/%d" % recall(clips, pick, lambda c, k, f: vis.get((c["file"], c["concept"], k, f)) in ok)
        out[(name, t, need)] = row
        print(f"p>={t} agree>={need}:", row)

# Self-check: the plain rule reproduces the sealed logs (clips/sealed-r6-real-m13.log, sealed-c5-clean-m13.log).
assert (out[("R6", 0.9, 1)]["V"], out[("R6", 0.9, 1)]["V+lean"],
        out[("R6", 0.9, 1)]["false_alarms"]) == ("23/25", "94/126", 230), out[("R6", 0.9, 1)]
assert (out[("C5", 0.9, 1)]["clean_flagged"], out[("C5", 0.9, 1)]["false_alarms"]) == ("33/64", 79), out[("C5", 0.9, 1)]
assert (out[("C5", 0.5, 1)]["clean_flagged"], out[("R6", 0.5, 1)]["V+lean"]) == ("45/64", "105/126")
assert (out[("R6-open", 0.5, 1)]["V"],
        out[("R6-open", 0.5, 1)]["V+lean"]) == ("23/25", "99/126"), out[("R6-open", 0.5, 1)]  # clips/sealed-r6-real.log
print("matches the sealed logs")

# The keyless samples (recorded with Muse Spark Open): judge calls at each threshold.
sample = [p(json.loads(f.read_text("utf-8"))["answer"]) for f in Path("samples/raw").glob("*.json")]
print("samples:", len(sample), "calls;", {t: sum(x >= t for x in sample) for t in (0.5, 0.9, 0.95)})
