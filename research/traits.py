# What the labelled glitches have in common, and what the missed ones have in common. Every labelled event of the
# judged lists gets traits:
# - how many of its object the count judge saw near it (several look-alikes or one);
# - its kind, where in the clip it falls, and its generator;
# - what the auditor's note and the dataset's description mention (hands, cover, liquids and see-through things,
#   small things, gradual change).
# Then, per trait, how often each judge (eval_tile.py alerts, p >= T) missed the visible ones (V or V-lean), with
# alerts of the named kind or of any kind.
# Usage: python research\traits.py <manifest.json> <judge verdicts.json> <count verdicts.json> --vis <v.json>...
#            -- <list.json>...
import json, re, sys
from collections import Counter
from pathlib import Path
from eval_tile import load, SHIFT, p
from eval_verify import BOTH
from eval_evidence import TOL

T = 0.5
WORDS = dict(hand=r"hand|hold|grab|pick|carr|pour|cut|slic|wash|stir|knead|fold|wip|scoop|press",
             cover=r"behind|cover|occlu|hidden|hide|block|beneath|inside",
             fluid=r"liquid|water|juice|sauce|soup|smoke|steam|foam|bubble|ice|glass|transparen|transluc|slime|oil|milk|dough|powder|flour",
             small=r"small|tiny|little|thin|crumb|drop",
             gradual=r"fade|gradual|slowly|ghost|blur")


def caught(ws, verdicts, kind, fr, anykind):
    return any(p(v) >= T and abs(m["frame"] + SHIFT.get(v["between"], 0) - fr) <= TOL
               and (anykind or kind in ({"vanish": ("vanish",), "appear": ("appear",)}.get(v["kind"]) or BOTH))
               for m in ws for v in verdicts.get(m["id"], []))


if __name__ == "__main__":
    cut = sys.argv.index("--")
    man, jv, cv = sys.argv[1:4]
    args, clips = sys.argv[4:cut], load(sys.argv[cut + 1:])
    vis = {}
    for i, a in enumerate(args):
        if a == "--vis":
            vis |= {(e["file"], e["concept"], e["kind"], e["frame"]): e
                    for e in json.loads(Path(args[i + 1]).read_text("utf-8"))["events"]}
    J, C = ({v["id"]: [v] for v in json.loads(Path(f).read_text("utf-8"))} for f in (jv, cv))
    JC = {k: J.get(k, []) + C.get(k, []) for k in J.keys() | C.keys()}
    windows = [m for m in json.loads(Path(man).read_text()) if m["id"] in J and m["id"] in C]
    text = {}
    for sel in ("clips/physion/selection-v3.json", "clips/physion/selection-v4.json"):
        text |= {f"physion/videos/{s['filename']}.mp4": s.get("reasoning", "")
                 for s in json.loads(Path(sel).read_text())}
    model = {c["file"]: c.get("model") for f in sys.argv[cut + 1:] for c in json.loads(Path(f).read_text())}

    rows = []
    for ci, c in enumerate(clips):
        ws = [m for m in windows if m["clip"] == ci]
        for kind, fr in c["truth"]:
            e = vis.get((c["file"], c["concept"], kind, fr), {})
            near = [n for m in ws if abs(m["frame"] - fr) <= TOL
                    for v in C.get(m["id"], []) for n in v.get("counts", [])]
            words = f"{e.get('note', '')} {text.get(c['file'], '')}".lower()
            r = dict(vis=e.get("visibility", "?"), kind=kind, model=model[c["file"]], concept=c["concept"],
                     many=max(near, default=0) >= 2,
                     where="start" if fr < 0.1 * c["n"] else "end" if fr > 0.9 * c["n"] else "middle",
                     **{w: bool(re.search(rx, words)) for w, rx in WORDS.items()})
            for name, ver in (("J", J), ("C", C), ("JC", JC)):
                r[name], r[name + "*"] = caught(ws, ver, kind, fr, False), caught(ws, ver, kind, fr, True)
            rows.append(r)

    seen = [r for r in rows if r["vis"] in ("V", "V-lean")]
    print(f"{len(rows)} labelled events ({Counter(r['vis'] for r in rows)}); visible (V or V-lean): {len(seen)}")
    print("\nTrait            | all events   | visible     | missed by J  JC   JC any kind (visible events with the trait)")
    traits = [("several look-alikes", lambda r: r["many"]), ("one of its kind", lambda r: not r["many"]),
              ("vanish", lambda r: r["kind"] == "vanish"), ("appear", lambda r: r["kind"] == "appear"),
              ("first 10% of clip", lambda r: r["where"] == "start"),
              ("last 10% of clip", lambda r: r["where"] == "end")]
    traits += [(f"mentions {w}", lambda r, w=w: r[w]) for w in WORDS]
    traits += [(f"model {m}", lambda r, m=m: r["model"] == m) for m in sorted({r["model"] for r in rows})]
    for name, has in traits:
        a, s = [r for r in rows if has(r)], [r for r in seen if has(r)]
        miss = lambda k: f"{sum(not r[k] for r in s):3d} ({sum(not r[k] for r in s) / max(len(s), 1):4.0%})"
        print(f"{name:17s}| {len(a):3d} ({len(a) / len(rows):4.0%}) | {len(s):3d} ({len(s) / len(seen):4.0%}) | {miss('J')} {miss('JC')} {miss('JC*')}")
    print("\nMost common objects:", Counter(r["concept"] for r in rows).most_common(15))
    print("Visible events missed by both judges even with any-kind alerts:")
    for r in seen:
        if not r["JC*"]:
            print("  ", r["vis"], r["kind"], r["concept"], "| many" if r["many"] else "| one", r["where"],
                  [w for w in WORDS if r[w]])
