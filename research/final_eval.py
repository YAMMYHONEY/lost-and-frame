# Pre-registered operating points (RESEARCH.md, 10-30) scored on one candidate manifest and its two verdict files.
# Candidates (verify_prep.py): SAM breaks off the frame edge (A), each object's top-3 in-place changes (P), its 3
# strongest evidence-track events (E). Verifiers (verify.mjs): boxed (v1) and box-free window (w1) Muse Spark prompts.
# Usage: python research\final_eval.py <manifest.json> [--chosen] <list.json>...   -> <manifest>-final.json
import json, sys
from pathlib import Path
from eval_evidence import load, score, chosen_names
from eval_verify import alerts

POINTS = {  # name: (candidate sources, verifier rule)
    "VC-Precise": ("A", "window"),     # SAM breaks the box-free verifier confirms
    "VC-Balanced": ("APE", "window"),  # any candidate the box-free verifier confirms
    "VC-Recall": ("APE", "either"),    # any candidate either verifier confirms
    "VC-Balanced+DWS": ("APEW", "window"),  # plus dense window scans of objects SAM barely sees (dws_prep.py)
    "VC-Recall+DWS": ("APEW", "either"),
}

if __name__ == "__main__":
    manifest_path, args = sys.argv[1], sys.argv[2:]
    clips = load([a for a in args if not a.startswith("--")], chosen_names() if "--chosen" in args else {})
    manifest = json.loads(Path(manifest_path).read_text())
    V = {s: {v["id"]: v
             for v in json.loads(Path(manifest_path.replace(".json", f"-verdicts{s}.json")).read_text("utf-8"))}
         for s in ("", "-window")}
    missing = [m["id"] for m in manifest
               if (m["source"] != "W" and m["id"] not in V[""]) or m["id"] not in V["-window"]]
    assert not missing, f"{len(missing)} moments lack a verdict, e.g. {missing[:3]}"

    by = {}
    for m in manifest:
        by.setdefault(m["clip"], []).append(m)

    def verified(c, sources, rule):
        out = []
        for m in by.get(clips.index(c), []):
            w, b = V["-window"][m["id"]], V[""].get(m["id"], {"violation": False})  # scan windows: no boxed verdict
            if m["source"] in sources:  # each confirming verdict contributes its own kind
                out += alerts(m, w) if w["violation"] else []
                out += alerts(m, b) if rule == "either" and b["violation"] else []
        return out

    rows = {
        "A: every SAM break": lambda c: [(k, f) for k, f, _ in c["A_all"]],
        "A without frame edge": lambda c: [(k, f) for k, f, _ in c["A"]],
        "candidates, unverified": lambda c: [a for m in by.get(clips.index(c), []) if m["source"] != "W"
                                             for a in alerts(m, {"kind": "none"})],
        **{name: (lambda s, r: lambda c: verified(c, s, r))(*p) for name, p in POINTS.items()},
    }
    result = {}
    for name, pick in rows.items():
        ca, tot, fa = score(clips, pick)
        result[name] = dict(caught=ca, events=tot, false_alarms=fa)
        print(f"{name:24} caught {ca}/{tot} ({ca / tot:.0%}), false alarms {fa}")
    cost = sum(V[s][m["id"]]["cost"] for s in V for m in manifest if m["id"] in V[s])
    print(f"Spark verification ${cost:.3f} for {len(clips)} objects (${cost / len(clips):.4f} each)")
    Path(manifest_path.replace(".json", "-final.json")).write_text(json.dumps(dict(result=result, cost=cost), indent=1))
