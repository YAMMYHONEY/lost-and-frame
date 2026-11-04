# Merge a blind visibility audit (audit_sheet.py sheets; one out_*.json per auditor: event number -> [class, note])
# into the v3-style file that eval_tile.py --vis reads. A second auditor's re-check of a sample (check.json, same
# shape) gives the agreement: the share of identical classes, and Cohen's kappa on V-or-V-lean against the rest.
# With a fourth path, it also writes the label list cut down to the objects with a V or V-lean event: the objects
# worth judging for visible-glitch recall. All their events are kept, so alerts at their other labelled moments are
# not false alarms.
# Usage: python research\visibility_merge.py <labels.json> <audit dir> <out.json> [<judged list.json>]
import glob, hashlib, json, sys
from pathlib import Path

CLASSES = ("V", "V-lean", "A", "N")


def kappa(a, b):  # two raters, two classes
    po = sum(x == y for x, y in zip(a, b)) / len(a)
    pe = sum((a.count(c) / len(a)) * (b.count(c) / len(b)) for c in (True, False))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


assert kappa([True, False], [True, False]) == 1.0 and kappa([True, False], [False, True]) == -1.0

if __name__ == "__main__":
    labels, audit, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    index = json.loads((audit / "index.json").read_text())
    calls = {}
    for f in sorted(glob.glob(str(audit / "out_*.json"))):
        part = {int(i): v for i, v in json.loads(Path(f).read_text("utf-8")).items()}
        assert not set(part) & set(calls), f"{f} repeats events"
        calls.update(part)
    assert set(calls) == {e["i"] for e in index} and all(v[0] in CLASSES for v in calls.values())

    events = [dict(i=e["i"], file=e["file"], concept=e["concept"], kind=e["kind"], frame=e["frame"],
                   visibility=calls[e["i"]][0], note=calls[e["i"]][1]) for e in index]
    out.write_text(json.dumps(dict(
        labels=labels, labels_sha256=hashlib.sha256(Path(labels).read_bytes()).hexdigest(),
        audit="8 frames at offsets -15..15 around each labelled frame (audit_sheet.py), viewed before any "
              "method ran on the set; one auditor (Claude) per sheet, blind to method outputs",
        classes=json.loads(Path("clips/physion-labels-v3-visibility.json").read_text())["classes"],
        events=events), indent=1))
    print({c: sum(e["visibility"] == c for e in events) for c in CLASSES}, "of", len(events))

    if len(sys.argv) > 4:
        seen = {(e["file"], e["concept"]) for e in events if e["visibility"] in ("V", "V-lean")}
        judged = [dict(c, concepts=[k for k in c["concepts"] if (c["file"], k) in seen],
                       events=[e for e in c["events"] if (c["file"], e["concept"]) in seen])
                  for c in json.loads(Path(labels).read_text()) if any((c["file"], k) in seen for k in c["concepts"])]
        Path(sys.argv[4]).write_text(json.dumps(judged, indent=1))
        print(len(judged), "clips,", sum(len(c["concepts"]) for c in judged), "objects worth judging")

    if (audit / "check.json").exists():
        check = {int(i): v[0] for i, v in json.loads((audit / "check.json").read_text("utf-8")).items()}
        a, b = [calls[i][0] for i in check], list(check.values())
        seen = lambda xs: [x in ("V", "V-lean") for x in xs]
        print(f"re-check of {len(check)}: same class {sum(x == y for x, y in zip(a, b)) / len(a):.0%},"
              f" kappa (V or V-lean vs rest) {kappa(seen(a), seen(b)):.2f}")
