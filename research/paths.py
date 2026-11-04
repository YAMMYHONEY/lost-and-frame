# Research outputs live in a short work folder, VC_WORK (default ./work): Windows long paths are off by default, and
# under a deep project folder Physion's 90-character clip names overflow MAX_PATH. Traces also get short names
# (prefix + hash, since clips of one prompt from different models share long prefixes).
import hashlib, os
from pathlib import Path

WORK = Path(os.environ.get("VC_WORK", "work"))
TRACES = WORK / "traces"


def trace_path(video, prompt):
    stem = Path(video).stem
    return TRACES / f"{stem[:12]}-{hashlib.sha1(stem.encode()).hexdigest()[:8]}.{prompt.replace(' ', '_')}.json"
