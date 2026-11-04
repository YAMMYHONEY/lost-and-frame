#!/usr/bin/env bash
# All remaining label-free GPU work for dev + sealed v2, one fresh process per clip (exit code 3 = "more to do"):
# a single long SAM 3.1 process grows to 14 GB and pages. Every step skips what already exists, so it can be rerun.
# Usage (project folder): bash research/run_chain.sh > research/gpu-chain3.log 2> research/gpu-chain3.err
P=${PYTHON:-python}
export SAM_MAX_OBJECTS=8 TRACE_MAX_NEW=1 PROBE_MAX_NEW=5
loop() { while :; do "$@"; rc=$?; [ $rc -eq 3 ] || return $rc; done; }
# Resource rule (label-blind, same for every split): a clip that does not finish within LIMIT seconds at 8 objects is
# retraced at 4. A crowded clip (dozens of beads) at 8 objects outgrew the 8 GB GPU into 20 GB of paged RAM; no dev
# clip comes near the limit (slowest: 283 s).
LIMIT=900
trace_loop() {
  while :; do
    timeout $LIMIT "$@"; rc=$?
    if [ $rc -eq 124 ]; then echo "timeout after ${LIMIT}s: retracing the stuck clip at 4 objects"; SAM_MAX_OBJECTS=4 "$@"; rc=$?; fi
    [ $rc -eq 3 ] || return $rc
  done
}
step() { echo "== $1 $(date +%H:%M)"; }

step "trace sealed v2 (given names)"; trace_loop $P research/trace.py --list clips/physion-labels-v2.json clips/physion-labels-v2-eval.json
step "probe sealed v2"; loop $P research/name_probe.py clips/physion-labels-v2.json
step "choose names (dev + v2)"; $P research/choose_names.py
step "trace renamed"; trace_loop $P research/trace.py --names research/names-chosen.json
step "exemplar dev"; trace_loop $P research/trace.py --boxes ${VC_WORK:-work}/localize/dev-boxes.json
step "localize v2"; $P research/localize_prep.py v2 clips/physion-labels-v2.json && node --env-file=.env research/localize.mjs ${VC_WORK:-work}/localize/v2.json
step "exemplar v2"; trace_loop $P research/trace.py --boxes ${VC_WORK:-work}/localize/v2-boxes.json
step "adopt exemplars"; $P research/exemplar_names.py ${VC_WORK:-work}/localize/dev-boxes.json ${VC_WORK:-work}/localize/v2-boxes.json
step "done"
