// Event-level scoring against human labels, the same for methods A, B and C.
// pred/truth: [{ concept, kind: 'vanish'|'appear', frame }]. A prediction matches the nearest
// unmatched label of the same concept and kind within ±tol frames (one-to-one). The default ±15 frames
// (~0.5 s at 30 fps) is close enough for a reviewer to find the moment; Spark's times were ~0.25 s off.
// burden: share of the clip a reviewer must watch, ±pad frames around each prediction.
export function scoreClip(pred, truth, { frameCount, tol = 15, pad = 15 }) {
  const used = new Set();
  let hit = 0;
  for (const p of pred) {
    let best = -1;
    truth.forEach((t, j) => {
      if (used.has(j) || t.concept !== p.concept || t.kind !== p.kind || Math.abs(t.frame - p.frame) > tol) return;
      if (best < 0 || Math.abs(t.frame - p.frame) < Math.abs(truth[best].frame - p.frame)) best = j;
    });
    if (best >= 0) {
      used.add(best);
      hit++;
    }
  }
  const watch = new Set();
  for (const p of pred) {
    for (let f = Math.max(0, p.frame - pad); f <= Math.min(frameCount - 1, p.frame + pad); f++) watch.add(f);
  }
  return { hit, miss: truth.length - hit, falseAlarm: pred.length - hit, burden: watch.size / frameCount };
}

// Success test fixed before the first evaluation: C raises at most half of A's false alarms and
// misses at most 10 percentage points more of the labeled events.
export function verdict({ A, C }, labeled) {
  const missRate = (m) => (labeled ? m.miss / labeled : 0);
  return { fewerFalseAlarms: C.falseAlarm <= A.falseAlarm / 2, missesHeld: missRate(C) - missRate(A) <= 0.1 };
}
