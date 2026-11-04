// Permanence audit: classify every appearance/disappearance of a tracked object.
// Events no ordinary cause explains (frame edge, occluder, scene cut, tracker hand-off,
// independent re-detection) are suspects; tiny or low-confidence ones are set aside as uncertain.

const BOX = /(\d+)<\|box;x1=([\d.-]+);y1=([\d.-]+);x2=([\d.-]+);y2=([\d.-]+);w=(\d+);h=(\d+)(?:;c=([\d.]+))?\|>/g;

// SAM 3.1 output: "<Nf>id<|box…|><|mask…|>,id<|box…|>…" per frame (images use <0f>). Masks are ignored.
// shortcut: boxes only; parse masks (@meta-sam/parser) if box overlap misjudges occlusion.
export function parseSamVideo(text) {
  const frames = new Map();
  let width = 0, height = 0;
  const parts = text.split(/<(\d+)f>/);
  for (let i = 1; i < parts.length; i += 2) {
    const objects = frames.get(+parts[i]) ?? [];
    for (const m of parts[i + 1].matchAll(BOX)) {
      const box = [+m[2], +m[3], +m[4], +m[5]];
      objects.push(m[8] ? { id: +m[1], box, c: +m[8] } : { id: +m[1], box });
      width = +m[6];
      height = +m[7];
    }
    frames.set(+parts[i], objects);
  }
  return { width, height, frames };
}

const area = ([x1, y1, x2, y2]) => Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
const inter = (a, b) => area([Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.min(a[2], b[2]), Math.min(a[3], b[3])]);
const iou = (a, b) => inter(a, b) / (area(a) + area(b) - inter(a, b) || 1);

export const DEFAULTS = {
  gap: 2,        // missing frames tolerated inside one track (flicker)
  edge: 0.02,    // box within this fraction of min(W,H) from the border = left/entered frame
  cutWindow: 2,  // frames around a scene cut
  cover: 0.3,    // share of the box covered by another object = occlusion
  maxOccluder: 0.25, // boxes over this share of the frame are ground/background (road, fence) and overlap everything
  moved: 0.2,    // an occluder must cover this much more of the spot than while the object was in view
  minArea: 0.0005, // boxes smaller than this share of the frame are too small to judge
  match: 0.5,    // IoU for "same object" across tracks or passes
  lookahead: 3,  // frames searched in the check pass
  minConf: 0.5,  // SAM gives one confidence per tracked object (same on every frame); below this = unsure what it is
  shrink: 0.5,   // box below this share of its recent peak = only a sliver is still visible
  lookback: 5,   // frames inside the track used for that peak
};

// pass: parseSamVideo() result. check: an independent detection (SAM on single frame images) near breaks.
export function auditTracks(pass, { frameCount, cuts = [], occluders = [], check, ...opt } = {}) {
  const o = { ...DEFAULTS, ...opt };
  const { width: W, height: H } = pass;
  const last = (frameCount ?? Math.max(-1, ...pass.frames.keys()) + 1) - 1;
  const margin = o.edge * Math.min(W, H);
  const at = (p, f) => p?.frames.get(f) ?? [];

  const tracks = new Map();
  for (const [f, objs] of pass.frames) for (const { id, box, c } of objs) {
    if (!tracks.has(id)) tracks.set(id, []);
    tracks.get(id).push({ f, box, c });
  }

  const events = [];
  const peak = (pts) => Math.max(0, ...pts.map((p) => area(p.box)));
  for (const [id, pts] of tracks) {
    pts.sort((a, b) => a.f - b.f);
    let s = 0; // first point of the current segment
    for (let i = 1; i <= pts.length; i++) {
      const prev = pts[i - 1], cur = pts[i];
      if (cur && cur.f - prev.f <= o.gap + 1) continue;
      const seg = pts.slice(s, i), start = seg[0];
      if (start.f > 0) events.push({ id, kind: 'appear', frame: start.f, box: start.box, c: start.c,
        peak: peak(seg.slice(1, 1 + o.lookback)) });
      if (prev.f < last) events.push({ id, kind: 'vanish', frame: prev.f, box: prev.box, c: prev.c,
        peak: peak(seg.slice(-1 - o.lookback, -1)) });
      s = i;
    }
  }

  const reasonFor = (e) => {
    const b = e.box, f = e.frame, vanish = e.kind === 'vanish';
    const shotStart = vanish ? f + 1 : f;
    if (cuts.some((c) => Math.abs(c - shotStart) <= o.cutWindow)) return 'cut';
    if (b[0] <= margin || b[1] <= margin || b[2] >= W - margin || b[3] >= H - margin) return 'edge';
    if (area(b) < o.minArea * W * H) return 'tiny';
    // Checked before occlusion: a new id on the same spot would otherwise "cover" the old one.
    const handoff = events.some((x) => x.id !== e.id && x.kind !== e.kind && iou(b, x.box) >= o.match &&
      (vanish ? x.frame > f && x.frame <= f + 1 + o.gap : x.frame < f && x.frame >= f - 1 - o.gap));
    if (handoff) return 'handoff';
    const side = vanish ? [f, f + 1] : [f - 1, f];
    // SAM boxes only the visible part, so an object sliding behind another shrinks to a sliver that
    // touches the occluder without overlapping it.
    // shortcut: an object that shrinks to nothing beside another one is excused even if it really vanished.
    const sliver = area(b) < o.shrink * e.peak;
    const grown = [b[0] - margin, b[1] - margin, b[2] + margin, b[3] + margin];
    const coverOf = (box) => inter(b, box) / area(b);
    const inView = Math.max(0, Math.min(last, vanish ? f - o.lookback : f + o.lookback));
    const covered = side.some((g) => [pass, ...occluders].some((p) => at(p, g).some((x) => {
      if ((p === pass && x.id === e.id) || area(x.box) > o.maxOccluder * W * H) return false;
      const c = coverOf(x.box);
      // The visible sliver of a hidden object lies outside the occluder, touching it. Generated objects
      // that shrink away inside another box are not hiding behind it.
      if (sliver && c < o.cover && inter(grown, x.box) > 0) return true;
      // Hiding needs a cause: the occluder came over the spot. A box that covered it all along is a
      // duplicate detection or a container. Sparse passes (one frame) have no history, so only slivers count.
      if (c < o.cover || p.sparse) return false;
      const earlier = at(p, inView).find((y) => y.id === x.id);
      return !earlier || coverOf(earlier.box) <= c - o.moved;
    })));
    if (covered) return 'occluded';
    for (let k = 1; k <= o.lookahead; k++) {
      if (at(check, vanish ? f + k : f - k).some((x) => iou(b, x.box) >= o.match)) return 'retracked';
    }
    if (e.c < o.minConf) return 'lowconf';
    return 'suspect';
  };

  for (const e of events) e.reason = reasonFor(e);
  events.sort((a, b) => a.frame - b.frame || a.id - b.id);
  const summary = {};
  for (const e of events) summary[e.reason] = (summary[e.reason] ?? 0) + 1;
  return {
    events,
    suspects: events.filter((e) => e.reason === 'suspect'),
    uncertain: events.filter((e) => e.reason === 'tiny' || e.reason === 'lowconf'),
    summary,
  };
}
