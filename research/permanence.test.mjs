import test from 'node:test';
import assert from 'node:assert/strict';
import { parseSamVideo, auditTracks } from './permanence.mjs';
import { readSse, videoCost } from './sam.mjs';

const N = 30, W = 640, H = 360;
const MID = [300, 150, 340, 190];

// spec: { id: [[fromFrame, toFrame, box, confidence?], ...] } → a parsed pass covering frames 0..N-1.
const mk = (spec) => {
  const frames = new Map(Array.from({ length: N }, (_, f) => [f, []]));
  for (const [id, runs] of Object.entries(spec)) for (const [a, b, box, c] of runs) {
    for (let f = a; f <= b; f++) frames.get(f).push({ id: +id, box: typeof box === 'function' ? box(f) : box, c });
  }
  return { width: W, height: H, frames };
};

const reasons = (r) => r.events.map((e) => `${e.kind}@${e.frame}:${e.reason}`);

test('parses the SAM 3.1 video sample from the docs', () => {
  const p = parseSamVideo('<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|><|mask;x=0;y=0;data=27,60,~ab|>,1<|box;x1=155;y1=228;x2=202;y2=254;w=320;h=334|><|mask;x=0;y=0;data=27,48,~cd|>\n<1f>');
  assert.equal(p.width, 320);
  assert.equal(p.height, 334);
  assert.deepEqual(p.frames.get(0), [{ id: 0, box: [211, 228, 270, 254] }, { id: 1, box: [155, 228, 202, 254] }]);
  assert.deepEqual(p.frames.get(1), []);
});

test('reads the confidence SAM adds with include_confidence', () => {
  const p = parseSamVideo(
    '<0f>1<|box;x1=444;y1=625;x2=679;y2=850;w=1800;h=1200;c=0.9609375|><|mask;x=0;y=0;c=0.9609375;data=226,236,!ab|>');
  assert.deepEqual(p.frames.get(0), [{ id: 1, box: [444, 625, 679, 850], c: 0.9609375 }]);
  assert.equal(p.width, 1800);
});

test('object vanishing mid-frame with no cause is a suspect', () => {
  const r = auditTracks(mk({ 0: [[0, 14, MID]] }), { frameCount: N });
  assert.deepEqual(reasons(r), ['vanish@14:suspect']);
  assert.equal(r.suspects.length, 1);
});

test('object appearing mid-frame from nothing is a suspect', () => {
  assert.deepEqual(reasons(auditTracks(mk({ 0: [[10, 29, MID]] }), { frameCount: N })), ['appear@10:suspect']);
});

test('a break in a track SAM itself was unsure of is set aside', () => {
  const r = auditTracks(mk({ 0: [[0, 14, MID, 0.2]] }), { frameCount: N });
  assert.deepEqual(reasons(r), ['vanish@14:lowconf']);
  assert.equal(r.suspects.length, 0);
  assert.equal(r.uncertain.length, 1);
});

test('leaving through the frame edge is explained', () => {
  const r = auditTracks(mk({ 0: [[0, 14, (f) => [300 + f * 22, 150, Math.min(W, 340 + f * 22), 190]]] }),
    { frameCount: N });
  assert.deepEqual(reasons(r), ['vanish@14:edge']);
});

test('an object that slides over it hides it', () => {
  const slidingIn = (f) => [Math.max(0, 620 - f * 22), 130, Math.min(W, 720 - f * 22), 220]; // from the right edge
  const r = auditTracks(mk({ 0: [[0, 14, MID]], 1: [[0, 29, slidingIn]] }), { frameCount: N });
  assert.deepEqual(reasons(r), ['vanish@14:occluded']);
});

test('a box that covered the spot all along (duplicate, container) is not an occluder', () => {
  const r = auditTracks(mk({ 0: [[0, 14, MID]], 1: [[0, 29, [280, 130, 380, 220]]] }), { frameCount: N });
  assert.deepEqual(reasons(r), ['vanish@14:suspect']);
});

test('an occluder seen on a single frame only counts when the object shrank behind it', () => {
  const once = { frames: new Map([[15, [{ id: 0, box: [280, 130, 380, 220] }]]]), sparse: true };
  assert.deepEqual(reasons(auditTracks(mk({ 0: [[0, 14, MID]] }), { frameCount: N, occluders: [once] })),
    ['vanish@14:suspect']);
});

test('sliding behind an object: the box shrinks to a sliver touching it', () => {
  const OCC = [340, 100, 440, 250];
  const slide = (f) => [305 + (f - 11) * 10, 150, 340, 190]; // visible part narrows toward x=340
  const r = auditTracks(mk({ 0: [[0, 10, MID], [11, 14, slide]], 1: [[0, 29, OCC]] }), { frameCount: N });
  assert.deepEqual(reasons(r), ['vanish@14:occluded']);
});

test('a background-sized box (ground, fence) does not count as an occluder', () => {
  const r = auditTracks(mk({ 0: [[0, 14, MID]], 1: [[0, 29, [0, 100, 640, 360]]] }), { frameCount: N });
  assert.deepEqual(reasons(r), ['vanish@14:suspect']);
});

test('an object shrinking away inside a bigger box is not sliding behind it', () => {
  // Shrinks in place, still judgeable.
  const fade = (f) => [310 + (f - 11), 160 + (f - 11), 330 - (f - 11), 180 - (f - 11)];
  const r = auditTracks(mk({ 0: [[0, 10, MID], [11, 14, fade]], 1: [[0, 29, [280, 130, 380, 220]]] }),
    { frameCount: N });
  assert.deepEqual(reasons(r), ['vanish@14:suspect']);
});

test('a full-size object vanishing next to another one is still a suspect', () => {
  const r = auditTracks(mk({ 0: [[0, 14, MID]], 1: [[0, 29, [340, 100, 440, 250]]] }), { frameCount: N });
  assert.deepEqual(reasons(r), ['vanish@14:suspect']);
});

test('occluder from a separate pass also explains it', () => {
  const r = auditTracks(mk({ 0: [[0, 14, MID]] }),
    { frameCount: N, occluders: [mk({ 7: [[12, 29, [280, 130, 380, 220]]] })] });
  assert.deepEqual(reasons(r), ['vanish@14:occluded']);
});

test('disappearing at a scene cut is explained', () => {
  assert.deepEqual(reasons(auditTracks(mk({ 0: [[0, 14, MID]] }), { frameCount: N, cuts: [15] })), ['vanish@14:cut']);
});

test('an independent pass that still sees the object marks a tracker miss', () => {
  const r = auditTracks(mk({ 0: [[0, 14, MID]] }), { frameCount: N, check: mk({ 0: [[0, 29, MID]] }) });
  assert.deepEqual(reasons(r), ['vanish@14:retracked']);
});

test('a short flicker inside a track is not an event', () => {
  assert.deepEqual(reasons(auditTracks(mk({ 0: [[0, 9, MID], [11, 29, MID]] }), { frameCount: N })), []);
});

test('id switch at the same place is a hand-off, not a violation', () => {
  const r = auditTracks(mk({ 0: [[0, 14, MID]], 1: [[15, 29, MID]] }), { frameCount: N });
  assert.deepEqual(reasons(r), ['vanish@14:handoff', 'appear@15:handoff']);
});

test('boxes too small to judge are set aside', () => {
  assert.deepEqual(reasons(auditTracks(mk({ 0: [[0, 14, [300, 150, 303, 153]]] }), { frameCount: N })),
    ['vanish@14:tiny']);
});

test('reads deltas from an event stream split at arbitrary points', async () => {
  const raw = 'event: x\r\ndata: {"type":"response.output_text.delta","delta":"<0f>0<|box"}\r\n\r\n' +
    'data: {"type":"response.output_text.delta","delta":";x1=1|>"}\n\ndata: {"type":"response.completed"}\n\n';
  const enc = new TextEncoder();
  const chunks = [raw.slice(0, 17), raw.slice(17, 90), raw.slice(90)].map((s) => enc.encode(s));
  assert.equal(await readSse(chunks), '<0f>0<|box;x1=1|>');
  await assert.rejects(readSse([enc.encode(raw.slice(0, 90))]), /before response.completed/); // dropped mid-stream
});

test('video cost bills at least 50 frames per request', () => {
  assert.equal(videoCost(30), 0.01);
  assert.equal(videoCost(150).toFixed(4), '0.0300');
});
