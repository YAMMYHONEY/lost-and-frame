// VC-Fuse: three witnesses that fail in different ways look at one clip (RESEARCH.md, sealed tests v5 and R6, 10-31).
//
// J, the scene judge: sees eight frames of each window and says whether the object vanished, appeared or changed
//    with no visible cause (research/verify.mjs, frames8 prompt). There is a window every 16 frames, and each one
//    spans ±15 frames.
// C, the counter: sees the same frames and counts the object in each one before it judges (count prompt).
// S, the spotter (precision mode, needs a CUDA GPU with SAM 3.1): SAM's detector proposes where one look-alike came
//    or went (research/detect.py, steps_prep.py), and the counter looks there up close. It runs only where J and C
//    found nothing.
//
// An alert is any witness answer at p >= threshold, placed at the pair of frames it named. Alerts of the same kind
// within 5 frames are one moment. Muse Spark's answers are cached by window, so checking a clip again costs nothing
// (and can replay without a key).
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { access, mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, relative } from 'node:path';
import { promisify } from 'node:util';
import { verify } from './research/verify.mjs';
import { frameCount, fps, normalized, windowFrames } from './video.mjs';

const run = promisify(execFile);

export const STEP = 16, HALF = 15, MERGE = 5;
export const OFF8 = [-15, -11, -7, -3, 3, 7, 11, 15];
export const PAIRS = ['A-B', 'B-C', 'C-D', 'D-E', 'E-F', 'F-G', 'G-H'];
export const SHIFT = Object.fromEntries(PAIRS.map((pair, i) => [pair, (OFF8[i] + OFF8[i + 1]) / 2]));
export const NAMES = { J: 'scene judge', C: 'counter', S: 'spotter (zoomed)' };
export const BACKENDS = {
  baseten: { J: 'open-frames8', C: 'open-frames8-count', S: 'open-zoom8-count' }, // Muse Spark Open: the sealed tests
  meta: { J: 'frames8', C: 'frames8-count', S: 'zoom8-count' }, // Muse Spark 1.3 on the Meta API: its own sealed runs
};

// Window centres, one every STEP frames, kept inside the clip (research/tile_prep.py).
export const centres = (n) => [
  ...new Set(Array.from(
    { length: Math.ceil((n + STEP - HALF) / STEP) },
    (_, i) => Math.min(HALF + i * STEP, n - 1 - HALF),
  )),
].sort((a, b) => a - b);

// How likely a verdict says the object really vanished or appeared.
export const p = (v) => (v.violation ? v.confidence / 100 : 1 - v.confidence / 100);

// The alert kinds a verdict raises. Any other kind, such as transform, raises both.
const kinds = (v) => ({ vanish: ['vanish'], appear: ['appear'] })[v.kind] ?? ['vanish', 'appear'];

// Every verdict at p >= t becomes an alert of its kind, at the frame pair it named (the window centre if none).
export const alerts = (judged, t) => judged
  .filter(({ v }) => p(v) >= t)
  .flatMap(({ w, v, witness }) => kinds(v).map((kind) => ({
    kind,
    frame: w.frame + (SHIFT[v.between] ?? 0),
    witness,
    w,
    v,
  })));

// Alerts of the same kind within MERGE frames are one moment, as the sealed tests scored them
// (research/eval_evidence.py).
export function moments(list) {
  const out = [];
  for (const a of [...list].sort((x, y) => x.kind.localeCompare(y.kind) || x.frame - y.frame)) {
    const m = out.find((o) => o.kind === a.kind && Math.abs(o.frame - a.frame) <= MERGE);
    if (m) m.by.push(a);
    else out.push({ kind: a.kind, frame: a.frame, by: [a] });
  }
  return out.sort((x, y) => x.frame - y.frame);
}

// Runs fn on every item, at most k at a time.
async function pool(items, k, fn) {
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(k, items.length) }, async () => {
    while (next < items.length) await fn(items[next++]);
  }));
}

const exists = (f) => access(f).then(() => true, () => false);
const slug = (s) => s.replace(/[^\p{L}\p{N}]+/gu, '_');

// Precision mode is offered only where the research spotter can run: Python with a CUDA GPU, SAM 3.1 and OpenCV.
export async function precisionReady(python) {
  try {
    await run(python, ['-c', 'import cv2, sam3, torch; assert torch.cuda.is_available()'], { timeout: 120e3 });
    return true;
  } catch {
    return false;
  }
}

// backend: 'baseten', 'meta' or 'recorded' (no key: saved answers only).
// maxCalls: a clip that needs more judge calls than this is refused before any call is made.
// Returns { source, frames, fps, backend, threshold, precision, calls, results: [{ concept, moments }] }.
export async function checkClip(original, concepts, {
  dir,
  key,
  backend,
  threshold = 0.9,
  precision = false,
  python = 'python',
  maxCalls = Infinity,
}) {
  const path = await normalized(original);
  const n = await frameCount(path);
  const rate = await fps(path);
  if (n < 2 * HALF + 1) {
    throw new Error(`The clip has ${n} frames; Lost & Frame needs at least ${2 * HALF + 1} (about 1.5 s).`);
  }

  const recorded = backend === 'recorded';
  const modes = BACKENDS[backend === 'meta' ? 'meta' : 'baseten'];
  const ask = (m, witness) => verify(m, modes[witness], { recorded });
  const windows = concepts.flatMap((concept, clip) => centres(n).map((frame) => ({
    id: `app-${key}-${slug(concept)}-t${String(frame).padStart(4, '0')}`,
    clip,
    concept,
    name: concept,
    frame,
    file: relative('clips', path).replaceAll('\\', '/'),
    kind: null,
    source: 'T',
    box: null,
    image: null,
    window: null,
  })));
  if (!recorded && windows.length * 2 > maxCalls) {
    throw new Error(`This clip needs ${windows.length * 2} judge calls, over the ${maxCalls} a check may make: use a shorter clip or fewer objects.`);
  }

  // Each window's eight frames are extracted once and reused when the clip is checked again.
  await mkdir(join(dir, 'w'), { recursive: true });
  await pool(windows, 4, async (w) => {
    const prefix = join(dir, 'w', w.id);
    w.frames8 = (await exists(`${prefix}-7.jpg`))
      ? OFF8.map((_, k) => `${prefix}-${k}.jpg`)
      : await windowFrames(path, w.frame, OFF8, prefix);
  });

  // The scene judge and the counter look at every window.
  const judged = [];
  await pool(
    windows.flatMap((w) => ['J', 'C'].map((witness) => ({ w, witness }))),
    8,
    async (j) => judged.push({ ...j, v: await ask(j.w, j.witness) }),
  );

  if (precision) {
    // The research spotter runs as it is: write its manifest and the two judges' verdicts, then run detect.py and
    // steps_prep.py. The file stem sits next to the clip's folder and is kept short, because steps_prep.py names a
    // folder and its crops after it, and OpenCV cannot write past Windows' 260-character path limit (it failed under
    // a 180-character project folder).
    const stem = join(
      dirname(dir),
      `${key.slice(0, 12)}${createHash('sha1').update(concepts.join(',')).digest('hex').slice(0, 4)}`,
    );
    const verdicts = (witness) => judged.filter((j) => j.witness === witness).map((j) => ({ id: j.w.id, ...j.v }));
    await writeFile(`${stem}.json`, JSON.stringify(windows));
    await writeFile(`${stem}-J.json`, JSON.stringify(verdicts('J')));
    await writeFile(`${stem}-C.json`, JSON.stringify(verdicts('C')));
    await writeFile(`${stem}-list.json`, JSON.stringify([{ file: windows[0].file, concepts }]));

    const env = { ...process.env, PYTHONPATH: 'research', PYTHONIOENCODING: 'utf-8' };
    await run(python, ['research/detect.py', `${stem}-list.json`], { env, maxBuffer: 64 << 20 });
    await run(python, ['research/steps_prep.py', `${stem}.json`, `${stem}-J.json`, `${stem}-C.json`], {
      env,
      maxBuffer: 64 << 20,
    });
    const spots = JSON.parse(await readFile(`${stem}-steps.json`, 'utf8')).filter((m) => m.zoom8);
    await pool(spots, 8, async (w) => judged.push({ w, witness: 'S', v: await ask(w, 'S') }));
  }

  const results = concepts.map((concept, clip) => ({
    concept,
    moments: moments(alerts(judged.filter((j) => j.w.clip === clip), threshold)),
  }));
  return { source: path, frames: n, fps: rate, backend, threshold, precision, calls: judged.length, results };
}
