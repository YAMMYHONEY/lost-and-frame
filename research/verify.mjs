// MLLM verifier for candidate moments (research/verify_prep.py). Muse Spark sees one image: four frames around the
// moment with the object's box in yellow (top row) and the same frames zoomed on the box (bottom row), and says
// whether something there vanished, appeared or turned into something else with no visible cause. It verifies a
// marked moment instead of searching a whole clip: asked to search, Spark found 0 of 20 events (sealed v2, method B).
// Usage: node --env-file=.env research/verify.mjs <manifest.json> [limit|all] [box|window|zoom|open-window|open-clip]
//   -> <manifest>-verdicts.json (box), else -verdicts-<mode>.json. open-*: Muse Spark Open on Baseten (spark_open.mjs).
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { askImages, sparkCost } from '../spark.mjs';
import { askOpen, openCost, imagePart, videoPart, OPEN_LEDGER } from '../spark_open.mjs';
import { pay } from '../budget.mjs';

const RESERVE = 0.03; // dollars per call, settled to the real cost after
const PARALLEL = Number(process.env.PARALLEL ?? 4); // more for long runs if the API keeps up
const VERSION = 'v1'; // part of the cache key: change it when the prompt or image layout changes
const WINDOW_VERSION = 'w1';

export const prompt = (name) =>
  `The top row shows four moments A, B, C, D (left to right, a fraction of a second apart) from an
AI-generated video. The yellow box marks where a "${name}" is or was. The bottom row shows the same four moments
zoomed in on the yellow box.

Between two consecutive moments, does an object at or inside the yellow box vanish, appear, or turn into a different
object (or suddenly lose or gain a part) with NO visible physical cause?
These are visible causes and do NOT count: a hand or tool picks it up, puts it down or carries it; it moves out of the
box or out of the frame; something passes in front of it; the camera moves, pans or zooms; it is cut, poured, folded or
eaten in a visible action; motion blur. Judge only what you can see.`;

// Box-free variant: candidate boxes often sit on the wrong object even when the moment is right (dev: a pizza slice
// vanished from the right pizza while the candidate box was on the left one), so ask about the whole short window.
export const windowPrompt = (name) =>
  `These are four moments A, B (top row) and C, D (bottom row), in that order and a fraction of
a second apart, from an AI-generated video.

Look at every ${name} in view. Between two consecutive moments, does any ${name} vanish, appear, or turn into a
different object (or suddenly lose or gain a part) with NO visible physical cause?
These are visible causes and do NOT count: a hand or tool picks it up, puts it down or carries it; it moves out of the
frame; something passes in front of it; the camera moves, pans or zooms; it is cut, poured, folded or eaten in a
visible action; motion blur. Judge only what you can see.`;

// v3: the same window plus a bottom row zoomed on the region that changed most (camera motion removed), weighted
// towards where the detector sees the object (research/change_zoom.py). The yellow box marks that region.
export const zoomPrompt = (name) => `${windowPrompt(name)}

The bottom row zooms into the yellow box, the region that changed most between the moments after removing camera
motion. It often shows the ${name}, but it may show something else (a moving hand): use it as a closer look, not as proof.`;

export const schema = {
  type: 'object', additionalProperties: false, required: ['violation', 'kind', 'between', 'confidence', 'reason'],
  properties: {
    violation: { type: 'boolean' },
    kind: { type: 'string', enum: ['vanish', 'appear', 'transform', 'none'] },
    between: { type: 'string', enum: ['A-B', 'B-C', 'C-D', 'none'] },
    confidence: { type: 'integer', minimum: 0, maximum: 100 },
    reason: { type: 'string' },
  },
};

// Muse Spark Open (Baseten) has no structured outputs: the same fields are asked for in words.
const JSON_ONLY = `Reply with only one JSON object and nothing else, with exactly these keys:
{"violation": true or false, "kind": "vanish" or "appear" or "transform" or "none",
 "between": "A-B" or "B-C" or "C-D" or "none", "confidence": an integer 0-100, "reason": "one sentence"}`;

// Video variant (Muse Spark Open samples up to 96 frames): the whole second around the moment instead of four stills,
// so a hand carrying something away is seen moving, not guessed from two frames.
export const clipPrompt = (name) => `This is a short clip (about one second) from an AI-generated video.

Look at every ${name} in view. During the clip, does any ${name} vanish, appear, or turn into a different object
(or suddenly lose or gain a part) with NO visible physical cause?
These are visible causes and do NOT count: a hand or tool picks it up, puts it down or carries it; it moves out of the
frame; something passes in front of it; the camera moves, pans or zooms; it is cut, poured, folded or eaten in a
visible action; motion blur. Judge only what you can see.
For "between", give the third of the clip where it happens: "A-B" first, "B-C" middle, "C-D" last.

${JSON_ONLY}`;

// Eight frames spanning ±15 frames (frames8_prep.py), the window in which events are scored.
export const frames8Prompt = (name) =>
  `These are eight moments A to H from an AI-generated video, in order, each a small fraction of a
second after the one before.

Look at every ${name} in view. Between two consecutive moments, does any ${name} vanish, appear, or turn into a
different object (or suddenly lose or gain a part) with NO visible physical cause?
These are visible causes and do NOT count: a hand or tool picks it up, puts it down or carries it; it moves out of the
frame; something passes in front of it; the camera moves, pans or zooms; it is cut, poured, folded or eaten in a
visible action; motion blur. Judge only what you can see.
For "between", give the pair of moments where it happens, for example "C-D".

${JSON_ONLY}`.replace('"A-B" or "B-C" or "C-D"', 'a pair such as "C-D"');

// Judge-only test on dev (windows centred on labelled events vs clean moments): the frames8 judge flags 44% of real
// events and 5% of clean moments, i.e. it is cautious, explaining real events away. Two ways to make it less so:
// a strict prior, and an inventory-diff procedure (perceive first, then compare, then look for a cause).
const STRICT = `AI-generated videos often make objects vanish, appear or change for no reason. Be strict: unless you can
clearly see what physically caused a change in these frames, count it as a violation.`;
export const strictPrompt = (name) =>
  frames8Prompt(name).replace('Judge only what you can see.', `Judge only what you can see.\n${STRICT}`);
export const inventoryPrompt = (name) =>
  frames8Prompt(name).replace('Judge only what you can see.', `Judge only what you can see.
Work in steps before answering: (1) list every ${name} you can see in A and in H, with where it is; (2) note each one
that is missing, new, or different in H, and also any that disappears or appears in between and comes back; (3) for
each, look at the moments in between for a visible cause from the list above. Any change without such a cause is a violation.`);

// Count first: 7 of the 9 visible events the sealed v4 test missed were one of several look-alikes (logs, rocks,
// mugs) vanishing or appearing while the judge answered "all remain". Counting every moment makes a one-object change
// explicit. No "motion blur" cause here: blur does not change a count, and it excused a fading arm in v4.
export const countPrompt = (name) =>
  `These are eight moments A to H from an AI-generated video, in order, each a small fraction of a
second after the one before.

First count: how many ${name} can you see in each moment, A to H? Count every one, including small, distant or partly
hidden ones and look-alikes next to each other.
Then compare consecutive moments. A count that goes up or down, or a ${name} that turns into a different object, is a
violation unless you can see what caused it: a hand or tool brings it in, takes it away or covers it; it moves in or out
of the frame; something passes in front of it; the camera moves, pans or zooms; it is cut, poured, folded or eaten in a
visible action. Judge only what you can see.
For "between", give the pair of moments where the first uncaused change happens, for example "C-D". For "kind", give
"appear" if the count goes up there, "vanish" if it goes down, "transform" if it turns into something else.

Reply with only one JSON object and nothing else, with exactly these keys:
{"counts": [eight integers, A to H], "violation": true or false, "kind": "vanish" or "appear" or "transform" or "none",
 "between": a pair such as "C-D" or "none", "confidence": an integer 0-100, "reason": "one sentence"}`;

// The spotter's windows (steps_prep.py): the same count question on a zoomed-in part of each frame, around a box SAM's
// detector count says came or went. Leaving that part is leaving the frame.
export const zoomCountPrompt = (name) => countPrompt(name).replace(
  'each a small fraction of a\nsecond after the one before.',
  'each a small fraction of a\nsecond after the one before. Each image is the same zoomed-in part of the frame.');

export const slowmoPrompt = (name) => clipPrompt(name).replace(
  'This is a short clip (about one second) from an AI-generated video.',
  'This is about one second of an AI-generated video, slowed down about 15 times so you can see every frame.');

const OPEN_RESERVE = 0.05;

const MODES = { // mode: cache version, manifest field, media type, provider, prompt
  box: { version: VERSION, field: 'image', prompt: (m) => prompt(m.name) },
  window: { version: WINDOW_VERSION, field: 'window', prompt: (m) => windowPrompt(m.concept) },
  zoom: { version: 'z1', field: 'zoom', prompt: (m) => zoomPrompt(m.concept) },
  'open-window': { version: 'ow1', field: 'window', open: true,
    prompt: (m) => `${windowPrompt(m.concept)}\n\n${JSON_ONLY}` },
  'open-clip': { version: 'oc1', field: 'video', mime: 'video/mp4', open: true, prompt: (m) => clipPrompt(m.concept) },
  'open-frames8': { version: 'of1', field: 'frames8', open: true, prompt: (m) => frames8Prompt(m.concept) },
  'open-frames8-high': { version: 'of1h', field: 'frames8', open: true, effort: 'high',
    prompt: (m) => frames8Prompt(m.concept) },
  'open-frames8-strict': { version: 'of1s', field: 'frames8', open: true, prompt: (m) => strictPrompt(m.concept) },
  'open-frames8-inventory': { version: 'of1i', field: 'frames8', open: true,
    prompt: (m) => inventoryPrompt(m.concept) },
  'open-frames8-count': { version: 'of1c', field: 'frames8', open: true, prompt: (m) => countPrompt(m.concept) },
  'open-zoom8-count': { version: 'oz1c', field: 'zoom8', open: true, prompt: (m) => zoomCountPrompt(m.concept) },
  // Full-resolution frames: judge misses on dev were tiny objects (an animal on a far road) and small in-hand swaps.
  'open-frames8-full': { version: 'of1f', field: 'frames8full', open: true, detail: 'original',
    prompt: (m) => frames8Prompt(m.concept) },
  // The ±15-frame window as slow motion (slowmo_prep.py, 2 fps): every frame reaches a sparsely sampling video judge.
  'open-slowmo': { version: 'osm1', field: 'slowmo', mime: 'video/mp4', open: true,
    prompt: (m) => slowmoPrompt(m.concept) },
  // Muse Spark 1.3 (Meta API) on the same eight frames, with the schema widened to eight frames' pairs.
  frames8: { version: 's8', field: 'frames8', schema8: true, prompt: (m) => frames8Prompt(m.concept) },
  // The app's witnesses for a Meta API key only (fuse.mjs); not run on the sealed sets.
  'frames8-count': { version: 's8c', field: 'frames8', schema8: true, prompt: (m) => countPrompt(m.concept) },
  'zoom8-count': { version: 's8z', field: 'zoom8', schema8: true, prompt: (m) => zoomCountPrompt(m.concept) },
};

const PAIRS = ['A-B', 'B-C', 'C-D', 'D-E', 'E-F', 'F-G', 'G-H'];
const schema8 = {
  ...schema, properties: { ...schema.properties, between: { type: 'string', enum: [...PAIRS, 'none'] } }
};

// Muse Spark Open's answers are free text: coerce them onto the schema the scorers expect.
const pick = (v, allowed) => (allowed.includes(v) ? v : 'none');
const tidy = (a) => ({
  violation: a.violation === true || a.violation === 'true',
  kind: pick(String(a.kind ?? '').toLowerCase(), ['vanish', 'appear', 'transform']),
  between: pick(String(a.between ?? '').toUpperCase(), PAIRS),
  confidence: Math.max(0, Math.min(100, Math.round(Number(a.confidence) || 0))),
  reason: String(a.reason ?? ''),
  ...(Array.isArray(a.counts) && { counts: a.counts.map(Number) }), // count mode only
});

// Every paid answer is saved in clips/raw under its key, so a rerun costs nothing (and the app's samples need no key).
// Keys carry object names from configs and from Spark's answers, so only safe file-name characters.
export async function cached(key, call) {
  const path = `clips/raw/${key.replace(/[^\w.+ -]+/g, '_')}`;
  try {
    return await readFile(path, 'utf8');
  } catch (e) {
    if (e.code !== 'ENOENT') throw e;
  }
  const text = await call();
  await mkdir('clips/raw', { recursive: true });
  await writeFile(`${path}.tmp`, text);
  await rename(`${path}.tmp`, path); // a run killed mid-write leaves no half reply behind
  return text;
}

// recorded: answer only from the cache (a run recorded earlier) and never call or pay; this is the app's keyless mode.
export async function verify(m, mode, { recorded = false } = {}) {
  const { version, field, mime = 'image/jpeg', open, effort, detail, schema8: wide, prompt: ask } = MODES[mode];
  const text = await cached(`verify-${version}-${m.id}.json`, async () => {
    if (recorded) throw new Error('no recorded answer for this clip: add an API key to .env to check new clips');
    const files = Array.isArray(m[field]) ? m[field] : [m[field]];
    const urls = await Promise.all(files.map(async (f) =>
      `data:${mime};base64,${(await readFile(f)).toString('base64')}`));
    const url = urls[0];
    if (open) {
      pay(OPEN_RESERVE, OPEN_LEDGER);
      const r = await askOpen(mime.startsWith('video') ? [videoPart(url)] : urls.map((u) => imagePart(u, detail)),
        ask(m), effort ? { effort } : undefined);
      pay(openCost(r.usage) - OPEN_RESERVE, { ...OPEN_LEDGER, force: true });
      return JSON.stringify(r);
    }
    pay(RESERVE);
    const r = await askImages(urls, ask(m), wide ? schema8 : schema);
    pay(sparkCost(r.usage) - RESERVE, { force: true }); // settle to the actual cost, even past the stop line
    return JSON.stringify(r);
  });
  const { answer, usage } = JSON.parse(text);
  return { id: m.id, ...tidy(answer), cost: open ? openCost(usage) : sparkCost(usage) };
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [manifestPath, limit, mode = 'box'] = process.argv.slice(2);
  const { field, open } = MODES[mode];
  // Every call reserves money before it is sent, so a run without the key would book reserves for calls that never ran.
  if (open && !process.env.BASETEN_API_KEY) throw new Error(
    'BASETEN_API_KEY is not set: run with node --env-file=.env');
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8')).filter((m) => m[field]) // scan windows have no box
    .slice(0, limit && limit !== 'all' ? Number(limit) : undefined);
  const out = [];
  let next = 0, failed = 0;
  await Promise.all(Array.from({ length: PARALLEL }, async () => {
    while (next < manifest.length) {
      const m = manifest[next++];
      try {
        out.push(await verify(m, mode));
      } catch (e) {
        failed++;
        console.error('FAILED', m.id, e.message.slice(0, 200));
        if (e.message.startsWith('budget stop')) next = manifest.length; // stop every worker
      }
    }
  }));
  await writeFile(manifestPath.replace(/\.json$/, mode === 'box' ? '-verdicts.json' : `-verdicts-${mode}.json`),
    JSON.stringify(out, null, 1));
  const cost = out.reduce((s, v) => s + v.cost, 0);
  console.log(`${out.length} verified, ${failed} failed, $${cost.toFixed(4)} (this run's calls, cached ones included)`);
}
