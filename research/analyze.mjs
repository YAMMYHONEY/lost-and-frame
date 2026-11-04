// One clip end to end, for every method on the same task:
//   A: SAM 3.1 tracking per concept, every break in a track is an alert.
//   B: Muse Spark asked the same question about the same objects.
//   C: Lost & Frame: A's breaks minus those explained by scene cuts, the frame edge, hand-offs,
//      occlusion (other tracks, then Spark-listed objects found by SAM in the frame after the break)
//      and frame-by-frame re-detection; tiny or low-confidence tracks are set aside as uncertain.
// Paid calls go through pay(cost, opts); raw replies are cached in clips/raw, so re-analysis is free.
import { stat } from 'node:fs/promises';
import { basename } from 'node:path';
import { parseSamVideo, auditTracks, DEFAULTS } from './permanence.mjs';
import { uploadVideo, segmentVideo, segmentImage, videoCost, IMAGE_COST } from './sam.mjs';
import { cached } from './verify.mjs';
import { askVideo, sparkCost } from '../spark.mjs';
import { normalized, frameCount, fps, frameUrl, sceneCuts } from '../video.mjs';

const SPARK_RESERVE = 0.03; // dollars reserved before a Spark call (5 s clips cost ~$0.005-0.013); settled after
const MAX_OCCLUDERS = 5;

// Method B: the same task Lost & Frame does, limited to the same object names.
const eventsPrompt = (concepts) =>
  `Watch this video closely. Find every moment where one of these objects disappears or appears
without a visible reason: ${concepts.join(', ')}.
Do NOT report an object that leaves through the edge of the frame, goes behind or comes out from behind
something, or changes because of a scene cut. Report only unexplained disappearances or appearances.
For each one give the object (exactly one of the names above), the kind ("vanish" or "appear"), and the time
in seconds from the start of the video when it happens. If there are none, return an empty list.`;

const eventsSchema = (concepts) => ({
  type: 'object', additionalProperties: false, required: ['events'],
  properties: {
    events: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false, required: ['object', 'kind', 'time'],
        properties: {
          object: { type: 'string', enum: concepts },
          kind: { type: 'string', enum: ['vanish', 'appear'] },
          time: { type: 'number' }
        },
      },
    },
  },
});

// Candidate occluders for the occlusion check: things in view that could hide another object.
const objectsPrompt = `List the distinct physical objects and people visible in this video that could hide
another object from view, as short noun phrases (e.g. "person", "table", "box"). At most 8.`;

const objectsSchema = {
  type: 'object', additionalProperties: false, required: ['objects'],
  properties: { objects: { type: 'array', items: { type: 'string' } } },
};

export async function analyzeClip(original, concepts, { pay }) {
  const path = await normalized(original);
  const frames = await frameCount(path), rate = await fps(path), cuts = await sceneCuts(path);
  const key = `${basename(path)}.${(await stat(path)).size}`;
  let fileId;
  const upload = async () => (fileId ??= await uploadVideo(path));
  const track = async (concept) => parseSamVideo(await cached(`${key}.${concept}.txt`, async () => {
    pay(videoCost(frames));
    return segmentVideo(await upload(), concept);
  }));
  const detect = async (concept, f) => parseSamVideo(await cached(`${key}.${concept}.f${f}.txt`, async () => {
    pay(IMAGE_COST);
    return segmentImage(await frameUrl(path, f), concept);
  }));
  // A failed Spark answer is reported, not fatal: B and the occluder list are optional to the audit.
  const ask = async (name, prompt, schema) => {
    try {
      return JSON.parse(await cached(`${key}.spark-${name}.json`, async () => {
        pay(SPARK_RESERVE);
        const r = await askVideo(await upload(), prompt, schema);
        pay(sparkCost(r.usage) - SPARK_RESERVE, { force: true }); // settle to the actual cost, even past the stop line
        return JSON.stringify(r);
      }));
    } catch (e) {
      if (/budget stop/.test(e.message)) throw e;
      return { error: e.message };
    }
  };

  // SAM 3.1 finds plain nouns ("ball", "wood") but often nothing for compound names ("wooden block").
  // Try the name, then its last word, on three frame images and track with the first one SAM finds.
  // Spark's "other names" were dropped: they named different objects (person -> truck, knife -> spoon).
  // Results stay keyed by the given name.
  const probe = [0, Math.floor(frames / 2), frames - 1];
  const seen = async (name) => {
    for (const f of probe) if ((await detect(name, f)).frames.get(0)?.length) return true;
    return false;
  };
  const samName = {};
  for (const concept of concepts) {
    const tries = [...new Set([concept, concept.trim().split(/\s+/).pop()])];
    let name;
    for (const n of tries) if (!name && await seen(n)) name = n;
    samName[concept] = name ?? concept;
  }

  const passes = {};
  for (const concept of concepts) passes[concept] = await track(samName[concept]);

  const b = await ask(`events.${concepts.join('+')}`, eventsPrompt(concepts), eventsSchema(concepts));
  const opts = (concept) => ({
    frameCount: frames, cuts,
    occluders: concepts.filter((c) => c !== concept).map((c) => passes[c])
  });

  // Re-detect each suspect frame by frame; candidates still unexplained get the occluder search.
  const checks = {}, open = {};
  for (const concept of concepts) {
    const check = checks[concept] = { frames: new Map() };
    for (const e of auditTracks(passes[concept], opts(concept)).suspects) {
      for (let k = 1; k <= DEFAULTS.lookahead; k++) {
        const f = e.kind === 'vanish' ? e.frame + k : e.frame - k;
        if (f < 0 || f >= frames || check.frames.has(f)) continue;
        const img = await detect(samName[concept], f);
        const p = passes[concept];
        if (img.width && (img.width !== p.width || img.height !== p.height)) {
          console.warn(`${path} frame ${f}: image is ${img.width}x${img.height}, video is ${p.width}x${p.height} (rotation?)`);
        }
        check.frames.set(f, img.frames.get(0) ?? []);
      }
    }
    open[concept] = auditTracks(passes[concept], { ...opts(concept), check }).suspects;
  }

  let listed = [];
  if (Object.values(open).some((s) => s.length)) {
    const answer = (await ask('objects', objectsPrompt, objectsSchema)).answer;
    const known = new Set([...concepts, ...Object.values(samName)].map((c) => c.toLowerCase()));
    listed = [...new Set((answer?.objects ?? []).map((o) => o.toLowerCase().trim()))]
      .filter((o) => o && !known.has(o))
      .slice(0, MAX_OCCLUDERS);
  }

  const results = [];
  for (const concept of concepts) {
    const extra = { frames: new Map(), sparse: true }; // one frame per break: no history for the occlusion check
    for (const e of open[concept]) {
      const g = e.kind === 'vanish' ? e.frame + 1 : e.frame - 1;
      if (g < 0 || g >= frames || extra.frames.has(g)) continue;
      const found = [];
      for (const name of listed) found.push(...((await detect(name, g)).frames.get(0) ?? []));
      extra.frames.set(g, found);
    }
    const o = opts(concept);
    const r = auditTracks(passes[concept], { ...o, occluders: [...o.occluders, extra], check: checks[concept] });
    const brief = ({ id, kind, frame, box, reason }) => ({ id, kind, frame, box, reason });
    results.push({
      concept,
      A: r.events.map(brief),
      B: b.error ? { error: b.error } : b.answer.events.filter((x) => x.object === concept)
        .map(({ kind, time }) => ({ kind, time, frame: Math.round(time * rate) })),
      C: r.suspects.map(brief),
      uncertain: r.uncertain.map(brief),
      reasons: r.summary,
    });
  }
  return { source: path, frames, fps: rate, cuts, samName, occluderNames: listed, sparkUsage: b.usage, results };
}
