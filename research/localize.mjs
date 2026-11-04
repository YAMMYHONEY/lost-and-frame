// Muse Spark points at an object SAM cannot ground by name: four frames of the clip, a box (or "not visible") per
// frame. The boxes become SAM 3.1 visual exemplars (trace.py --boxes); SAM must then confirm by tracking there.
// Usage: node --env-file=.env research/localize.mjs <manifest.json>   -> <manifest>-boxes.json
import { readFile, writeFile } from 'node:fs/promises';
import { askImages, sparkCost } from '../spark.mjs';
import { pay } from '../budget.mjs';
import { cached } from './verify.mjs';

const RESERVE = 0.05;
const VERSION = 'v1';
const LETTERS = ['A', 'B', 'C', 'D'];

const prompt = (name) => `These are four frames A, B, C, D (in order) from one video. For each frame, find the ${name}.
If a ${name} is visible, give its bounding box as [x1, y1, x2, y2] in normalized coordinates (0 = left/top edge,
1 = right/bottom edge) and how sure you are (0-100). If there are several, give the clearest one. If no ${name} is
visible in that frame, set visible to false and box to [0, 0, 0, 0].`;

const schema = {
  type: 'object', additionalProperties: false, required: ['frames'],
  properties: {
    frames: {
      type: 'array', minItems: 4, maxItems: 4,
      items: {
        type: 'object', additionalProperties: false, required: ['frame', 'visible', 'box', 'confidence'],
        properties: {
          frame: { type: 'string', enum: LETTERS },
          visible: { type: 'boolean' },
          box: { type: 'array', minItems: 4, maxItems: 4, items: { type: 'number' } },
          confidence: { type: 'integer', minimum: 0, maximum: 100 },
        },
      },
    },
  },
};

const manifestPath = process.argv[2];
const out = [];
for (const m of JSON.parse(await readFile(manifestPath, 'utf8'))) {
  const text = await cached(`localize-${VERSION}-${m.id}.json`, async () => {
    const urls = await Promise.all(m.images.map(async (i) =>
      `data:image/jpeg;base64,${(await readFile(i.image)).toString('base64')}`));
    pay(RESERVE);
    const r = await askImages(urls, prompt(m.name), schema);
    pay(sparkCost(r.usage) - RESERVE, { force: true });
    return JSON.stringify(r);
  });
  const { answer, usage } = JSON.parse(text);
  const boxes = answer.frames.map((a) => ({ ...a, frame: m.images[LETTERS.indexOf(a.frame)].frame }))
    .filter((a) => a.visible && a.box[2] > a.box[0] && a.box[3] > a.box[1] && a.box.every((v) => v >= 0 && v <= 1));
  out.push({ id: m.id, key: m.key, file: m.file, name: m.name, boxes, cost: sparkCost(usage) });
  console.log(m.id, m.name, boxes.map((b) => `f${b.frame}:${b.confidence}`).join(' ') || 'not visible');
}
await writeFile(manifestPath.replace(/\.json$/, '-boxes.json'), JSON.stringify(out, null, 1));
console.log(`$${out.reduce((s, o) => s + o.cost, 0).toFixed(4)} for ${out.length} objects`);
