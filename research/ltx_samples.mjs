// The app's sample clips, generated with LTX-2.5 through its official Hugging Face Space (Gradio REST API), with the
// caller's HF_TOKEN for the ZeroGPU quota. License: Lightricks, LTX-2.x Community License ("Licensor claims no rights
// in the Output"); outputs must be disclosed as machine generated. The scenes are ones generators often get wrong:
// look-alikes and hands (RESEARCH.md, 10-31: 72% of permanence glitches happen among several look-alikes).
// Fixed seeds, 5 s (the Space's maximum), 24 fps.
// Usage: node --env-file=.env research/ltx_samples.mjs   -> samples/<name>.mp4, samples/ltx-prompts.json
import { mkdir, writeFile } from 'node:fs/promises';

const SPACE = 'https://lightricks-ltx-2-5.hf.space/gradio_api';
const SCENES = [
  ['cups', 7, 'Close-up of two hands shuffling three red plastic cups on a wooden table, a small white ball hidden under one cup. Static camera, bright natural light.'],
  ['oranges', 11, 'A person juggling four oranges in a bright kitchen. Static camera.'],
  ['candies', 13, 'Close-up of hands sorting colorful round candies into three small white bowls on a table. Static camera.'],
  ['carrot', 17, 'A chef slicing a carrot into round slices on a wooden cutting board, close-up. Static camera.'],
];
const auth = { Authorization: `Bearer ${process.env.HF_TOKEN}` };

async function run(prompt, seed) {
  // Inputs in order: prompt, image, height, width, seconds, seed, decoder, auto length, randomize seed, enhance prompt.
  const call = await fetch(`${SPACE}/call/run`, {
    method: 'POST',
    headers: { ...auth, 'Content-Type': 'application/json' },
    body: JSON.stringify({ data: [prompt, null, 576, 1024, 5, seed, 'conv', false, false, true] })
  });
  if (!call.ok) throw new Error(`call failed ${call.status}: ${(await call.text()).slice(0, 200)}`);
  const { event_id: id } = await call.json();
  const stream = await (await fetch(`${SPACE}/call/run/${id}`, { headers: auth })).text();
  const done = stream.split('\n\n').find((e) => e.startsWith('event: complete'));
  if (!done) throw new Error(`no result: ${stream.slice(-300)}`);
  return JSON.parse(done.slice(done.indexOf('data:') + 5));
}

if (!process.env.HF_TOKEN) throw new Error('HF_TOKEN is not set: run with node --env-file=.env');
await mkdir('samples', { recursive: true });
const log = [];
for (const [name, seed, prompt] of SCENES) {
  const [video, sent, seedUsed, length] = await run(prompt, seed);
  const file = video.video ?? video; // a Video component returns { video: FileData, subtitles }
  const res = await fetch(file.url, { headers: auth });
  if (!res.ok) throw new Error(`download failed ${res.status} for ${name}`);
  await writeFile(`samples/${name}.mp4`, Buffer.from(await res.arrayBuffer()));
  log.push({
    file: `samples/${name}.mp4`, model: 'LTX-2.5 (Lightricks/LTX-2.5 Space)',
    date: new Date().toISOString().slice(0, 10), prompt, prompt_sent: sent, seed: seedUsed, length
  });
  console.log(name, 'done', length);
}
await writeFile('samples/ltx-prompts.json', JSON.stringify(log, null, 1));
