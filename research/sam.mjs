// Meta Model API client for SAM 3.1 video tracking and image detection. Reads MODEL_API_KEY
// from the environment (run with `node --env-file=.env`); the key is never logged.
import { readFile } from 'node:fs/promises';
import { basename } from 'node:path';

const BASE = 'https://api.meta.ai/v1';
const auth = () => ({ Authorization: `Bearer ${process.env.MODEL_API_KEY}` });

// Published prices: video $0.20 per 1k frames (min 50 frames per request), images $2.50 per 1k.
export const videoCost = (frames) => Math.max(50, frames) * 0.2 / 1000;
export const IMAGE_COST = 2.5 / 1000;

export async function uploadVideo(path) {
  const type = path.toLowerCase().endsWith('.mov') ? 'video/quicktime' : 'video/mp4';
  const form = new FormData();
  form.append('purpose', 'user_data');
  form.append('file', new Blob([await readFile(path)], { type }), basename(path));
  const res = await fetch(`${BASE}/files`, { method: 'POST', headers: auth(), body: form });
  if (!res.ok) throw new Error(`upload failed ${res.status}: ${await res.text()}`);
  return (await res.json()).id;
}

// One concept per request: SAM 3.1 returns nothing for lists like "cat, cup".
async function segment(prompt, media, metadata) {
  const res = await fetch(`${BASE}/responses`, {
    method: 'POST',
    headers: { ...auth(), 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'sam-3.1',
      stream: true,
      metadata: { ...metadata, include_confidence: 'true' },
      input: [{ type: 'message', role: 'user', content: [{ type: 'input_text', text: prompt }, media] }],
    }),
  });
  if (!res.ok) throw new Error(`sam failed ${res.status}: ${await res.text()}`);
  return readSse(res.body);
}

// prompt: a noun phrase ("cat"). Forward only: the API default ("both") would also track backward.
export const segmentVideo = (fileId, prompt) =>
  segment(prompt, { type: 'input_video', file_id: fileId }, { propagation_direction: 'forward' });

// imageUrl: a public URL or a base64 data URL. No tracker memory, so it is independent of any video pass.
export const segmentImage = (imageUrl, prompt) => segment(prompt, { type: 'input_image', image_url: imageUrl });

// Concatenates response.output_text.delta events from a Responses API event stream. A stream that
// ends without response.completed (dropped connection) throws, so a cut-off reply is never cached.
export async function readSse(body) {
  const dec = new TextDecoder();
  let text = '', buf = '', done = false;
  for await (const chunk of body) {
    buf += dec.decode(chunk, { stream: true }).replace(/\r/g, '');
    for (let i; (i = buf.indexOf('\n\n')) >= 0; buf = buf.slice(i + 2)) {
      const data = buf.slice(0, i).split('\n')
        .filter((l) => l.startsWith('data:'))
        .map((l) => l.slice(5).trim())
        .join('\n');
      if (!data || data === '[DONE]') continue;
      const ev = JSON.parse(data);
      if (ev.type === 'response.output_text.delta') text += ev.delta;
      else if (ev.type === 'response.completed') done = true;
      else if (ev.type === 'error' || ev.type === 'response.failed') throw new Error(`stream error: ${data}`);
    }
  }
  if (!done) throw new Error('stream ended before response.completed');
  return text;
}
