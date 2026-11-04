// Muse Spark Open, Meta's open-weights model, on Baseten. It uses OpenAI-compatible Chat Completions with the
// hackathon key (BASETEN_API_KEY), which is valid only for this model. It takes images and short videos (up to 96
// sampled frames) and always reasons. It has no structured outputs, so the prompt asks for one JSON object and the
// answer is parsed from the text. Spending goes to its own ledger (hackathon credits, $100), separate from the Meta
// API ledger.
const BASE = 'https://inference.baseten.co/v1';
export const OPEN_MODEL = 'meta-models/muse-spark-open-1026';
export const OPEN_LEDGER = { file: 'clips/ledger-baseten.json', stop: 100 };

// Baseten list price on 2026-10-23: input $1.25/M, output $4.25/M. Reasoning tokens are billed as completion tokens.
export const openCost = ({ prompt_tokens = 0, completion_tokens = 0 }) =>
  prompt_tokens * 1.25e-6 + completion_tokens * 4.25e-6;

// detail: 'auto' (default), 'low', 'high' or 'original' (Baseten vision docs).
export const imagePart = (url, detail) => ({ type: 'image_url', image_url: detail ? { url, detail } : { url } });
export const videoPart = (url) => ({ type: 'video_url', video_url: { url } });

export async function askOpen(parts, prompt, { effort = 'low', maxTokens = 6000 } = {}) {
  const res = await fetch(`${BASE}/chat/completions`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${process.env.BASETEN_API_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: OPEN_MODEL,
      reasoning_effort: effort,
      max_tokens: maxTokens,
      messages: [{ role: 'user', content: [{ type: 'text', text: prompt }, ...parts] }],
    }),
  });
  if (!res.ok) throw new Error(`spark-open failed ${res.status}: ${(await res.text()).slice(0, 500)}`);

  const body = await res.json();
  const text = body.choices?.[0]?.message?.content ?? '';
  const json = text.match(/\{[\s\S]*\}/)?.[0]; // the object is usually on the last line; take the outermost braces
  if (!json) throw new Error(`spark-open returned no JSON: ${text.slice(0, 300)}`);
  return { answer: JSON.parse(json), usage: body.usage ?? {} };
}
