// Muse Spark 1.3 on the Meta Model API (Responses API). It sends a prompt with images or an uploaded video and gets
// the answer in a JSON schema. It uses the same key and base URL as SAM. The "contributor" variant, whose data trains
// Meta's models, is never used.
const BASE = 'https://api.meta.ai/v1';
const MODEL = 'muse-spark-1.3';

// Published prices per token: input $1.25/M, output $4.25/M (reasoning tokens count as output).
export const sparkCost = ({ input_tokens = 0, output_tokens = 0 }) => input_tokens * 1.25e-6 + output_tokens * 4.25e-6;

export const askVideo = (fileId, prompt, schema) => ask([{ type: 'input_file', file_id: fileId }], prompt, schema);

// imageUrls: data: URLs or https URLs, sent in order after the prompt.
export const askImages = (imageUrls, prompt, schema) =>
  ask(imageUrls.map((image_url) => ({ type: 'input_image', image_url })), prompt, schema);

async function ask(media, prompt, schema) {
  const res = await fetch(`${BASE}/responses`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${process.env.MODEL_API_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: MODEL,
      input: [{ type: 'message', role: 'user', content: [{ type: 'input_text', text: prompt }, ...media] }],
      text: { format: { type: 'json_schema', name: 'answer', schema, strict: true } },
    }),
  });
  if (!res.ok) throw new Error(`spark failed ${res.status}: ${await res.text()}`);

  const body = await res.json();
  const text = body.output?.flatMap((o) => o.content ?? []).find((c) => c.type === 'output_text')?.text;
  if (text === undefined) throw new Error(`spark returned no text: ${JSON.stringify(body).slice(0, 500)}`);
  return { answer: JSON.parse(text), usage: body.usage ?? {} };
}
