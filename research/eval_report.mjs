// Review page for the first method (SAM tracking, methods A, B and C of eval.mjs): per clip the alert counts, and for
// every C suspect the frames just before and after with the box.
// Usage: node research/eval_report.mjs <clips/list-eval.json>   → writes <result>.html next to it (frames in a folder).
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { esc, page } from '../report.mjs';
import { framePng } from '../video.mjs';

const REASON = {
  suspect: 'unexplained', edge: 'crossed the frame edge', occluded: 'hidden behind another object',
  cut: 'scene cut', handoff: 'tracker switched IDs', retracked: 'still there when re-detected',
  tiny: 'too small to judge', lowconf: 'SAM unsure what this object is',
};

// report: [{ file, source, fps, samName, results }] as analyzeClip/eval.mjs produce. Frames are written
// to dir and linked as href/<name>. Returns the page's body HTML.
async function renderReport(report, dir, href) {
  await mkdir(dir, { recursive: true });
  const rows = [], cards = [];
  for (const clip of report) {
    for (const r of clip.results) {
      const b = r.B.error ? 'error' : r.B.length;
      const as = clip.samName?.[r.concept] && clip.samName[r.concept] !== r.concept
        ? ` <small>(SAM: “${esc(clip.samName[r.concept])}”)</small>` : '';
      rows.push(`<tr><td>${esc(clip.file)}</td><td>${esc(r.concept)}${as}</td><td>${r.A.length}</td><td>${b}</td><td><b>${r.C.length}</b></td><td>${r.uncertain.length}</td></tr>`);
      // Listed once per object, under its first card; uncertain breaks are not "explained".
      let explained = r.A.filter((x) => !['suspect', 'tiny', 'lowconf'].includes(x.reason))
        .map((x) => `${x.kind} @ ${x.frame}: ${REASON[x.reason] ?? x.reason}`);
      for (const e of r.C) {
        const pair = e.kind === 'vanish' ? [e.frame, e.frame + 1] : [e.frame - 1, e.frame];
        const imgs = [];
        for (const f of pair) {
          const name = `${clip.file}.${r.concept}.${e.kind}.${e.frame}.f${f}.png`.replace(/[^\w.-]+/g, '_');
          await writeFile(`${dir}/${name}`,
            await framePng(clip.source ?? `clips/${clip.file}`, f, { width: 480, box: e.box }));
          imgs.push(`<figure><img src="${href}/${name}" alt="frame ${f}"><figcaption>frame ${f} · ${(f / clip.fps).toFixed(2)} s</figcaption></figure>`);
        }
        cards.push(`<section class="card"><h3>${esc(clip.file)} · ${esc(r.concept)} ${e.kind === 'vanish' ? 'vanishes' : 'appears'} at ${(e.frame / clip.fps).toFixed(2)} s</h3>
<p class="why">Not the frame edge · no scene cut · nothing in front of it · ${e.kind === 'vanish' ? 'not found again in the next frames' : 'not found in the frames before'}</p>
<div class="pair">${imgs.join('')}</div>
${explained.length ? `<details><summary>${explained.length} other break(s) explained</summary><ul>${explained.map((x) => `<li>${esc(x)}</li>`).join('')}</ul></details>` : ''}</section>`);
        explained = [];
      }
    }
  }
  return `<h1>Lost &amp; Frame review</h1>
<p class="sub">A = every break in a SAM track · B = Muse Spark asked the same question · C = Lost &amp; Frame (what is left to review)</p>
<table><thead><tr><th>Clip</th><th>Object</th><th>A</th><th>B</th><th>C</th><th>Uncertain</th></tr></thead><tbody>${rows.join('')}</tbody></table>
${cards.length ? cards.join('\n') : '<p>Nothing to review: every break was explained.</p>'}`;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const input = process.argv[2];
  if (!input) throw new Error('usage: node research/eval_report.mjs <clips/list-eval.json>');
  const dir = input.replace(/(\.json)?$/, '-frames');
  const out = input.replace(/(\.json)?$/, '.html');
  const { report } = JSON.parse(await readFile(input, 'utf8'));
  await writeFile(out, page('Lost & Frame review', await renderReport(report, dir, dir.split(/[\\/]/).pop())));
  console.log(`wrote ${out}`);
}
