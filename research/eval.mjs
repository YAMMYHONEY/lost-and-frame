// Run methods A, B and C (analyze.mjs) on every clip in a list and, where the list has labels, score each
// method against them (score.mjs). Prints the list's sha256 so a sealed label file can be checked.
// Usage: node --env-file=.env research/eval.mjs <clips/list.json>
// The list format is in research/labels.example.json; "events" may be left out for a plain run.
// Writes <list>-eval.json, which research/eval_report.mjs turns into a review page.
// Paid calls stop before this run passes CAP or the running total passes the ledger's stop line.
// Needs ffmpeg/ffprobe (or FFMPEG/FFPROBE env paths).
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { analyzeClip } from './analyze.mjs';
import { pay, spentThisRun } from '../budget.mjs';
import { scoreClip, verdict } from './score.mjs';

const CAP = 6; // dollars for one run
const METHODS = ['A', 'B', 'C'];

const listPath = process.argv[2];
if (!listPath) throw new Error('usage: node --env-file=.env research/eval.mjs <clips/list.json>');
const raw = await readFile(listPath);
console.log(`list ${listPath} sha256 ${createHash('sha256').update(raw).digest('hex')}`);
const clips = JSON.parse(raw);
const total = Object.fromEntries(METHODS.map((m) => [m, { hit: 0, miss: 0, falseAlarm: 0, burden: 0 }]));
const rows = [], report = [];
let labeled = 0;
try {
  for (const { file, kind, concepts, events = [] } of clips) {
    const r = await analyzeClip(`clips/${file}`, concepts, { pay: (cost, o) => pay(cost, { runCap: CAP, ...o }) });
    report.push({ file, kind, ...r });
    const truth = events.map((e) => ({ concept: e.concept, kind: e.kind, frame: Math.round(e.time * r.fps) }));
    labeled += truth.length;
    const row = { file, labeled: truth.length };
    for (const m of METHODS) {
      // A Spark error counts as no answer (all labeled events missed), and is marked in the row.
      const failed = r.results.some((x) => x[m].error);
      const pred = r.results.flatMap((x) =>
        (x[m].error ? [] : x[m]).map((e) => ({ concept: x.concept, kind: e.kind, frame: e.frame })));
      const s = scoreClip(pred, truth, { frameCount: r.frames });
      for (const k of ['hit', 'miss', 'falseAlarm', 'burden']) total[m][k] += s[k];
      row[m] = failed ? 'error' : `${s.hit} hit / ${s.falseAlarm} false`;
    }
    rows.push(row);
  }
} finally {
  const n = rows.length || 1;
  const summary = Object.fromEntries(
    METHODS.map((m) => [m, { ...total[m], burden: +(total[m].burden / n).toFixed(3) }]));
  const result = {
    list: listPath, clips: rows.length, labeled, summary, verdict: verdict(summary, labeled), spent: spentThisRun(),
    rows, report
  };
  await writeFile(listPath.replace(/(\.json)?$/, '-eval.json'), JSON.stringify(result, null, 2));
  console.table(rows);
  console.table(summary);
  console.log('verdict', result.verdict, `spent about $${result.spent.toFixed(4)}`);
}
