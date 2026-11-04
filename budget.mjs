// The running spend across all runs, kept in a ledger file. Each paid call reserves its cost first (so a failed call
// still counts), and a call is refused once the total would pass the stop line.
import { readFileSync, writeFileSync } from 'node:fs';

export const STOP = 115; // dollars: the Meta API ledger's stop line, raised from 105 on 11-04 for H17 (user approved)
let run = 0;

// force: record money already spent (for example, a call that cost more than it reserved) without refusing.
// stop, file: a separate ledger with its own stop line (the Baseten hackathon credits, see spark_open.mjs).
export function pay(cost, {
  runCap = Infinity,
  force = false,
  stop = STOP,
  file = process.env.LEDGER ?? 'clips/ledger.json',
} = {}) {
  let total = 0;
  try {
    total = JSON.parse(readFileSync(file, 'utf8')).total;
  } catch (e) {
    if (e.code !== 'ENOENT') throw e;
  }
  if (!force && total + cost > stop) {
    throw new Error(`budget stop: total $${total.toFixed(4)} + $${cost.toFixed(4)} would pass the $${stop} stop line`);
  }
  if (!force && run + cost > runCap) {
    throw new Error(`budget stop: this run $${run.toFixed(4)} + $${cost.toFixed(4)} would pass its $${runCap} cap`);
  }
  run += cost;
  writeFileSync(file, JSON.stringify({ total: total + cost }));
}

export const spentThisRun = () => run;
