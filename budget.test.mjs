import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { pay, STOP, spentThisRun } from './budget.mjs';

test('the ledger keeps a running total and refuses calls past the stop line or the run cap', () => {
  const file = join(mkdtempSync(join(tmpdir(), 'ledger-')), 'ledger.json');
  pay(0.5, { file });
  pay(0.25, { file });
  assert.equal(JSON.parse(readFileSync(file, 'utf8')).total, 0.75);
  writeFileSync(file, JSON.stringify({ total: STOP - 0.01 }));
  assert.throws(() => pay(0.02, { file }), /stop line/);
  assert.equal(JSON.parse(readFileSync(file, 'utf8')).total, STOP - 0.01); // refused call is not recorded
  assert.throws(() => pay(0.001, { file, runCap: spentThisRun() }), /run/);
  pay(0.02, { file, force: true }); // money already spent is recorded even past the stop line
  assert.ok(JSON.parse(readFileSync(file, 'utf8')).total > STOP);
});
