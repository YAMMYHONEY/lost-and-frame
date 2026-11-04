import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { listZip, extract, joinReaders } from './zipfetch.mjs';

let zip, files;
try {
  const dir = mkdtempSync(join(tmpdir(), 'zip-'));
  files = {
    'a.txt': 'hello '.repeat(500),
    'b.bin': Buffer.from(Array.from({ length: 3000 }, (_, i) => (i * 7) % 256))
  };
  for (const [n, c] of Object.entries(files)) writeFileSync(join(dir, n), c);
  zip = join(dir, 'test.zip');
  execFileSync('powershell.exe', ['-NoProfile', '-Command',
    `Compress-Archive -Path '${join(dir, 'a.txt')}','${join(dir, 'b.bin')}' -DestinationPath '${zip}'`]);
} catch {
  zip = null;
}

test('lists a zip and extracts entries through byte-range reads only',
  { skip: !zip && 'cannot build a test zip here' }, async () => {
    const buf = readFileSync(zip);
    const ranges = [];
    const read = async (s, e) => {
      ranges.push(e - s + 1);
      return buf.subarray(s, e + 1);
    };
    const entries = await listZip(read, buf.length);
    assert.deepEqual([...entries.keys()].sort(), ['a.txt', 'b.bin']);
    assert.equal((await extract(read, entries.get('a.txt'))).toString(), files['a.txt']);
    assert.deepEqual(await extract(read, entries.get('b.bin')), files['b.bin']);
    assert.ok(ranges.every((n) => n <= buf.length));
  });

test('reads a zip cut into byte parts as one archive', { skip: !zip && 'cannot build a test zip here' }, async () => {
  const buf = readFileSync(zip);
  const cuts = [0, 700, 1900, buf.length]; // parts of uneven size; entries and the central directory straddle them
  const parts = cuts.slice(1).map((end, i) => {
    const part = buf.subarray(cuts[i], end);
    return { size: part.length, read: async (s, e) => part.subarray(s, e + 1) };
  });
  const { read, size } = joinReaders(parts);
  assert.equal(size, buf.length);
  const entries = await listZip(read, size);
  assert.equal((await extract(read, entries.get('a.txt'))).toString(), files['a.txt']);
  assert.deepEqual(await extract(read, entries.get('b.bin')), files['b.bin']);
});
