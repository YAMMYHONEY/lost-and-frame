import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { mkdir, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { centres, SHIFT, alerts, moments, checkClip } from './fuse.mjs';

const ffmpeg = process.env.FFMPEG ?? 'ffmpeg';
let found = true;
try {
  execFileSync(ffmpeg, ['-version'], { stdio: 'ignore' });
} catch {
  found = false;
}

test('windows and pair positions match the research code (tile_prep.py, eval_tile.py)', () => {
  assert.deepEqual(centres(31), [15]);
  assert.deepEqual(centres(64), [15, 31, 47, 48]);
  assert.deepEqual(centres(116), [15, 31, 47, 63, 79, 95, 100]);
  assert.deepEqual(centres(187), [15, 31, 47, 63, 79, 95, 111, 127, 143, 159, 171]);
  assert.deepEqual([SHIFT['A-B'], SHIFT['D-E'], SHIFT['G-H']], [-13, 0, 13]);
});

test('alerts sit at the named pair, transform alerts both kinds, nearby same-kind alerts are one moment', () => {
  const w = (frame) => ({ frame });
  const v = (violation, kind, between, confidence) => ({ violation, kind, between, confidence, reason: '' });
  const judged = [
    { w: w(31), witness: 'J', v: v(true, 'vanish', 'D-E', 95) }, // vanish at 31
    { w: w(31), witness: 'C', v: v(true, 'vanish', 'E-F', 90) }, // vanish at 36: the same moment
    { w: w(47), witness: 'J', v: v(true, 'appear', 'B-C', 80) }, // appear at 38, only when sensitive
    { w: w(79), witness: 'C', v: v(true, 'transform', 'none', 92) }, // both kinds at the centre
    { w: w(95), witness: 'J', v: v(false, 'none', 'none', 97) }, // a confident "no"
  ];

  const strict = moments(alerts(judged, 0.9));
  assert.deepEqual(
    strict.map((m) => [m.kind, m.frame, m.by.map((a) => a.witness).join('')]),
    [['vanish', 31, 'JC'], ['appear', 79, 'C'], ['vanish', 79, 'C']],
  );

  const sensitive = moments(alerts(judged, 0.5));
  assert.deepEqual(
    sensitive.map((m) => [m.kind, m.frame]),
    [['vanish', 31], ['appear', 38], ['appear', 79], ['vanish', 79]],
  );
});

test(
  'a clip checks end to end from recorded answers, and says so when an answer is missing',
  { skip: !found && 'ffmpeg not found' },
  async () => {
    const dir = mkdtempSync(join(tmpdir(), 'fuse-'));
    const clip = join(dir, 'gray.mp4');
    const key = `test${process.pid}`;
    execFileSync(ffmpeg, [
      '-v', 'error', '-f', 'lavfi', '-i', 'color=c=gray:s=320x180:r=30',
      '-frames:v', '64', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', clip,
    ]);

    // Saved answers for every window: "no" everywhere, except the judge sees the ball vanish in the window at 31.
    const answer = (a) => JSON.stringify({ answer: a, usage: { prompt_tokens: 0, completion_tokens: 0 } });
    const no = { violation: false, kind: 'none', between: 'none', confidence: 96, reason: 'nothing changes' };
    const cache = (v, frame) => `clips/raw/verify-${v}-app-${key}-ball-t${String(frame).padStart(4, '0')}.json`;
    await mkdir('clips/raw', { recursive: true });
    const files = [];
    for (const f of [15, 31, 47, 48]) {
      for (const v of ['of1', 'of1c']) {
        files.push(cache(v, f));
        await writeFile(cache(v, f), answer(f === 31 && v === 'of1'
          ? { violation: true, kind: 'vanish', between: 'C-D', confidence: 93, reason: 'the ball is gone' }
          : no));
      }
    }

    try {
      const r = await checkClip(clip, ['ball'], { dir: join(dir, 'out'), key, backend: 'recorded' });
      assert.equal(r.frames, 64);
      assert.deepEqual(
        r.results[0].moments.map((m) => [m.kind, m.frame, m.by[0].v.reason]),
        [['vanish', 26, 'the ball is gone']],
      );

      await rm(cache('of1c', 48));
      await assert.rejects(
        checkClip(clip, ['ball'], { dir: join(dir, 'out'), key, backend: 'recorded' }),
        /no recorded answer/,
      );
    } finally {
      await Promise.all(files.map((f) => rm(f, { force: true })));
      await rm(dir, { recursive: true, force: true });
    }
  },
);
