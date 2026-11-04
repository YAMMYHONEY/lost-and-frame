import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { frameCount, sceneCuts, framePng, normalized } from './video.mjs';

const ffmpeg = process.env.FFMPEG ?? 'ffmpeg';
let found = true;
try {
  execFileSync(ffmpeg, ['-version'], { stdio: 'ignore' });
} catch {
  found = false;
}

test('counts frames, finds the scene cut, and extracts a frame', { skip: !found && 'ffmpeg not found' }, async () => {
  const clip = join(mkdtempSync(join(tmpdir(), 'cut-')), 'cut.mp4');
  // 45 frames of a gray scene, then 45 frames of a blue scene: the cut is frame 45
  execFileSync(ffmpeg, [
    '-v', 'error',
    '-f', 'lavfi', '-i', 'color=c=gray:s=320x180:r=30:d=1.5',
    '-f', 'lavfi', '-i', 'color=c=blue:s=320x180:r=30:d=1.5',
    '-filter_complex', '[0][1]concat=n=2:v=1', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', clip,
  ]);
  assert.equal(await frameCount(clip), 90);
  assert.deepEqual(await sceneCuts(clip), [45]);
  assert.ok((await framePng(clip, 50)).length > 100);
});

test(
  'a clean MP4 is used as is; a rotated phone-style clip is made upright once',
  { skip: !found && 'ffmpeg not found' },
  async () => {
    const dir = mkdtempSync(join(tmpdir(), 'norm-'));
    const clip = join(dir, 'clean.mp4');
    const mov = join(dir, 'phone.mov');
    execFileSync(ffmpeg, [
      '-v', 'error', '-f', 'lavfi', '-i', 'color=c=gray:s=320x180:r=30:d=1',
      '-c:v', 'libx264', '-pix_fmt', 'yuv420p', clip,
    ]);
    execFileSync(ffmpeg, ['-v', 'error', '-display_rotation', '90', '-i', clip, '-c', 'copy', mov]);

    assert.equal(await normalized(clip), clip);
    const out = await normalized(mov);
    assert.equal(out, join(dir, 'phone.norm.mp4'));
    assert.equal(await frameCount(out), 30);
    const png = await framePng(out, 0);
    assert.deepEqual([png.readUInt32BE(16), png.readUInt32BE(20)], [180, 320]); // PNG width and height: now upright
  },
);
