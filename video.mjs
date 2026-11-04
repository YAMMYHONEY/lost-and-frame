// ffmpeg and ffprobe helpers. The programs come from FFMPEG and FFPROBE, or else from PATH. Frame numbers are in
// decode order, which matched SAM's numbering on a constant-frame-rate test clip.
import { execFile } from 'node:child_process';
import { stat } from 'node:fs/promises';
import { promisify } from 'node:util';

const run = promisify(execFile);
const FFMPEG = () => process.env.FFMPEG ?? 'ffmpeg';
const FFPROBE = () => process.env.FFPROBE ?? 'ffprobe';

export async function frameCount(path) {
  const { stdout } = await run(FFPROBE(), [
    '-v', 'error', '-select_streams', 'v:0', '-count_packets',
    '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', path,
  ]);
  const n = parseInt(stdout, 10);
  if (!(n > 0)) throw new Error(`could not count frames of ${path}`);
  return n;
}

// A rotation flag or a variable frame rate (common on phones) would make SAM's frames and ffmpeg's disagree, and
// Spark takes MP4 only. Such clips are re-encoded once to an upright, constant-rate H.264 MP4 next to the original
// (<name>.norm.mp4). Clips that are already fine are used as they are.
export async function normalized(path) {
  const { stdout } = await run(FFPROBE(), [
    '-v', 'error', '-select_streams', 'v:0',
    '-show_entries', 'stream=codec_name,r_frame_rate,avg_frame_rate:stream_side_data=rotation', '-of', 'json', path,
  ]);
  const s = JSON.parse(stdout).streams?.[0] ?? {};
  const rotated = (s.side_data_list ?? []).some((d) => d.rotation && d.rotation % 360 !== 0);
  if (/\.mp4$/i.test(path) && s.codec_name === 'h264' && s.r_frame_rate === s.avg_frame_rate && !rotated) return path;

  const out = path.replace(/\.[^.\\/]+$/, '.norm.mp4');
  // Reuse an earlier conversion unless the original has changed since.
  try {
    if ((await stat(out)).mtimeMs >= (await stat(path)).mtimeMs) return out;
  } catch {}
  const rate = /^[1-9]\d*\/[1-9]\d*$/.test(s.avg_frame_rate ?? '') ? s.avg_frame_rate : '30';
  await run(FFMPEG(), [
    '-v', 'error', '-y', '-i', path, '-an', '-fps_mode', 'cfr', '-r', rate,
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', out,
  ]);
  return out;
}

export async function fps(path) {
  const { stdout } = await run(FFPROBE(), [
    '-v', 'error', '-select_streams', 'v:0',
    '-show_entries', 'stream=avg_frame_rate', '-of', 'csv=p=0', path,
  ]);
  const [n, d] = stdout.trim().split('/').map(Number);
  if (!(n > 0 && d > 0)) throw new Error(`could not read frame rate of ${path}`);
  return n / d;
}

// shortcut: decode order may differ from SAM's numbering on variable-frame-rate phone clips; check there.
// box: [x1, y1, x2, y2] in source pixels, drawn in yellow (visible on red objects too) before scaling.
export async function framePng(path, f, { width, box } = {}) {
  const vf = `select=eq(n\\,${f})` +
    (box ? `,drawbox=x=${box[0]}:y=${box[1]}:w=${box[2] - box[0]}:h=${box[3] - box[1]}:color=yellow:t=4` : '') +
    (width ? `,scale=${width}:-2` : '');
  const { stdout } = await run(FFMPEG(), [
    '-v', 'error', '-i', path, '-vf', vf, '-frames:v', '1',
    '-f', 'image2pipe', '-c:v', 'png', '-',
  ], { encoding: 'buffer', maxBuffer: 256 << 20 });
  return stdout;
}

export const frameUrl = async (path, f) => `data:image/png;base64,${(await framePng(path, f)).toString('base64')}`;

// A judge window's frames (research/frames8_prep.py): the centre frame plus each offset, at 640x360, lettered A, B, …
// in white at the top left, and written as <prefix>-0.jpg, <prefix>-1.jpg, … in one pass. select keeps decode order,
// so the k-th selected frame gets the k-th letter. Offsets must be distinct and inside the clip.
export async function windowFrames(path, centre, offsets, prefix) {
  const select = offsets.map((o) => `eq(n\\,${centre + o})`).join('+');
  const letters = offsets.map((_, k) => `drawtext=text=${'ABCDEFGH'[k]}:x=12:y=10:fontsize=40:fontcolor=white:enable=eq(n\\,${k})`);
  await run(FFMPEG(), [
    '-v', 'error', '-y', '-i', path,
    '-vf', [`select=${select}`, 'scale=640:360', ...letters].join(','),
    '-fps_mode', 'passthrough', '-q:v', '2', '-start_number', '0', `${prefix}-%d.jpg`,
  ]);
  return offsets.map((_, k) => `${prefix}-${k}.jpg`);
}

// Frames where the picture changes completely. setpts=N/TB makes each timestamp equal the frame's decode index, so
// showinfo reports frame numbers instead of times.
export async function sceneCuts(path, threshold = 0.3) {
  const { stderr } = await run(FFMPEG(), [
    '-hide_banner', '-nostats', '-i', path, '-an',
    '-vf', `setpts=N/TB,select=gt(scene\\,${threshold}),showinfo`,
    '-fps_mode', 'passthrough', '-f', 'null', '-',
  ], { maxBuffer: 64 << 20 });
  return [...stderr.matchAll(/pts_time:(\d+)/g)].map((m) => +m[1]);
}
