// Pull selected files out of a remote zip with HTTP range requests, so a 25.6 GB archive costs only
// its central directory plus the chosen entries. Handles ZIP64 (archives over 4 GB); stored or deflated.
// read(start, end) returns bytes [start, end] inclusive as a Buffer; size is the archive length.
// CLI: node research/zipfetch.mjs <zip url> <out dir> <list.json>...
// Fetches every "file" the lists name (by base name), after checking the total against MAX_MB (default 300), so an
// oversized pull stops before it starts.
import { inflateRawSync } from 'node:zlib';
import { readFile, writeFile, mkdir, rename, stat } from 'node:fs/promises';
import { basename, join } from 'node:path';
import { pathToFileURL } from 'node:url';

const u64 = (b, o) => Number(b.readBigUInt64LE(o));

export async function listZip(read, size) {
  const tailStart = Math.max(0, size - 65557);
  const tail = await read(tailStart, size - 1);
  const e = tail.lastIndexOf(Buffer.from([0x50, 0x4b, 0x05, 0x06]));
  if (e < 0) throw new Error('not a zip: end of central directory not found');
  let count = tail.readUInt16LE(e + 10), cdSize = tail.readUInt32LE(e + 12), cdOffset = tail.readUInt32LE(e + 16);
  if (count === 0xffff || cdSize === 0xffffffff || cdOffset === 0xffffffff) {
    const loc = e - 20; // ZIP64 end-of-central-directory locator sits right before the classic record
    if (loc < 0 || tail.readUInt32LE(loc) !== 0x07064b50) throw new Error('ZIP64 locator not found');
    const recAt = u64(tail, loc + 8);
    const rec = await read(recAt, recAt + 55);
    if (rec.readUInt32LE(0) !== 0x06064b50) throw new Error('ZIP64 record not found');
    count = u64(rec, 32);
    cdSize = u64(rec, 40);
    cdOffset = u64(rec, 48);
  }
  const cd = await read(cdOffset, cdOffset + cdSize - 1);
  const entries = new Map();
  for (let p = 0, i = 0; i < count; i++) {
    if (cd.readUInt32LE(p) !== 0x02014b50) throw new Error(`bad central directory entry ${i}`);
    const nameLen = cd.readUInt16LE(p + 28), extraLen = cd.readUInt16LE(p + 30), commentLen = cd.readUInt16LE(p + 32);
    const ent = {
      name: cd.toString('utf8', p + 46, p + 46 + nameLen), method: cd.readUInt16LE(p + 10),
      compSize: cd.readUInt32LE(p + 20), size: cd.readUInt32LE(p + 24), offset: cd.readUInt32LE(p + 42),
    };
    // ZIP64 extra field (0x0001) holds the 64-bit values, in this order, for those set to 0xFFFFFFFF.
    for (let x = p + 46 + nameLen; x < p + 46 + nameLen + extraLen; x += 4 + cd.readUInt16LE(x + 2)) {
      if (cd.readUInt16LE(x) !== 0x0001) continue;
      let q = x + 4;
      for (const k of ['size', 'compSize', 'offset']) if (ent[k] === 0xffffffff) {
        ent[k] = u64(cd, q);
        q += 8;
      }
    }
    entries.set(ent.name, ent);
    p += 46 + nameLen + extraLen + commentLen;
  }
  return entries;
}

export async function extract(read, ent) {
  const head = await read(ent.offset, ent.offset + 29);
  if (head.readUInt32LE(0) !== 0x04034b50) throw new Error(`bad local header for ${ent.name}`);
  const start = ent.offset + 30 + head.readUInt16LE(26) + head.readUInt16LE(28);
  const data = ent.compSize ? await read(start, start + ent.compSize - 1) : Buffer.alloc(0);
  if (ent.method !== 0 && ent.method !== 8) throw new Error(`unsupported compression ${ent.method} for ${ent.name}`);
  const out = ent.method === 8 ? inflateRawSync(data) : data;
  // A misread ZIP64 field would otherwise yield a silently truncated video.
  if (out.length !== ent.size) throw new Error(
    `size mismatch for ${ent.name}: got ${out.length}, expected ${ent.size}`);
  return out;
}

// read() over HTTP; also returns the archive size from Content-Range.
export async function httpReader(url) {
  const read = async (start, end) => {
    const res = await fetch(url, { headers: { Range: `bytes=${start}-${end}` } });
    if (res.status !== 206) throw new Error(`range request failed ${res.status} for bytes ${start}-${end}`);
    return Buffer.from(await res.arrayBuffer());
  };
  const res = await fetch(url, { headers: { Range: 'bytes=0-0' } });
  const size = Number(res.headers.get('content-range')?.split('/')[1]);
  if (!(size > 0)) throw new Error(`no archive size from ${url} (status ${res.status})`);
  return { read, size };
}

// One zip cut into raw byte parts (videos.zip.001, .002, ...): archive offsets run on across the parts in order.
export function joinReaders(parts) {
  const starts = parts.map((_, i) => parts.slice(0, i).reduce((a, p) => a + p.size, 0));
  const read = async (start, end) => {
    const bufs = [];
    for (let i = 0; i < parts.length; i++) {
      const lo = Math.max(start, starts[i]), hi = Math.min(end, starts[i] + parts[i].size - 1);
      if (lo <= hi) bufs.push(await parts[i].read(lo - starts[i], hi - starts[i]));
    }
    return Buffer.concat(bufs);
  };
  return { read, size: starts.at(-1) + parts.at(-1).size };
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [url, outDir, ...lists] = process.argv.slice(2); // url: one zip, or its parts joined by commas
  if (!lists.length) throw new Error('usage: node zipfetch.mjs <zip url> <out dir> <list.json>...');
  const want = new Set();
  for (const l of lists) for (const c of JSON.parse(await readFile(l, 'utf8'))) want.add(basename(c.file));
  const { read, size } = joinReaders(await Promise.all(url.split(',').map(httpReader)));
  const byBase = new Map([...(await listZip(read, size)).values()].map((e) => [basename(e.name), e]));
  const missing = [...want].filter((n) => !byBase.has(n));
  if (missing.length) throw new Error(`not in the zip: ${missing.join(', ')}`);
  const mb = [...want].reduce((a, n) => a + byBase.get(n).compSize, 0) / 1e6;
  console.log(`${want.size} files, ${mb.toFixed(1)} MB to fetch (archive ${(size / 1e9).toFixed(1)} GB)`);
  if (mb > Number(process.env.MAX_MB ?? 300)) throw new Error(`stopping: ${mb.toFixed(1)} MB is over MAX_MB`);
  await mkdir(outDir, { recursive: true });
  for (const n of want) {
    const path = join(outDir, n); // our own list's base name, never the archive's path
    if (await stat(path).then(() => true, () => false)) continue;
    await writeFile(`${path}.tmp`, await extract(read, byBase.get(n)));
    await rename(`${path}.tmp`, path);
    process.stdout.write('.');
  }
  console.log(' done');
}
