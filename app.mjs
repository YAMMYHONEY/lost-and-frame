// Lost & Frame's local web app.
//
// Upload a clip or pick a sample, name the objects to watch, and get the moments where one of them may vanish or
// appear with no visible cause (VC-Fuse, see fuse.mjs). API keys stay in .env on this computer and are never sent to
// the page. Without a key, the samples replay their recorded answers (samples/raw), so the whole check still runs.
//
// Usage: node --env-file-if-exists=.env --env-file-if-exists=ffmpeg.env app.mjs
// Then open http://localhost:8787.
import { createServer } from 'node:http';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { cp, readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { checkClip, precisionReady } from './fuse.mjs';
import { renderFuse, page, esc } from './report.mjs';
import { framePng } from './video.mjs';

const PORT = Number(process.env.PORT ?? 8787);
const MAX_CALLS = 300; // judge calls per check, about $2 on either backend
const MAX_BYTES = 200e6;
const DIR = 'clips/app';
const PYTHON = process.env.PYTHON ?? 'python';

const backend = () => (process.env.BASETEN_API_KEY ? 'baseten' : process.env.MODEL_API_KEY ? 'meta' : 'recorded');

let precisionOk = false; // set at start: true when this computer has a CUDA GPU with SAM 3.1 (README, precision mode)
let samples = [];

// Page text in English and Korean. ?lang=ko or ?lang=en switches the language, and a cookie keeps the choice.
const T = {
  en: {
    eyebrow: 'Object permanence check for AI video',
    sub: `Upload an AI-generated clip and name the objects to watch. Lost &amp; Frame shows each moment where one of them may
disappear or appear for no visible reason, with the two frames side by side. A check takes a minute or two.`,
    noKey: 'There is no API key in .env, so only the samples below can be checked. Their answers were saved in advance. The README explains how to add a key.',
    clip: 'Clip',
    drop: 'Drop an MP4 or MOV file here, or click to choose one',
    dropHint: 'up to 200 MB',
    sample: 'or pick a sample',
    sampleHint: 'made with LTX-2.5, plays on hover',
    objects: 'Objects to watch',
    objectsHint: 'English nouns separated by commas, like cup, ball. A sample fills this in.',
    setting: 'Setting',
    strict: ['Strict', 'fewer false alarms'],
    sensitive: ['Sensitive', 'finds more, with more false alarms'],
    precision: 'Precision mode (SAM 3.1 spotter)',
    noGpuHint: ' needs a CUDA GPU with SAM 3.1 (see the README)',
    check: 'Check',
    checking: 'Checking…',
    unit: ' s',
    wait: 'The model is looking at the frames. This usually takes a minute or two.',
    how: 'How it works',
    tested: 'Test results',
    witnesses: [
      ['Scene judge', 'Every 16 frames, Muse Spark looks at eight frames and says whether the object disappeared or appeared with nothing to explain it.'],
      ['Counter', 'Counts the object in each frame, so a drop from three to two is easy to see.'],
      ['Spotter', 'Uses SAM 3.1 to find where a copy came or went, including weak detections SAM usually discards. Needs a GPU.'],
    ],
    stat: 'In a sealed test it caught 23 of 25 clearly visible glitches from real video generators, with all three checks on.',
    statNote: 'It also finds moments in clips that have no glitch, so look at each one yourself.',
    privacy: 'Your video stays on this computer. Only still frames go to the model, sent with your own key.',
    review: 'Lost & Frame review',
    another: '← Check another clip',
    notFound: 'Not found.',
    home: 'Back',
    tooBig: 'That file is over 200 MB.',
    busy: 'Another check is running. Try again in a minute.',
    notVideo: 'Please choose an MP4 or MOV file.',
    noClip: 'Please choose an MP4 or MOV file, or a sample.',
    badObjects: 'Name one to four objects in English letters, separated by commas.',
    noGpu: 'Precision mode needs a CUDA GPU with SAM 3.1 on this computer. See the README.',
    failed: 'Something went wrong: ',
  },
  ko: {
    eyebrow: 'AI 영상 대상 영속성 검사',
    sub: `AI로 만든 클립을 올리고 지켜볼 물체 이름을 적어 주세요. 그 물체가 까닭 없이 사라지거나 나타난 것 같은 순간을 찾아서
앞뒤 프레임 두 장을 나란히 보여 줘요. 1~2분쯤 걸려요.`,
    noKey: '.env에 API 키가 없어서 아래 샘플만 확인할 수 있어요. 샘플은 모델의 답을 미리 받아 뒀어요. 키를 넣는 방법은 README에 있어요.',
    clip: '클립',
    drop: 'MP4나 MOV 파일을 끌어다 놓거나 눌러서 고르세요',
    dropHint: '200 MB까지',
    sample: '또는 샘플 고르기',
    sampleHint: 'LTX-2.5로 만든 영상, 마우스를 올리면 재생돼요',
    objects: '지켜볼 물체',
    objectsHint: '영어 명사를 쉼표로 나눠 적어 주세요. 예: cup, ball. 샘플을 고르면 자동으로 채워져요.',
    setting: '설정',
    strict: ['엄격', '오경보가 적어요'],
    sensitive: ['민감', '더 많이 찾지만 오경보도 늘어요'],
    precision: '정밀 모드 (SAM 3.1 스포터)',
    noGpuHint: ' SAM 3.1이 설치된 CUDA GPU가 필요해요. 자세한 내용은 README에 있어요.',
    check: '확인',
    checking: '확인 중…',
    unit: '초',
    wait: '모델이 프레임을 보고 있어요. 보통 1~2분 걸려요.',
    how: '작동 방식',
    tested: '시험 결과',
    witnesses: [
      ['장면 판정기', '16프레임마다 Muse Spark가 프레임 8장을 보고, 물체가 까닭 없이 사라지거나 나타났는지 판단해요.'],
      ['개수 판정기', '프레임마다 물체 개수를 세어요. 세 개가 두 개로 줄면 바로 드러나요.'],
      ['스포터', 'SAM 3.1로 물체 하나가 들어오거나 나간 자리를 찾아요. SAM이 보통 버리는 약한 탐지도 써요. GPU가 필요해요.'],
    ],
    stat: '실제 영상 생성기가 만든 글리치 중 눈에 분명히 보이는 25개로 봉인 시험을 했고, 23개를 잡았어요. 세 검사를 모두 켠 결과예요.',
    statNote: '글리치가 없는 클립에서도 순간을 찾아낼 때가 있어요. 하나씩 직접 보고 판단해 주세요.',
    privacy: '영상은 이 컴퓨터에만 있어요. 모델에는 정지 프레임만 내 키로 보내요.',
    review: 'Lost & Frame 리뷰',
    another: '← 다른 클립 확인',
    notFound: '없는 페이지예요.',
    home: '처음으로',
    tooBig: '파일이 200 MB를 넘어요.',
    busy: '다른 확인이 진행 중이에요. 1분 뒤에 다시 해 주세요.',
    notVideo: 'MP4나 MOV 파일을 골라 주세요.',
    noClip: 'MP4나 MOV 파일, 또는 샘플을 골라 주세요.',
    badObjects: '물체 이름을 영어로 1~4개, 쉼표로 나눠 적어 주세요.',
    noGpu: '정밀 모드를 쓰려면 이 컴퓨터에 SAM 3.1이 설치된 CUDA GPU가 있어야 해요. 자세한 내용은 README에 있어요.',
    failed: '문제가 생겼어요: ',
  },
};

// The form looks like a lab slip. Samples sit on a contact sheet, and the chosen one gets a grease-pencil ring.
const formPage = (lang, msg = '') => {
  const t = T[lang];
  return page('Lost & Frame', `<header class="top"><span>${t.eyebrow}</span>
<nav class="lang" aria-label="Language">${lang === 'ko' ? '<a href="/?lang=en">English</a><b>한국어</b>' : '<b>English</b><a href="/?lang=ko">한국어</a>'}</nav></header>
<h1 class="display">Lost <span class="amp">&amp;<svg viewBox="0 0 120 100" aria-hidden="true"><path pathLength="400" d="M64 10C30 6 8 28 12 54c4 26 30 40 58 36 28-4 44-24 40-48C106 18 84 4 56 8 38 11 24 19 16 31"/></svg></span> Frame</h1>
<p class="sub">${t.sub}</p>
${msg ? `<p class="card note">${msg}</p>` : ''}
${backend() === 'recorded' ? `<p class="card note">${t.noKey}</p>` : ''}
<div class="layout"><form method="post" action="/check" enctype="multipart/form-data" class="slip">
<fieldset class="field"><legend class="label"><i>01</i>${t.clip}</legend><label class="drop">
<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 16V4M7 9l5-5 5 5M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/></svg>
<span>${t.drop} <span class="hint">(${t.dropHint})</span><span class="name"></span></span>
<input class="vh" type="file" name="video" accept=".mp4,.mov"></label>
${samples.length ? `<span class="or">${t.sample} · ${t.sampleHint}</span><div class="sheet">
${samples.map((s, i) => `<label class="sample"><input class="vh" type="radio" name="sample" value="${i}" data-objects="${esc(s.objects)}">
<video src="/samples/${i}.mp4" poster="/samples/${i}.png" muted loop playsinline preload="none"></video>
<span>${String(i + 1).padStart(2, '0')} ▸ ${esc(((lang === 'ko' && s.title_ko) || s.title).replace(/\s*\(LTX-2\.5\)$/, ''))} · ${esc(s.objects)}</span></label>`).join('\n')}
</div>` : ''}</fieldset>
<div class="field"><label><span class="label"><i>02</i>${t.objects} <span class="hint">${t.objectsHint}</span></span>
<input class="text" name="objects" placeholder="cup, cat" size="40"></label></div>
<fieldset class="field"><legend class="label"><i>03</i>${t.setting}</legend><div class="seg">
<label><input class="vh" type="radio" name="sensitivity" value="strict" checked><b>${t.strict[0]}</b><small>${t.strict[1]}</small></label>
<label><input class="vh" type="radio" name="sensitivity" value="sensitive"><b>${t.sensitive[0]}</b><small>${t.sensitive[1]}</small></label>
</div><label class="switch"><input class="vh" type="checkbox" name="precision"${precisionOk ? '' : ' disabled'}><span class="knob"></span>
<span>${t.precision}${precisionOk ? '' : `<span class="hint">${t.noGpuHint}</span>`}</span></label></fieldset>
<button class="submit" data-check="${t.check}" data-checking="${t.checking}" data-unit="${t.unit}">${t.check}</button></form>
<aside class="notes"><h2>${t.how}</h2><ol>${t.witnesses.map(([h, p]) => `<li><b>${h}</b>${p}</li>`).join('')}</ol>
<h2>${t.tested}</h2><p class="stat">${t.stat}</p><p>${t.statNote}</p><p class="hint">${t.privacy}</p></aside></div>
<div class="overlay" hidden><div class="strip"></div><b>${t.checking}</b><p>${t.wait}</p></div>`, lang);
};

const posters = new Map(); // sample → its poster PNG (a promise), made once with ffmpeg

let busy = false;

export async function handler(req, res) {
  const send = (status, body, type = 'text/html; charset=utf-8') => {
    res.writeHead(status, { 'Content-Type': type });
    res.end(body);
  };

  // Answer only this computer's own page. Otherwise any web site open in the browser could post here (or reach the
  // app through DNS rebinding) and spend the user's key.
  const self = [`localhost:${req.socket.localPort}`, `127.0.0.1:${req.socket.localPort}`];
  if (
    !self.includes(req.headers.host) ||
    (req.headers.origin && !self.some((h) => req.headers.origin === `http://${h}`))
  ) {
    return send(403, 'Forbidden', 'text/plain; charset=utf-8');
  }

  let lang = /(?:^|;\s*)lang=(en|ko)(?:;|$)/.exec(req.headers.cookie ?? '')?.[1] ?? 'en';
  const form = (msg) => formPage(lang, msg);

  try {
    const url = new URL(req.url, 'http://localhost');
    const asked = url.searchParams.get('lang');
    if (asked === 'en' || asked === 'ko') {
      lang = asked;
      res.setHeader('Set-Cookie', `lang=${lang}; Path=/; Max-Age=31536000; SameSite=Lax`);
    }
    const t = T[lang];

    if (req.method === 'GET' && url.pathname === '/') return send(200, form());

    const img = url.pathname.match(/^\/frames\/(\w+)\/([\w.-]+\.png)$/); // no slashes, so the path stays inside DIR
    if (req.method === 'GET' && img) return send(200, await readFile(join(DIR, img[1], img[2])), 'image/png');

    // shortcut: whole-file responses without Range support, so Safari shows only the poster.
    // Add Range support if the previews must play there.
    const media = url.pathname.match(/^\/samples\/(\d)\.(mp4|png)$/);
    const s = media && samples[media[1]];
    if (req.method === 'GET' && s && media[2] === 'mp4') return send(200, await readFile(s.file), 'video/mp4');
    if (req.method === 'GET' && s) {
      if (!posters.has(s)) posters.set(s, framePng(s.file, 24, { width: 480 }));
      return send(200, await posters.get(s), 'image/png');
    }

    if (req.method !== 'POST' || url.pathname !== '/check') {
      return send(404, page('Not found', `<p>${t.notFound} <a href="/">${t.home}</a></p>`, lang));
    }
    if (Number(req.headers['content-length'] ?? Infinity) > MAX_BYTES) return send(413, form(t.tooBig));
    if (busy) return send(429, form(t.busy));

    const data = await new Request('http://localhost/check', {
      method: 'POST',
      headers: { 'content-type': req.headers['content-type'] ?? '' },
      body: req,
      duplex: 'half',
    }).formData();
    const sample = samples[Number(data.get('sample') || NaN)];
    const video = data.get('video');
    const named = String(data.get('objects') ?? '') || (sample?.objects ?? '');
    const concepts = named.split(',').map((s) => s.trim().toLowerCase()).filter(Boolean);

    let buf, name;
    if (video instanceof File && video.size) {
      if (!/\.(mp4|mov)$/i.test(video.name)) return send(400, form(t.notVideo));
      [buf, name] = [Buffer.from(await video.arrayBuffer()), video.name];
    } else if (sample) {
      [buf, name] = [await readFile(sample.file), sample.file.split('/').pop()];
    } else {
      return send(400, form(t.noClip));
    }

    // English letters only: the judge and SAM read English nouns, and the answer cache keeps only ASCII in file
    // names (two Korean names would share one file and one answer).
    if (!concepts.length || concepts.length > 4 || !concepts.every((c) => /^[a-z0-9 '-]{1,40}$/.test(c))) {
      return send(400, form(t.badObjects));
    }
    const precision = data.get('precision') === 'on';
    if (precision && !precisionOk) return send(400, form(t.noGpu));

    busy = true;
    try {
      // The folder is named after the clip's content, so checking the same clip again reuses the saved answers
      // (no cost, and no key for the samples).
      const id = createHash('sha1').update(buf).digest('hex').slice(0, 16);
      await mkdir(join(DIR, id), { recursive: true });
      const path = join(DIR, id, `${id}${name.slice(name.lastIndexOf('.')).toLowerCase()}`);
      await writeFile(path, buf);
      const r = await checkClip(path, concepts, {
        dir: join(DIR, id),
        key: id,
        backend: backend(),
        precision,
        python: PYTHON,
        maxCalls: MAX_CALLS,
        threshold: data.get('sensitivity') === 'sensitive' ? 0.5 : 0.9,
      });
      const body = await renderFuse({ file: name, ...r }, join(DIR, id), `/frames/${id}`, lang);
      return send(
        200,
        page(t.review, `<header class="top"><a class="back" href="/">${t.another}</a></header>${body}`, lang),
      );
    } finally {
      busy = false;
    }
  } catch (e) {
    return send(500, form(`${T[lang].failed}${esc(e.message)}`));
  }
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    execFileSync(process.env.FFMPEG ?? 'ffmpeg', ['-version'], { stdio: 'ignore' });
  } catch {
    console.warn('ffmpeg not found: install it or set FFMPEG/FFPROBE (e.g. in ffmpeg.env)');
  }

  // The recorded sample answers join the answer cache, so the samples can be checked without a key.
  try {
    samples = JSON.parse(await readFile('samples/samples.json', 'utf8'));
    await cp('samples/raw', 'clips/raw', { recursive: true, force: false });
  } catch (e) {
    if (e.code !== 'ENOENT') throw e;
  }

  console.log(`judge: ${{
    baseten: 'Muse Spark Open (BASETEN_API_KEY)',
    meta: 'Muse Spark 1.3 (MODEL_API_KEY)',
    recorded: 'no key: samples replay their recorded answers',
  }[backend()]}`);
  precisionOk = await precisionReady(PYTHON);
  console.log(`precision mode: ${precisionOk ? 'available' : 'off (no CUDA GPU with SAM 3.1 for ' + PYTHON + ')'}`);
  createServer(handler).listen(PORT, '127.0.0.1', () => console.log(`Lost & Frame on http://localhost:${PORT}`));
}
