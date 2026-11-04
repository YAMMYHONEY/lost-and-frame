// The app's pages: the review page for a VC-Fuse result (renderFuse), and the shell and styles every page shares (page).
import { writeFile, mkdir } from 'node:fs/promises';
import { framePng } from './video.mjs';
import { OFF8, PAIRS, NAMES } from './fuse.mjs';

export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

export const STYLE = `<style>
:root {
  color-scheme: light; --paper: #f4f1ea; --ink: #1a1917; --muted: #6b665d; --rule: #d6cfc2; --red: #c8321e;
  --film: #141210; --edge: #e2a03a; --hole: var(--paper);
  --serif: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  --mono: ui-monospace, "Cascadia Mono", "SF Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark; --paper: #151412; --ink: #ece8df; --muted: #9b958a; --rule: #35322d; --red: #ff5a3c;
    --film: #050505; --hole: #4a463f;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0 auto; max-width: 1080px; padding: 18px 20px 64px; background: var(--paper); color: var(--ink);
  font: 15px/1.6 system-ui, -apple-system, "Segoe UI", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
}
html[lang=ko] body { word-break: keep-all; }
html[lang=ko] :is(.top, .sample span, figcaption, .notes h2) { letter-spacing: 0; }
a { color: inherit; text-underline-offset: 3px; }
:focus-visible { outline: 2px solid var(--red); outline-offset: 2px; }
.mono, .label i, th, figcaption { font-family: var(--mono); }
.top {
  display: flex; justify-content: space-between; align-items: baseline; gap: 12px; flex-wrap: wrap;
  padding-bottom: 10px; border-bottom: 1px solid var(--ink);
  font-family: var(--mono); font-size: 12px; letter-spacing: .04em; text-transform: uppercase;
}
.lang a, .lang b { margin-left: 12px; }
.lang a { color: var(--muted); text-decoration: none; }
.lang b { text-decoration: underline 2px var(--red); }
.back { text-decoration: none; }
.back:hover { color: var(--red); }
h1 { font: 400 34px/1.1 var(--serif); letter-spacing: -.01em; margin: 26px 0 8px; }
.display { font-size: clamp(48px, 9vw, 104px); line-height: .95; letter-spacing: -.03em; margin: 34px 0 18px; }
.amp { position: relative; display: inline-block; font-style: italic; color: var(--red); padding: 0 .04em; }
.amp svg {
  position: absolute; inset: -14% -30% -6% -26%; width: 156%; height: 120%; overflow: visible;
  fill: none; stroke: var(--red); stroke-width: 2.5; stroke-linecap: round;
}
.sub { color: var(--muted); max-width: 640px; margin: 0 0 30px; font-size: 16px; }
.layout {
  display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(0, 1fr); gap: 0; border-top: 1px solid var(--rule);
}
.slip { padding: 22px 28px 26px 0; border-right: 1px solid var(--rule); }
.notes { padding: 22px 0 0 28px; font-size: 14px; }
.notes h2 {
  margin: 0 0 12px; font: 600 12px var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted);
}
.notes ol { margin: 0 0 28px; padding: 0; list-style: none; counter-reset: n; }
.notes li { counter-increment: n; padding: 10px 0 10px 34px; border-top: 1px dashed var(--rule); position: relative; }
.notes li::before {
  content: counter(n, decimal-leading-zero); position: absolute; left: 0; top: 11px;
  font: 12px var(--mono); color: var(--red);
}
.notes li b { display: block; }
.notes p { margin: 0 0 12px; }
.stat { font: 400 22px/1.3 var(--serif); }
.card.note {
  padding: 12px 14px; border: 1px solid var(--ink); border-left: 4px solid var(--red);
  margin: 0 0 22px; background: transparent;
}
.field { margin: 0 0 26px; border: 0; padding: 0; min-width: 0; }
.label { display: block; font-weight: 700; margin-bottom: 10px; padding: 0; }
.label i { font-style: normal; font-weight: 400; font-size: 12px; color: var(--red); margin-right: 8px; }
.hint { color: var(--muted); font-size: 13px; font-weight: 400; }
.drop {
  position: relative; display: flex; align-items: center; gap: 14px;
  padding: 16px 18px; border: 1px dashed var(--muted); cursor: pointer; transition: border-color .15s, background .15s;
}
.drop:hover, .drop.over { border: 1px solid var(--ink); background: color-mix(in srgb, var(--ink) 4%, transparent); }
.drop:focus-within { outline: 2px solid var(--red); outline-offset: 2px; }
.drop svg { flex: none; }
.drop .name { display: block; font-family: var(--mono); font-size: 13px; color: var(--red); word-break: break-all; }
.vh { position: absolute; width: 1px; height: 1px; opacity: 0; overflow: hidden; }
.or { display: block; margin: 14px 0 8px; font-size: 13px; color: var(--muted); }
.sheet {
  display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 22px 10px; padding: 18px 12px 12px;
  background:
    radial-gradient(circle, var(--hole) 2.5px, transparent 3px) 0 3px / 16px 10px repeat-x,
    radial-gradient(circle, var(--hole) 2.5px, transparent 3px) 0 calc(100% - 3px) / 16px 10px repeat-x,
    var(--film);
}
.sample { position: relative; display: block; cursor: pointer; padding-bottom: 20px; }
.sample video {
  display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: cover; background: #222;
  filter: saturate(.9); transition: filter .2s;
}
.sample:hover video { filter: none; }
.sample span {
  position: absolute; left: 0; right: 0; bottom: 0; font: 11px var(--mono); color: var(--edge);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; letter-spacing: .03em;
}
.sample:has(input:checked)::after {
  content: ""; position: absolute; inset: -7px -6px 12px; border: 3px solid var(--red); pointer-events: none;
  border-radius: 255px 18px 225px 18px / 18px 225px 18px 255px; transform: rotate(-.8deg);
}
.sample:has(input:focus-visible) { outline: 2px solid var(--edge); outline-offset: 3px; }
.text {
  width: 100%; padding: 10px 2px; border: 0; border-bottom: 1px solid var(--ink); background: transparent;
  color: var(--ink); font: 16px var(--mono);
}
.text:focus { outline: none; border-bottom: 2px solid var(--red); }
.seg { display: grid; grid-template-columns: 1fr 1fr; border: 1px solid var(--ink); }
.seg label { position: relative; padding: 10px 14px; cursor: pointer; }
.seg label + label { border-left: 1px solid var(--ink); }
.seg label:has(input:checked) { background: var(--ink); color: var(--paper); }
.seg label:has(input:focus-visible) { outline: 2px solid var(--red); outline-offset: -4px; }
.seg b { display: block; }
.seg small { font-size: 12px; opacity: .8; }
.switch { display: flex; align-items: center; gap: 10px; margin-top: 14px; cursor: pointer; font-size: 14px; }
.knob { flex: none; width: 16px; height: 16px; border: 1.5px solid var(--ink); display: grid; place-items: center; }
.switch input:checked + .knob::after { content: "✕"; color: var(--red); font-weight: 700; font-size: 13px; line-height: 1; }
.switch input:focus-visible + .knob { outline: 2px solid var(--red); outline-offset: 2px; }
.switch:has(input:disabled) { cursor: not-allowed; color: var(--muted); }
.switch:has(input:disabled) .knob { border-style: dashed; }
button {
  font: 600 14px var(--mono); letter-spacing: .06em; text-transform: uppercase;
  border: 1px solid var(--ink); border-radius: 0; padding: 14px 22px;
  background: var(--ink); color: var(--paper); cursor: pointer; transition: background .15s, border-color .15s;
}
button:hover { background: var(--red); border-color: var(--red); }
button:disabled { opacity: .7; cursor: wait; }
.submit { width: 100%; }
.submit::after { content: "  →"; }
.ghost { padding: 4px 10px; background: transparent; color: var(--ink); font-size: 12px; }
.ghost:hover { background: transparent; color: var(--red); }
.ghost[aria-pressed=true] { background: var(--red); border-color: var(--red); color: #fff; }
table { width: 100%; border-collapse: collapse; margin: 0 0 30px; font-size: 14px; }
th, td { padding: 8px 4px; text-align: left; border-bottom: 1px solid var(--rule); }
thead th { border-bottom: 1px solid var(--ink); }
th { font-size: 12px; font-weight: 400; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
td:nth-child(n+3) { text-align: right; font-variant-numeric: tabular-nums; }
.card { margin: 0 0 34px; }
.card header { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
.card h3 { margin: 0 0 4px; font: 400 22px/1.25 var(--serif); }
.why { margin: 0 0 12px; color: var(--muted); font: italic 15px/1.5 var(--serif); max-width: 820px; }
.pair {
  display: flex; flex-wrap: wrap; gap: 10px; padding: 18px 12px 16px;
  background:
    radial-gradient(circle, var(--hole) 2.5px, transparent 3px) 0 3px / 16px 10px repeat-x,
    radial-gradient(circle, var(--hole) 2.5px, transparent 3px) 0 calc(100% - 3px) / 16px 10px repeat-x,
    var(--film);
}
figure { margin: 0; flex: 1 1 300px; background: var(--film); }
img { width: 100%; display: block; }
.pair img { cursor: zoom-in; }
figcaption { color: var(--edge); font-size: 11px; letter-spacing: .03em; padding: 4px 0 8px; }
.pair.blinking { display: grid; grid-template-columns: min(100%, 640px); justify-content: center; }
.pair.blinking figure { grid-area: 1 / 1; }
.pair.blinking.alt figure:last-child { visibility: hidden; }
details { margin-top: 8px; color: var(--muted); font-size: 13px; }
.overlay {
  position: fixed; inset: 0; z-index: 10; display: grid; place-content: center; justify-items: center; gap: 16px;
  text-align: center; background: color-mix(in srgb, var(--paper) 92%, transparent);
}
.overlay[hidden] { display: none; }
.overlay p { margin: 0; color: var(--muted); }
.overlay b { font: 600 15px var(--mono); letter-spacing: .06em; }
.strip {
  width: min(420px, 80vw); height: 74px;
  background:
    radial-gradient(circle, var(--hole) 2.5px, transparent 3px) 0 3px / 16px 10px repeat-x,
    radial-gradient(circle, var(--hole) 2.5px, transparent 3px) 0 calc(100% - 3px) / 16px 10px repeat-x,
    linear-gradient(var(--film) 0 14px, transparent 14px calc(100% - 14px), var(--film) 0),
    repeating-linear-gradient(90deg, #5c564c 0 74px, var(--film) 74px 82px);
}
dialog.lightbox { padding: 0; border: 0; background: transparent; max-width: 94vw; }
dialog.lightbox::backdrop { background: rgba(0,0,0,.86); }
dialog.lightbox img { width: auto; max-width: 94vw; max-height: 90vh; cursor: zoom-out; }
@media (max-width: 820px) {
  .layout { grid-template-columns: 1fr; }
  .slip { padding-right: 0; border-right: 0; border-bottom: 1px solid var(--rule); }
  .notes { padding-left: 0; }
}
@media (max-width: 480px) {
  body { padding: 14px 16px 48px; }
  .seg { grid-template-columns: 1fr; }
  .seg label + label { border-left: 0; border-top: 1px solid var(--ink); }
}
@media (prefers-reduced-motion: no-preference) {
  .amp path { stroke-dasharray: 400; stroke-dashoffset: 400; animation: draw .9s .5s cubic-bezier(.4,0,.2,1) forwards; }
  .strip { animation: roll .8s linear infinite; }
}
@keyframes draw { to { stroke-dashoffset: 0; } }
@keyframes roll { to { background-position: 0 3px, 0 calc(100% - 3px), 0 0, -82px 0; } }
</style>`;

// Shared page behaviour: the upload drop zone, sample previews, the checking overlay, blink compare and a frame
// lightbox. No template literals inside: this whole script sits in one.
const SCRIPT = `<script>
const form = document.querySelector('form[action="/check"]');
if (form) {
  const file = form.querySelector('input[type=file]');
  const drop = form.querySelector('.drop');
  const shown = drop.querySelector('.name');
  const objects = form.querySelector('input[name=objects]');
  const button = form.querySelector('button');
  const overlay = document.querySelector('.overlay');
  const radios = [...form.querySelectorAll('input[name=sample]')];

  // A chosen file replaces the chosen sample.
  const picked = () => {
    shown.textContent = file.files[0] ? file.files[0].name : '';
    if (file.files.length) radios.forEach((r) => { r.checked = false; });
  };
  file.addEventListener('change', picked);
  drop.addEventListener('dragover', (e) => {
    e.preventDefault();
    drop.classList.add('over');
  });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('over');
    file.files = e.dataTransfer.files;
    picked();
  });

  // A chosen sample clears the file and shows its objects in the field. Samples play while the pointer is on them.
  for (const r of radios) {
    const card = r.closest('.sample');
    const video = card.querySelector('video');
    r.addEventListener('change', () => {
      file.value = '';
      shown.textContent = '';
      objects.placeholder = r.dataset.objects;
    });
    card.addEventListener('pointerenter', () => video.play().catch(() => {}));
    card.addEventListener('pointerleave', () => video.pause());
  }

  // While a check runs, the overlay counts the seconds.
  let timer;
  form.addEventListener('submit', () => {
    const t0 = Date.now();
    const out = overlay.querySelector('b');
    button.disabled = true;
    button.textContent = out.textContent = button.dataset.checking;
    overlay.hidden = false;
    timer = setInterval(() => {
      out.textContent = button.dataset.checking + ' ' + Math.round((Date.now() - t0) / 1000) + button.dataset.unit;
    }, 1000);
  });

  // The browser's Back button restores this page as it was left, with the button on "Checking…" and the overlay up.
  addEventListener('pageshow', () => {
    clearInterval(timer);
    button.disabled = false;
    button.textContent = button.dataset.check;
    overlay.hidden = true;
  });
}

// Blink: show a moment's two frames in turn, in the same place.
for (const b of document.querySelectorAll('.blink')) {
  let timer;
  b.addEventListener('click', () => {
    const pair = b.closest('.card').querySelector('.pair');
    const on = pair.classList.toggle('blinking');
    clearInterval(timer);
    pair.classList.remove('alt');
    if (on) timer = setInterval(() => pair.classList.toggle('alt'), 450);
    b.textContent = on ? b.dataset.stop : b.dataset.start;
    b.setAttribute('aria-pressed', on);
  });
}

// Lightbox: clicking a frame opens it at full size.
const box = document.createElement('dialog');
box.className = 'lightbox';
box.addEventListener('click', () => box.close());
document.body.append(box);
for (const img of document.querySelectorAll('.pair img')) {
  img.addEventListener('click', () => {
    box.replaceChildren(img.cloneNode());
    box.showModal();
  });
}
</script>`;

// The review page for a VC-Fuse result (fuse.mjs, checkClip). It shows every moment a witness flagged, with the two
// frames it compared (written to dir as PNG and linked as href/<name>) and its one-sentence reason, and returns the
// body HTML. With lang 'ko' the page is in Korean; the judge's reasons stay in English, as the judge wrote them.
export async function renderFuse(clip, dir, href, lang = 'en') {
  const ko = lang === 'ko';
  const names = ko ? { J: '장면 판정기', C: '개수 판정기', S: '스포터(확대)' } : NAMES;
  await mkdir(dir, { recursive: true });

  const rows = [];
  const cards = [];
  for (const r of clip.results) {
    rows.push(`<tr><td>${esc(clip.file)}</td><td>${esc(r.concept)}</td><td><b>${r.moments.length}</b></td></tr>`);
    for (const m of r.moments) {
      // The two frames the first witness compared, or 3 frames on each side of its window centre if it named none.
      const first = m.by[0];
      const i = PAIRS.indexOf(first.v.between);
      const pair = i >= 0
        ? [first.w.frame + OFF8[i], first.w.frame + OFF8[i + 1]]
        : [first.w.frame - 3, first.w.frame + 3];
      const imgs = [];
      for (const f of pair) {
        const name = `${r.concept}.${m.kind}.${m.frame}.f${f}.png`.replace(/[^\w.-]+/g, '_');
        await writeFile(`${dir}/${name}`, await framePng(clip.source, f, { width: 480 }));
        const frame = ko ? `${f}번 프레임` : `frame ${f}`;
        const s = (f / clip.fps).toFixed(2);
        imgs.push(`<figure><img src="${href}/${name}" alt="${frame}"><figcaption>${frame} · ${s}${ko ? '초' : ' s'}</figcaption></figure>`);
      }

      const who = esc([...new Set(m.by.map((a) => names[a.witness]))].join(', '));
      const at = (m.frame / clip.fps).toFixed(2);
      const vanish = m.kind === 'vanish';
      const concept = esc(r.concept);
      const blink = ko ? 'data-start="깜빡 비교" data-stop="멈춤">깜빡 비교' : 'data-start="Blink" data-stop="Stop">Blink';
      cards.push(`<section class="card"><header><h3>${ko ? `${concept}: ${at}초쯤 ${vanish ? '사라짐' : '나타남'}` : `${concept} ${vanish ? 'vanishes' : 'appears'} around ${at} s`}</h3>
<button type="button" class="ghost blink" aria-pressed="false" ${blink}</button></header>
<p class="why">${ko ? `${who}가 찾은 순간` : `Flagged by the ${who}`} · “${esc(first.v.reason)}”</p>
<div class="pair">${imgs.join('')}</div></section>`);
    }
  }

  const strict = clip.threshold >= 0.9;
  const recorded = clip.backend === 'recorded';
  if (ko) {
    const mode = `${clip.precision ? '정밀 모드(장면 판정기, 개수 판정기, 스포터)' : '기본 모드(장면 판정기, 개수 판정기)'} · ${strict ? '엄격' : '민감'}${recorded ? ' · 저장된 답, 키 없음' : ''}`;
    return `<h1>Lost &amp; Frame 리뷰</h1>
<p class="hint mono">${mode}</p>
<p class="sub">지켜보는 물체가 까닭 없이 사라지거나 나타난 것 같은 순간이에요. 두 프레임을 직접 보고 판단해 주세요. 따옴표 안 설명은 모델이 영어로 쓴 원문이에요.</p>
<table><thead><tr><th>클립</th><th>물체</th><th>순간</th></tr></thead><tbody>${rows.join('')}</tbody></table>
${cards.length ? cards.join('\n') : '<p>이 물체들에서는 찾은 순간이 없어요.</p>'}`;
  }
  const mode = `${clip.precision ? 'precision mode (scene judge, counter, spotter)' : 'basic mode (scene judge, counter)'} · ${strict ? 'strict' : 'sensitive'}${recorded ? ' · saved answers, no key' : ''}`;
  return `<h1>Lost &amp; Frame review</h1>
<p class="hint mono">${esc(mode)}</p>
<p class="sub">These are the moments where a watched object may have disappeared or appeared for no visible reason. Look at the two frames before you decide.</p>
<table><thead><tr><th>Clip</th><th>Object</th><th>Moments</th></tr></thead><tbody>${rows.join('')}</tbody></table>
${cards.length ? cards.join('\n') : '<p>No moments found for these objects.</p>'}`;
}

// The shell every page shares: the head, the styles and the script.
export const page = (title, body, lang = 'en') => `<!doctype html><html lang="${lang}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>${esc(title)}</title>${STYLE}</head><body>${body}${SCRIPT}</body></html>`;
