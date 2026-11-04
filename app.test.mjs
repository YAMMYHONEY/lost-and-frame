import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { handler } from './app.mjs';

test('serves the form and rejects bad uploads before any paid call', async () => {
  const server = createServer(handler).listen(0, '127.0.0.1');
  await new Promise((r) => server.once('listening', r));
  const base = `http://127.0.0.1:${server.address().port}`;
  const post = (name, objects, headers = {}) => {
    const fd = new FormData();
    fd.append('video', new File(['x'], name));
    fd.append('objects', objects);
    return fetch(`${base}/check`, { method: 'POST', body: fd, headers });
  };

  try {
    const home = await fetch(base);
    assert.equal(home.status, 200);
    assert.match(await home.text(), /<form method="post" action="\/check"/);

    const wrongType = await post('notes.txt', 'cup');
    assert.equal(wrongType.status, 400);
    assert.match(await wrongType.text(), /MP4 or MOV/);
    for (const bad of ['', 'a, b, c, d, e', 'cup/../../x', '<script>', '컵']) {
      assert.equal((await post('clip.mp4', bad)).status, 400);
    }
    assert.equal((await fetch(`${base}/frames/abc/..%2f..%2f.env`)).status, 404);
    assert.equal((await post('clip.mp4', 'cup', { origin: 'https://example.com' })).status, 403); // another site's page

    const ko = await fetch(`${base}/?lang=ko`);
    assert.match(ko.headers.get('set-cookie'), /^lang=ko;/);
    assert.match(await ko.text(), /<html lang="ko">[\s\S]*확인 중…/);
    assert.match(await (await post('notes.txt', 'cup', { cookie: 'lang=ko' })).text(), /MP4나 MOV/);
    assert.equal((await fetch(`${base}/?lang=constructor`)).headers.get('set-cookie'), null);
  } finally {
    server.close();
  }
});
