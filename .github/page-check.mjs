/* Loads the real page in a real browser and fails on anything a person would
   see as "the app does nothing".
 *
 * The static check next door catches one shape of the 0.1.1 bug: $ called above
 * the line that defines it. This catches the class. Any uncaught error at the
 * top of that script stops the whole thing, and the page still paints, so it
 * looks finished with every control dead. That is what two testers reported and
 * what no assertion on the HTML text would have caught in general.
 *
 * It runs on one operating system on purpose. Script order is not
 * platform-specific; discovery is, and the three-way matrix covers that.
 */
import { chromium } from 'playwright';

const base = process.argv[2] || 'http://127.0.0.1:8471/';
const fail = [];

// What /api/device returns for a Tiiny handed over by the launcher over the USB
// cable: the longest values this panel ever has to show, taken from the field
// report that opened this. The serial and the vhost name are the real shapes.
const FOUND = {
  host: '172.17.7.177', port: 80, plane: 'given', transport: 'vhost',
  vhost: 'p8800.api.tiiny', source: 'TIINY_BASE', serial: 'TNYM26072400300011Q',
  key_set: true, key_hint: 'xxxx\u2026xxxx', key_source: 'TIINY_KEY',
  candidates: ['tiiny', 'openai.api.tiiny', 'tiiny.local'],
  config: '/home/runner/.config/tiiny-bench.json',
  device: { name: 'a Tiiny on a desk', model: 'Tiiny z01', os: '0.1.34',
            service: '0.1.30', ram: '80 GB', storage: '1 TB',
            serial: 'TNYM26072400300011Q' }
};

for (const [width, height] of [[2000, 1200], [1440, 900], [390, 844]]) {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width, height } });
  const page = await ctx.newPage();

  const errors = [];
  const api = [];
  page.on('pageerror', e => errors.push('uncaught: ' + e.message));
  page.on('console', m => {
    // An uncaught exception is the signal. "Failed to load resource" is the
    // browser reporting an HTTP status, and some of those are deliberate: a
    // fresh install with no key gets 503 from /api/catalog and the page is
    // meant to carry on. Failing on those would make this check cry wolf, and
    // a check that cries wolf gets switched off.
    if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) {
      errors.push('console: ' + m.text());
    }
  });
  page.on('response', r => {
    const u = new URL(r.url());
    if (u.pathname.startsWith('/api/')) api.push(`${r.request().method()} ${u.pathname}`);
  });

  await page.goto(base, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(6000);

  const at = `${width}x${height}`;

  if (errors.length) fail.push(`${at}: ${errors.length} page error(s): ${errors.join(' | ')}`);

  // The whole point. A page that never asks is a page that is not running.
  if (!api.some(c => c === 'GET /api/device')) {
    fail.push(`${at}: the page never requested /api/device. Saw: ${api.join(', ') || 'nothing'}`);
  }

  // The find button has to be reachable and has to do something.
  await page.evaluate(() => { location.hash = 'conn'; });
  await page.waitForTimeout(500);
  const panel = await page.evaluate(() => {
    const shown = id => {
      const el = document.getElementById(id);
      if (!el) return 'absent';
      const r = el.getBoundingClientRect();
      return (r.width > 0 && r.height > 0) ? 'visible' : 'hidden';
    };
    return { credetect: shown('credetect'), connhost: shown('connhost'), connkey: shown('connkey') };
  });
  for (const [id, state] of Object.entries(panel)) {
    if (state !== 'visible') fail.push(`${at}: #${id} is ${state} on the Connection panel`);
  }

  const before = api.length;
  await page.evaluate(() => document.getElementById('credetect').click());
  await page.waitForTimeout(12000);
  if (!api.slice(before).some(c => c === 'POST /api/device')) {
    fail.push(`${at}: the find button sent nothing. Saw after the click: ${api.slice(before).join(', ') || 'nothing'}`);
  }

  // A panel somebody opens on a phone must not be off the right edge.
  const overflow = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth, inner: window.innerWidth }));
  if (overflow.scrollWidth > overflow.inner + 1) {
    fail.push(`${at}: the page is ${overflow.scrollWidth}px wide in a ${overflow.inner}px viewport`);
  }

  // 0.1.2 painted a found box in the display face the dashboard tiles use, which
  // is sized for a single number, so a tester at 2000px read "172.17.7.1" off a
  // box that lives at 172.17.7.177. A runner has no device, so /api/device is
  // answered with the record a real one produces, longest values and all, and
  // the page renders it through its own code path. Nothing in the app is
  // reached into; the reply is the only thing replaced.
  await page.route('**/api/device', route =>
    route.fulfill({ contentType: 'application/json', body: JSON.stringify(FOUND) }));
  await page.goto(base + '?tiles#conn', { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(3000);

  const tiles = await page.evaluate(() =>
    Array.from(document.querySelectorAll('#connstats .stat .v')).map(v => ({
      text: v.textContent, scrollWidth: v.scrollWidth, clientWidth: v.clientWidth })));

  if (tiles.length < 5) fail.push(`${at}: the found panel drew ${tiles.length} tiles`);

  // The bug, as an assertion: no tile value may be wider than the box holding it.
  const clipped = tiles.filter(t => t.scrollWidth > t.clientWidth);
  if (clipped.length) {
    fail.push(`${at}: stat tile value clipped: ` + clipped
      .map(t => `"${t.text}" (${t.scrollWidth} > ${t.clientWidth})`).join(' | '));
  }

  // The whole value has to be there too, not an ellipsis standing in for it.
  for (const want of ['172.17.7.177', 'p8800.api.tiiny', 'TNYM26072400300011Q']) {
    if (!tiles.some(t => t.text.includes(want))) {
      fail.push(`${at}: no tile reads ${want}. Saw: ${tiles.map(t => t.text).join(' / ')}`);
    }
  }

  // And the words in those tiles are for a person, not this suite's own
  // environment variable names.
  const jargon = tiles.map(t => t.text).filter(t => /TIINY_BASE|TIINY_KEY|TIINY_HOST/.test(t));
  if (jargon.length) fail.push(`${at}: a tile shows a variable name: ${jargon.join(', ')}`);

  // Long values wrap; they must not push the page sideways on a phone either.
  const wide = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth, inner: window.innerWidth }));
  if (wide.scrollWidth > wide.inner + 1) {
    fail.push(`${at}: with a box found the page is ${wide.scrollWidth}px in a ${wide.inner}px viewport`);
  }

  console.log(`${at}: ${errors.length} page errors, ${api.length} api calls, panel ${JSON.stringify(panel)}, scrollWidth ${overflow.scrollWidth}, ${clipped.length} clipped tiles`);
  await browser.close();
}

if (fail.length) {
  console.error('\nFAILED:');
  for (const f of fail) console.error('  ' + f);
  process.exit(1);
}
console.log('\nok: the page runs, asks for the device, and the find button works');
