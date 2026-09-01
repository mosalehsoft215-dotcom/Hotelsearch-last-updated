/**
 * Tests for chat_render.js, on node's built-in runner:  node --test tests/
 *
 * There is no bundler and no npm in this project, so there is no jsdom either.
 * The shim below is the smallest DOM that the renderer actually uses —
 * createElement, createTextNode, createDocumentFragment, appendChild,
 * setAttribute — plus a serialiser, so assertions can be made against real node
 * structure rather than against a string the renderer happened to build.
 *
 * That the renderer needs only this much is the point: it never assembles
 * markup, so there is nothing here to emulate but the node API.
 */
const test = require('node:test');
const assert = require('node:assert');
const R = require('../chat_render.js');

/* --- the smallest DOM the renderer touches ------------------------------- */

const VOID = new Set(['br', 'hr']);

function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function makeNode(tag) {
  return {
    nodeType: 1, tagName: tag, className: '', attrs: {}, childNodes: [],
    appendChild(child) { this.childNodes.push(child); return child; },
    setAttribute(k, v) { this.attrs[k] = String(v); },
    get textContent() { return this.childNodes.map(textOf).join(''); },
    set textContent(v) { this.childNodes = v === '' ? [] : [makeText(v)]; }
  };
}
function makeText(value) {
  return { nodeType: 3, data: String(value), childNodes: [] };
}
function makeFragment() {
  return {
    nodeType: 11, tagName: '#fragment', childNodes: [],
    appendChild(child) { this.childNodes.push(child); return child; }
  };
}
const doc = {
  createElement: makeNode,
  createTextNode: makeText,
  createDocumentFragment: makeFragment
};

function textOf(node) {
  if (!node) return '';
  if (node.nodeType === 3) return node.data;
  return (node.childNodes || []).map(textOf).join('');
}
function html(node) {
  if (node.nodeType === 3) return esc(node.data);
  const inner = (node.childNodes || []).map(html).join('');
  if (node.nodeType === 11) return inner;
  const attrs = Object.keys(node.attrs || {})
    .map(k => ` ${k}="${esc(node.attrs[k])}"`).join('');
  const cls = node.className ? ` class="${esc(node.className)}"` : '';
  const tag = node.tagName;
  if (VOID.has(tag)) return `<${tag}${cls}${attrs}>`;
  return `<${tag}${cls}${attrs}>${inner}</${tag}>`;
}
function tagsIn(node, tag, found = []) {
  if (node.nodeType === 1 && node.tagName === tag) found.push(node);
  (node.childNodes || []).forEach(c => tagsIn(c, tag, found));
  return found;
}
const md = src => R.renderMarkdown(src, doc);

/* --- 1-6: markdown ------------------------------------------------------- */

test('bold renders as strong', () => {
  const out = md('**Recommended hotels**');
  const strong = tagsIn(out, 'strong');
  assert.equal(strong.length, 1);
  assert.equal(textOf(strong[0]), 'Recommended hotels');
  assert.ok(!html(out).includes('**'), 'literal ** must not survive');
});

test('italic renders as em, both spellings', () => {
  assert.equal(tagsIn(md('a *very* good rate'), 'em').length, 1);
  assert.equal(textOf(tagsIn(md('a _very_ good rate'), 'em')[0]), 'very');
});

test('headings render as heading elements, shifted below the page outline', () => {
  const out = md('# Options\n## Detail');
  assert.equal(tagsIn(out, 'h3').length, 1);
  assert.equal(tagsIn(out, 'h4').length, 1);
  assert.equal(tagsIn(out, 'h1').length, 0, 'a message must not own the page h1');
  assert.ok(!html(out).includes('#'));
});

test('unordered markdown renders as a real list', () => {
  const out = md('- Refundable\n- Breakfast included');
  const ul = tagsIn(out, 'ul');
  assert.equal(ul.length, 1);
  const li = tagsIn(ul[0], 'li');
  assert.deepEqual(li.map(textOf), ['Refundable', 'Breakfast included']);
});

test('ordered markdown renders as a real ordered list', () => {
  const out = md('1. Search\n2. Compare\n3. Confirm');
  const ol = tagsIn(out, 'ol');
  assert.equal(ol.length, 1);
  assert.equal(tagsIn(ol[0], 'li').length, 3);
  assert.equal(tagsIn(out, 'ul').length, 0);
});

test('a GFM table renders as an actual table', () => {
  const out = md('| Hotel | Price |\n|---|---|\n| A | $100 |\n| B | $120 |');
  const table = tagsIn(out, 'table');
  assert.equal(table.length, 1);
  assert.deepEqual(tagsIn(out, 'th').map(textOf), ['Hotel', 'Price']);
  assert.deepEqual(tagsIn(out, 'td').map(textOf), ['A', '$100', 'B', '$120']);
  assert.ok(!html(out).includes('|'), 'literal pipes must not survive');
});

test('inline and fenced code are preserved literally', () => {
  assert.equal(textOf(tagsIn(md('use `checkIn` here'), 'code')[0]), 'checkIn');
  const fenced = md('```\nline one\n**not bold**\n```');
  assert.equal(tagsIn(fenced, 'pre').length, 1);
  assert.equal(textOf(tagsIn(fenced, 'code')[0]), 'line one\n**not bold**');
  assert.equal(tagsIn(fenced, 'strong').length, 0, 'markup inside code stays text');
});

test('links render safely with noopener noreferrer', () => {
  const a = tagsIn(md('see [GOV.UK](https://www.gov.uk/foreign-travel-advice/oman)'), 'a');
  assert.equal(a.length, 1);
  assert.equal(a[0].attrs.href, 'https://www.gov.uk/foreign-travel-advice/oman');
  assert.equal(a[0].attrs.rel, 'noopener noreferrer');
  assert.equal(a[0].attrs.target, '_blank');
  assert.equal(textOf(a[0]), 'GOV.UK');
});

test('unsafe url schemes lose the link and keep only the text', () => {
  for (const bad of ['javascript:alert(1)', 'JaVaScRiPt:alert(1)',
                     'data:text/html,<script>x</script>', 'vbscript:msgbox(1)']) {
    const out = md(`click [here](${bad})`);
    assert.equal(tagsIn(out, 'a').length, 0, bad);
    assert.ok(textOf(out).includes('here'));
    assert.ok(!html(out).toLowerCase().includes('javascript:'), bad);
  }
});

test('raw HTML in model output never becomes HTML', () => {
  const nasty = '<script>alert(1)</script><img src=x onerror=alert(1)>' +
                '<b>bold?</b> <a href="javascript:alert(1)">x</a>';
  const out = md(nasty);
  assert.equal(tagsIn(out, 'script').length, 0);
  assert.equal(tagsIn(out, 'img').length, 0);
  assert.equal(tagsIn(out, 'b').length, 0);
  assert.equal(tagsIn(out, 'a').length, 0);
  /* It survives as visible text, escaped by the serialiser — which is what a
     browser does with a text node. */
  assert.ok(textOf(out).includes('<script>alert(1)</script>'));
  assert.ok(html(out).includes('&lt;script&gt;'));
});

/* --- the acceptance case from the brief, exactly as written -------------- */

test('the acceptance answer renders bold, a table and a bullet list', () => {
  const out = md([
    '**Recommended hotels**', '',
    '| Hotel | Price |', '|---|---|', '| A | $100 |', '| B | $120 |', '',
    '- Refundable', '- Breakfast included'
  ].join('\n'));
  const markup = html(out);
  assert.equal(textOf(tagsIn(out, 'strong')[0]), 'Recommended hotels');
  assert.equal(tagsIn(out, 'table').length, 1);
  assert.equal(tagsIn(out, 'ul').length, 1);
  assert.equal(tagsIn(out, 'li').length, 2);
  assert.ok(!markup.includes('**'), 'no literal asterisks');
  assert.ok(!markup.includes('|'), 'no literal pipes');
});

/* --- 7-10: one component per block type ---------------------------------- */

const HOTEL = {
  type: 'hotel_option', hotel_name: 'Carawan Al Fahad', stars: 4,
  location: 'Riyadh, Saudi Arabia', price_per_night: 120.5, total_price: 361.5,
  currency: 'USD', board: 'Breakfast Included', refundable: true,
  cancellation_summary: 'Free until 8 Sep 2026'
};
const FLIGHT = {
  type: 'flight_option', airline: 'Saudia', flight_number: 'SV1234',
  origin: 'JED', destination: 'MCT', departure: '2026-09-10T08:00',
  arrival: '2026-09-10T11:15', duration: '3h 15m', stops: 0,
  total_price: 410, currency: 'USD'
};
const SUMMARY = {
  type: 'booking_summary', title: 'Confirmed rate',
  items: [{ label: 'Hotel', value: 'Carawan Al Fahad' }, { label: 'Nights', value: '3' }],
  total: 361.5, currency: 'USD'
};
const TABLE = {
  type: 'table', columns: ['Hotel', 'Total'],
  rows: [['A', 100], ['B', 120.5], ['C', null]]
};
const blocks = (list) => R.renderBlocks(list, doc);

test('hotel_option renders a hotel card', () => {
  const out = blocks([HOTEL]);
  const markup = html(out);
  assert.ok(markup.includes('blk-hotel'));
  assert.ok(textOf(out).includes('Carawan Al Fahad'));
  assert.ok(textOf(out).includes('Breakfast Included'));
  assert.ok(textOf(out).includes('Refundable'));
  assert.ok(textOf(out).includes('361.50 USD'));
  assert.ok(textOf(out).includes('★'), 'four stars shown');
});

test('flight_option renders a flight card', () => {
  const out = blocks([FLIGHT]);
  assert.ok(html(out).includes('blk-flight'));
  assert.ok(textOf(out).includes('JED → MCT'));
  assert.ok(textOf(out).includes('Saudia SV1234'));
  assert.ok(textOf(out).includes('Direct'));
  assert.ok(textOf(out).includes('410 USD'));
});

test('booking_summary renders a summary card with its total', () => {
  const out = blocks([SUMMARY]);
  assert.ok(html(out).includes('blk-summary'));
  assert.ok(textOf(out).includes('Confirmed rate'));
  assert.ok(textOf(out).includes('Carawan Al Fahad'));
  assert.ok(html(out).includes('blk-total'));
  assert.ok(textOf(out).includes('361.50 USD'));
});

test('table renders a data table, including a null cell', () => {
  const out = blocks([TABLE]);
  assert.ok(html(out).includes('blk-table'));
  assert.deepEqual(tagsIn(out, 'th').map(textOf), ['Hotel', 'Total']);
  assert.equal(tagsIn(out, 'tr').length, 4);         /* header + three rows */
  assert.deepEqual(tagsIn(out, 'td').map(textOf),
                   ['A', '100', 'B', '120.5', 'C', '']);
});

/* --- 11-15: robustness --------------------------------------------------- */

test('answer and blocks render together, prose first', () => {
  const frag = R.renderAssistantMessage(
    { output: '**Three options.** The first is cheapest.', blocks: [HOTEL, HOTEL] }, doc);
  const kids = frag.childNodes;
  assert.equal(kids[0].className, 'md', 'markdown comes first');
  assert.equal(kids[1].className, 'blocks', 'blocks come second');
  assert.equal(tagsIn(kids[1], 'div').filter(d => d.className.includes('blk-hotel')).length, 2);
  assert.ok(textOf(kids[0]).includes('Three options.'));
});

test('a message with no blocks renders exactly the markdown and nothing else', () => {
  for (const value of [null, undefined, [], 'nonsense', 42]) {
    const frag = R.renderAssistantMessage({ output: 'Riyadh is hot in September.', blocks: value }, doc);
    assert.equal(frag.childNodes.length, 1, JSON.stringify(value));
    assert.equal(frag.childNodes[0].className, 'md');
    assert.ok(textOf(frag).includes('Riyadh is hot'));
  }
});

test('an unknown block type is ignored and never throws', () => {
  const out = blocks([
    { type: 'wat', payload: 'whatever' },
    HOTEL,
    { type: 'car_rental', vendor: 'x' },
    { nope: true },
    null,
    'a string'
  ]);
  /* the one known block still rendered; the rest were dropped quietly */
  assert.equal(tagsIn(out, 'div').filter(d => d.className.includes('blk-hotel')).length, 1);
  assert.ok(!textOf(out).includes('whatever'));
});

test('null optional fields do not crash any card', () => {
  const sparse = [
    { type: 'hotel_option', hotel_name: 'Bare Hotel', stars: null, location: null,
      price_per_night: null, total_price: null, currency: null, board: null,
      refundable: null, cancellation_summary: null },
    { type: 'flight_option', origin: 'JED', destination: 'MCT', airline: null,
      flight_number: null, departure: null, arrival: null, duration: null,
      stops: null, total_price: null, currency: null },
    { type: 'booking_summary', title: 'Quote', items: [], total: null, currency: null },
    { type: 'table', columns: [], rows: [] }
  ];
  const out = blocks(sparse);
  assert.equal(tagsIn(out, 'div').filter(d => d.className.startsWith('blk ')).length, 4);
  assert.ok(textOf(out).includes('Bare Hotel'));
  assert.ok(textOf(out).includes('JED → MCT'));
  assert.ok(textOf(out).includes('Quote'));
  assert.ok(!textOf(out).includes('null'), 'a null must never be printed');
  assert.ok(!textOf(out).includes('★'), 'no stars when stars is null');
  assert.ok(!textOf(out).includes('Direct'), 'no stop count when stops is null');
});

test('a card that throws does not take the message down', () => {
  /* items is the wrong shape entirely; the summary card must survive it */
  const out = blocks([{ type: 'booking_summary', title: 'Odd', items: 'not-a-list' }, HOTEL]);
  assert.ok(textOf(out).includes('Carawan Al Fahad'), 'the good card still rendered');
});

test('rendering is pure: it builds a fragment and touches nothing else', () => {
  /* what message history depends on — an assistant message cannot reach out and
     alter earlier ones, because the renderer only ever returns new nodes */
  const first = R.renderAssistantMessage({ output: 'first', blocks: [HOTEL] }, doc);
  const second = R.renderAssistantMessage({ output: 'second', blocks: null }, doc);
  assert.ok(textOf(first).includes('first'));
  assert.ok(textOf(first).includes('Carawan Al Fahad'));
  assert.ok(textOf(second).includes('second'));
  assert.ok(!textOf(second).includes('Carawan Al Fahad'));
  assert.ok(!textOf(first).includes('second'));
});

test('enrichment provenance renders when the turn carried it', () => {
  const frag = R.renderAssistantMessage({
    output: 'Oman has no travel restrictions.',
    sources: [{ url: 'https://www.gov.uk/foreign-travel-advice/oman', host: 'gov.uk',
                domain: 'advisory', observed_at: '2026-09-01T00:00:00+00:00',
                valid_until: '2026-09-02T00:00:00+00:00', is_stale: false }]
  }, doc);
  const a = tagsIn(frag, 'a');
  assert.equal(a.length, 1);
  assert.equal(a[0].attrs.rel, 'noopener noreferrer');
  assert.ok(textOf(frag).includes('gov.uk'));
  assert.ok(textOf(frag).includes('advisory'));
  assert.ok(textOf(frag).includes('seen 2026-09-01'));
});

test('a source with an unsafe url is shown as text, never as a link', () => {
  const frag = R.renderAssistantMessage({
    output: 'x', sources: [{ url: 'javascript:alert(1)', host: 'evil' }]
  }, doc);
  assert.equal(tagsIn(frag, 'a').length, 0);
  assert.ok(textOf(frag).includes('evil'));
});
