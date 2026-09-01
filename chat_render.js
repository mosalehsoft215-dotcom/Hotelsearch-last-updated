/**
 * Rendering for one assistant message: the markdown answer, then the structured
 * blocks underneath it.
 *
 * The markdown grammar is imported from the yarvel-ai-assistant frontend's
 * services/markdown.js — same feature set, same link-scheme allowlist, same
 * table/list/heading rules — with one deliberate change. That version builds an
 * HTML string and hands it to dangerouslySetInnerHTML, escaping the source
 * first. This one builds DOM nodes and never assembles markup from model output
 * at all, because chat_ui.html states the rule twice: build nodes, never
 * innerHTML. Model text only ever reaches createTextNode, so there is no path
 * by which a tag in an answer becomes a tag on the page — raw HTML cannot
 * render, and no sanitiser has to be trusted to have caught everything.
 *
 * Exported for tests via module.exports when running under node; attached to
 * window otherwise, since the page has no bundler and no build step.
 */
(function (root) {
  'use strict';

  /* Only schemes that cannot execute. Anything else — javascript:, data:,
     vbscript: — loses its href and renders as plain text, which is what the
     imported version does too. */
  var SAFE_URL = /^(https?:|mailto:)/i;

  function textNode(doc, text) {
    return doc.createTextNode(String(text == null ? '' : text));
  }

  function el(doc, tag, cls) {
    var node = doc.createElement(tag);
    if (cls) node.className = cls;
    return node;
  }

  /* --- inline: bold, italic, code, links ---------------------------------- */

  var INLINE_RULES = [
    { re: /`([^`\n]+)`/, make: function (doc, m) {
        var n = el(doc, 'code', 'md-code');
        n.appendChild(textNode(doc, m[1]));
        return n;                        /* code is literal: no nesting */
      } },
    { re: /\*\*([\s\S]+?)\*\*/, make: function (doc, m) {
        var n = el(doc, 'strong');
        n.appendChild(inlineFragment(doc, m[1]));
        return n;
      } },
    { re: /__([\s\S]+?)__/, make: function (doc, m) {
        var n = el(doc, 'strong');
        n.appendChild(inlineFragment(doc, m[1]));
        return n;
      } },
    { re: /\[([^\]]+)\]\(([^)\s]+)\)/, make: function (doc, m) {
        if (!SAFE_URL.test(m[2])) return textNode(doc, m[1]);
        var a = el(doc, 'a', 'md-link');
        a.setAttribute('href', m[2]);
        a.setAttribute('target', '_blank');
        a.setAttribute('rel', 'noopener noreferrer');
        a.appendChild(inlineFragment(doc, m[1]));
        return a;
      } },
    /* Group 1 is the character before the marker, which is part of the match
       and has to be put back as text. */
    { lead: 1, re: /(^|[^*\w])\*([^*\s][^*\n]*?)\*/, make: function (doc, m) {
        var n = el(doc, 'em');
        n.appendChild(inlineFragment(doc, m[2]));
        return n;
      } },
    { lead: 1, re: /(^|[^_\w])_([^_\s][^_\n]*?)_/, make: function (doc, m) {
        var n = el(doc, 'em');
        n.appendChild(inlineFragment(doc, m[2]));
        return n;
      } }
  ];

  function inlineFragment(doc, source) {
    var frag = doc.createDocumentFragment();
    var rest = String(source == null ? '' : source);
    var guard = 0;
    while (rest && guard++ < 5000) {
      var best = null;
      for (var i = 0; i < INLINE_RULES.length; i++) {
        var rule = INLINE_RULES[i];
        var m = rule.re.exec(rest);
        if (!m) continue;
        var lead = rule.lead && m[rule.lead] ? m[rule.lead].length : 0;
        /* Compare where the marker starts, not where the match starts, or a
           rule that swallows a leading character always looks earlier. */
        var at = m.index + lead;
        if (best === null || at < best.at) best = { rule: rule, m: m, at: at, lead: lead };
      }
      if (best === null) { frag.appendChild(textNode(doc, rest)); break; }
      var before = rest.slice(0, best.m.index) +
                   (best.lead ? best.m[best.rule.lead] : '');
      if (before) frag.appendChild(textNode(doc, before));
      frag.appendChild(best.rule.make(doc, best.m));
      rest = rest.slice(best.m.index + best.m[0].length);
    }
    return frag;
  }

  /* --- blocks: headings, lists, tables, quotes, code, paragraphs ---------- */

  var splitRow = function (line) {
    return line.replace(/^\|/, '').replace(/\|$/, '').split('|').map(function (c) {
      return c.trim();
    });
  };
  var isTableSep = function (line) {
    return line.indexOf('-') !== -1 && line.indexOf('|') !== -1 &&
           /^\|?[\s:|-]+\|?$/.test(line);
  };
  var isUl = function (line) { return /^[-*+]\s+/.test(line); };
  var isOl = function (line) { return /^\d+\.\s+/.test(line); };

  function renderMarkdown(source, doc) {
    doc = doc || root.document;
    var out = el(doc, 'div', 'md');
    var lines = String(source == null ? '' : source).split('\n');
    var para = [];

    function flushPara() {
      if (!para.length) return;
      var p = el(doc, 'p', 'md-p');
      for (var i = 0; i < para.length; i++) {
        if (i) p.appendChild(el(doc, 'br'));
        p.appendChild(inlineFragment(doc, para[i]));
      }
      out.appendChild(p);
      para = [];
    }

    var i = 0;
    while (i < lines.length) {
      var line = lines[i].trim();

      if (!line) { flushPara(); i++; continue; }

      if (/^```/.test(line)) {                      /* fenced code */
        flushPara();
        var code = [];
        i++;
        while (i < lines.length && !/^```/.test(lines[i].trim())) { code.push(lines[i]); i++; }
        if (i < lines.length) i++;                  /* closing fence */
        var pre = el(doc, 'pre', 'md-pre');
        var codeEl = el(doc, 'code');
        codeEl.appendChild(textNode(doc, code.join('\n')));
        pre.appendChild(codeEl);
        out.appendChild(pre);
        continue;
      }

      if (/^(-{3,}|\*{3,}|_{3,})$/.test(line)) {
        flushPara(); out.appendChild(el(doc, 'hr', 'md-hr')); i++; continue;
      }

      if (/^>(\s|$)/.test(line)) {
        flushPara();
        var quoted = [];
        while (i < lines.length) {
          var qm = lines[i].trim().match(/^>\s?(.*)$/);
          if (!qm) break;
          quoted.push(qm[1]); i++;
        }
        var quote = el(doc, 'blockquote', 'md-quote');
        for (var q = 0; q < quoted.length; q++) {
          if (q) quote.appendChild(el(doc, 'br'));
          quote.appendChild(inlineFragment(doc, quoted[q]));
        }
        out.appendChild(quote);
        continue;
      }

      var h = line.match(/^(#{1,6})\s+(.*)$/);
      if (h) {
        flushPara();
        /* Shifted down two levels: a message is not the page's outline, so an
           answer's "#" must not become an h1 next to the app's own headings. */
        var heading = el(doc, 'h' + Math.min(h[1].length + 2, 6), 'md-h');
        heading.appendChild(inlineFragment(doc, h[2]));
        out.appendChild(heading);
        i++; continue;
      }

      if (line.indexOf('|') !== -1 && i + 1 < lines.length &&
          isTableSep(lines[i + 1].trim())) {
        flushPara();
        var header = splitRow(line);
        i += 2;
        var body = [];
        while (i < lines.length && lines[i].trim() && lines[i].indexOf('|') !== -1) {
          body.push(splitRow(lines[i].trim())); i++;
        }
        out.appendChild(tableNode(doc, header, body, 'md-table'));
        continue;
      }

      if (isUl(line) || isOl(line)) {
        flushPara();
        var ordered = isOl(line);
        var match = ordered
          ? function (l) { return l.match(/^\d+\.\s+(.*)$/); }
          : function (l) { return l.match(/^[-*+]\s+(.*)$/); };
        var list = el(doc, ordered ? 'ol' : 'ul', 'md-list');
        while (i < lines.length) {
          var im = match(lines[i].trim());
          if (!im) break;
          var li = el(doc, 'li');
          li.appendChild(inlineFragment(doc, im[1]));
          list.appendChild(li);
          i++;
        }
        out.appendChild(list);
        continue;
      }

      para.push(line); i++;
    }
    flushPara();
    return out;
  }

  function tableNode(doc, columns, rows, cls) {
    var wrap = el(doc, 'div', 'md-table-wrap');
    var table = el(doc, 'table', cls || 'md-table');
    var thead = el(doc, 'thead');
    var hrow = el(doc, 'tr');
    (columns || []).forEach(function (c) {
      var th = el(doc, 'th');
      th.appendChild(inlineFragment(doc, c));
      hrow.appendChild(th);
    });
    thead.appendChild(hrow);
    var tbody = el(doc, 'tbody');
    (rows || []).forEach(function (r) {
      var tr = el(doc, 'tr');
      (r || []).forEach(function (c) {
        var td = el(doc, 'td');
        td.appendChild(inlineFragment(doc, c == null ? '' : c));
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(thead); table.appendChild(tbody);
    wrap.appendChild(table);
    return wrap;
  }

  /* --- structured blocks -------------------------------------------------- */

  function money(amount, currency) {
    if (amount == null || amount === '' || isNaN(Number(amount))) return null;
    var n = Number(amount);
    var shown = (Math.round(n * 100) / 100).toLocaleString('en-US',
      { minimumFractionDigits: n % 1 ? 2 : 0, maximumFractionDigits: 2 });
    return currency ? shown + ' ' + currency : shown;
  }

  function card(doc, kind) {
    return el(doc, 'div', 'blk blk-' + kind);
  }

  function cardTitle(doc, parent, text, badge) {
    var head = el(doc, 'div', 'blk-head');
    var title = el(doc, 'div', 'blk-title');
    title.appendChild(textNode(doc, text));
    head.appendChild(title);
    if (badge) {
      var b = el(doc, 'span', 'blk-badge');
      b.appendChild(textNode(doc, badge));
      head.appendChild(b);
    }
    parent.appendChild(head);
  }

  /* label/value rows. A null value is skipped entirely rather than shown as
     "null" — every optional field on every block reaches here. */
  function factRow(doc, parent, label, value) {
    if (value == null || value === '') return;
    var row = el(doc, 'div', 'blk-row');
    var k = el(doc, 'span', 'blk-k');
    k.appendChild(textNode(doc, label));
    var v = el(doc, 'span', 'blk-v');
    v.appendChild(textNode(doc, value));
    row.appendChild(k); row.appendChild(v);
    parent.appendChild(row);
  }

  function stars(value) {
    var n = Number(value);
    if (value == null || isNaN(n) || n <= 0) return null;
    var whole = Math.max(0, Math.min(5, Math.round(n)));
    return '★'.repeat(whole) + '☆'.repeat(5 - whole);
  }

  function HotelOptionCard(doc, b) {
    var node = card(doc, 'hotel');
    cardTitle(doc, node, b.hotel_name || 'Hotel',
      b.refundable === true ? 'Refundable' : (b.refundable === false ? 'Non-refundable' : null));
    factRow(doc, node, 'Rating', stars(b.stars));
    factRow(doc, node, 'Location', b.location);
    factRow(doc, node, 'Board', b.board);
    factRow(doc, node, 'Per night', money(b.price_per_night, b.currency));
    factRow(doc, node, 'Total', money(b.total_price, b.currency));
    factRow(doc, node, 'Cancellation', b.cancellation_summary);
    return node;
  }

  function FlightOptionCard(doc, b) {
    var node = card(doc, 'flight');
    var route = (b.origin || '?') + ' → ' + (b.destination || '?');
    cardTitle(doc, node, route,
      b.stops == null ? null : (b.stops === 0 ? 'Direct' : b.stops + ' stop' + (b.stops > 1 ? 's' : '')));
    var carrier = [b.airline, b.flight_number].filter(Boolean).join(' ');
    factRow(doc, node, 'Flight', carrier || null);
    factRow(doc, node, 'Departs', b.departure);
    factRow(doc, node, 'Arrives', b.arrival);
    factRow(doc, node, 'Duration', b.duration);
    factRow(doc, node, 'Total', money(b.total_price, b.currency));
    return node;
  }

  function BookingSummaryCard(doc, b) {
    var node = card(doc, 'summary');
    cardTitle(doc, node, b.title || 'Summary', null);
    (Array.isArray(b.items) ? b.items : []).forEach(function (item) {
      if (item && typeof item === 'object') factRow(doc, node, item.label, item.value);
    });
    var total = money(b.total, b.currency);
    if (total != null) {
      var row = el(doc, 'div', 'blk-row blk-total');
      var k = el(doc, 'span', 'blk-k');
      k.appendChild(textNode(doc, 'Total'));
      var v = el(doc, 'span', 'blk-v');
      v.appendChild(textNode(doc, total));
      row.appendChild(k); row.appendChild(v);
      node.appendChild(row);
    }
    return node;
  }

  function DataTableBlock(doc, b) {
    var node = card(doc, 'table');
    node.appendChild(tableNode(doc, b.columns || [], b.rows || [], 'md-table'));
    return node;
  }

  var BLOCK_RENDERERS = {
    hotel_option: HotelOptionCard,
    flight_option: FlightOptionCard,
    booking_summary: BookingSummaryCard,
    table: DataTableBlock
  };

  /**
   * Blocks for one message. An unknown type is skipped and a card that throws
   * is dropped — a display addition must never be able to take down the
   * message it was decorating, or the conversation.
   */
  function renderBlocks(blocks, doc) {
    doc = doc || root.document;
    var wrap = el(doc, 'div', 'blocks');
    if (!Array.isArray(blocks)) return wrap;
    blocks.forEach(function (b) {
      if (!b || typeof b !== 'object') return;
      var make = BLOCK_RENDERERS[b.type];
      if (!make) return;                       /* unknown type: ignore safely */
      try {
        wrap.appendChild(make(doc, b));
      } catch (err) {
        /* keep the rest of the message */
      }
    });
    return wrap;
  }

  /** Where enrichment claims came from and how fresh they are, when the turn
   *  carried any. Purely a display of metadata the tools already returned. */
  function renderSources(sources, doc) {
    doc = doc || root.document;
    var wrap = el(doc, 'div', 'srcs');
    if (!Array.isArray(sources) || !sources.length) return wrap;
    var head = el(doc, 'div', 'srcs-head');
    head.appendChild(textNode(doc, 'Sources'));
    wrap.appendChild(head);
    sources.forEach(function (s) {
      if (!s || typeof s !== 'object') return;
      var row = el(doc, 'div', 'srcs-row');
      if (s.url && SAFE_URL.test(s.url)) {
        var a = el(doc, 'a', 'md-link');
        a.setAttribute('href', s.url);
        a.setAttribute('target', '_blank');
        a.setAttribute('rel', 'noopener noreferrer');
        a.appendChild(textNode(doc, s.host || s.url));
        row.appendChild(a);
      } else {
        row.appendChild(textNode(doc, s.host || '(source)'));
      }
      var bits = [];
      if (s.domain) bits.push(s.domain);
      if (s.observed_at) bits.push('seen ' + String(s.observed_at).slice(0, 10));
      if (s.is_stale) bits.push('stale');
      else if (s.valid_until) bits.push('fresh to ' + String(s.valid_until).slice(0, 10));
      if (bits.length) {
        var meta = el(doc, 'span', 'srcs-meta');
        meta.appendChild(textNode(doc, ' · ' + bits.join(' · ')));
        row.appendChild(meta);
      }
      wrap.appendChild(row);
    });
    return wrap;
  }

  /**
   * One assistant message: markdown answer, then blocks, then sources.
   * The order is fixed — prose first, structured data under it.
   */
  function renderAssistantMessage(data, doc) {
    doc = doc || root.document;
    data = data || {};
    var frag = doc.createDocumentFragment();
    frag.appendChild(renderMarkdown(data.output || '', doc));
    var blocks = renderBlocks(data.blocks, doc);
    if (blocks.childNodes.length) frag.appendChild(blocks);
    var sources = renderSources(data.sources, doc);
    if (sources.childNodes.length) frag.appendChild(sources);
    return frag;
  }

  var api = {
    renderMarkdown: renderMarkdown,
    renderBlocks: renderBlocks,
    renderSources: renderSources,
    renderAssistantMessage: renderAssistantMessage,
    HotelOptionCard: HotelOptionCard,
    FlightOptionCard: FlightOptionCard,
    BookingSummaryCard: BookingSummaryCard,
    DataTableBlock: DataTableBlock
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ChatRender = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
