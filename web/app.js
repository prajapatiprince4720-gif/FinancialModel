/* EquityLens front end.
 *
 * Plain ES2017 — no modules, no bundler, no CDN, so this runs identically in
 * Safari, Chrome, Firefox and Edge straight off the server with no build step.
 *
 * Deliberately avoided for older-Safari parity:
 *   - optional chaining / nullish coalescing (Safari < 13.1)
 *   - Array.prototype.at (Safari < 15.4)
 *   - top-level await, dynamic import
 *   - String.replaceAll (Safari < 13.1)
 */
(function () {
  'use strict';

  // ── Small helpers ─────────────────────────────────────────────────────────

  function $(id) { return document.getElementById(id); }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }

  function clear(node) {
    while (node.firstChild) { node.removeChild(node.firstChild); }
  }

  /* Everything user- or model-supplied is inserted via textContent, never
     innerHTML, so a company name or an LLM answer cannot inject markup. */
  function setText(node, text) {
    clear(node);
    node.appendChild(document.createTextNode(text === null || text === undefined ? '' : String(text)));
  }

  function fmtNum(value, decimals) {
    if (value === null || value === undefined || value === '') { return '—'; }
    var num = Number(value);
    if (!isFinite(num)) { return '—'; }
    return num.toLocaleString('en-IN', {
      minimumFractionDigits: decimals || 0,
      maximumFractionDigits: decimals || 0
    });
  }

  function fmtRupees(value, decimals) {
    if (value === null || value === undefined) { return '—'; }
    var num = Number(value);
    if (!isFinite(num)) { return '—'; }
    return '₹' + fmtNum(num, decimals === undefined ? 2 : decimals);
  }

  function fmtPct(value, decimals, withSign) {
    if (value === null || value === undefined) { return '—'; }
    var num = Number(value);
    if (!isFinite(num)) { return '—'; }
    var d = decimals === undefined ? 1 : decimals;
    var sign = withSign && num > 0 ? '+' : '';
    return sign + num.toFixed(d) + '%';
  }

  function verdictClass(verdict) {
    var v = String(verdict || '').toUpperCase();
    if (v.indexOf('BUY') >= 0) { return 'pill pill--buy'; }
    if (v.indexOf('SELL') >= 0) { return 'pill pill--sell'; }
    if (v.indexOf('HOLD') >= 0) { return 'pill pill--hold'; }
    return 'pill pill--none';
  }

  function api(path, options) {
    var opts = options || {};
    opts.headers = opts.headers || {};
    opts.headers['Accept'] = 'application/json';
    return fetch(path, opts).then(function (response) {
      return response.text().then(function (body) {
        var data = null;
        try { data = body ? JSON.parse(body) : null; } catch (e) { data = null; }
        if (!response.ok) {
          var message = data && data.detail ? data.detail
            : 'Request failed (' + response.status + ' ' + response.statusText + ')';
          var error = new Error(message);
          error.status = response.status;
          throw error;
        }
        return data;
      });
    });
  }

  function postJSON(path, payload) {
    return api(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  }

  function banner(container, kind, message) {
    clear(container);
    if (!message) { return; }
    var box = el('div', 'banner banner--' + kind);
    box.appendChild(document.createTextNode(message));
    container.appendChild(box);
  }

  function busy(button, isBusy, busyLabel) {
    if (!button) { return; }
    if (isBusy) {
      button.dataset.label = button.textContent;
      button.disabled = true;
      clear(button);
      button.appendChild(el('span', 'spinner' + (button.className.indexOf('ghost') >= 0 ? ' spinner--dark' : '')));
      button.appendChild(document.createTextNode(busyLabel || 'Working…'));
    } else {
      button.disabled = false;
      setText(button, button.dataset.label || busyLabel || 'Done');
    }
  }

  // ── App state ─────────────────────────────────────────────────────────────

  var state = {
    stocks: [],
    leaderboard: [],
    report: null,
    faqPairs: [],
    coverage: null
  };

  // ── Tabs ──────────────────────────────────────────────────────────────────

  var TABS = ['research', 'leaderboard', 'ask', 'coverage'];
  var loaded = {};

  function selectTab(name) {
    TABS.forEach(function (tab) {
      var button = $('tab-' + tab);
      var panel = $('panel-' + tab);
      var active = tab === name;
      button.setAttribute('aria-selected', active ? 'true' : 'false');
      panel.hidden = !active;
    });
    if (!loaded[name]) {
      loaded[name] = true;
      if (name === 'leaderboard') { loadLeaderboard(); }
      if (name === 'coverage') { loadCoverage(); }
    }
  }

  TABS.forEach(function (tab) {
    $('tab-' + tab).addEventListener('click', function () { selectTab(tab); });
  });

  // ── Bootstrap ─────────────────────────────────────────────────────────────

  function loadHealth() {
    return api('/api/health').then(function (data) {
      var macro = data.macro || {};
      var strip = $('macroStrip');
      clear(strip);
      strip.appendChild(document.createTextNode('India 10Y GSEC '));
      var rate = el('strong', null, fmtPct(macro.rf_pct, 2));
      strip.appendChild(rate);
      strip.appendChild(document.createTextNode(
        ' · ' + (macro.is_live ? 'live · ' + (macro.source || '') : 'NOT LIVE')
      ));

      var health = data.data_health || {};
      var notes = [];
      if (!macro.is_live) {
        notes.push('The risk-free rate is a hardcoded fallback, so every intrinsic value is approximate.');
      }
      if (health.stale_count) {
        notes.push(health.stale_count + ' of ' + health.companies +
          ' companies have fundamentals older than ' + health.stale_threshold_days +
          ' days (oldest ' + Math.round(health.oldest_age_days) +
          ' days). Refresh with: python main.py cache --refresh');
      }
      banner($('globalBanner'), 'warn', notes.join(' '));
      return data;
    }).catch(function (error) {
      banner($('globalBanner'), 'error', 'Could not reach the API: ' + error.message);
    });
  }

  function loadDisclaimer() {
    return api('/api/disclaimer').then(function (data) {
      setText($('disclaimerText'), data.full);
    }).catch(function () { /* non-fatal */ });
  }

  function loadStocks() {
    return api('/api/stocks').then(function (data) {
      state.stocks = data.stocks || [];
      var picker = $('stockPicker');
      var askPicker = $('askSymbol');
      clear(picker);
      var placeholder = el('option', null, 'Select a company…');
      placeholder.value = '';
      picker.appendChild(placeholder);

      state.stocks.forEach(function (stock) {
        var option = el('option', null, stock.name + '  (' + stock.symbol + ')');
        option.value = stock.symbol;
        picker.appendChild(option);

        var askOption = el('option', null, stock.name);
        askOption.value = stock.symbol;
        askPicker.appendChild(askOption);
      });
    }).catch(function (error) {
      banner($('globalBanner'), 'error', 'Could not load the company list: ' + error.message);
    });
  }

  // ── Research panel ────────────────────────────────────────────────────────

  function statTile(label, value, sub, valueClass) {
    var tile = el('div', 'stat');
    tile.appendChild(el('div', 'stat__label', label));
    tile.appendChild(el('div', 'stat__value' + (valueClass ? ' ' + valueClass : ''), value));
    if (sub) { tile.appendChild(el('div', 'stat__sub', sub)); }
    return tile;
  }

  function renderReport(data) {
    var area = $('reportArea');
    clear(area);
    state.report = data;

    var val = data.valuation || {};
    var score = data.score;

    // Header
    var head = el('div', 'card');
    var headRow = el('div', 'card__head');
    var title = el('div');
    title.appendChild(el('h2', null, data.name));
    title.appendChild(el('div', 'card__note', 'NSE: ' + data.symbol + ' · ' + data.sector));
    headRow.appendChild(title);
    var spacer = el('div');
    spacer.style.flex = '1 1 auto';
    headRow.appendChild(spacer);
    headRow.appendChild(el('span', verdictClass(val.verdict), val.verdict || 'N/A'));
    head.appendChild(headRow);

    if (data.warnings && data.warnings.length) {
      data.warnings.forEach(function (warning) {
        var box = el('div', 'banner banner--warn', warning);
        head.appendChild(box);
      });
    }

    var mos = val.margin_of_safety_pct;
    var grid = el('div', 'grid');
    grid.appendChild(statTile('Market price', fmtRupees(val.cmp)));
    grid.appendChild(statTile('Intrinsic value', fmtRupees(val.intrinsic_value),
      val.model + ' model'));
    grid.appendChild(statTile('Margin of safety', fmtPct(mos, 1, true),
      mos === null || mos === undefined ? 'no price data'
        : (mos >= 0 ? 'trading below model value' : 'trading above model value'),
      mos === null || mos === undefined ? '' : (mos >= 0 ? 'stat__value--pos' : 'stat__value--neg')));
    grid.appendChild(statTile('Valuation range',
      fmtRupees(val.iv_low, 0) + ' – ' + fmtRupees(val.iv_high, 0), 'P10–P90, Monte Carlo'));
    grid.appendChild(statTile('Discount rate', fmtPct(val.wacc_pct, 2),
      val.model === 'RIM' ? 'cost of equity' : 'WACC'));
    if (score) {
      grid.appendChild(statTile('Quality score', score.composite + ' / 100',
        'grade ' + score.grade));
    }
    head.appendChild(grid);
    area.appendChild(head);

    // Score breakdown
    if (score) {
      var scoreCard = el('div', 'card');
      var scoreHead = el('div', 'card__head');
      scoreHead.appendChild(el('span', 'card__title', 'Quality breakdown'));
      scoreHead.appendChild(el('span', 'card__note',
        'Percentiles are relative to the other 49 Nifty 50 companies.'));
      scoreCard.appendChild(scoreHead);

      var scoreGrid = el('div', 'grid');
      [
        ['Composite', score.composite, 100],
        ['Quality', score.quality, 100],
        ['Growth', score.growth, 100],
        ['Earnings quality', score.earnings_quality, 100],
        ['Piotroski F-Score', score.piotroski, 9]
      ].forEach(function (row) {
        var tile = el('div', 'stat');
        tile.appendChild(el('div', 'stat__label', row[0]));
        tile.appendChild(el('div', 'stat__value', row[1] + ' / ' + row[2]));
        var bar = el('div', 'bar');
        var fill = el('div', 'bar__fill');
        var pct = Math.max(0, Math.min(100, (Number(row[1]) / Number(row[2])) * 100));
        fill.style.width = pct.toFixed(1) + '%';
        bar.appendChild(fill);
        tile.appendChild(bar);
        scoreGrid.appendChild(tile);
      });
      scoreCard.appendChild(scoreGrid);

      var ratios = el('div', 'grid');
      ratios.style.marginTop = '12px';
      ratios.appendChild(statTile('Revenue CAGR 3Y', fmtPct(score.revenue_cagr_3y, 1, true)));
      ratios.appendChild(statTile('EPS CAGR 3Y', fmtPct(score.eps_cagr_3y, 1, true)));
      ratios.appendChild(statTile('FCF CAGR 3Y', fmtPct(score.fcf_cagr_3y, 1, true)));
      ratios.appendChild(statTile('Debt / equity', Number(score.de_ratio).toFixed(2) + 'x'));
      ratios.appendChild(statTile('ROCE percentile', fmtNum(score.roce_percentile, 0) + 'th'));
      scoreCard.appendChild(ratios);
      area.appendChild(scoreCard);
    }

    // Financial table
    var fin = data.financials;
    if (fin && fin.years && fin.years.length) {
      var finCard = el('div', 'card');
      var finHead = el('div', 'card__head');
      finHead.appendChild(el('span', 'card__title', 'Five-year financials'));
      finHead.appendChild(el('span', 'card__note', '₹ Crore unless marked otherwise'));
      finCard.appendChild(finHead);

      var scroll = el('div', 'table-scroll');
      var table = el('table');
      var thead = el('thead');
      var headTr = el('tr');
      headTr.appendChild(el('th', null, 'Metric'));
      fin.years.forEach(function (year) { headTr.appendChild(el('th', null, year)); });
      thead.appendChild(headTr);
      table.appendChild(thead);

      var tbody = el('tbody');
      [
        [fin.is_bank ? 'Revenue' : 'Sales', fin.sales, 0, false],
        ['Net profit', fin.net_profit, 0, false],
        ['EPS (₹)', fin.eps, 2, false],
        [fin.opm_label, fin.opm, 1, true],
        [fin.roce_label, fin.roce, 1, true],
        ['Cash from operations', fin.cfo, 0, false],
        ['Free cash flow', fin.fcf, 0, false]
      ].forEach(function (row) {
        var tr = el('tr');
        tr.appendChild(el('td', null, row[0]));
        var values = row[1] || [];
        fin.years.forEach(function (_, index) {
          var raw = values[index];
          tr.appendChild(el('td', null, row[3] ? fmtPct(raw, row[2]) : fmtNum(raw, row[2])));
        });
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      scroll.appendChild(table);
      finCard.appendChild(scroll);
      area.appendChild(finCard);
    }

    // Model notes + provenance
    var infoCard = el('div', 'card');
    var infoHead = el('div', 'card__head');
    infoHead.appendChild(el('span', 'card__title', 'Model notes and data provenance'));
    infoCard.appendChild(infoHead);

    if (val.notes && val.notes.length) {
      var noteList = el('ul', 'provenance');
      val.notes.forEach(function (note) { noteList.appendChild(el('li', null, '• ' + note)); });
      infoCard.appendChild(noteList);
    }

    var provList = el('ul', 'provenance');
    provList.style.marginTop = '8px';
    (data.provenance || []).forEach(function (line) {
      var flagged = line.indexOf('STALE') >= 0 || line.indexOf('NOT LIVE') >= 0;
      provList.appendChild(el('li', flagged ? 'flag' : null, line));
    });
    infoCard.appendChild(provList);

    var thesisHolder = el('div');
    thesisHolder.id = 'thesisHolder';
    infoCard.appendChild(thesisHolder);
    area.appendChild(infoCard);

    $('thesisBtn').disabled = false;
    $('pdfBtn').disabled = false;
    $('docxBtn').disabled = false;
  }

  function loadReport() {
    var symbol = $('stockPicker').value;
    if (!symbol) {
      banner($('globalBanner'), 'info', 'Choose a company first.');
      return;
    }
    var button = $('loadReportBtn');
    busy(button, true, 'Valuing…');
    var area = $('reportArea');
    clear(area);
    var loadingCard = el('div', 'card');
    loadingCard.appendChild(el('p', 'muted', 'Running the valuation model and 10,000 Monte Carlo draws…'));
    for (var i = 0; i < 4; i++) {
      var skeleton = el('div', 'skeleton');
      skeleton.style.marginTop = '8px';
      loadingCard.appendChild(skeleton);
    }
    area.appendChild(loadingCard);

    api('/api/report/' + encodeURIComponent(symbol))
      .then(renderReport)
      .catch(function (error) {
        clear(area);
        banner(area, 'error', 'Could not build the report: ' + error.message);
      })
      .then(function () { busy(button, false, 'Generate research'); });
  }

  function loadThesis() {
    if (!state.report) { return; }
    var button = $('thesisBtn');
    busy(button, true, 'Writing…');
    postJSON('/api/thesis', { symbol: state.report.symbol })
      .then(function (data) {
        var holder = $('thesisHolder');
        if (!holder) { return; }
        clear(holder);
        holder.appendChild(el('h3', 'card__title', 'AI investment thesis'));
        var body = el('div', 'answer');
        setText(body, data.thesis);
        holder.appendChild(body);
        holder.appendChild(el('p', 'card__note',
          'Written by a language model from the data above. Verify any figure before acting on it.'));
      })
      .catch(function (error) {
        var holder = $('thesisHolder');
        if (holder) { banner(holder, 'error', 'Thesis generation failed: ' + error.message); }
      })
      .then(function () { busy(button, false, 'Add AI thesis'); });
  }

  /* Downloads go through a normal navigation rather than fetch + blob:
     Safari blocks programmatic blob downloads in several configurations, and a
     plain location change lets the browser's own download manager handle it. */
  function download(format) {
    if (!state.report) { return; }
    var url = '/api/export/' + encodeURIComponent(state.report.symbol) +
      '?fmt=' + encodeURIComponent(format);
    var frame = $('dlFrame');
    if (!frame) {
      frame = document.createElement('iframe');
      frame.id = 'dlFrame';
      frame.style.display = 'none';
      document.body.appendChild(frame);
    }
    frame.src = url;
    banner($('globalBanner'), 'info',
      'Preparing the ' + format.toUpperCase() + ' — your browser will download it shortly.');
    window.setTimeout(function () { banner($('globalBanner'), 'info', ''); loadHealth(); }, 6000);
  }

  $('loadReportBtn').addEventListener('click', loadReport);
  $('thesisBtn').addEventListener('click', loadThesis);
  $('pdfBtn').addEventListener('click', function () { download('pdf'); });
  $('docxBtn').addEventListener('click', function () { download('docx'); });
  $('stockPicker').addEventListener('change', function () {
    $('thesisBtn').disabled = true;
    $('pdfBtn').disabled = true;
    $('docxBtn').disabled = true;
  });

  // ── Leaderboard ───────────────────────────────────────────────────────────

  function renderLeaderboard() {
    var filter = $('lbFilter').value.toLowerCase();
    var tbody = $('lbTable').getElementsByTagName('tbody')[0];
    clear(tbody);

    var rows = state.leaderboard.filter(function (row) {
      if (!filter) { return true; }
      return (row.name + ' ' + row.symbol + ' ' + row.sector).toLowerCase().indexOf(filter) >= 0;
    });

    if (!rows.length) {
      var empty = el('tr');
      var cell = el('td', 'center muted', 'No companies match that filter.');
      cell.colSpan = 10;
      empty.appendChild(cell);
      tbody.appendChild(empty);
      return;
    }

    rows.forEach(function (row) {
      var tr = el('tr');
      var nameCell = el('td');
      var link = el('button', 'row-link', row.name);
      link.type = 'button';
      link.addEventListener('click', function () {
        $('stockPicker').value = row.symbol;
        selectTab('research');
        loadReport();
      });
      nameCell.appendChild(link);
      tr.appendChild(nameCell);
      tr.appendChild(el('td', null, row.sector));
      tr.appendChild(el('td', null, Number(row.composite).toFixed(1)));
      tr.appendChild(el('td', null, row.grade));
      tr.appendChild(el('td', null, Number(row.quality).toFixed(1)));
      tr.appendChild(el('td', null, Number(row.growth).toFixed(1)));
      tr.appendChild(el('td', null, row.piotroski + '/9'));
      tr.appendChild(el('td', null, fmtPct(row.revenue_cagr_3y, 1, true)));
      tr.appendChild(el('td', null, fmtPct(row.eps_cagr_3y, 1, true)));
      tr.appendChild(el('td', null, Number(row.de_ratio).toFixed(2) + 'x'));
      tbody.appendChild(tr);
    });
  }

  function loadLeaderboard() {
    api('/api/leaderboard').then(function (data) {
      state.leaderboard = data.rows || [];
      renderLeaderboard();
    }).catch(function (error) {
      var tbody = $('lbTable').getElementsByTagName('tbody')[0];
      clear(tbody);
      var tr = el('tr');
      var cell = el('td', 'center muted', 'Could not load rankings: ' + error.message);
      cell.colSpan = 10;
      tr.appendChild(cell);
      tbody.appendChild(tr);
    });
  }

  $('lbFilter').addEventListener('input', renderLeaderboard);
  $('lbRefresh').addEventListener('click', loadLeaderboard);

  // ── Ask ───────────────────────────────────────────────────────────────────

  function renderFaq() {
    var filter = $('faqFilter').value.toLowerCase();
    var list = $('faqList');
    clear(list);

    var pairs = state.faqPairs.filter(function (pair) {
      if (!filter) { return true; }
      return (pair.question + ' ' + pair.answer).toLowerCase().indexOf(filter) >= 0;
    });

    if (!pairs.length) {
      list.appendChild(el('p', 'muted', 'No pre-answered questions match.'));
      return;
    }

    pairs.slice(0, 80).forEach(function (pair) {
      var item = el('div', 'qa__item');
      var question = el('button', 'qa__q');
      question.type = 'button';
      question.appendChild(el('span', 'qa__marker', 'Q'));
      question.appendChild(document.createTextNode(pair.question));

      var answer = el('div', 'qa__a');
      setText(answer, pair.answer);
      answer.hidden = true;

      question.addEventListener('click', function () { answer.hidden = !answer.hidden; });
      item.appendChild(question);
      item.appendChild(answer);
      list.appendChild(item);
    });

    if (pairs.length > 80) {
      list.appendChild(el('p', 'muted',
        'Showing 80 of ' + pairs.length + ' — narrow the search to see more.'));
    }
  }

  function loadFaq(symbol) {
    var card = $('faqCard');
    if (!symbol) { card.hidden = true; return; }
    api('/api/faq/' + encodeURIComponent(symbol)).then(function (data) {
      state.faqPairs = data.pairs || [];
      card.hidden = state.faqPairs.length === 0;
      setText($('faqNote'), data.company + ' · ' + data.answered + ' of ' +
        data.expected + ' questions answered (' + data.coverage_pct + '%)' +
        (data.trained ? '' : ' · still training'));
      renderFaq();
    }).catch(function () { card.hidden = true; });
  }

  function ask() {
    var question = $('askInput').value.trim();
    if (question.length < 3) {
      banner($('askArea'), 'info', 'Type a question first.');
      return;
    }
    var button = $('askBtn');
    busy(button, true, 'Thinking…');
    var area = $('askArea');
    clear(area);

    postJSON('/api/ask', { question: question, symbol: $('askSymbol').value })
      .then(function (data) {
        clear(area);
        var box = el('div', 'answer');
        setText(box, data.answer);
        area.appendChild(box);
        area.appendChild(el('p', 'card__note', data.disclaimer));
      })
      .catch(function (error) {
        clear(area);
        banner(area, 'error', error.message);
      })
      .then(function () { busy(button, false, 'Ask'); });
  }

  $('askBtn').addEventListener('click', ask);
  $('askInput').addEventListener('keydown', function (event) {
    if (event.key === 'Enter') { ask(); }
  });
  $('askSymbol').addEventListener('change', function () { loadFaq(this.value); });
  $('faqFilter').addEventListener('input', renderFaq);

  // ── Coverage ──────────────────────────────────────────────────────────────

  function loadCoverage() {
    api('/api/training-status').then(function (data) {
      state.coverage = data;
      var summary = $('coverageSummary');
      clear(summary);
      var grid = el('div', 'grid');
      grid.appendChild(statTile('Companies trained',
        data.trained_count + ' / ' + data.total,
        'at least ' + data.threshold_pct + '% of the question bank'));
      grid.appendChild(statTile('Partly trained', data.partial.length));
      grid.appendChild(statTile('Not started', data.missing.length));
      grid.appendChild(statTile('Questions remaining', fmtNum(data.questions_remaining, 0)));
      summary.appendChild(grid);

      var tbody = $('coverageTable').getElementsByTagName('tbody')[0];
      clear(tbody);
      var symbols = Object.keys(data.coverage).sort(function (a, b) {
        return data.coverage[b].answered - data.coverage[a].answered;
      });
      var names = {};
      state.stocks.forEach(function (stock) { names[stock.symbol] = stock.name; });

      symbols.forEach(function (symbol) {
        var row = data.coverage[symbol];
        var pct = row.expected ? (row.answered / row.expected) * 100 : 0;
        var tr = el('tr');
        tr.appendChild(el('td', null, names[symbol] || symbol));
        tr.appendChild(el('td', null, row.answered));
        tr.appendChild(el('td', null, row.expected));
        var pctCell = el('td', null, pct.toFixed(0) + '%');
        tr.appendChild(pctCell);
        var status = el('td');
        status.appendChild(el('span',
          row.trained ? 'pill pill--buy' : (row.answered ? 'pill pill--hold' : 'pill pill--none'),
          row.trained ? 'trained' : (row.answered ? 'partial' : 'pending')));
        tr.appendChild(status);
        tbody.appendChild(tr);
      });
    }).catch(function (error) {
      banner($('coverageSummary'), 'error', 'Could not load coverage: ' + error.message);
    });

    api('/api/health').then(function (data) {
      var health = data.data_health || {};
      var area = $('healthArea');
      clear(area);
      var grid = el('div', 'grid');
      grid.appendChild(statTile('Fundamentals stale',
        health.stale_count + ' / ' + health.companies,
        'older than ' + health.stale_threshold_days + ' days'));
      grid.appendChild(statTile('Oldest data',
        health.oldest_age_days === null ? '—' : Math.round(health.oldest_age_days) + ' days'));
      grid.appendChild(statTile('Newest data',
        health.newest_age_days === null ? '—' : Math.round(health.newest_age_days) + ' days'));
      grid.appendChild(statTile('Risk-free rate',
        health.rf_is_live ? 'live' : 'fallback', health.rf_label,
        health.rf_is_live ? 'stat__value--pos' : 'stat__value--neg'));
      area.appendChild(grid);
      if (health.stale_count) {
        area.appendChild(el('p', 'card__note',
          'Refresh fundamentals with:  python main.py cache --refresh'));
      }
    }).catch(function () { /* already reported by the global banner */ });
  }

  // ── Go ────────────────────────────────────────────────────────────────────

  selectTab('research');
  loadHealth();
  loadDisclaimer();
  loadStocks();
})();
