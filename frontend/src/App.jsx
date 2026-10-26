'use client';
// BloomBot UI. The markup is generated from the Claude Design file (src/views/AppView.jsx); this
// class is the design's logic, extended to call the real API. When the API is not running it falls
// back to the recorded runs in src/designData.js ("Replay").

import React from 'react';
import { view } from './views/AppView.jsx';
import { BB } from './designData.js';
import { api, apiBase } from './api.js';
import * as A from './adapt.js';
import { SAMPLE_OS_PAPER } from './samples.js';

export default class App extends React.Component {
  static defaultProps = { frame: 'auto', traceSpeed: 260 };

  state = { ready: true, page: 'check', theme: 'light', llm: true, winW: 1280,
    checkStage: 'upload', paper: null, blueprint: 'balanced', resolved: {}, dragging: false, pasteText: '', fileName: '', assessed: false, reportT: 1,
    clsText: '', clsKey: 'bridge', showMasked: false, showReasoning: true, parseOpen: false,
    jobStatus: 'idle', jobN: 0, rwStatus: {}, rwText: {}, afterT: 1,
    labelCount: 0, labelIdx: 0, lastLabel: null, activeProfile: 'none',
    node: 'Lexis', step: 0,
    // live API
    online: null, llmAvailable: null, llmModel: '', error: '', busy: '',
    apiPaper: null, report: null, D: {}, clsLive: null, labels: {}, profileId: null, progress: 0 };

  componentDidMount() {
    this.setState({ paper: BB.PAPER.map(q => ({ ...q })), clsText: BB.BRIDGE.text, winW: window.innerWidth,
      theme: (() => { try { return localStorage.getItem('bb.theme') || 'light'; } catch (e) { return 'light'; } })() });
    this.onResize = () => this.setState({ winW: window.innerWidth });
    this.onKey = e => {
      if (this.state.page !== 'profile') return;
      const tag = (e.target && e.target.tagName) || '';
      if (/INPUT|TEXTAREA|SELECT/.test(tag)) return;
      const n = parseInt(e.key, 10); if (n >= 1 && n <= 6) { e.preventDefault(); this.label(n); }
    };
    window.addEventListener('resize', this.onResize); window.addEventListener('keydown', this.onKey);
    this.apiReady = this.checkApi();
  }
  componentDidUpdate(_, prev) {
    if (prev.jobN !== this.state.jobN) {  // keep the newest trace event in view
      const logEl = document.querySelector('[role="log"]');
      if (logEl) logEl.scrollTop = logEl.scrollHeight;
    }
  }
  componentWillUnmount() { clearInterval(this.poll); clearTimeout(this.timer); cancelAnimationFrame(this.raf); clearTimeout(this.pt); this.dead = true;
    window.removeEventListener('resize', this.onResize); window.removeEventListener('keydown', this.onKey); }

  // ---------- API ----------
  live() { return this.state.online === true; }
  set(patch) { return new Promise(r => this.setState(patch, r)); }
  async checkApi() {
    try {
      const m = await api.meta();
      this.setState({ online: true, llmAvailable: !!m.llm.available, llmModel: m.llm.model, llm: !!m.llm.available });
    } catch (e) { this.setState({ online: false }); }
  }
  useLlm() { return this.state.llm ? null : false; }
  fail(e) { this.setState({ error: e.message || String(e), busy: '' }); }
  bpId() { return A.BPID[this.state.blueprint]; }

  anim(key) { const t0 = performance.now(); cancelAnimationFrame(this.raf);
    const f = now => { const t = Math.min(1, (now - t0) / 900); this.setState({ [key]: 1 - Math.pow(1 - t, 3) }); if (t < 1) this.raf = requestAnimationFrame(f); };
    this.setState({ [key]: 0 }); this.raf = requestAnimationFrame(f); }
  go(p) { this.setState({ page: p, parseOpen: false }); try { window.scrollTo(0, 0); } catch (e) {} }
  label(n) {
    const q = BB.POOL[this.state.labelIdx % BB.POOL.length];
    this.setState(s => ({ labelCount: s.labelCount + 1, labelIdx: s.labelIdx + 1, lastLabel: n, labels: { ...s.labels, [q]: A.L[n - 1] } }));
  }

  // Load a parsed paper (live) or the recorded sample (replay).
  async parse(name, source) {
    await this.apiReady;
    this.setState({ checkStage: 'parsing', fileName: name || 'pasted text', error: '' });
    if (!this.live()) {
      if (source !== 'sample') this.setState({ error: 'The API is not running, so BloomBot cannot read your own paper. Showing the recorded sample paper instead. Start it with `bloombot serve`.' });
      clearTimeout(this.pt);
      this.pt = setTimeout(() => this.setState({ checkStage: 'edit', apiPaper: null, report: null, D: {}, paper: BB.PAPER.map(q => ({ ...q })) }), 900);
      return;
    }
    try {
      const paper = source && source.name ? await api.parseFile(source) : await api.parseText(source === 'sample' ? SAMPLE_OS_PAPER : this.state.pasteText,
        source === 'sample' ? 'Operating Systems' : 'Pasted paper');
      if (!paper.questions.length) throw new Error('No questions were found. Check the text has numbered questions (Q1, 1., 1 a) …).');
      const rows = A.paperRows(paper);
      await this.set({ checkStage: 'edit', apiPaper: paper, report: null, paper: rows, resolved: {},
        D: { SECTIONS: A.sections(rows), TITLE: A.paperTitle(paper, rows), OR_NOTE: A.orNote(rows) } });
    } catch (e) { this.setState({ checkStage: 'upload' }); this.fail(e); }
  }

  async assessPaper() {
    const s = this.state;
    if (!this.live() || !s.apiPaper) {
      this.setState({ checkStage: 'report', assessed: true }); this.anim('reportT'); try { window.scrollTo(0, 0); } catch (e) {}
      return true;
    }
    this.setState({ busy: 'assess', error: '' });
    try {
      const paper = A.paperForApi(s.apiPaper, s.paper);
      const [report, bps] = await Promise.all([
        api.assess({ paper, blueprint_id: this.bpId(), use_llm: this.useLlm(), profile_id: s.profileId }),
        this.blueprints || (this.blueprints = fetch(apiBase() + '/v1/blueprints').then(r => r.json())),
      ]);
      const bp = bps.find(b => b.id === this.bpId());
      const rows = A.paperRows(paper, report).map(r => ({ ...r, locked: (s.paper.find(x => x.id === r.id) || {}).locked }));
      await this.set(st => ({ busy: '', report, apiPaper: paper, paper: rows, checkStage: 'report', assessed: true, resolved: {},
        D: { ...st.D, SECTIONS: A.sections(rows), TITLE: A.paperTitle(paper, rows), OR_NOTE: A.orNote(rows), DIST_BEFORE: A.dist(report.distribution),
          BPAPI: bp, BP: { ...BB.BP, [st.blueprint]: A.blueprint(bp, report) } } }));
      this.anim('reportT'); try { window.scrollTo(0, 0); } catch (e) {}
      return true;
    } catch (e) { this.fail(e); return false; }
  }

  async startJob() {
    await this.apiReady;
    if (this.live()) return this.startLiveJob();
    const B = BB; clearTimeout(this.timer);
    const st = {}; B.RW.forEach(r => st[r.id] = 'pending');
    this.setState(s => ({ page: 'rebalance', jobStatus: 'running', jobN: 0, rwStatus: st, rwText: {}, assessed: true, D: {}, paper: B.PAPER.map(q => ({ ...q })) }));
    const scale = (this.props.traceSpeed ?? 260) / 260;
    const tick = () => {
      const n = this.state.jobN + 1;
      if (n >= B.TRACE.length) { this.setState({ jobN: B.TRACE.length, jobStatus: 'done' }); this.anim('afterT'); return; }
      this.setState({ jobN: n });
      const ms = B.TRACE[n].ms;
      this.timer = setTimeout(tick, Math.max(110, Math.min(900, ms / 3)) * scale);
    };
    clearTimeout(this.timer); this.timer = setTimeout(tick, 300 * scale);
    try { window.scrollTo(0, 0); } catch (e) {}
  }

  async startLiveJob() {
    clearInterval(this.poll);
    this.setState({ page: 'rebalance', error: '' });
    try { window.scrollTo(0, 0); } catch (e) {}
    if (!this.state.apiPaper) await this.parse('sample paper', 'sample');
    if (!this.state.report && !(await this.assessPaper())) return;
    await this.set({ page: 'rebalance' });
    const s = this.state;
    try {
      const paper = A.paperForApi(s.apiPaper, s.paper);
      const jobs = await api.plan({ report: s.report, blueprint_id: this.bpId() });
      const locked = new Set(s.paper.filter(q => q.locked).map(q => q.id));
      const plan = jobs.map(A.rwFromJob);
      const st = {}; plan.forEach(r => st[r.id] = 'pending');
      this.setState(x => ({ jobStatus: 'running', jobN: 0, rwStatus: st, rwText: {}, progress: 0.02,
        D: { ...x.D, TRACE: [], RW: plan, PLAN: plan, JOB_ID: '…', HANDBACK: null } }));
      const job = await api.rebalance({ paper, blueprint_id: this.bpId(), use_llm: this.useLlm(), profile_id: s.profileId });
      this.setState(x => ({ D: { ...x.D, JOB_ID: job.id } }));
      const open = plan.filter(r => !locked.has(r.id)).length || 1;
      this.poll = setInterval(async () => {
        let j;
        try { j = await api.job(job.id); } catch (e) { return; }
        if (this.dead) return;
        const trace = j.trace.filter(e => e.agent);
        const rows = trace.map(A.traceRow);
        const checked = new Set(trace.filter(e => e.agent === 'Verifier' && e.detail.passed).map(e => String(e.detail.question_id)));
        const architect = trace.some(e => e.agent === 'Architect');
        this.setState(x => ({ jobN: rows.length, progress: architect ? 0.15 + 0.85 * checked.size / open : Math.min(0.14, 0.02 + rows.length * 0.02),
          D: { ...x.D, TRACE: rows, RW: A.rwProgress(x.D.PLAN || [], trace) } }));
        if (j.status === 'failed') { clearInterval(this.poll); this.setState({ jobStatus: 'idle' }); this.fail(new Error(j.error || 'The rebalance job failed.')); }
        if (j.status === 'done') { clearInterval(this.poll); this.finishLiveJob(j, rows); }
      }, 700);
    } catch (e) { this.setState({ jobStatus: 'idle' }); this.fail(e); }
  }

  finishLiveJob(j, rows) {
    const res = j.result, s = this.state;
    const rw = res.rewrites.map(A.rwFromResult);
    const failed = rw.filter(r => r.handedBack);
    const after = {};
    res.after.items.forEach(it => { after[it.question.id] = { level: it.prediction.level, conf: it.prediction.confidence }; });
    const bp = s.D.BPAPI;
    this.setState(x => ({ jobStatus: 'done', jobN: rows.length, progress: 1,
      rwStatus: Object.fromEntries(rw.map(r => [r.id, r.handedBack ? 'pending' : (x.rwStatus[r.id] || 'pending')])),
      D: { ...x.D, TRACE: rows, RW: rw.filter(r => !r.handedBack), FAILED: failed, HANDBACK: failed[0] || null,
        DIST_BEFORE: A.dist(res.before.distribution), DIST_AFTER: A.dist(res.after.distribution),
        AFTER_REVIEW: res.after.needs_review, AFTER_PRED: after, AFTER_COMPLIANT: res.after.compliant,
        BP: bp ? { ...x.D.BP, [x.blueprint]: A.blueprint(bp, res.before, res.after) } : x.D.BP } }));
    this.anim('afterT');
  }

  async runLiveClassify(text) {
    this.setState({ busy: 'classify', error: '' });
    try {
      const [c, an] = await Promise.all([api.classify({ questions: [text], use_llm: this.useLlm(), profile_id: this.state.profileId }), api.analyze(text)]);
      this.setState({ busy: '', clsLive: A.example(c.predictions[0], an), clsKey: null, showMasked: false });
    } catch (e) { this.fail(e); }
  }

  async useProfile(id) {
    if (id === 'none' || !this.live()) { this.setState({ activeProfile: id, profileId: null }); return; }
    const ex = Object.entries(this.state.labels).map(([question, level]) => ({ question, level }));
    if (ex.length < 5) { this.setState({ error: `Label at least 5 different questions first (you have ${ex.length}). 25–100 is recommended.` }); return; }
    this.setState({ busy: 'profile', error: '' });
    try {
      const p = await api.createProfile('My college · CSE', ex);
      this.setState({ busy: '', activeProfile: 'mine', profileId: p.id, report: null });
    } catch (e) { this.fail(e); }
  }

  download(name, text, type) { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; document.body.appendChild(a); a.click(); a.remove(); }
  B() { return { ...BB, ...this.state.D }; }
  title() { return this.state.D.TITLE || 'Operating Systems · 50 marks'; }
  applied() { const B = this.B(), s = this.state; const out = {};
    B.RW.forEach(r => { const q = (s.paper || []).find(x => x.id === r.id); if (q && q.locked) return; const st = s.rwStatus[r.id];
      if (st === 'accepted' || st === 'edited') out[r.id] = s.rwText[r.id] || r.rewrite; }); return out; }
  paperMd() { const B = this.B(), ap = this.applied();
    let md = `# ${this.title()}\n\nRebalanced with BloomBot against ${B.BP[this.state.blueprint].name}.\n\n`;
    B.SECTIONS.forEach(sec => { md += `## ${sec.title} — ${sec.note}\n\n`;
      (this.state.paper || B.PAPER).filter(q => q.section === sec.id).forEach(q => { md += `- **${q.number}** (${q.marks} marks${q.co ? ', ' + q.co : ''}) ${ap[q.id] || q.text}\n`; }); md += '\n'; });
    return md; }
  reportObj() { const B = this.B(), bp = B.BP[this.state.blueprint];
    return { paper: this.title(), blueprint: bp.name, before: { distribution: B.DIST_BEFORE, checks: bp.checks.map(c => ({ bound: c.label, share: c.b })) },
      after: { distribution: B.DIST_AFTER, checks: bp.checks.map(c => ({ bound: c.label, share: c.a })) },
      rewrites: B.RW.map(r => ({ q: r.q, from: B.L[r.from - 1], to: B.L[r.to - 1], status: this.state.rwStatus[r.id] || 'pending', text: this.state.rwText[r.id] || r.rewrite })),
      trace: this.state.D.TRACE || null }; }
  h() { return React.createElement; }
  lineChart(series, o) {
    const h = this.h(), W = o.w || 640, H = o.h || 300, pl = 44, pr = 168, pt = 16, pb = 36;
    const xs = o.xs, xMax = xs[xs.length - 1], x = i => o.proportional ? pl + xs[i] / xMax * (W - pl - pr) : pl + i * (W - pl - pr) / (xs.length - 1), y = v => pt + (o.max - v) / (o.max - o.min) * (H - pt - pb);
    const els = [];
    for (let v = o.min; v <= o.max + 1e-9; v += o.step) { els.push(h('line', { key: 'g' + v, x1: pl, x2: W - pr, y1: y(v), y2: y(v), style: { stroke: 'var(--line)' } }), h('text', { key: 't' + v, x: pl - 8, y: y(v) + 4, textAnchor: 'end', style: { fill: 'var(--ink3)', font: "11px 'JetBrains Mono',monospace" } }, v.toFixed(2))); }
    xs.forEach((k, i) => els.push(h('text', { key: 'x' + i, x: x(i) + (o.proportional && i === 1 ? 6 : 0), y: H - 12, textAnchor: 'middle', style: { fill: 'var(--ink3)', font: "11px 'JetBrains Mono',monospace" } }, 'k=' + k)));
    if (o.marker != null) els.push(h('line', { key: 'mk', x1: x(o.marker), x2: x(o.marker), y1: pt, y2: H - pb, style: { stroke: 'var(--accent)', strokeDasharray: '4 4', strokeWidth: 1.5 } }));
    const ends = series.map(s => ({ s, y: y(s.values[s.values.length - 1]) })).sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 14) ends[i].y = ends[i - 1].y + 14;
    series.forEach((s, si) => {
      els.push(h('polyline', { key: 'l' + si, points: s.values.map((v, i) => x(i) + ',' + y(v)).join(' '), fill: 'none', style: { stroke: s.color, strokeWidth: s.width || 1.5, strokeDasharray: s.dash || 'none', opacity: s.opacity || 1 } }));
      s.values.forEach((v, i) => els.push(h('circle', { key: 'c' + si + i, cx: x(i), cy: y(v), r: s.bold ? 4 : 2.5, style: { fill: s.color, opacity: s.opacity || 1 } }, h('title', null, `${s.name} k=${xs[i]}: ${v.toFixed(3)}`))));
    });
    ends.forEach((e, i) => els.push(h('text', { key: 'e' + i, x: W - pr + 10, y: e.y + 4, style: { fill: e.s.bold ? 'var(--ink)' : 'var(--ink2)', font: (e.s.bold ? '600 ' : '') + "12px 'Instrument Sans',sans-serif" } }, `${e.s.name} ${e.s.values[e.s.values.length - 1].toFixed(3)}`)));
    return h('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', role: 'img', 'aria-label': o.aria, style: { display: 'block', maxWidth: W + 'px' } }, els);
  }
  arcSvg() {
    const h = this.h(), toks = ['Explain', 'how', 'engineers', 'evaluate', 'the', 'safety', 'of', 'an', 'old', 'bridge', '.'];
    const arcs = [[0, 3, 'ccomp'], [3, 1, 'advmod'], [3, 2, 'nsubj'], [3, 5, 'dobj'], [5, 4, 'det'], [5, 6, 'prep'], [6, 9, 'pobj'], [9, 7, 'det'], [9, 8, 'amod'], [0, 10, 'punct']];
    const gap = 78, W = toks.length * gap + 20, base = 236, y0 = 214, x = i => 20 + i * gap + gap / 2;
    const els = [h('line', { key: 'root', x1: x(0) - 8, x2: x(0) - 8, y1: 22, y2: y0, style: { stroke: 'var(--accent)', strokeWidth: 2 } }), h('text', { key: 'rt', x: x(0) - 8, y: 14, textAnchor: 'middle', style: { fill: 'var(--accent)', font: "700 11px 'JetBrains Mono',monospace" } }, 'ROOT')];
    arcs.forEach(([a, b, lab], k) => { const x1 = x(a) + (a === 0 ? 6 : 0), x2 = x(b), d = Math.abs(b - a), ht = 18 + d * 17, yy = y0;
      els.push(h('path', { key: 'p' + k, d: `M${x1},${yy} C${x1},${yy - ht} ${x2},${yy - ht} ${x2},${yy}`, fill: 'none', style: { stroke: 'var(--ink3)', strokeWidth: 1.3 } }),
        h('path', { key: 'a' + k, d: `M${x2 - 4},${yy - 7} L${x2},${yy} L${x2 + 4},${yy - 7}`, fill: 'none', style: { stroke: 'var(--ink3)', strokeWidth: 1.3 } }),
        h('text', { key: 'l' + k, x: (x1 + x2) / 2, y: yy - ht * 0.75 - 3, textAnchor: 'middle', style: { fill: 'var(--ink2)', font: "11px 'JetBrains Mono',monospace", paintOrder: 'stroke', stroke: 'var(--surface)', strokeWidth: 4 } }, lab)); });
    toks.forEach((t, i) => { const role = i === 0 ? 'directive' : i === 3 ? 'trap' : null;
      els.push(h('text', { key: 'w' + i, x: x(i), y: base, textAnchor: 'middle', style: { fill: role === 'directive' ? 'var(--accent)' : role === 'trap' ? 'var(--warn)' : 'var(--ink)', font: (role ? '700 ' : '') + "15px 'Instrument Sans',sans-serif" } }, t));
      if (role) els.push(h('text', { key: 'r' + i, x: x(i), y: base + 18, textAnchor: 'middle', style: { fill: 'var(--ink3)', font: "10px 'JetBrains Mono',monospace" } }, role === 'directive' ? 'directive' : 'trap: not the task')); });
    return h('svg', { viewBox: `0 0 ${W} 270`, width: W, height: 270, role: 'img', 'aria-label': 'Dependency parse: ROOT is explain; evaluate is an embedded clause', style: { display: 'block' } }, els);
  }
  ring(count) { const h = this.h(), r = 54, c = 2 * Math.PI * r, p = Math.min(count / 100, 1);
    const tick = f => { const a = f * 2 * Math.PI - Math.PI / 2; return h('line', { key: 'k' + f, x1: 70 + Math.cos(a) * 46, y1: 70 + Math.sin(a) * 46, x2: 70 + Math.cos(a) * 62, y2: 70 + Math.sin(a) * 62, style: { stroke: 'var(--ink)', strokeWidth: 2 } }); };
    return h('svg', { viewBox: '0 0 140 140', width: 140, height: 140, role: 'img', 'aria-label': `${count} of 100 labelled` },
      h('circle', { cx: 70, cy: 70, r, fill: 'none', style: { stroke: 'var(--track)', strokeWidth: 10 } }),
      h('circle', { cx: 70, cy: 70, r, fill: 'none', transform: 'rotate(-90 70 70)', strokeDasharray: `${c * p} ${c}`, strokeLinecap: 'round', style: { stroke: 'var(--accent)', strokeWidth: 10, transition: 'stroke-dasharray .4s' } }),
      tick(0.25), tick(0.5),
      h('text', { x: 70, y: 70, textAnchor: 'middle', style: { fill: 'var(--ink)', font: "500 34px 'Newsreader',serif" } }, String(count)),
      h('text', { x: 70, y: 90, textAnchor: 'middle', style: { fill: 'var(--ink3)', font: "11px 'JetBrains Mono',monospace" } }, 'of 100'));
  }

  renderVals() {
    const s = this.state, B = this.B(), live = this.live();
    const narrow = this.props.frame === 'phone' || s.winW < 820;
    const base = { theme: s.theme, frameW: this.props.frame === 'phone' ? '390px' : '100%', frameShadow: this.props.frame === 'phone' ? '0 0 0 1px var(--line), 0 20px 60px rgba(0,0,0,.12)' : 'none',
      shellDir: narrow ? 'column' : 'row', pad: narrow ? '20px 16px 72px' : '36px 48px 96px', narrow, wide: !narrow, showRail: !narrow, showTopbar: narrow,
      h1fs: narrow ? '34px' : '48px' };
    if (!s.paper) return { ...base, ready: false };
    const L = B.L, FG = n => n >= 5 ? '#fff' : '#1b1f2a', chip = n => ({ n, name: L[n - 1] || '—', bg: `var(--l${n})`, fg: FG(n) }), N = x => L.indexOf(x) + 1;
    const pct = v => Math.round(v * 100) + '%';
    const navItem = (id, num, label) => ({ id, num, label, go: () => this.go(id), bg: s.page === id ? 'var(--accent-soft)' : 'transparent', color: s.page === id ? 'var(--ink)' : 'var(--ink2)', weight: s.page === id ? 600 : 500, current: s.page === id ? 'page' : undefined });
    const navTeacher = [navItem('check', '01', 'Check a paper'), navItem('classify', '02', 'Classify a question'), navItem('rebalance', '03', 'Rebalance'), navItem('profile', '04', 'College profile')];
    const navDemo = [navItem('how', '05', 'How it works'), navItem('eval', '06', 'Evaluation'), navItem('data', '07', 'Data')];
    const navSheet = [navItem('sheet', '··', 'Component sheet')];
    const titles = { check: 'Check a paper', classify: 'Classify a question', rebalance: 'Rebalance', profile: 'College profile', how: 'How it works', eval: 'Evaluation', data: 'Data', sheet: 'Component sheet' };
    const bp = B.BP[s.blueprint];
    const qById = id => s.paper.find(q => q.id === id);
    const paperTitle = this.title();

    // ---------- Check ----------
    const lockedIds = s.paper.filter(q => q.locked).map(q => q.id);
    const sections = B.SECTIONS.map(sec => {
      const qs = s.paper.filter(q => q.section === sec.id);
      return { ...sec, rows: qs.map((q, i) => {
        const inG = !!q.group, first = inG && (i === 0 || qs[i - 1].group !== q.group);
        const upd = patch => this.setState(st => ({ paper: st.paper.map(x => x.id === q.id ? { ...x, ...patch } : x), report: patch.locked === undefined ? null : st.report }));
        return { ...q, inGroup: inG, showOr: inG && !first, groupLabel: first ? `Answer either Q${q.id} or Q${qs[i + 1] ? qs[i + 1].id : ''} · one ${q.marks}-mark slot` : '', showGroupLabel: first,
          rowBg: inG ? 'var(--surface2)' : 'transparent', groupBorder: inG ? 'var(--accent)' : 'transparent',
          hasTag: !!q.tag, noTag: !q.tag, tagChip: q.tag ? chip(N(q.tag)) : chip(1), coTxt: q.co || '—', hasContext: !!q.context, onText: e => upd({ text: e.target.value }), onMarks: e => upd({ marks: +e.target.value || 0 }),
          toggleLock: () => upd({ locked: !q.locked }), lockLabel: q.locked ? '🔒 Locked' : 'Lock', lockPressed: q.locked ? 'true' : 'false',
          lockBg: q.locked ? 'var(--ink)' : 'transparent', lockFg: q.locked ? 'var(--bg)' : 'var(--ink2)',
          lockTitle: q.locked ? 'Locked: BloomBot will never rewrite this question' : 'Lock so BloomBot never rewrites it' };
      }) };
    });
    const blueprints = ['aicte', 'balanced'].map(id => ({ id, label: BB.BP[id].label, sub: BB.BP[id].sub, pick: () => this.setState({ blueprint: id, report: s.apiPaper ? null : s.report }),
      border: s.blueprint === id ? 'var(--accent)' : 'var(--line)', bg: s.blueprint === id ? 'var(--accent-soft)' : 'var(--surface)', dot: s.blueprint === id ? 'var(--accent)' : 'transparent', checked: s.blueprint === id ? 'true' : 'false' }));
    const checkRows = (key, t) => bp.checks.map(c => { const v = key === 'b' ? c.b : c.a; const pass = key === 'a' ? c.aOk !== false : !!c.bOk;
      return { label: c.label, shareTxt: Math.round(v * t) + '%', icon: pass ? '✓' : '✗', color: pass ? 'var(--good)' : 'var(--bad)', bg: pass ? 'var(--goodbg)' : 'var(--badbg)', note: key === 'b' ? c.bn : c.an }; });
    const failing = bp.checks.filter(c => !c.bOk).length;
    const tableRows = s.paper.map(q => { const res = s.resolved[q.id]; const lvl = res ? N(res) : q.pred; const agree = !!q.tag && N(q.tag) === lvl, disagree = !!q.tag && !agree;
      return { ...q, chip: chip(lvl), confTxt: q.conf.toFixed(2), confColor: q.conf < 0.6 ? 'var(--warn)' : 'var(--ink2)', lowConf: q.conf < 0.6, hasContext: !!q.context,
        agree, disagree, untagged: !q.tag, tagName: q.tag, predName: L[lvl - 1], tagBg: disagree ? 'var(--warnbg)' : 'transparent',
        flag: q.review && !res, resolvedFlag: !!res, resolvedTxt: res ? 'Teacher: ' + res : '', noFlag: !q.review && !res, bySource: res ? 'teacher' : 'predicted' }; });
    const reviewQueue = s.paper.filter(q => q.review).map(q => { const res = s.resolved[q.id];
      return { ...q, resolved: !!res, open: !res, resolvedText: res ? `Resolved — you chose ${res}.` : '',
        votes: q.votes.map(v => ({ ...v, chip: chip(N(v.level)), confTxt: v.confidence.toFixed(2), role: { Lexis: 'Classifier · weight 1', Archivist: 'Evidence only · weight 0', Socrates: 'Local LLM · weight 0.5' }[v.agent] })),
        choices: [...new Set(q.votes.map(v => v.level))].map(l => ({ label: 'Use ' + l, pick: () => this.setState(st => ({ resolved: { ...st.resolved, [q.id]: l } })) })),
        undo: () => this.setState(st => { const r = { ...st.resolved }; delete r[q.id]; return { resolved: r }; }) }; });
    const openReviews = reviewQueue.filter(r => r.open).length;
    const check = {
      checkUpload: s.checkStage === 'upload', checkParsing: s.checkStage === 'parsing', checkEdit: s.checkStage === 'edit', checkReport: s.checkStage === 'report',
      dropBorder: s.dragging ? 'var(--accent)' : 'var(--line)', dropBg: s.dragging ? 'var(--accent-soft)' : 'var(--surface)',
      onDragOver: e => { e.preventDefault(); if (!s.dragging) this.setState({ dragging: true }); }, onDragLeave: () => this.setState({ dragging: false }),
      onDrop: e => { e.preventDefault(); const f = e.dataTransfer.files[0]; this.setState({ dragging: false }); if (f) this.parse(f.name, f); },
      onFile: e => { const f = e.target.files[0]; if (f) this.parse(f.name, f); e.target.value = ''; },
      pasteText: s.pasteText, onPaste: e => this.setState({ pasteText: e.target.value }),
      parsePaste: () => { if (!s.pasteText.trim()) { this.setState({ error: 'Paste some questions first, or try the sample paper.' }); return; } this.parse('pasted text', 'paste'); },
      useSample: () => this.parse('sample OS paper', 'sample'), fileName: s.fileName,
      sections, tableWide: !narrow && s.winW >= 1240, tableNarrow: narrow || s.winW < 1240, rowCols: narrow ? '40px minmax(0,1fr) 56px' : '52px minmax(0,1fr) 64px 52px 118px 104px',
      blueprints, assess: () => { if (!s.busy) this.assessPaper(); }, assessLabel: s.busy === 'assess' ? 'Assessing…' : 'Assess',
      backToEdit: () => this.setState({ checkStage: 'edit' }), uploadNew: () => this.setState({ checkStage: 'upload' }),
      bpName: bp.name, markers: bp.markers, distBefore: B.DIST_BEFORE, checksBefore: checkRows('b', s.reportT),
      failCount: failing, verdictSub: failing ? `${failing} of ${bp.checks.length} bound${bp.checks.length > 1 ? 's' : ''} fail against ${bp.name}.` : `All ${bp.checks.length} bound${bp.checks.length > 1 ? 's' : ''} met against ${bp.name}.`,
      verdictText: failing ? 'Not compliant' : 'Compliant', verdictIcon: failing ? '✗' : '✓', verdictBg: failing ? 'var(--badbg)' : 'var(--goodbg)', verdictBorder: failing ? 'var(--bad)' : 'var(--good)',
      paperTitle, paperNote: s.apiPaper ? `Read from ${s.fileName}.` : 'Recorded sample paper (the API is not running).', orNote: B.OR_NOTE || 'OR-alternatives share one slot: Q2/Q3 are one 10-mark slot; if they get different levels, each counts 5 marks.',
      reviewQueue, openReviews, reviewTitle: openReviews ? `${openReviews} question${openReviews > 1 ? 's' : ''} to resolve first` : 'Review queue clear',
      tableRows, tableCols: '44px minmax(220px,1fr) 48px 150px 52px 160px 92px',
      lockedNote: lockedIds.length ? `${lockedIds.length} locked: ${lockedIds.map(i => 'Q' + i).join(', ')} — never rewritten.` : 'No questions locked.',
      profileName: s.activeProfile === 'mine' ? `My college · CSE (k = ${Object.keys(s.labels).length || s.labelCount})` : 'No profile (generic model)',
      rebalance: () => this.startJob()
    };

    // ---------- Classify ----------
    const ex = B.EXAMPLES.find(e => e.key === s.clsKey);
    const cls = s.clsLive || (ex ? ex.ex : null);
    const clsVals = { clsText: s.clsText, onClsText: e => this.setState({ clsText: e.target.value }),
      examples: B.EXAMPLES.map(e => ({ label: e.label, pick: () => this.setState({ clsText: e.ex.text, clsKey: e.key, clsLive: null, showMasked: false }),
        bg: !s.clsLive && s.clsKey === e.key ? 'var(--ink)' : 'var(--surface)', fg: !s.clsLive && s.clsKey === e.key ? 'var(--bg)' : 'var(--ink)', pressed: s.clsKey === e.key ? 'true' : 'false', trap: e.key === 'bridge' })),
      runClassify: async () => { const t = s.clsText.trim(); if (!t || s.busy) return;
        await this.apiReady;
        if (this.live()) { this.runLiveClassify(t); return; }
        const m = B.EXAMPLES.find(e => e.ex.text.toLowerCase() === t.toLowerCase()); this.setState({ clsKey: m ? m.key : null, clsLive: null }); },
      clsBtn: s.busy === 'classify' ? 'Classifying…' : 'Classify',
      clsKnown: !!cls, clsUnknown: !cls };
    if (cls) {
      let dirSeen = false;
      const toks = cls.text.split(/\s+/).map(w => { const k = w.toLowerCase().replace(/[^a-z\-]/g, '');
        const isDir = cls.directive && !dirSeen && k === cls.directive.verb; if (isDir) dirSeen = true;
        const role = isDir ? 'd' : cls.embedded.includes(k) ? 't' : null;
        return { t: w, bg: role === 'd' ? 'var(--accent-soft)' : 'transparent', border: role === 'd' ? '2px solid var(--accent)' : role === 't' ? '2px dashed var(--warn)' : '2px solid transparent', weight: role ? 700 : 400, tip: role === 'd' ? 'Directive: the task' : role === 't' ? 'Embedded Bloom verb — trap: not the task' : '' }; });
      const asked = cls.votes.some(v => v.agent === 'Socrates');
      const lex = cls.votes.find(v => v.agent === 'Lexis'), arch = cls.votes.find(v => v.agent === 'Archivist'), soc = cls.votes.find(v => v.agent === 'Socrates');
      const why = lex.confidence < 0.6 ? `Lexis unsure (${lex.confidence.toFixed(2)} < 0.6)` : (arch && arch.level !== lex.level ? `Lexis and the Archivist disagree (${lex.level} vs ${arch.level})` : 'Lexis unsure');
      const sorted = cls.proba.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
      const fooled = !!soc && cls.embedded.length > 0 && soc.level !== lex.level;
      const voteList = cls.votes.map(v => { const skip = v.agent === 'Socrates' && !s.llm, isFooled = fooled && v.agent === 'Socrates';
          return { ...v, chip: chip(N(v.level)), confTxt: v.confidence.toFixed(2), role: { Lexis: 'EmbedLex classifier', Archivist: 'Nearest labelled questions', Socrates: 'Qwen3-4B · reasoning first' }[v.agent],
            weight: { Lexis: 'weight 1', Archivist: 'weight 0 · evidence only', Socrates: 'weight 0.5' }[v.agent],
            skip, live: !skip, notAsked: false, fooled: isFooled && !skip, border: isFooled && !skip ? 'var(--bad)' : 'var(--line)',
            hasReasoning: !!v.reasoning && !skip, reasoning: v.reasoning, showReasoning: s.showReasoning, reasonLabel: s.showReasoning ? 'Hide reasoning' : 'Show reasoning',
            toggleReasoning: () => this.setState({ showReasoning: !s.showReasoning }) }; });
      if (!asked) voteList.push({ agent: 'Socrates', role: 'Qwen3-4B · reasoning first', weight: 'weight 0.5', skip: false, live: false, notAsked: true,
        notAskedText: !s.llm ? 'Socrates is off — fast mode (no LLM).' : `Socrates not asked — Lexis confident (${lex.confidence.toFixed(2)})${arch && arch.level === lex.level ? ' and the Archivist agrees' : ''}.`, border: 'var(--line)' });
      Object.assign(clsVals, { cls, clsProba: cls.proba, clsChip: chip(N(cls.level)), clsConf: cls.conf.toFixed(2), clsConfPct: pct(cls.conf),
        clsFlag: cls.needsReview, clsNoFlag: !cls.needsReview, clsOrder: cls.lowerOrder ? 'Lower order' : 'Higher order',
        isTrapHero: fooled, closeCall: `${L[sorted[0][1]]} ${sorted[0][0].toFixed(2)} vs ${L[sorted[1][1]]} ${sorted[1][0].toFixed(2)}`,
        lexTxt: `${lex.level} ${lex.confidence.toFixed(2)}`, socTxt: soc ? `${soc.level} ${soc.confidence.toFixed(2)}` : '', archTxt: arch ? `${arch.level} ${arch.confidence.toFixed(2)}` : '',
        tokens: toks, hasDirective: !!cls.directive, noDirective: !cls.directive,
        directiveText: cls.directive ? `${cls.directive.verb} · ${cls.directive.kind}${cls.directive.obj ? ' · object “' + cls.directive.obj + '”' : ''}${cls.directive.negated ? ' · negated' : ' · not negated'}` : 'none',
        hasPhrase: !!(cls.directive && cls.directive.phrase), phraseText: cls.directive && cls.directive.phrase ? `“${cls.directive.phrase}”` : '',
        hasTrap: cls.embedded.length > 0, trapText: cls.embedded.join(', '), frameText: cls.frame ? cls.frame + (cls.wh ? ` · wh-word “${cls.wh}”` : '') : 'none', evidenceText: cls.evidence,
        hasMasked: !!cls.masked, masked: cls.masked, showMasked: s.showMasked, toggleMasked: () => this.setState({ showMasked: !s.showMasked }), maskedLabel: s.showMasked ? 'Hide what the model sees' : 'What the model sees',
        votes: voteList,
        voteCols: 'repeat(auto-fit,minmax(min(100%,260px),1fr))',
        askedNote: asked ? (s.llm ? `Socrates was asked because ${why}. Pooled: Lexis × 1 + Socrates × 0.5; the Archivist is evidence only.` : 'Fast mode: Socrates (the LLM) is off, so the result rests on Lexis alone.') : (!s.llm ? 'Fast mode: Socrates (the LLM) is off, so the result rests on Lexis alone.' : `Socrates not asked — Lexis confident (${lex.confidence.toFixed(2)}).`),
        hasNeighbours: !!(cls.neighbours && cls.neighbours.length), neighbours: (cls.neighbours || []).map(n => ({ ...n, chip: chip(N(n.level)), sim: n.similarity.toFixed(3), simW: pct(n.similarity) })),
        hasParse: !!cls.parse, openParse: () => this.setState({ parseOpen: true }) });
    }
    const drawer = { parseOpen: s.parseOpen, closeParse: () => this.setState({ parseOpen: false }), arc: s.parseOpen || (s.page === 'how' && s.step === 1) ? this.arcSvg() : null };

    // ---------- Rebalance ----------
    const TR = B.TRACE, n = s.jobN, ev = TR.slice(0, n), seen = new Set(ev.map(e => e.agent)), cur = s.jobStatus === 'running' && ev.length ? ev[ev.length - 1].agent : null;
    let cum = 0;
    const fmtMs = ms => ms >= 1000 ? (ms / 1000).toFixed(2) + ' s' : ms.toFixed(ms < 10 ? 1 : 0) + ' ms';
    const traceRows = ev.map((e, i) => { cum += e.ms || 0; return { i: String(i + 1).padStart(2, '0'), agent: e.agent, action: e.action, detail: e.detail, ms: fmtMs(e.ms), barW: Math.max(2, Math.min(100, Math.log10(1 + e.ms) / Math.log10(3000) * 100)) + '%', latest: i === ev.length - 1 && s.jobStatus === 'running', rowBg: i === ev.length - 1 && s.jobStatus === 'running' ? 'var(--accent-soft)' : 'transparent' }; });
    if (n > 0) seen.add('Scribe');
    const stages = B.STAGES.map((st, i) => { const done = seen.has(st) && st !== cur, act = st === cur;
      return { name: st, showArrow: i > 0, bg: act ? 'var(--accent)' : done ? 'var(--accent-soft)' : 'var(--surface)', fg: act ? 'var(--accent-ink)' : done ? 'var(--ink)' : 'var(--ink3)', border: act || done ? 'var(--accent)' : 'var(--line)', anim: act ? 'bbpulse 1.2s infinite' : 'none', mark: done ? '✓' : act ? '●' : '○' }; });
    const rwVisibleAt = r => { const idx = TR.findIndex(e => e.agent === 'Verifier' && e.qid === r.id && e.passed); return idx >= 0 && n > idx; };
    const archIdx = TR.findIndex(e => e.agent === 'Architect');
    const showPlan = live ? !!B.PLAN : (archIdx >= 0 && n > archIdx);
    const planSrc = B.PLAN || B.RW;
    const planRows = planSrc.map(r => { const q = qById(r.id); return { q: r.q, from: chip(r.from), to: chip(r.to), reason: (r.reason || '').replace('<=', '≤').replace('>=', '≥'), marks: r.marks + ' marks', locked: !!(q && q.locked), open: !(q && q.locked) }; });
    const reasons = bp.checks.filter(c => !c.bOk).map(c => `${c.label} of marks — currently ${c.b}%`);
    const rwCards = B.RW.filter(r => !(qById(r.id) || {}).locked && rwVisibleAt(r)).map(r => ({ id: r.id, rw: { ...r, rewrite: s.rwText[r.id] || r.rewrite }, status: s.rwStatus[r.id] || 'pending',
      onAction: (st, text) => this.setState(x => ({ rwStatus: { ...x.rwStatus, [r.id]: st }, rwText: text != null ? { ...x.rwText, [r.id]: text } : x.rwText })) }));
    const notApplied = B.RW.filter(r => (qById(r.id) || {}).locked || ['rejected'].includes(s.rwStatus[r.id])).length + (B.FAILED ? B.FAILED.length : 0);
    const pending = B.RW.filter(r => !(qById(r.id) || {}).locked && (s.rwStatus[r.id] || 'pending') === 'pending').length;
    const afterCompliant = B.AFTER_COMPLIANT !== undefined ? B.AFTER_COMPLIANT : true;
    const afterReview = B.AFTER_REVIEW || [];
    const hand = B.HANDBACK || BB.HANDBACK;
    const reb = { jobIdle: s.jobStatus === 'idle', jobRunning: s.jobStatus === 'running', jobDone: s.jobStatus === 'done', notIdle: s.jobStatus !== 'idle', startJob: () => this.startJob(),
      startLabel: live ? (s.apiPaper ? 'Start rebalance job' : 'Start on the sample OS paper') : 'Replay the recorded job',
      idleText: live ? (s.apiPaper ? `Rebalance ${paperTitle} against ${bp.name}. The job runs on the API and streams every agent step here; review each rewrite as it lands.` : 'Rebalance opens from a checked paper. Start it on the sample OS paper to watch the agents work live — the job runs on the API and you can review each rewrite as it lands.')
        : 'The API is not running, so this replays a recorded run on the sample OS paper. Start `bloombot serve` (and Ollama) to run it live.',
      jobLabel: s.jobStatus === 'running' ? (live ? `Running · ${n} events` : `Running · ${n} of ${TR.length} events`) : `Done · ${TR.length} events`, jobId: B.JOB_ID,
      afterReviewTxt: afterReview.length ? `After rewriting, ${afterReview.map(i => 'Q' + i).join(' and ')} ${afterReview.length > 1 ? 'are' : 'is'} flagged ⚑ for review — check ${afterReview.length > 1 ? 'them' : 'it'} before you submit.` : 'After rewriting, no question is flagged for review.',
      stages, traceRows, totalMs: (cum / 1000).toFixed(1) + ' s', progressW: live ? Math.round(s.progress * 100) + '%' : Math.round(n / TR.length * 100) + '%',
      showPlan, planRows, planTitle: planSrc.length ? `${planSrc.length} rewrite${planSrc.length > 1 ? 's' : ''} needed` : 'No rewrites needed — the paper already meets the blueprint',
      planWhy: reasons.length ? 'Why: ' + reasons.join('; ') + '.' : '',
      rwCards, hasCards: rwCards.length > 0, handBack: { ...hand }, handStatus: s.rwStatus[hand.id] || 'pending',
      onHand: st => this.setState(x => ({ rwStatus: { ...x.rwStatus, [hand.id]: st } })),
      acceptAll: () => this.setState(x => { const r = { ...x.rwStatus }; B.RW.forEach(w => { if (!(qById(w.id) || {}).locked && (r[w.id] || 'pending') === 'pending' && rwVisibleAt(w)) r[w.id] = 'accepted'; }); return { rwStatus: r }; }),
      pendingCount: pending, hasPending: pending > 0, pendingText: pending ? `${pending} rewrite${pending > 1 ? 's' : ''} waiting for your decision` : 'All rewrites decided',
      distAfter: B.DIST_AFTER, beforeAfter: bp.checks.map(c => { const bOk = !!c.bOk, aOk = c.aOk !== false;
        return { label: c.label, b: Math.round(c.b) + '%', a: Math.round((c.a ?? 0) * s.afterT) + '%', bn: c.bn, an: c.an,
          bIcon: bOk ? '✓' : '✗', bColor: bOk ? 'var(--good)' : 'var(--bad)', aIcon: aOk ? '✓' : '✗', aColor: aOk ? 'var(--good)' : 'var(--bad)' }; }),
      afterBadge: afterCompliant ? `✓ Compliant · ${bp.name}` : `✗ Still not compliant · ${bp.name}`, afterColor: afterCompliant ? 'var(--good)' : 'var(--bad)', afterBg: afterCompliant ? 'var(--goodbg)' : 'var(--badbg)',
      afterNote: notApplied ? `${notApplied} planned rewrite${notApplied > 1 ? 's' : ''} not applied (locked, rejected or handed back). The “after” bar assumes every accepted rewrite — assess again before you submit.` : `Projection with all ${B.RW.length} rewrites applied.`,
      afterWarn: notApplied > 0, afterOk: notApplied === 0,
      exportMd: () => this.download('rebalanced-paper.md', this.paperMd(), 'text/markdown'),
      exportReport: () => { const o = this.reportObj(); this.download('bloombot-report.md', `# BloomBot report\n\n${o.paper}\n\nBlueprint: ${o.blueprint}\n\n| Bound | Before | After |\n|---|---:|---:|\n` + bp.checks.map(c => `| ${c.label} | ${c.b}% ${c.bOk ? '✓' : '✗'} | ${c.a}% ${c.aOk !== false ? '✓' : '✗'} |`).join('\n') + '\n\n## Rewrites\n\n' + o.rewrites.map(r => `- ${r.q} ${r.from} → ${r.to} (${r.status}): ${r.text}`).join('\n') + '\n', 'text/markdown'); },
      exportJson: () => this.download('bloombot-report.json', JSON.stringify(this.reportObj(), null, 2), 'application/json') };

    // ---------- Profile ----------
    const k = s.labelCount, ms = [25, 50, 100].find(m => k < m);
    const kIdx = k >= 100 ? 4 : k >= 50 ? 3 : k >= 25 ? 2 : k >= 10 ? 1 : 0;
    const prof = { labelCount: k, labelQ: B.POOL[s.labelIdx % B.POOL.length], labelNo: s.labelIdx + 1,
      labelButtons: L.map((nm, i) => ({ n: i + 1, name: nm, bg: `var(--l${i + 1})`, fg: FG(i + 1), pick: () => this.label(i + 1), ring: s.lastLabel === i + 1 ? '0 0 0 3px var(--ink)' : 'none' })),
      ring: this.ring(k), milestone: ms ? `${ms - k} more to reach k = ${ms}` : 'k = 100 reached — the full expected gain.',
      lastText: s.lastLabel ? `Last label: ${s.lastLabel} ${L[s.lastLabel - 1]}` : 'Press 1–6 or click a level',
      skip: () => this.setState(x => ({ labelIdx: x.labelIdx + 1 })),
      gain: s.page === 'profile' ? this.lineChart([{ name: 'Mean (6 targets)', values: B.GAIN.mean, color: 'var(--accent)', width: 3, bold: true }, { name: 'Indian boards', values: B.GAIN.boards, color: 'var(--l5)', width: 2 }], { xs: [0, 10, 25, 50, 100], proportional: true, min: 0.55, max: 0.72, step: 0.05, marker: kIdx, aria: 'Expected gain: mean weighted F1 from 0.627 at k=0 to 0.694 at k=100; Indian boards 0.578 to 0.674' }) : null,
      kNowTxt: `You are at k = ${k}. Expected mean weighted F1 at k = ${[0, 10, 25, 50, 100][kIdx]}: ${B.GAIN.mean[kIdx].toFixed(3)}.`,
      profiles: [{ id: 'none', name: 'No profile (generic model)', count: 0 }, { id: 'mine', name: 'My college · CSE', count: Object.keys(s.labels).length }].map(p => ({ ...p,
        countTxt: p.id === 'mine' ? `${p.count} different questions labelled${s.profileId ? ' · profile ' + s.profileId : ''}` : 'built-in model',
        sel: s.activeProfile === p.id, notSel: s.activeProfile !== p.id, use: () => this.useProfile(p.id), border: s.activeProfile === p.id ? 'var(--accent)' : 'var(--line)' })) };

    // ---------- How ----------
    const node = id => ({ id, sub: B.NODES[id].sub, pick: () => this.setState({ node: id }), bg: s.node === id ? 'var(--ink)' : 'var(--surface)', fg: s.node === id ? 'var(--bg)' : 'var(--ink)', subFg: s.node === id ? 'var(--bg)' : 'var(--ink3)',
      borderStyle: id === 'Socrates' ? 'dashed' : 'solid', pressed: s.node === id ? 'true' : 'false' });
    const how = { lane1: ['Scribe', 'Lexis', 'Archivist', 'Socrates', 'Arbiter'].map((id, i) => ({ ...node(id), showArrow: i > 0 })), lane2: ['Conductor', 'Architect', 'Composer'].map((id, i) => ({ ...node(id), showArrow: i > 0 })),
      verifiers: ['Level check', 'Critic', 'Student'].map(node), nodeName: s.node, nodeInfo: B.NODES[s.node],
      steps: B.STEPS.map((st, i) => ({ n: i + 1, title: st[0], pick: () => this.setState({ step: i }), bg: s.step === i ? 'var(--accent)' : i < s.step ? 'var(--accent-soft)' : 'var(--surface)', fg: s.step === i ? 'var(--accent-ink)' : 'var(--ink)', current: s.step === i ? 'step' : undefined })),
      stepTitle: B.STEPS[s.step][0], stepCaption: B.STEPS[s.step][1], stepNum: s.step + 1,
      prevStep: () => this.setState({ step: Math.max(0, s.step - 1) }), nextStep: () => this.setState({ step: Math.min(8, s.step + 1) }),
      prevDisabled: s.step === 0, nextDisabled: s.step === 8,
      bridgeText: B.BRIDGE.text, bridgeMasked: B.BRIDGE.masked, priorProba: B.BRIDGE.prior, lexisProba: B.BRIDGE.votes[0].proba, socratesProba: B.BRIDGE.votes[2].proba, archProba: B.BRIDGE.votes[1].proba, finalProba: B.BRIDGE.proba,
      lexStepTxt: `Lexis · ${B.BRIDGE.votes[0].level} ${B.BRIDGE.votes[0].confidence.toFixed(2)}`, archStepTxt: `Archivist · ${B.BRIDGE.votes[1].level} ${B.BRIDGE.votes[1].confidence.toFixed(2)} · weight 0`, socStepTxt: `Socrates · ${B.BRIDGE.votes[2].level} ${B.BRIDGE.votes[2].confidence.toFixed(2)} · fooled by the trap`,
      bridgeReasoning: B.BRIDGE.votes[2].reasoning };
    for (let i = 0; i < 9; i++) how['step' + (i + 1)] = s.step === i;

    // ---------- Eval ----------
    const E = B.EV, sx = (v, lo, hi) => ((v - lo) / (hi - lo) * 100) + '%';
    const evals = {
      dumbbell: E.dumbbell.map(r => ({ model: r[0], same: r[1].toFixed(3), other: r[2].toFixed(3), drop: r[3].toFixed(3), x1: sx(r[1], 0.3, 0.8), x2: sx(r[2], 0.3, 0.8), lineL: sx(Math.min(r[1], r[2]), 0.3, 0.8), lineW: ((Math.abs(r[1] - r[2])) / 0.5 * 100) + '%', weight: r[4] ? 700 : 500, rowBg: r[4] ? 'var(--accent-soft)' : 'transparent' })),
      dbTicks: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8].map(v => ({ v: v.toFixed(1), left: sx(v, 0.3, 0.8) })),
      loso: E.loso.map(r => ({ model: r[0], v: r[1].toFixed(3), w: (r[1] / 0.7 * 100) + '%', bg: r[2] ? 'var(--accent)' : 'var(--ink3)', weight: r[2] ? 700 : 500 })),
      perSource: E.perSource.map(r => ({ src: r[0], v: r[1].toFixed(3), ci: `[${r[2].toFixed(3)}, ${r[3].toFixed(3)}]`, lo: sx(r[2], 0.35, 0.8), w: ((r[3] - r[2]) / 0.45 * 100) + '%', dot: sx(r[1], 0.35, 0.8), low: r[1] < 0.5 })),
      ciTicks: [0.4, 0.5, 0.6, 0.7, 0.8].map(v => ({ v: v.toFixed(1), left: sx(v, 0.35, 0.8) })),
      heat: E.heat.map(([verb, rows]) => ({ verb, rows: rows.map(([src, vals]) => ({ src, cells: vals.map((v, i) => ({ v: v.toFixed(2), bg: `color-mix(in oklch, var(--heat) ${Math.round(v * 100)}%, var(--surface))`, fg: v > 0.5 ? 'var(--surface)' : v === 0 ? 'var(--ink3)' : 'var(--ink)', title: `${verb} · ${src} · ${L[i]} ${v.toFixed(2)}` })) })) })),
      levelHeads: L.map((nm, i) => ({ n: i + 1, name: nm, short: nm.slice(0, 3), bg: `var(--l${i + 1})`, fg: FG(i + 1) })),
      kchart: s.page === 'eval' ? this.lineChart(E.kcurve.map(([name, values]) => name === 'Mean' ? { name, values, color: 'var(--accent)', width: 3.5, bold: true } : name.startsWith('Indian') ? { name, values, color: name === 'Indian boards' ? 'var(--l5)' : 'var(--l4)', width: 2 } : { name, values, color: 'var(--ink3)', width: 1.25, opacity: 0.7 }), { xs: [0, 10, 25, 50, 100], min: 0.40, max: 0.80, step: 0.1, w: 680, h: 320, aria: 'Weighted F1 rises with k labelled questions per college, for all six targets; mean 0.627 to 0.694' }) : null,
      kTable: E.kcurve.map(([name, v]) => ({ name, vals: v.map(x => x.toFixed(3)), weight: name === 'Mean' ? 700 : 400 })),
      traps: E.traps.map(r => ({ model: r[0], v: r[1].toFixed(1), w: (r[1] * 100) + '%', bg: r[2] ? 'var(--accent)' : 'var(--ink3)', weight: r[2] ? 700 : 500 })),
      llmRows: E.llm.map(r => ({ model: r[0], v: r[1].toFixed(3), w: (r[1] / 0.7 * 100) + '%', t: r[2], bg: r[3] ? 'var(--accent)' : 'var(--ink3)', weight: r[3] ? 700 : 500 })),
      syco: E.syco.map(r => ({ mode: r[0], v: r[1] + '%', w: r[1] + '%' })),
      combos: E.combos.map(r => ({ name: r[0], v: r[1].toFixed(3), w: (r[1] / 0.7 * 100) + '%', bg: r[2] ? 'var(--accent)' : r[1] < 0.5 ? 'var(--bad)' : 'var(--ink3)', weight: r[2] ? 700 : 500 })),
      rwStats: E.rw.map(r => ({ k: r[0], v: r[1] })),
      evalExamples: [{ rw: BB.RW[3], status: 'accepted' }, { rw: BB.RW[1], status: 'accepted' }, { rw: BB.HANDBACK, status: 'pending' }]
    };
    const data = { sources: E.sources.map(r => ({ name: r[0], rows: r[1].toLocaleString('en-IN'), role: r[2], test: r[2] !== 'train', lic: r[3], origin: r[4], roleBg: r[2] !== 'train' ? 'var(--warnbg)' : 'var(--surface2)' })) };

    const sv = B.BRIDGE.votes[2];
    const sheet = { sheetChips: L.map((nm, i) => chip(i + 1)), sampleProba: B.BRIDGE.proba, sampleTrace: BB.TRACE.slice(0, 6).map((e, i) => ({ i: String(i + 1).padStart(2, '0'), agent: e.agent, action: e.action, detail: e.detail, ms: e.ms >= 1000 ? (e.ms / 1000).toFixed(2) + ' s' : e.ms.toFixed(1) + ' ms', barW: Math.max(2, Math.min(100, Math.log10(1 + e.ms) / Math.log10(3000) * 100)) + '%' })),
      sampleVote: { agent: sv.agent, role: 'Qwen3-4B · weight 0.5', chip: chip(N(sv.level)), confTxt: sv.confidence.toFixed(2), proba: sv.proba }, sampleRw: BB.RW[0] };

    // ---------- Header: API and LLM status ----------
    const apiState = s.online === null ? 'checking' : s.online ? 'live' : 'replay';
    const status = {
      apiLabel: { checking: 'Connecting…', live: 'Live API', replay: 'Replay · recorded' }[apiState], apiShort: { checking: '…', live: 'Live', replay: 'Replay' }[apiState],
      apiDot: { checking: 'var(--ink3)', live: 'var(--good)', replay: 'var(--warn)' }[apiState],
      apiTitle: s.online ? `Connected to ${apiBase() || window.location.origin}. Click to re-check.` : `No API at ${apiBase() || 'this origin'} — showing recorded runs. Start \`bloombot serve\`, then click to re-check (or add ?api=http://host:port).`,
      recheckApi: () => { this.setState({ online: null }); this.apiReady = this.checkApi(); },
      hasError: !!s.error, errorText: s.error, clearError: () => this.setState({ error: '' }),
    };
    const llmOff = live && s.llmAvailable === false;
    return { ...base, ready: true, ...check, ...clsVals, ...drawer, ...reb, ...prof, ...how, ...evals, ...data, ...sheet, ...status,
      navTeacher, navDemo, navSheet, navAll: [...navTeacher, ...navDemo, ...navSheet], pageTitle: titles[s.page],
      isCheck: s.page === 'check', isClassify: s.page === 'classify', isRebalance: s.page === 'rebalance', isProfile: s.page === 'profile', isHow: s.page === 'how', isEval: s.page === 'eval', isData: s.page === 'data', isSheet: s.page === 'sheet',
      llmLabel: llmOff ? 'LLM not running · fast mode' : s.llm ? 'LLM second opinion on' : 'Fast mode (no LLM)', llmDot: !llmOff && s.llm ? 'var(--good)' : 'var(--ink3)', llmPressed: s.llm ? 'true' : 'false',
      llmTitle: llmOff ? 'Ollama is not running, so Socrates and the LLM writer are off. Start `ollama serve`, then re-check the API.' : live ? `Socrates and the Composer use ${s.llmModel}. Click for fast mode (no LLM).` : 'Recorded runs used the LLM. Click to see fast mode.',
      toggleLlm: () => { if (!llmOff) this.setState({ llm: !s.llm }); },
      themeLabel: s.theme === 'light' ? 'Dark theme' : 'Light theme', themeIcon: s.theme === 'light' ? '◐' : '◑',
      toggleTheme: () => { const t = s.theme === 'light' ? 'dark' : 'light'; try { localStorage.setItem('bb.theme', t); } catch (e) {} this.setState({ theme: t }); },
      goCheck: () => this.go('check'), goClassify: () => this.go('classify'), goHow: () => this.go('how'), goEval: () => this.go('eval'), goProfile: () => this.go('profile') };
  }

  render() {
    const v = this.renderVals();
    if (!v.ready) return <div data-theme={v.theme} style={{ minHeight: '100vh', background: 'var(--bg)' }} />;
    return view(v);
  }
}
