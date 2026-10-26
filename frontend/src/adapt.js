// Turns API responses into the shapes the design's screens read (see src/designData.js).

export const L = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create'];
export const levelN = name => L.indexOf(name) + 1;
const r2 = x => Math.round(x * 100) / 100;
const pctInt = x => Math.round(x * 100);
const probaArr = p => (Array.isArray(p) ? p : L.map(l => p[l] || 0));
const short = (t, n = 60) => (t && t.length > n ? t.slice(0, n - 1) + '…' : t || '');

// Design blueprint keys ↔ API blueprint ids.
export const BPID = { aicte: 'aicte-2018-example', balanced: 'balanced-ug' };

function vote(v) {
  return { agent: v.agent, level: v.level, confidence: r2(v.confidence), proba: probaArr(v.proba), reasoning: v.reasoning || null };
}

// A parsed Paper (+ an optional PaperReport) → the design's PAPER rows.
export function paperRows(paper, report) {
  const items = {};
  (report ? report.items : []).forEach(it => { items[it.question.id] = it.prediction; });
  return paper.questions.map(q => {
    const p = items[q.id];
    return {
      id: q.id, number: q.number, text: q.text, context: q.context, marks: q.marks, section: q.section || 'A',
      group: q.group, option: q.option, co: q.course_outcome, tag: q.tagged_level, locked: !!q.locked,
      pred: p ? levelN(p.level) : 0, conf: p ? r2(p.confidence) : 0, proba: p ? probaArr(p.proba) : [0, 0, 0, 0, 0, 0],
      review: p ? !!p.needs_review : false, votes: p ? p.votes.map(vote) : [], neighbours: p ? p.neighbours : [],
    };
  });
}

// Edited rows → the Paper the API expects.
export function paperForApi(paper, rows) {
  const byId = Object.fromEntries(rows.map(r => [r.id, r]));
  return { ...paper, questions: paper.questions.map(q => {
    const r = byId[q.id];
    return r ? { ...q, text: r.text, marks: Number(r.marks) || 0, locked: !!r.locked } : q;
  }) };
}

function slotMarks(rows) {
  const seen = new Set();
  let total = 0;
  rows.forEach(r => {
    if (r.group) { if (seen.has(r.group)) return; seen.add(r.group); }
    total += Number(r.marks) || 0;
  });
  return total;
}

export function sections(rows) {
  const ids = [...new Set(rows.map(r => r.section || 'A'))];
  return ids.map(id => {
    const qs = rows.filter(r => (r.section || 'A') === id);
    const groups = [...new Set(qs.filter(r => r.group).map(r => r.group))];
    const marks = slotMarks(qs);
    const same = qs.length > 1 && !groups.length && qs.every(r => r.marks === qs[0].marks);
    const or = groups.map(g => qs.filter(r => r.group === g).map(r => 'Q' + r.id).join(' or ')).join('; ');
    return { id, title: `Part ${id}`, note: same ? `Answer all · ${qs.length} × ${qs[0].marks} marks` : `Answer all${or ? '; ' + or : ''} · ${marks} marks` };
  });
}

export function paperTitle(paper, rows) {
  const t = (paper.title || 'Question paper').replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
  const nice = t.charAt(0).toUpperCase() + t.slice(1);
  return `${nice} · ${slotMarks(rows)} marks`;
}

export function orNote(rows) {
  const g = rows.find(r => r.group);
  if (!g) return 'Every question counts by its marks.';
  const alts = rows.filter(r => r.group === g.group);
  return `OR-alternatives share one slot: ${alts.map(r => 'Q' + r.id).join('/')} are one ${g.marks}-mark slot; if they get different levels, each counts ${r2(g.marks / alts.length)} marks.`;
}

export const dist = d => L.map(l => pctInt(d[l] || 0));

const op = b => (b.max_share != null ? '≤' : '≥');
const share = b => pctInt(b.max_share != null ? b.max_share : b.min_share);

function note(c) {
  const b = c.bound, s = pctInt(c.share), lim = share(b), d = Math.abs(s - lim);
  if (b.max_share != null) return c.ok ? (d === 0 ? 'at the cap' : `${d} points under the cap`) : `${d} points over`;
  return c.ok ? (d === 0 ? 'meets the floor' : `${d} points clear`) : `${d} points under`;
}

// API blueprint + report checks → the design's BP entry (before; `after` fills the a/an columns).
export function blueprint(bp, report, after) {
  const first = bp.bounds.map(b => b.levels);
  const markers = bp.bounds.map((b, i) => {
    const lv = first[i].map(levelN).sort((x, y) => x - y);
    const label = `${b.label.split(' (')[0]} ${op(b)} ${share(b)}%`;
    if (b.max_share != null && lv[0] === 1 && lv.every((n, k) => n === k + 1)) return { at: share(b), label };
    if (b.min_share != null && lv[lv.length - 1] === 6 && lv.every((n, k) => n === 7 - lv.length + k)) return { at: 100 - share(b), label };
    return null;
  }).filter(Boolean);
  const checks = report.checks.map((c, i) => {
    const a = after ? after.checks[i] : null;
    return { label: `${c.bound.label} ${op(c.bound)} ${share(c.bound)}%`, b: pctInt(c.share), bn: note(c), bOk: c.ok,
      a: a ? pctInt(a.share) : null, an: a ? note(a) : '', aOk: a ? a.ok : null };
  });
  return { name: bp.name.replace(/\s*\(illustrative\)/, ''), label: bp.name, sub: bp.description.split('.')[0], markers, checks, compliant: report.compliant };
}

const CHECK_NAME = { level: 'Level', form: 'Form', topic: 'Topic', not_duplicate: 'Not duplicate', student: 'Student' };
export const checks = cs => (cs || []).map(c => ({ k: CHECK_NAME[c.name] || c.name, pass: c.passed, detail: c.name === 'student' && c.passed ? 'passed' : short(c.detail, 90) }));

// A RewriteJob (from the plan) → a design RW entry with nothing drafted yet.
export const rwFromJob = j => ({ q: 'Q' + j.question_id, id: j.question_id, from: levelN(j.from_level), to: levelN(j.to_level), marks: j.marks,
  reason: j.reason, attempts: 0, original: j.original, rewrite: '', checks: [] });

export function rwFromResult(r) {
  const last = r.attempts[r.attempts.length - 1] || {};
  return { ...rwFromJob(r.job), attempts: r.attempts.length, rewrite: r.final_question || last.question || '', rationale: last.rationale,
    checks: checks(last.checks), handedBack: !r.accepted };
}

// One trace event → a design trace row.
export function traceRow(e) {
  const d = e.detail || {};
  const q = d.question_id != null ? 'Q' + d.question_id : '';
  const text = {
    'Lexis classified': () => `${d.n} question${d.n === 1 ? '' : 's'} · ${d.model}`,
    'Archivist retrieved neighbours': () => `k = ${d.k}`,
    'Socrates answered': () => `${d.level || 'no answer'}${d.confidence != null ? ' ' + d.confidence.toFixed(2) : ''} · “${short(d.question, 48)}”`,
    'Socrates reasoned votes': () => `${d.n} unsure question${d.n === 1 ? '' : 's'} · ${d.model}`,
    'Arbiter pooled votes': () => `${d.n} pooled · ${d.sent_to_llm} sent to LLM · ${d.needs_review} flagged ⚑`,
    'Conductor assessed paper': () => `${d.questions} questions · ${d.compliant ? 'compliant' : 'not compliant'} · ${d.needs_review} ⚑`,
    'Architect planned rewrites': () => `${d.jobs} rewrite${d.jobs === 1 ? '' : 's'} · blueprint ${d.blueprint}`,
    'Composer drafting': () => `${q} → ${d.to_level} · attempt ${d.attempt}${d.feedback && d.feedback.length ? ' · fixing: ' + short(d.feedback.join('; '), 60) : ''}`,
    'Composer drafted': () => `${q} · attempt ${d.attempt}`,
    'Verifier checked': () => `${q} · attempt ${d.attempt} · ${d.passed ? 'all checks passed' : 'failed: ' + (d.failed || []).join(', ')}`,
  }[`${e.agent} ${e.action}`];
  return { agent: e.agent, action: e.action, detail: text ? text() : short(JSON.stringify(d), 80), ms: e.ms || 0,
    qid: d.question_id != null ? String(d.question_id) : null, passed: e.agent === 'Verifier' ? !!d.passed : undefined };
}

// Live job trace → RW entries filled in as Composer and Verifier events arrive.
export function rwProgress(plan, trace) {
  const out = plan.map(r => ({ ...r }));
  const by = Object.fromEntries(out.map(r => [r.id, r]));
  trace.forEach(e => {
    const d = e.detail || {}, r = by[String(d.question_id)];
    if (!r) return;
    if (e.agent === 'Composer' && e.action === 'drafted') { r.rewrite = d.text || ''; r.attempts = d.attempt; }
    if (e.agent === 'Verifier') { r.checks = checks(d.checks); r.attempts = d.attempt; }
  });
  return out;
}

// /v1/classify prediction + /v1/analyze → the design's classify example.
export function example(pred, an) {
  const d0 = (an.directives || [])[0];
  return {
    key: 'live', text: pred.question, level: pred.level, conf: r2(pred.confidence), needsReview: pred.needs_review, lowerOrder: pred.lower_order,
    proba: probaArr(pred.proba),
    directive: d0 ? { verb: d0.lemma, kind: d0.kind, negated: d0.negated, obj: d0.obj || null, phrase: d0.phrase || null } : null,
    embedded: an.embedded_bloom_verbs || [], frame: an.frame, wh: an.wh_word, evidence: pred.evidence || an.evidence, masked: an.masked,
    prior: probaArr(an.prior), votes: pred.votes.map(vote), neighbours: pred.neighbours, parse: false,
  };
}
