// Recorded runs of every sample paper (scripts/record_samples.py), served like the API so the
// offline page runs the same code as the live one. Loaded only when needed.

let rec = null, data = null;
export const loadRecorded = () => rec || (rec = import('./recorded/samples.json').then(m => m.default));
export const loadDataset = () => data || (data = import('./recorded/dataset.json').then(m => m.default));

const jobs = new Map();
const MAX_REPLAY_MS = 20000;  // a recorded rebalance replays on its own timeline, at most this long

const offline = what => Promise.reject(new Error(`${what} needs BloomBot running: click the Replay chip to connect it.`));

// A client for one recorded sample: the methods App uses on the live API.
export function recordedClient(R, sampleId) {
  const run = R.runs[sampleId];
  return {
    samples: async () => R.samples,
    sample: async id => structuredClone(R.runs[id].paper),
    blueprints: async () => R.blueprints,
    assessJob: async body => {
      const id = `rec-assess-${jobs.size}`;
      jobs.set(id, { status: 'done', kind: 'assess', trace: [], result: structuredClone(run.assess[body.blueprint_id]) });
      return { id };
    },
    plan: async body => structuredClone(run.plan[body.blueprint_id]),
    rebalance: async body => {
      const id = `rec-rebalance-${jobs.size}`;
      jobs.set(id, { replay: run.rebalance[body.blueprint_id], started: Date.now() });
      return { id, kind: 'rebalance', status: 'running', trace: [] };
    },
    job: async id => {
      const j = jobs.get(id);
      if (!j.replay) return j;
      const trace = j.replay.trace.filter(e => e.agent);
      const t0 = Date.parse(trace[0].at), total = Date.parse(trace[trace.length - 1].at) - t0;
      const speed = Math.max(1, total / MAX_REPLAY_MS), elapsed = (Date.now() - j.started) * speed;
      if (elapsed > total + 300) return { ...j.replay, id };
      return { ...j.replay, id, status: 'running', result: null, trace: trace.filter(e => Date.parse(e.at) - t0 <= elapsed) };
    },
    cancelJob: async () => ({}),
    rereport: () => offline('Changing a recorded report'),
  };
}

// The dataset explorer on the bundled sample: the same shape as GET /v1/dataset.
export function datasetPage(D, { source, level, q, offset = 0, limit = 25 }) {
  const public_ = new Set(D.sources.filter(s => s.public).map(s => s.id));
  let pub = D.items.filter(i => public_.has(i.source));
  if (q) { const needle = q.toLowerCase(); pub = pub.filter(i => i.question.toLowerCase().includes(needle)); }
  const bySourceRows = level ? pub.filter(i => i.level === level) : pub;
  const byLevelRows = source ? pub.filter(i => i.source === source) : pub;
  const hits = level ? byLevelRows.filter(i => i.level === level) : byLevelRows;
  const count = (rows, key, vals) => Object.fromEntries(vals.map(v => [v, rows.filter(r => r[key] === v).length]));
  return {
    total: hits.length, offset, limit, items: hits.slice(offset, offset + limit),
    by_level: count(byLevelRows, 'level', ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']),
    by_source: count(bySourceRows, 'source', [...public_].sort()), sources: D.sources, sampled: true,
  };
}
