// Thin client for the BloomBot HTTP API (docs/API.md).
//
// Where the API lives: `?api=http://host:port` (remembered), else NEXT_PUBLIC_BLOOMBOT_API,
// else the same origin when the page is served by `bloombot serve`, else http://127.0.0.1:8000.
// NEXT_PUBLIC_BLOOMBOT_API=replay (the hosted demo) looks for no API at all: apiBase() is null.

export function apiBase() {
  if (typeof window === 'undefined') return '';
  try {
    const q = new URLSearchParams(window.location.search).get('api');
    if (q) { localStorage.setItem('bb.api', q); return q.replace(/\/$/, ''); }
    const saved = localStorage.getItem('bb.api');
    if (saved) return saved;
  } catch (e) { /* storage blocked: fall through */ }
  if (process.env.NEXT_PUBLIC_BLOOMBOT_API === 'replay') return null;
  if (process.env.NEXT_PUBLIC_BLOOMBOT_API) return process.env.NEXT_PUBLIC_BLOOMBOT_API;
  return window.location.port === '8000' ? '' : 'http://127.0.0.1:8000';
}

async function req(path, opts = {}, timeoutMs = 180000) {
  const base = apiBase();
  if (base === null) throw new Error(`${path}: this build replays recorded runs only`);
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetch(base + path, { ...opts, signal: ctl.signal });
    const body = await res.json().catch(() => null);
    if (!res.ok) {
      const detail = body && body.detail ? (typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)) : res.statusText;
      throw new Error(`${path}: ${detail}`);
    }
    return body;
  } catch (e) {
    if (e.name === 'AbortError') throw new Error(`${path}: no answer after ${Math.round(timeoutMs / 1000)} s`);
    throw e;
  } finally {
    clearTimeout(t);
  }
}

const post = (path, body, timeoutMs) => req(path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }, timeoutMs);

export const api = {
  meta: () => req('/v1/meta', {}, 60000),  // the first call waits while the API loads its models
  parseFile: file => { const f = new FormData(); f.append('file', file); return req('/v1/papers/parse', { method: 'POST', body: f }); },
  parseText: (text, title) => post('/v1/papers/parse-text', { text, title }),
  assess: body => post('/v1/papers/assess', body),
  plan: body => post('/v1/papers/plan', body),
  classify: body => post('/v1/classify', body),
  analyze: question => post('/v1/analyze', { question }),
  rebalance: body => post('/v1/papers/rebalance', body),
  job: id => req(`/v1/jobs/${id}`, {}, 15000),
  createProfile: (name, examples) => post('/v1/profiles', { name, examples }),
};
