// Thin client for the BloomBot HTTP API (docs/API.md).
//
// Where the API lives: `?api=http://host:port` or a local server the teacher connected (both
// remembered), else NEXT_PUBLIC_BLOOMBOT_API, else the same origin when the page is served by
// `bloombot serve`, else http://127.0.0.1:8000.
// NEXT_PUBLIC_BLOOMBOT_API=replay (the hosted demo) looks for no API at all until the teacher
// clicks "Connect to local BloomBot", so ordinary visitors never see a local-network prompt.

const KEY = 'bb.api';
export const LOCAL_API = 'http://127.0.0.1:8000';
export const REPLAY_BUILD = process.env.NEXT_PUBLIC_BLOOMBOT_API === 'replay';

export function savedApi() {
  try { return localStorage.getItem(KEY); } catch (e) { return null; }
}
export function saveApi(url) { try { localStorage.setItem(KEY, url.replace(/\/$/, '')); } catch (e) { /* storage blocked */ } }
export function clearApi() { try { localStorage.removeItem(KEY); } catch (e) { /* storage blocked */ } }

export function apiBase() {
  if (typeof window === 'undefined') return '';
  try {
    const q = new URLSearchParams(window.location.search).get('api');
    if (q) { saveApi(q); return q.replace(/\/$/, ''); }
  } catch (e) { /* fall through */ }
  const saved = savedApi();
  if (saved) return saved;
  if (REPLAY_BUILD) return null;
  if (process.env.NEXT_PUBLIC_BLOOMBOT_API) return process.env.NEXT_PUBLIC_BLOOMBOT_API;
  return window.location.port === '8000' ? '' : LOCAL_API;
}

// Where "Connect to local BloomBot" looks: the hosted demo always means this computer.
export const connectTarget = () => (REPLAY_BUILD ? LOCAL_API : apiBase() ?? LOCAL_API);

// Why a local API did not answer: "safari" (it blocks http://127.0.0.1 from an https page),
// "blocked" (Chrome's local-network permission was denied) or "down" (nothing listening).
export async function whyUnreachable() {
  const ua = typeof navigator !== 'undefined' ? navigator.userAgent : '';
  if (/^((?!chrome|chromium|crios|android|edg).)*safari/i.test(ua)) return 'safari';
  if (navigator.permissions) {
    for (const name of ['loopback-network', 'local-network-access', 'local-network']) {
      try { if ((await navigator.permissions.query({ name })).state === 'denied') return 'blocked'; } catch (e) { /* unknown name */ }
    }
  }
  return 'down';
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
  probe: base => fetch(`${base}/v1/meta`, { signal: AbortSignal.timeout(60000) }).then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); }),
  parseFile: file => { const f = new FormData(); f.append('file', file); return req('/v1/papers/parse', { method: 'POST', body: f }); },
  parseText: (text, title) => post('/v1/papers/parse-text', { text, title }),
  assess: body => post('/v1/papers/assess', body),
  assessJob: body => post('/v1/papers/assess-job', body),
  plan: body => post('/v1/papers/plan', body),
  classify: body => post('/v1/classify', body),
  analyze: question => post('/v1/analyze', { question }),
  rebalance: body => post('/v1/papers/rebalance', body),
  job: id => req(`/v1/jobs/${id}`, {}, 15000),
  cancelJob: id => req(`/v1/jobs/${id}`, { method: 'DELETE' }, 15000),
  cancelJobOnExit: id => { const b = apiBase(); if (b !== null) fetch(`${b}/v1/jobs/${id}`, { method: 'DELETE', keepalive: true }).catch(() => {}); },
  createProfile: (name, examples) => post('/v1/profiles', { name, examples }),
};
