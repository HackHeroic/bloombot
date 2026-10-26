// Turns a design inline-style string ("display:flex;gap:8px") into a React style object, cached.
const cache = new Map();

export function sx(css) {
  let out = cache.get(css);
  if (out) return out;
  out = {};
  for (const decl of css.split(';')) {
    const i = decl.indexOf(':');
    if (i < 0) continue;
    const prop = decl.slice(0, i).trim();
    const val = decl.slice(i + 1).trim();
    if (!prop || val === '' || val === 'undefined') continue;
    const key = prop.startsWith('--') ? prop : prop.replace(/^-ms-/, 'ms-').replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    out[key] = val;
  }
  if (cache.size > 5000) cache.clear();
  cache.set(css, out);
  return out;
}
