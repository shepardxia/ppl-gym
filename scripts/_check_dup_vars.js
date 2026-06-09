#!/usr/bin/env node
// Read WebPPL/JS source on stdin, parse with webppl's bundled esprima,
// emit JSON: {ok: true, dupes: [{name, lines}, ...]} or {ok: false, error}.
//
// Duplicate detection is "top-level `var X` declared more than once" — the
// case that hits us is overlapping source_block_indices stitched together
// in assemble_curated.py. We don't try to be clever about scoped vars.

const path = require('path');
const fs = require('fs');
const { execSync } = require('child_process');

function findEsprima() {
  // Prefer webppl's bundled copy (always present where execute_webppl runs).
  try {
    const wp = execSync('which webppl', { encoding: 'utf8' }).trim();
    const real = fs.realpathSync(wp);
    let d = path.dirname(real);
    while (d !== '/') {
      const cand = path.join(d, 'node_modules', 'esprima');
      if (fs.existsSync(cand)) return cand;
      d = path.dirname(d);
    }
  } catch (e) { /* fall through */ }
  // Fallback: let node resolve from this script's own dirs.
  return 'esprima';
}

let esprima;
try {
  esprima = require(findEsprima());
} catch (e) {
  process.stdout.write(JSON.stringify({ ok: false, error: `esprima not found: ${e.message}` }));
  process.exit(0);
}

let code = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => { code += chunk; });
process.stdin.on('end', () => {
  let ast;
  try {
    ast = esprima.parse(code, { tolerant: true, loc: true });
  } catch (e) {
    process.stdout.write(JSON.stringify({ ok: false, error: `parse failed: ${e.message}` }));
    return;
  }
  const seen = {};
  for (const node of ast.body) {
    if (node.type !== 'VariableDeclaration') continue;
    for (const d of node.declarations) {
      if (!d.id || d.id.type !== 'Identifier') continue;
      const name = d.id.name;
      if (!seen[name]) seen[name] = [];
      seen[name].push(node.loc.start.line);
    }
  }
  const dupes = Object.entries(seen)
    .filter(([, ls]) => ls.length > 1)
    .map(([name, lines]) => ({ name, lines }));
  process.stdout.write(JSON.stringify({ ok: true, dupes }));
});
