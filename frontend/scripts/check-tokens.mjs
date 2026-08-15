#!/usr/bin/env node
/**
 * check-tokens.mjs — fail the build when a component invents its own styling.
 *
 * A design system that is only a convention lasts until the first hurried
 * afternoon. This is a grep, not a linter: it cannot tell a good grey from a
 * bad one, and it does not need to. What it catches is the one thing that
 * actually destroys a token set — a component file that stops asking
 * tokens.css what a card looks like and decides for itself.
 *
 * Allowed everywhere: var(--token), 0, 1px hairlines, and percentages.
 * Allowed only in tokens.css: hex colours, rem/px sizes, font stacks.
 *
 *     node scripts/check-tokens.mjs
 */

import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";

const ROOT = new URL("..", import.meta.url).pathname;
const SRC = join(ROOT, "src");
const EXEMPT = ["src/styles/tokens.css", "src/styles/global.css"];

/** @type {{name: string, re: RegExp, hint: string}[]} */
const RULES = [
  {
    name: "hex colour",
    re: /#[0-9a-fA-F]{3,8}\b/g,
    hint: "use var(--n-*), var(--accent*) or var(--danger|warn|ok)*",
  },
  {
    // 0 and 1px are allowed: zero has no unit to tokenise, and 1px is the
    // hairline, which is a physical constant of the display rather than a
    // design decision.
    name: "raw length",
    re: /(?<![\w-])(?!0)(?!1px)\d*\.?\d+(px|rem|em)\b/g,
    hint: "use var(--s-*) for space, var(--t-*) for type, var(--r) for radius",
  },
  {
    name: "raw font-family",
    re: /font-family:(?!\s*var\()/g,
    hint: "use var(--font-heading|body|mono)",
  },
  {
    name: "raw font-weight",
    re: /font-weight:(?!\s*var\()\s*[a-z0-9]/g,
    hint: "use var(--w-body|label|heading)",
  },
  {
    name: "raw box-shadow",
    re: /box-shadow:(?!\s*(var\(|none))/g,
    hint: "use var(--sh-sm|md)",
  },
];

/** @param {string} dir @returns {string[]} */
const walk = (dir) =>
  readdirSync(dir).flatMap((entry) => {
    const full = join(dir, entry);
    return statSync(full).isDirectory()
      ? walk(full)
      : /\.(css|tsx|ts)$/.test(entry)
        ? [full]
        : [];
  });

let failures = 0;
for (const file of walk(SRC)) {
  const rel = relative(ROOT, file);
  if (EXEMPT.includes(rel)) continue;

  const lines = readFileSync(file, "utf8").split("\n");
  lines.forEach((line, i) => {
    // A comment explaining a value is not a value.
    const code = line.replace(/\/\*.*?\*\//g, "").replace(/\/\/.*$/, "");
    for (const rule of RULES) {
      const hits = code.match(rule.re);
      if (!hits) continue;
      failures += 1;
      console.error(
        `${rel}:${i + 1}  ${rule.name}: ${hits.join(", ")}\n    → ${rule.hint}`,
      );
    }
  });
}

if (failures) {
  console.error(`\n${failures} token violation(s).`);
  process.exit(1);
}
console.log("tokens ok");
