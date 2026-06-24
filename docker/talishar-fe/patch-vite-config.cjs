#!/usr/bin/env node
/** Allow Docker-internal hostnames (e.g. talishar-fe) for Playwright replay capture. */
const fs = require("fs");
const path = require("path");

const root = process.argv[2] || "/app";
const candidates = ["vite.config.ts", "vite.config.js", "vite.config.mjs"];

for (const name of candidates) {
  const configPath = path.join(root, name);
  if (!fs.existsSync(configPath)) continue;

  const src = fs.readFileSync(configPath, "utf8");
  if (/allowedHosts/.test(src)) {
    process.exit(0);
  }

  const replaced = src.replace(/(\bserver\s*:\s*\{)/, "$1\n    allowedHosts: true,");
  if (replaced === src) {
    console.warn(`docker/talishar-fe: could not patch ${name} — no server block`);
    process.exit(1);
  }

  fs.writeFileSync(configPath, replaced);
  console.log(`docker/talishar-fe: patched ${name} (server.allowedHosts = true)`);
  process.exit(0);
}

console.warn("docker/talishar-fe: no vite.config.* found — skipping allowedHosts patch");
process.exit(0);
