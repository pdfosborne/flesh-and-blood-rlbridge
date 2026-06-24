#!/bin/sh
set -e

cp -f /config/talishar-fe.env /app/.env

if [ ! -x node_modules/.bin/vite ]; then
  echo "Installing Talishar-FE dependencies..."
  npm install
fi

# Playwright in fab-bridge opens http://talishar-fe:5173 — Vite must allow that Host header.
node /patch-vite-config.cjs /app

echo "Starting Talishar-FE on http://0.0.0.0:5173"
exec npm run dev -- --host 0.0.0.0 --port 5173
