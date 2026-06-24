#!/bin/sh
set -e

cp -f /config/talishar-fe.env /app/.env

if [ ! -x node_modules/.bin/vite ]; then
  echo "Installing Talishar-FE dependencies..."
  npm install
fi

echo "Starting Talishar-FE on http://0.0.0.0:5173"
exec npm run dev -- --host 0.0.0.0 --port 5173
