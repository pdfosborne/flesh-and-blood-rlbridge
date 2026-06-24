FROM node:20-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/talishar-fe/entrypoint.sh /entrypoint.sh
COPY docker/talishar-fe/patch-vite-config.cjs /patch-vite-config.cjs
RUN chmod +x /entrypoint.sh

EXPOSE 5173

HEALTHCHECK --interval=5s --timeout=3s --retries=30 --start-period=120s \
  CMD curl -fsS http://127.0.0.1:5173/ >/dev/null || exit 1

ENTRYPOINT ["/entrypoint.sh"]
