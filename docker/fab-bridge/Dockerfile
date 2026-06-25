FROM python:3.12-bookworm

WORKDIR /app

# Set TORCH_CUDA=1 via docker-compose.gpu.yml when a host GPU is available.
ARG TORCH_CUDA=0

# Optional: C++ engine builds during training runs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md runtime_defaults.py start_talishar.py ./
COPY src ./src
COPY fab_bridge ./fab_bridge
COPY fab_tui ./fab_tui
COPY fab_gui ./fab_gui
COPY rl_agents ./rl_agents
COPY scripts ./scripts
COPY runscripts ./runscripts
COPY Talishar/Assets ./Talishar/Assets
COPY Talishar/GeneratedCode ./Talishar/GeneratedCode

RUN pip install --no-cache-dir -e ".[gui,cpp]" \
    && if [ "$TORCH_CUDA" = "1" ]; then \
         pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124; \
       else \
         pip install --no-cache-dir torch; \
       fi \
    && python -m playwright install --with-deps chromium

EXPOSE 8765

CMD ["fab-bridge", "gui", "--host", "0.0.0.0", "--no-browser"]
