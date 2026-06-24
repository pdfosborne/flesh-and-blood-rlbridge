FROM python:3.12-bookworm

WORKDIR /app

# Optional: C++ engine builds during training runs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
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

RUN pip install --no-cache-dir -e ".[gui,train,cpp]"

EXPOSE 8765

CMD ["fab-bridge", "gui", "--host", "0.0.0.0", "--no-browser"]
