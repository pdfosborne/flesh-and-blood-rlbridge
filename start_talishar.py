#!/usr/bin/env python3
"""Start the Talishar backend (Docker Compose) and Talishar-FE Vite dev server.

Uses the repo-root ``docker-compose.yml`` (RLStep rl-bridge overlay). For training,
also applies ``docker-compose.training.yml`` (tmpfs Games, Apache/OPcache tuning).

Usage:
    python start_talishar.py              # start everything
    python start_talishar.py --backend-only
    python start_talishar.py --backend-only --shards 4
    python start_talishar.py --fe-only
    python start_talishar.py --down
    python start_talishar.py --backend-only --no-training   # dev profile without training overlay
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from fab_bridge.paths import configure_import_paths, repo_root
from fab_bridge.talishar_env import (
    apply_training_env_values,
    clear_training_env,
    load_env_file,
    shard_game_urls,
    write_training_env,
)

configure_import_paths()
REPO_ROOT = repo_root()
TALISHAR_DIR = REPO_ROOT / "Talishar"
FE_DIR = REPO_ROOT / "Talishar-FE"
FE_LOCAL_ENV = REPO_ROOT / "docker" / "talishar-fe" / "talishar-fe.local.env"
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
TRAINING_COMPOSE_FILE = REPO_ROOT / "docker-compose.training.yml"
SHARD_COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.shard.yml"
SHARD_STATE_FILE = REPO_ROOT / "Talishar" / "HostFiles" / "shard_count.txt"
BACKEND_URL = "http://localhost:8080"
GAME_URL = "http://localhost:8080/game"
FE_URL = "http://localhost:5173"
PHPMYADMIN_URL = "http://localhost:5001"
DEFAULT_SHARD_BASE_PORT = 8080
DEFAULT_SHARD_BASE_REDIS_PORT = 6382
SHARD_PROJECT_PREFIX = "fab-rl-bridge-shard"
SHARD_STARTUP_WAIT_SECONDS = 60
MYSQL_ROOT_PASSWORD = "secret"
MYSQL_DATABASE = "fabonline"
MYSQL_SCHEMA_PROBE_TABLE = "savedsettings"
MYSQL_INIT_WAIT_SECONDS = 120
MYSQL_INIT_SQL = TALISHAR_DIR / "Database" / "00_database.sql"


def _shard_project_name(shard_index: int) -> str:
    if shard_index == 0:
        return "fab-rl-bridge"
    return f"{SHARD_PROJECT_PREFIX}-{shard_index}"


def _mysql_container_name(shard_index: int) -> str:
    return f"{_shard_project_name(shard_index)}-mysql-server-1"


def _shard_mysql_volume_name(shard_index: int) -> str:
    return f"fabrlbridge_shard_{shard_index}_mysql"


def _header(msg: str) -> None:
    print(f"\n=== {msg} ===")


def _reachable(url: str, timeout: float = 2) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return int(resp.status) < 500
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _wait_for(url: str, seconds: int, label: str) -> bool:
    print(f"  Waiting for {label}...", end="", flush=True)
    for _ in range(seconds):
        if _reachable(url):
            print()
            return True
        time.sleep(1)
        print(".", end="", flush=True)
    print()
    return False


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)  # noqa: S603


def _compose_file_args(*, training: bool) -> list[str]:
    args = ["-f", str(COMPOSE_FILE)]
    if training and TRAINING_COMPOSE_FILE.is_file():
        args.extend(["-f", str(TRAINING_COMPOSE_FILE)])
    return args


def _compose_cmd(*compose_args: str, training: bool) -> list[str]:
    return [
        "docker",
        "compose",
        *_compose_file_args(training=training),
        *compose_args,
    ]


def _shard_game_url(shard_index: int, *, base_port: int = DEFAULT_SHARD_BASE_PORT) -> str:
    urls = shard_game_urls(shards=shard_index + 1, base_port=base_port)
    return urls[shard_index]


def _read_shard_count() -> int:
    try:
        text = SHARD_STATE_FILE.read_text(encoding="utf-8").strip()
        return max(1, int(text))
    except (OSError, TypeError, ValueError):
        return 1


def _write_shard_count(n_shards: int) -> None:
    SHARD_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SHARD_STATE_FILE.write_text(f"{max(1, int(n_shards))}\n", encoding="utf-8")


def _shard_compose_cmd(
    shard_index: int,
    *compose_args: str,
    training: bool,
    base_port: int = DEFAULT_SHARD_BASE_PORT,
) -> list[str]:
    files = ["-f", str(COMPOSE_FILE)]
    if training and TRAINING_COMPOSE_FILE.is_file():
        files.extend(["-f", str(TRAINING_COMPOSE_FILE)])
    if shard_index > 0 and SHARD_COMPOSE_FILE.is_file():
        files.extend(["-f", str(SHARD_COMPOSE_FILE)])
    project = (
        "fab-rl-bridge"
        if shard_index == 0
        else f"{SHARD_PROJECT_PREFIX}-{shard_index}"
    )
    cmd = ["docker", "compose", "-p", project, *files, *compose_args]
    return cmd


def _shard_compose_env(
    shard_index: int,
    *,
    base_port: int = DEFAULT_SHARD_BASE_PORT,
    base_redis_port: int = DEFAULT_SHARD_BASE_REDIS_PORT,
) -> dict[str, str]:
    env = os.environ.copy()
    if shard_index > 0:
        env["TALISHAR_SHARD_PORT"] = str(int(base_port) + int(shard_index))
        env["TALISHAR_SHARD_INDEX"] = str(int(shard_index))
        env["TALISHAR_SHARD_REDIS_PORT"] = str(
            int(base_redis_port) + int(shard_index)
        )
        env["TALISHAR_INITDB_DIR"] = str(TALISHAR_DIR / "Database")
        env["TALISHAR_HOSTFILES_SEED_DIR"] = str(TALISHAR_DIR / "HostFiles")
    return env


def _run_compose(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, env=env)  # noqa: S603


def _run_docker(
    cmd: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        cmd,
        cwd=cwd,
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )


def _mysql_table_exists(shard_index: int, table_name: str) -> bool | None:
    """Return whether *table_name* exists; None when MySQL is not ready yet."""
    container = _mysql_container_name(shard_index)
    probe = (
        "SELECT COUNT(*) FROM information_schema.tables "
        f"WHERE table_schema='{MYSQL_DATABASE}' AND table_name='{table_name}'"
    )
    result = _run_docker(
        [
            "docker",
            "exec",
            container,
            "mysql",
            "-uroot",
            f"-p{MYSQL_ROOT_PASSWORD}",
            "-N",
            "-B",
            "-e",
            probe,
        ],
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").lower()
        if any(
            token in stderr
            for token in (
                "is not running",
                "no such container",
                "can't connect",
                "cannot connect",
                "connection refused",
                "server has gone away",
            )
        ):
            return None
        return False
    try:
        return int((result.stdout or "").strip()) > 0
    except ValueError:
        return False


def _mysql_is_responsive(shard_index: int) -> bool:
    result = _run_docker(
        [
            "docker",
            "exec",
            _mysql_container_name(shard_index),
            "mysqladmin",
            "ping",
            "-uroot",
            f"-p{MYSQL_ROOT_PASSWORD}",
            "--silent",
        ],
    )
    return result.returncode == 0


def _wait_for_mysql_table(
    shard_index: int,
    table_name: str,
    *,
    timeout: int = MYSQL_INIT_WAIT_SECONDS,
) -> bool:
    print(f"  Waiting for MySQL schema ({table_name})...", end="", flush=True)
    for _ in range(max(1, int(timeout))):
        state = _mysql_table_exists(shard_index, table_name)
        if state is True:
            print(" ready.")
            return True
        time.sleep(1)
        print(".", end="", flush=True)
    print(" timed out.")
    return False


def _apply_mysql_init_sql(shard_index: int) -> bool:
    """Apply Talishar schema SQL when entrypoint init scripts did not run."""
    if not MYSQL_INIT_SQL.is_file():
        print(
            f"  WARNING: MySQL init SQL not found: {MYSQL_INIT_SQL}",
            flush=True,
        )
        return False
    container = _mysql_container_name(shard_index)
    result = subprocess.run(  # noqa: S603
        [
            "docker",
            "exec",
            "-i",
            container,
            "mysql",
            "-uroot",
            f"-p{MYSQL_ROOT_PASSWORD}",
            MYSQL_DATABASE,
        ],
        input=MYSQL_INIT_SQL.read_bytes(),
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        print(f"  WARNING: MySQL init SQL failed: {stderr[:500]}", flush=True)
        return False
    return True


def _remove_docker_volume(volume_name: str) -> bool:
    result = _run_docker(["docker", "volume", "rm", volume_name])
    return result.returncode == 0


def _recreate_shard_mysql_volume(
    shard_index: int,
    *,
    training: bool,
    base_port: int,
) -> None:
    label = f"shard {shard_index}"
    volume = _shard_mysql_volume_name(shard_index)
    env = _shard_compose_env(shard_index, base_port=base_port)

    print(f"  Recreating {label} MySQL volume ({volume})…")
    stop_cmd = _shard_compose_cmd(
        shard_index,
        "stop",
        "web-server",
        "mysql-server",
        training=training,
        base_port=base_port,
    )
    _run_docker(stop_cmd, env=env)

    rm_cmd = _shard_compose_cmd(
        shard_index,
        "rm",
        "-f",
        "mysql-server",
        training=training,
        base_port=base_port,
    )
    _run_docker(rm_cmd, env=env)

    if not _remove_docker_volume(volume):
        print(f"  NOTE: volume {volume} was not removed (may already be absent).")

    up_mysql = _shard_compose_cmd(
        shard_index,
        "up",
        "-d",
        "mysql-server",
        training=training,
        base_port=base_port,
    )
    _run_compose(up_mysql, cwd=REPO_ROOT, env=env)

    if not _wait_for_mysql_table(shard_index, MYSQL_SCHEMA_PROBE_TABLE):
        print(f"  Applying Talishar schema SQL to {label} MySQL…")
        if _apply_mysql_init_sql(shard_index) and _wait_for_mysql_table(
            shard_index,
            MYSQL_SCHEMA_PROBE_TABLE,
            timeout=30,
        ):
            return
        raise RuntimeError(
            f"{label} MySQL failed to initialize after volume recreate "
            f"(missing `{MYSQL_SCHEMA_PROBE_TABLE}`)."
        )


def _ensure_shard_mysql_schema(
    shard_index: int,
    *,
    training: bool,
    base_port: int,
) -> None:
    """Ensure Talishar's MySQL schema exists (Start.php needs savedsettings)."""
    label = "shard 0 (primary)" if shard_index == 0 else f"shard {shard_index}"

    if _wait_for_mysql_table(shard_index, MYSQL_SCHEMA_PROBE_TABLE):
        print(f"  {label} MySQL schema: ok (`{MYSQL_SCHEMA_PROBE_TABLE}`)")
        return

    if shard_index == 0:
        print(
            f"  WARNING: {label} MySQL is missing `{MYSQL_SCHEMA_PROBE_TABLE}`. "
            "Apply Talishar/Database/00_database.sql to the primary mysql-data volume.",
            flush=True,
        )
        return

    if not _mysql_is_responsive(shard_index):
        print(
            f"  WARNING: {label} MySQL did not become ready; "
            f"could not verify `{MYSQL_SCHEMA_PROBE_TABLE}`.",
            flush=True,
        )
        return

    print(
        f"  WARNING: {label} MySQL is missing `{MYSQL_SCHEMA_PROBE_TABLE}`; "
        "applying schema SQL…",
        flush=True,
    )
    if _apply_mysql_init_sql(shard_index) and _wait_for_mysql_table(
        shard_index,
        MYSQL_SCHEMA_PROBE_TABLE,
        timeout=30,
    ):
        print(f"  {label} MySQL schema: applied")
        return

    print(
        f"  WARNING: {label} MySQL still missing `{MYSQL_SCHEMA_PROBE_TABLE}` "
        "(recreating shard volume).",
        flush=True,
    )
    _recreate_shard_mysql_volume(shard_index, training=training, base_port=base_port)
    print(f"  {label} MySQL schema: recreated")

    up_web = _shard_compose_cmd(
        shard_index,
        "up",
        "-d",
        "--build",
        "web-server",
        training=training,
        base_port=base_port,
    )
    env = _shard_compose_env(shard_index, base_port=base_port)
    _run_compose(up_web, cwd=REPO_ROOT, env=env)


def _start_backend_shard(
    shard_index: int,
    *,
    training: bool,
    base_port: int,
) -> None:
    label = "shard 0 (primary)" if shard_index == 0 else f"shard {shard_index}"
    print(f"  Starting {label} on port {base_port + shard_index}…")
    cmd = _shard_compose_cmd(
        shard_index,
        "up",
        "-d",
        "--build",
        "web-server",
        training=training,
        base_port=base_port,
    )
    env = _shard_compose_env(shard_index, base_port=base_port)
    _run_compose(cmd, cwd=REPO_ROOT, env=env)
    _ensure_shard_mysql_schema(shard_index, training=training, base_port=base_port)
    game_url = _shard_game_url(shard_index, base_port=base_port)
    root_url = game_url.rsplit("/game", 1)[0]
    wait_secs = SHARD_STARTUP_WAIT_SECONDS + (30 if shard_index == 0 else 0)
    if _wait_for(root_url + "/", wait_secs, f"{label} backend"):
        print(f"  {label} is up ({game_url})")
        if _reachable(game_url + "/APIs/RLStep.php", timeout=3):
            print(f"  {label} RLStep overlay: available")
        else:
            print(f"  {label} RLStep overlay: not detected")
    else:
        print(f"  {label} did not respond within {wait_secs}s")


def _stop_backend_shards(*, training: bool, n_shards: int) -> None:
    for shard_index in range(max(1, int(n_shards)) - 1, -1, -1):
        label = "shard 0 (primary)" if shard_index == 0 else f"shard {shard_index}"
        print(f"  Stopping {label}…")
        cmd = _shard_compose_cmd(shard_index, "down", training=training)
        try:
            _run_compose(cmd, cwd=REPO_ROOT)
        except subprocess.CalledProcessError as exc:
            print(f"  WARNING: docker compose down failed for {label} (exit {exc.returncode})")


def _install_rl_bridge_overlay_on_host(talishar_dir: Path) -> None:
    """Copy rl-bridge PHP overlays onto the host tree (mirrors container entrypoint)."""
    rl_bridge = REPO_ROOT / "docker" / "talishar" / "rl-bridge"
    if not rl_bridge.is_dir():
        return
    dest_apis = talishar_dir / "APIs"
    dest_apis.mkdir(parents=True, exist_ok=True)
    src_apis = rl_bridge / "APIs"
    if src_apis.is_dir():
        for src in src_apis.glob("*.php"):
            dest = dest_apis / src.name
            if not dest.exists() or src.read_bytes() != dest.read_bytes():
                dest.write_bytes(src.read_bytes())
    for src in rl_bridge.glob("*.php"):
        dest = talishar_dir / src.name
        if not dest.exists() or src.read_bytes() != dest.read_bytes():
            dest.write_bytes(src.read_bytes())


def _npm_executable() -> str:
    """Resolve npm on PATH (required on Windows where ``npm`` is ``npm.cmd``)."""
    npm = shutil.which("npm")
    if npm is None:
        raise FileNotFoundError(
            "npm not found on PATH. Install Node.js 18+ to run Talishar-FE."
        )
    return npm


def _npm_cmd(*args: str) -> list[str]:
    return [_npm_executable(), *args]


def _load_env_file(path: Path) -> dict[str, str]:
    return load_env_file(path)


def _vite_process_env() -> dict[str, str]:
    """Environment for native Talishar-FE Vite (config lives in fab-rl-bridge only)."""
    env = os.environ.copy()
    env.update(_load_env_file(FE_LOCAL_ENV))
    return env


def _prepare_talishar_runtime_files(talishar_dir: Path) -> None:
    """Create gitignored HostFiles Talishar needs before the PHP server starts."""
    host_files = talishar_dir / "HostFiles"
    host_files.mkdir(parents=True, exist_ok=True)

    redirector = host_files / "Redirector.php"
    template = host_files / "RedirectorTemplate.php"
    if template.is_file() and not redirector.is_file():
        redirector.write_bytes(template.read_bytes())

    counter = host_files / "GameIDCounter.txt"
    if not counter.is_file():
        counter.write_text("1\n", encoding="utf-8")
    if sys.platform != "win32":
        counter.chmod(0o666)

    games = talishar_dir / "Games"
    games.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        games.chmod(0o777)

    api_keys_dir = talishar_dir / "APIKeys"
    api_keys_dir.mkdir(parents=True, exist_ok=True)
    api_keys = api_keys_dir / "APIKeys.php"
    stub = REPO_ROOT / "docker" / "talishar" / "APIKeys.local.php"
    if stub.is_file() and not api_keys.is_file():
        api_keys.write_bytes(stub.read_bytes())


def _start_vite_detached(fe_dir: Path) -> None:
    cmd = _npm_cmd(
        "run",
        "dev",
        "--",
        "--host",
        "0.0.0.0",
        "--port",
        "5173",
        "--strictPort",
    )
    vite_env = _vite_process_env()
    if sys.platform == "win32":
        subprocess.Popen(  # noqa: S603
            cmd,
            cwd=fe_dir,
            env=vite_env,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
    else:
        subprocess.Popen(  # noqa: S603
            cmd,
            cwd=fe_dir,
            env=vite_env,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def run(
    *,
    backend_only: bool = False,
    fe_only: bool = False,
    down: bool = False,
    training: bool = True,
    shards: int = 1,
    shard_base_port: int = DEFAULT_SHARD_BASE_PORT,
) -> int:
    n_shards = max(1, int(shards))
    if down:
        _header("Stopping Talishar backend containers")
        if not COMPOSE_FILE.is_file():
            print(f"ERROR: compose file not found: {COMPOSE_FILE}", file=sys.stderr)
            return 1
        stop_count = n_shards if n_shards > 1 else _read_shard_count()
        _stop_backend_shards(training=training, n_shards=stop_count)
        clear_training_env()
        if SHARD_STATE_FILE.is_file():
            try:
                SHARD_STATE_FILE.unlink()
            except OSError:
                pass
        print("Backend stopped.")
        return 0

    if not fe_only:
        _header("Starting Talishar backend (Docker Compose)")
        if not COMPOSE_FILE.is_file():
            print(f"ERROR: compose file not found: {COMPOSE_FILE}", file=sys.stderr)
            return 1

        from fab_bridge.talishar_setup import ensure_talishar  # noqa: PLC0415

        try:
            ensure_talishar(clone=True, quiet=True)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: Talishar setup failed: {exc}", file=sys.stderr)
            return 1

        _prepare_talishar_runtime_files(TALISHAR_DIR)
        _install_rl_bridge_overlay_on_host(TALISHAR_DIR)

        if training and TRAINING_COMPOSE_FILE.is_file():
            print("  Training overlay: docker-compose.training.yml (tmpfs Games, Apache/OPcache)")
        else:
            print("  Training overlay: off")
        if n_shards > 1:
            print(f"  Multi-shard mode: {n_shards} backend(s) from port {shard_base_port}")
            print(
                "  Extra shards use isolated Redis/MySQL volumes "
                f"(Redis host ports {DEFAULT_SHARD_BASE_REDIS_PORT + 1}…)"
            )

        try:
            for shard_index in range(n_shards):
                _start_backend_shard(
                    shard_index,
                    training=training,
                    base_port=shard_base_port,
                )
            _write_shard_count(n_shards)
        except subprocess.CalledProcessError as exc:
            print(f"ERROR: docker compose up failed (exit {exc.returncode})", file=sys.stderr)
            return exc.returncode or 1
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        from flesh_and_blood_rlbridge.talishar_backend_pool import (  # noqa: PLC0415
            probe_backend_health,
        )

        healthy_urls: list[str] = []
        for shard_index in range(n_shards):
            game_url = _shard_game_url(shard_index, base_port=shard_base_port)
            ok, reason = probe_backend_health(game_url, timeout=5.0)
            if ok:
                healthy_urls.append(game_url)
            else:
                label = (
                    "shard 0 (primary)"
                    if shard_index == 0
                    else f"shard {shard_index}"
                )
                print(
                    f"  WARNING: {label} ({game_url}) not ready for training: {reason}",
                    flush=True,
                )

        if not healthy_urls:
            print(
                "ERROR: no Talishar backends passed the RLStep health probe.",
                file=sys.stderr,
            )
            return 1

        backend_urls = healthy_urls
        print("Backend containers started.")
        for url in backend_urls:
            print(f"  Game engine : {url}")
        if n_shards == 1:
            print(f"  phpMyAdmin  : {PHPMYADMIN_URL}")

    if not backend_only:
        _header("Starting Talishar-FE (Vite dev server)")
        if not FE_DIR.is_dir():
            print(f"ERROR: Talishar-FE directory not found: {FE_DIR}", file=sys.stderr)
            return 1

        try:
            if not (FE_DIR / "node_modules").is_dir():
                print("  node_modules not found - running npm install...")
                _run(_npm_cmd("install"), cwd=FE_DIR)

            if _reachable(FE_URL) or _reachable("http://127.0.0.1:5173"):
                print(f"  Vite dev server already running at {FE_URL}")
            else:
                _start_vite_detached(FE_DIR)
                if sys.platform == "win32":
                    print("  Vite dev server launched in a new window.")
                else:
                    print("  Vite dev server launched in the background.")
                print(f"  Frontend URL : {FE_URL}")
                print("  FE config    : docker/talishar-fe/talishar-fe.local.env (via process env)")

                if _wait_for(FE_URL, 20, "Vite to start") or _wait_for(
                    "http://127.0.0.1:5173", 5, "Vite on 127.0.0.1"
                ):
                    print("  Frontend is up.")
                else:
                    print("  Frontend did not respond yet - it may need a few more seconds.")
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    training_env_values: dict[str, str] = {}
    if not fe_only:
        training_env_values = write_training_env(
            urls=backend_urls,
            fe_url=FE_URL,
            reserve_eval_shard=len(backend_urls) >= 2,
            reserve_render_shard=len(backend_urls) >= 3,
        )
        apply_training_env_values(training_env_values)

    _header("Talishar stack ready")
    if not fe_only:
        if len(backend_urls) > 1:
            print(
                f"  Backends : {len(backend_urls)} healthy shard(s) "
                f"({backend_urls[0]} …)"
            )
        else:
            print(f"  Backend  : {backend_urls[0]}")
    if not backend_only:
        print(f"  Frontend : {FE_URL}")

    print()
    if training_env_values:
        from fab_bridge.talishar_env import training_env_path  # noqa: PLC0415

        print("Training URLs saved for scripts/TUI (auto-loaded on next run):")
        print(f"  {training_env_path()}")
        if "TALISHAR_URLS" in training_env_values:
            print(f"  TALISHAR_URLS={training_env_values['TALISHAR_URLS']}")
        if "TALISHAR_EVAL_URL" in training_env_values:
            print(f"  TALISHAR_EVAL_URL={training_env_values['TALISHAR_EVAL_URL']}")
            print("  (last eval shard reserved for checkpoint / eval traffic only)")
        if "TALISHAR_RENDER_URL" in training_env_values:
            print(f"  TALISHAR_RENDER_URL={training_env_values['TALISHAR_RENDER_URL']}")
            print("  (last shard reserved for optimal-policy live render only)")
        print(f"  TALISHAR_URL={training_env_values.get('TALISHAR_URL', '')}")
        print(f"  TALISHAR_FE_URL={training_env_values.get('TALISHAR_FE_URL', FE_URL)}")
    else:
        print("Set TALISHAR_URL before training if the backend was started separately.")
    print()
    print("To stop the backend containers:  python start_talishar.py --down")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start or stop the Talishar stack.")
    parser.add_argument(
        "--backend-only",
        action="store_true",
        help="skip the FE dev server",
    )
    parser.add_argument(
        "--fe-only",
        action="store_true",
        help="skip Docker (backend already up)",
    )
    parser.add_argument(
        "--down",
        action="store_true",
        help="stop backend containers",
    )
    parser.add_argument(
        "--training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "apply docker-compose.training.yml (tmpfs Games, high Apache concurrency, "
            "OPcache training profile); default on"
        ),
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=1,
        help="number of independent Talishar backend containers (ports base, base+1, …)",
    )
    parser.add_argument(
        "--shard-base-port",
        type=int,
        default=DEFAULT_SHARD_BASE_PORT,
        help="host port for shard 0 (default 8080)",
    )
    args = parser.parse_args(argv)

    if sum([args.backend_only, args.fe_only, args.down]) > 1:
        parser.error("use at most one of --backend-only, --fe-only, or --down")

    return run(
        backend_only=args.backend_only,
        fe_only=args.fe_only,
        down=args.down,
        training=bool(args.training),
        shards=max(1, int(args.shards)),
        shard_base_port=int(args.shard_base_port),
    )


if __name__ == "__main__":
    raise SystemExit(main())
