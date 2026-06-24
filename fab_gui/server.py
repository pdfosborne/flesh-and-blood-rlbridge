"""HTTP server for the FAB RL Bridge web GUI."""

from __future__ import annotations

import json
import mimetypes
import re
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from fab_tui.config import EnvironmentSettings, REPO_ROOT

from fab_gui import api as gui_api

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_PORT = 8765


def _json_response(handler: BaseHTTPRequestHandler, payload: Any, *, status: int = 200) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    data = json.loads(raw.decode("utf-8"))
    return data if isinstance(data, dict) else {}


def _serve_bytes(
    handler: BaseHTTPRequestHandler,
    data: bytes,
    *,
    content_type: str,
    cache_seconds: int = 3600,
) -> None:
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", f"public, max-age={cache_seconds}")
    handler.end_headers()
    handler.wfile.write(data)


def _serve_file(handler: BaseHTTPRequestHandler, path: Path) -> None:
    if not path.is_file():
        handler.send_error(HTTPStatus.NOT_FOUND)
        return
    content = path.read_bytes()
    mime, _ = mimetypes.guess_type(str(path))
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", mime or "application/octet-stream")
    handler.send_header("Content-Length", str(len(content)))
    handler.end_headers()
    handler.wfile.write(content)


class GuiRequestHandler(BaseHTTPRequestHandler):
    env = EnvironmentSettings()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path in {"/", "/index.html"}:
            return _serve_file(self, STATIC_DIR / "index.html")
        if path.startswith("/static/"):
            rel = unquote(path[len("/static/") :])
            return _serve_file(self, STATIC_DIR / rel)
        if path.startswith("/api/card-image/"):
            card_id = unquote(path[len("/api/card-image/") :])
            if not card_id:
                return _json_response(self, {"error": "card id required"}, status=400)
            payload = gui_api.fetch_card_image(self.env, card_id)
            if payload is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data, mime = payload
            return _serve_bytes(self, data, content_type=mime)

        if path == "/api/config":
            return _json_response(
                self,
                {
                    "talishar_url": self.env.talishar_url,
                    "talishar_fe_url": self.env.talishar_fe_url,
                    "assets_path": self.env.assets_path,
                },
            )
        if path == "/api/precons":
            return _json_response(self, {"precons": gui_api.list_precons(self.env)})
        if path == "/api/saved-decks":
            return _json_response(self, {"decks": gui_api.list_saved_decks_api()})

        query = parse_qs(parsed.query)

        if path == "/api/deck/load":
            deck_path = Path((query.get("path") or [""])[0]).expanduser()
            if not deck_path.is_file():
                return _json_response(self, {"error": "Deck file not found"}, status=404)
            return _json_response(self, gui_api.load_deck_payload(deck_path, self.env))

        if path == "/api/cards/search":
            q = (query.get("q") or [""])[0]
            fmt = (query.get("format") or ["silver_age"])[0]
            limit = int((query.get("limit") or ["24"])[0])
            hits = gui_api.search_cards(
                q, game_format=fmt, talishar_url=self.env.talishar_url, limit=limit
            )
            return _json_response(self, {"cards": hits})

        if path == "/api/equipment/search":
            q = (query.get("q") or [""])[0]
            fmt = (query.get("format") or ["silver_age"])[0]
            hero_id = (query.get("hero_id") or [""])[0]
            hero_class = (query.get("hero_class") or [""])[0]
            slot = (query.get("slot") or [None])[0]
            limit = int((query.get("limit") or ["24"])[0])
            hits = gui_api.search_equipment(
                q,
                game_format=fmt,
                hero_id=hero_id,
                hero_class=hero_class,
                talishar_url=self.env.talishar_url,
                slot=slot,
                limit=limit,
            )
            return _json_response(self, {"equipment": hits})

        if path == "/api/equipment/alternatives":
            fmt = (query.get("format") or ["silver_age"])[0]
            hero_id = (query.get("hero_id") or [""])[0]
            hero_class = (query.get("hero_class") or [""])[0]
            slot = (query.get("slot") or [""])[0]
            if not slot:
                return _json_response(self, {"error": "slot required"}, status=400)
            hits = gui_api.equipment_alternatives_for_slot(
                game_format=fmt,
                hero_id=hero_id,
                hero_class=hero_class,
                talishar_url=self.env.talishar_url,
                slot=slot,
            )
            return _json_response(self, {"equipment": hits, "slot": slot})

        if path == "/api/equipment/loadout":
            header = (query.get("equipment_header") or [""])[0]
            hero_id = (query.get("hero_id") or [""])[0]
            hero_class = (query.get("hero_class") or [""])[0]
            fmt = (query.get("format") or ["silver_age"])[0]
            rows = gui_api.equipment_loadout(
                header,
                hero_id=hero_id,
                hero_class=hero_class,
                game_format=fmt,
                talishar_url=self.env.talishar_url,
            )
            return _json_response(self, {"loadout": rows})

        match = re.fullmatch(r"/api/runs/([^/]+)/dashboard", path)
        if match:
            run_id = match.group(1)
            html_path = gui_api.dashboard_path(run_id)
            if html_path is None or not html_path.is_file():
                return _json_response(self, {"error": "Dashboard not ready"}, status=404)
            return _serve_file(self, html_path)

        match = re.fullmatch(r"/api/runs/([^/]+)/status", path)
        if match:
            status = gui_api.run_status(match.group(1))
            if status is None:
                return _json_response(self, {"error": "Run not found"}, status=404)
            return _json_response(self, status)

        match = re.fullmatch(r"/api/runs/([^/]+)/results", path)
        if match:
            results = gui_api.run_results(match.group(1))
            if results is None:
                return _json_response(self, {"error": "Run not found"}, status=404)
            return _json_response(self, results)

        match = re.fullmatch(r"/api/runs/([^/]+)/replay\.gif", path)
        if match:
            run_id = match.group(1)
            out_dir = gui_api.resolve_run_out_dir(run_id)
            if out_dir is None:
                return _json_response(self, {"error": "Run not found"}, status=404)
            gif_path = gui_api.replay_gif_path(out_dir)
            if gif_path is None or not gif_path.is_file():
                return _json_response(self, {"error": "Replay GIF not ready"}, status=404)
            return _serve_file(self, gif_path)

        match = re.fullmatch(r"/api/runs/([^/]+)/replay-status", path)
        if match:
            status = gui_api.replay_render_status(match.group(1))
            if status is None:
                return _json_response(self, {"error": "Run not found"}, status=404)
            return _json_response(self, status)

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            body = _read_json_body(self)
        except json.JSONDecodeError:
            return _json_response(self, {"error": "Invalid JSON body"}, status=400)

        if path == "/api/import/precon":
            deck_name = str(body.get("deck_name") or "").strip()
            if not deck_name:
                return _json_response(self, {"error": "deck_name required"}, status=400)
            try:
                deck_path = gui_api.import_precon(deck_name, self.env)
                payload = gui_api.load_deck_payload(deck_path, self.env)
                return _json_response(self, payload)
            except Exception as exc:  # noqa: BLE001
                return _json_response(self, {"error": str(exc)}, status=400)

        if path == "/api/import/fabrary":
            source = str(body.get("url_or_slug") or "").strip()
            if not source:
                return _json_response(self, {"error": "url_or_slug required"}, status=400)
            try:
                deck_path = gui_api.import_fabrary(source)
                payload = gui_api.load_deck_payload(deck_path, self.env)
                return _json_response(self, payload)
            except Exception as exc:  # noqa: BLE001
                return _json_response(self, {"error": str(exc)}, status=400)

        if path == "/api/opponent/precon":
            deck_name = str(body.get("deck_name") or "").strip()
            if not deck_name:
                return _json_response(self, {"error": "deck_name required"}, status=400)
            return _json_response(self, gui_api.opponent_from_precon(deck_name, self.env))

        if path == "/api/opponent/fabrary":
            source = str(body.get("url_or_slug") or "").strip()
            if not source:
                return _json_response(self, {"error": "url_or_slug required"}, status=400)
            try:
                return _json_response(self, gui_api.opponent_from_fabrary(source, self.env))
            except Exception as exc:  # noqa: BLE001
                return _json_response(self, {"error": str(exc)}, status=400)

        if path == "/api/guide-baseline":
            try:
                result = gui_api.compute_guide_baseline(
                    card_pool={str(k): int(v) for k, v in (body.get("card_pool") or {}).items()},
                    opponent_hero_id=str(body.get("opponent_hero_id") or ""),
                    hero_id=str(body.get("hero_id") or ""),
                    hero_class=str(body.get("hero_class") or ""),
                    game_format=str(body.get("game_format") or "silver_age"),
                    equipment_header=str(body.get("equipment_header") or ""),
                )
                fmt = str(body.get("game_format") or "silver_age")
                result["deck_entries"] = gui_api.deck_counts_to_entries(
                    result["baseline_deck"], game_format=fmt, talishar_url=self.env.talishar_url
                )
                opponent = body.get("opponent")
                if isinstance(opponent, dict) and opponent.get("opponent_deck"):
                    result["opponent_guide"] = gui_api.apply_opponent_guide_sideboard(
                        self.env,
                        opponent,
                        player_hero_id=str(body.get("hero_id") or ""),
                    )
                return _json_response(self, result)
            except Exception as exc:  # noqa: BLE001
                return _json_response(self, {"error": str(exc)}, status=400)

        if path == "/api/deck/swap":
            result = gui_api.try_swap(
                {str(k): int(v) for k, v in (body.get("deck") or {}).items()},
                {str(k): int(v) for k, v in (body.get("card_pool") or {}).items()},
                str(body.get("out_card") or ""),
                str(body.get("in_card") or ""),
            )
            if result is None:
                return _json_response(self, {"error": "Invalid swap"}, status=400)
            fmt = str(body.get("game_format") or "silver_age")
            return _json_response(
                self,
                {
                    "deck": result["deck"],
                    "card_pool": result["card_pool"],
                    "deck_entries": gui_api.deck_counts_to_entries(
                        result["deck"], game_format=fmt, talishar_url=self.env.talishar_url
                    ),
                },
            )

        if path == "/api/deck/save":
            try:
                saved_path = gui_api.save_deck_api(
                    baseline_deck={str(k): int(v) for k, v in (body.get("deck") or {}).items()},
                    card_pool={str(k): int(v) for k, v in (body.get("card_pool") or {}).items()},
                    equipment_header=str(body.get("equipment_header") or ""),
                    hero_id=str(body.get("hero_id") or ""),
                    hero_class=str(body.get("hero_class") or ""),
                    game_format=str(body.get("game_format") or "silver_age"),
                    label=str(body.get("label") or "Saved list"),
                    opponent_hero_id=str(body.get("opponent_hero_id") or ""),
                    baseline_label=str(body.get("baseline_label") or "GUI baseline"),
                )
                return _json_response(self, {"path": saved_path})
            except Exception as exc:  # noqa: BLE001
                return _json_response(self, {"error": str(exc)}, status=400)

        if path == "/api/equipment/replace":
            try:
                header = str(body.get("equipment_header") or "")
                new_header = gui_api.replace_equipment_slot(
                    header,
                    slot_index=int(body.get("slot_index") or 0),
                    replacement_card_id=str(body.get("replacement_card_id") or ""),
                    hero_id=str(body.get("hero_id") or ""),
                    hero_class=str(body.get("hero_class") or ""),
                    game_format=str(body.get("game_format") or "silver_age"),
                )
                if new_header is None:
                    return _json_response(self, {"error": "Invalid equipment replacement"}, status=400)
                rows = gui_api.equipment_loadout(
                    new_header,
                    hero_id=str(body.get("hero_id") or ""),
                    hero_class=str(body.get("hero_class") or ""),
                    game_format=str(body.get("game_format") or "silver_age"),
                    talishar_url=self.env.talishar_url,
                )
                return _json_response(self, {"equipment_header": new_header, "loadout": rows})
            except Exception as exc:  # noqa: BLE001
                return _json_response(self, {"error": str(exc)}, status=400)

        if path == "/api/training/start":
            try:
                session_path = gui_api.persist_session_deck(body)
                variants = body.get("variants") or []
                spec_kwargs = {
                    "max_parallel": int(body.get("max_parallel", 0)),
                    "play_episodes": int(body.get("play_episodes", 5000)),
                    "final_eval_episodes": int(body.get("final_eval_episodes", 50)),
                    "build_cpp_engine": bool(body.get("build_cpp_engine", True)),
                    "workers": body.get("workers"),
                    "render_replay_gif": bool(body.get("render_replay_gif", True)),
                }
                if spec_kwargs["workers"] is not None:
                    spec_kwargs["workers"] = int(spec_kwargs["workers"])
                run = gui_api.start_training(
                    env=self.env,
                    starting_deck_path=session_path,
                    opponent_hero_id=str(body.get("opponent_hero_id") or ""),
                    opponent_deck=str(body.get("opponent_deck") or ""),
                    baseline_deck={str(k): int(v) for k, v in (body.get("deck") or {}).items()},
                    card_pool={str(k): int(v) for k, v in (body.get("card_pool") or {}).items()},
                    equipment_header=str(body.get("equipment_header") or ""),
                    baseline_label=str(body.get("baseline_label") or "Baseline"),
                    variants=variants,
                    spec_kwargs=spec_kwargs,
                )
                manifest_path = run.out_dir / "candidates_manifest.json"
                if manifest_path.is_file():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["gui_run_id"] = run.run_id
                    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                return _json_response(self, run.to_dict())
            except Exception as exc:  # noqa: BLE001
                return _json_response(self, {"error": str(exc)}, status=500)

        match = re.fullmatch(r"/api/runs/([^/]+)/render-replay", path)
        if match:
            try:
                payload = gui_api.start_replay_render(match.group(1), self.env)
                return _json_response(self, payload)
            except Exception as exc:  # noqa: BLE001
                return _json_response(self, {"error": str(exc)}, status=400)

        return _json_response(self, {"error": "Not found"}, status=404)


def run_gui(*, host: str = "127.0.0.1", port: int = DEFAULT_PORT, open_browser: bool = True) -> int:
    """Start the web GUI and block until interrupted."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    server = ThreadingHTTPServer((host, port), GuiRequestHandler)
    url = f"http://{host}:{port}/"
    print("=" * 62)
    print("  FAB RL Bridge — Web GUI")
    print("=" * 62)
    print(f"  Open: {url}")
    print("  Press Ctrl+C to stop")
    print("=" * 62)

    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down GUI server…")
    finally:
        server.server_close()
    return 0
