"""Tests for Talishar multi-shard naming helpers in start_talishar.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import start_talishar as st


def test_shard_compose_env_sets_absolute_init_paths() -> None:
    env = st._shard_compose_env(2, base_port=8080)
    assert env["TALISHAR_INITDB_DIR"].endswith("Talishar\\Database") or env[
        "TALISHAR_INITDB_DIR"
    ].endswith("Talishar/Database")
    assert env["TALISHAR_HOSTFILES_SEED_DIR"].endswith("HostFiles")
    assert st._shard_project_name(0) == "fab-rl-bridge"
    assert st._shard_project_name(2) == "fab-rl-bridge-shard-2"
    assert st._mysql_container_name(1) == "fab-rl-bridge-shard-1-mysql-server-1"
    assert st._shard_mysql_volume_name(3) == "fabrlbridge_shard_3_mysql"


def test_ensure_shard_mysql_schema_recreates_stale_shard_volume() -> None:
    with (
        patch.object(st, "_wait_for_mysql_table", side_effect=[False, True]),
        patch.object(st, "_mysql_is_responsive", return_value=True),
        patch.object(st, "_apply_mysql_init_sql", return_value=False),
        patch.object(st, "_recreate_shard_mysql_volume") as recreate,
        patch.object(st, "_run_compose") as up_web,
    ):
        st._ensure_shard_mysql_schema(2, training=True, base_port=8080)

    recreate.assert_called_once_with(2, training=True, base_port=8080)
    up_web.assert_called_once()


def test_ensure_shard_mysql_schema_applies_sql_before_recreate() -> None:
    with (
        patch.object(st, "_wait_for_mysql_table", side_effect=[False, True]),
        patch.object(st, "_mysql_is_responsive", return_value=True),
        patch.object(st, "_apply_mysql_init_sql", return_value=True) as apply_sql,
        patch.object(st, "_recreate_shard_mysql_volume") as recreate,
    ):
        st._ensure_shard_mysql_schema(2, training=True, base_port=8080)

    apply_sql.assert_called_once_with(2)
    recreate.assert_not_called()


def test_ensure_shard_mysql_schema_skips_recreate_when_ready() -> None:
    with (
        patch.object(st, "_wait_for_mysql_table", return_value=True),
        patch.object(st, "_recreate_shard_mysql_volume") as recreate,
    ):
        st._ensure_shard_mysql_schema(2, training=True, base_port=8080)

    recreate.assert_not_called()


def test_mysql_table_exists_parses_count() -> None:
    result = MagicMock(returncode=0, stdout="1\n", stderr="")
    with patch.object(st, "_run_docker", return_value=result):
        assert st._mysql_table_exists(1, "savedsettings") is True

    missing = MagicMock(returncode=0, stdout="0\n", stderr="")
    with patch.object(st, "_run_docker", return_value=missing):
        assert st._mysql_table_exists(1, "savedsettings") is False
