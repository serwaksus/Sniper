#!/usr/bin/env python3
"""SQLite storage layer for DOTM Sniper. Replaces JSON files for high-contention data."""
import json
import os
import sqlite3
import threading
import time
from typing import Any

from config import DB_PATH, POSITIONS_FILE, HYPOTHESIS_DB_FILE, SETTINGS_FILE

_local = threading.local()

def _get_conn() -> sqlite3.Connection:
    """Get a thread-local connection."""
    if not hasattr(_local, 'conn') or _local.conn is None:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return _local.conn

def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS kv_store (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS positions (
            slug TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS hypotheses (
            slug TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            resolved INTEGER DEFAULT 0,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT NOT NULL,
            data TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_predictions_slug ON predictions(slug);
        CREATE INDEX IF NOT EXISTS idx_hypotheses_resolved ON hypotheses(resolved);
    """)
    conn.commit()

def load_positions() -> dict[str, dict[str, Any]]:
    """Load all positions as a dict (same format as positions.json)."""
    conn = _get_conn()
    rows = conn.execute("SELECT slug, data FROM positions").fetchall()
    return {row['slug']: json.loads(row['data']) for row in rows}

def _validate_position(data: Any) -> None:
    if not isinstance(data, dict):
        return
    for key in ("entry_price", "shares", "stop_loss", "high_price"):
        val = data.get(key)
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError(f"Position field '{key}' must be numeric, got {type(val).__name__}: {val}")
    shares = data.get("shares")
    if shares is not None and shares < 0:
        raise ValueError(f"Position shares must be >= 0, got {shares}")

def save_positions(positions_dict: dict[str, dict[str, Any]]) -> None:
    conn = _get_conn()
    now = time.time()
    conn.execute("BEGIN IMMEDIATE")
    current_slugs = [row[0] for row in conn.execute("SELECT slug FROM positions").fetchall()]
    for slug in current_slugs:
        if slug not in positions_dict:
            conn.execute("DELETE FROM positions WHERE slug = ?", (slug,))
    for slug, data in positions_dict.items():
        try:
            _validate_position(data)
        except ValueError as e:
            import logging
            logging.getLogger(__name__).error(f"[DB-VALIDATE] {e}")
            raise
        conn.execute(
            "INSERT OR REPLACE INTO positions (slug, data, updated_at) VALUES (?, ?, ?)",
            (slug, json.dumps(data, default=str), now)
        )
    conn.commit()

def update_position(slug: str, data: dict[str, Any]) -> None:
    conn = _get_conn()
    now = time.time()
    try:
        _validate_position(data)
    except ValueError as e:
        import logging
        logging.getLogger(__name__).error(f"[DB-VALIDATE] {e}")
        raise
    conn.execute("BEGIN IMMEDIATE")
    existing = conn.execute("SELECT data FROM positions WHERE slug = ?", (slug,)).fetchone()
    if existing:
        merged = json.loads(existing['data'])
        merged.update(data)
        conn.execute(
            "INSERT OR REPLACE INTO positions (slug, data, updated_at) VALUES (?, ?, ?)",
            (slug, json.dumps(merged, default=str), now)
        )
    else:
        conn.execute(
            "INSERT OR REPLACE INTO positions (slug, data, updated_at) VALUES (?, ?, ?)",
            (slug, json.dumps(data, default=str), now)
        )
    conn.commit()

def delete_position(slug: str) -> None:
    """Delete a single position."""
    conn = _get_conn()
    conn.execute("DELETE FROM positions WHERE slug = ?", (slug,))
    conn.commit()

def merge_save_positions(updated_positions: dict[str, dict[str, Any]]) -> None:
    conn = _get_conn()
    now = time.time()
    for slug, data in updated_positions.items():
        try:
            _validate_position(data)
        except ValueError as e:
            import logging
            logging.getLogger(__name__).error(f"[DB-VALIDATE] {e}")
            raise
        existing = conn.execute(
            "SELECT data FROM positions WHERE slug = ?", (slug,)
        ).fetchone()
        if existing:
            existing_data = json.loads(existing['data'])
            existing_data.update(data)
            data = existing_data
        conn.execute(
            "INSERT OR REPLACE INTO positions (slug, data, updated_at) VALUES (?, ?, ?)",
            (slug, json.dumps(data, default=str), now)
        )
    conn.commit()

def load_hypotheses() -> dict[str, dict[str, dict[str, Any]]]:
    """Load hypothesis database."""
    conn = _get_conn()
    rows = conn.execute("SELECT slug, data FROM hypotheses").fetchall()
    return {"hypotheses": {row['slug']: json.loads(row['data']) for row in rows}}

def save_hypotheses(db_dict: dict[str, Any]) -> None:
    conn = _get_conn()
    now = time.time()
    hypotheses = db_dict.get("hypotheses", {})
    conn.execute("BEGIN IMMEDIATE")
    current_slugs = [row[0] for row in conn.execute("SELECT slug FROM hypotheses").fetchall()]
    for slug in current_slugs:
        if slug not in hypotheses:
            conn.execute("DELETE FROM hypotheses WHERE slug = ?", (slug,))
    for slug, data in hypotheses.items():
        resolved = 1 if data.get("resolved") else 0
        conn.execute(
            "INSERT OR REPLACE INTO hypotheses (slug, data, resolved, updated_at) VALUES (?, ?, ?, ?)",
            (slug, json.dumps(data, default=str), resolved, now)
        )
    conn.commit()

def load_kv(key: str, default: Any = None) -> Any:
    """Load a key-value pair."""
    conn = _get_conn()
    row = conn.execute("SELECT value FROM kv_store WHERE key = ?", (key,)).fetchone()
    if row:
        return json.loads(row['value'])
    return default

def save_kv(key: str, value: Any) -> None:
    """Save a key-value pair."""
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, ?, ?)",
        (key, json.dumps(value, default=str), time.time())
    )
    conn.commit()

def load_position(slug: str) -> dict[str, Any] | None:
    """Load a single position by slug. Returns None if not found."""
    conn = _get_conn()
    row = conn.execute("SELECT data FROM positions WHERE slug = ?", (slug,)).fetchone()
    return json.loads(row['data']) if row else None

def load_hypothesis(slug: str) -> dict[str, Any] | None:
    """Load a single hypothesis by slug. Returns None if not found."""
    conn = _get_conn()
    row = conn.execute("SELECT data FROM hypotheses WHERE slug = ?", (slug,)).fetchone()
    return json.loads(row['data']) if row else None

def update_hypothesis(slug: str, data: dict[str, Any]) -> None:
    conn = _get_conn()
    now = time.time()
    resolved = 1 if data.get("resolved") else 0
    conn.execute("BEGIN IMMEDIATE")
    existing = conn.execute("SELECT data FROM hypotheses WHERE slug = ?", (slug,)).fetchone()
    if existing:
        merged = json.loads(existing['data'])
        merged.update(data)
        conn.execute(
            "INSERT OR REPLACE INTO hypotheses (slug, data, resolved, updated_at) VALUES (?, ?, ?, ?)",
            (slug, json.dumps(merged, default=str), resolved, now)
        )
    else:
        conn.execute(
            "INSERT OR REPLACE INTO hypotheses (slug, data, resolved, updated_at) VALUES (?, ?, ?, ?)",
            (slug, json.dumps(data, default=str), resolved, now)
        )
    conn.commit()

def delete_hypothesis(slug: str) -> None:
    """Delete a single hypothesis."""
    conn = _get_conn()
    conn.execute("DELETE FROM hypotheses WHERE slug = ?", (slug,))
    conn.commit()

def count_positions() -> int:
    """Count active positions."""
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) as cnt FROM positions").fetchone()
    return row['cnt']

def count_resolved_hypotheses() -> int:
    """Count resolved hypotheses."""
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) as cnt FROM hypotheses WHERE resolved = 1").fetchone()
    return row['cnt']

def position_slugs() -> list[str]:
    """Get all position slugs."""
    conn = _get_conn()
    rows = conn.execute("SELECT slug FROM positions").fetchall()
    return [row['slug'] for row in rows]

def hypothesis_slugs() -> list[str]:
    """Get all hypothesis slugs."""
    conn = _get_conn()
    rows = conn.execute("SELECT slug FROM hypotheses").fetchall()
    return [row['slug'] for row in rows]

SETTINGS_KEY = "bot_settings"


def load_settings() -> dict[str, Any]:
    return load_kv(SETTINGS_KEY, {})


def save_settings(settings: dict[str, Any]) -> None:
    save_kv(SETTINGS_KEY, settings)


def auto_migrate() -> None:
    """One-time migration from JSON files to SQLite. Called at startup."""
    init_db()

    positions_json = POSITIONS_FILE
    hypothesis_json = HYPOTHESIS_DB_FILE
    settings_json = SETTINGS_FILE

    if count_positions() == 0 and os.path.exists(positions_json):
        try:
            migrate_json_to_sqlite(positions_json, "positions")
            os.rename(positions_json, positions_json + ".migrated")
        except Exception as e:
            print(f"[DB] Migration of positions.json failed: {e}")

    if count_resolved_hypotheses() == 0 and os.path.exists(hypothesis_json):
        conn = _get_conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM hypotheses").fetchone()
        if row['cnt'] == 0:
            try:
                migrate_json_to_sqlite(hypothesis_json, "hypotheses")
                os.rename(hypothesis_json, hypothesis_json + ".migrated")
            except Exception as e:
                print(f"[DB] Migration of hypothesis_db.json failed: {e}")

    existing_settings = load_kv(SETTINGS_KEY)
    if (existing_settings is None or existing_settings == {}) and os.path.exists(settings_json):
        try:
            with open(settings_json) as f:
                data = json.load(f)
            save_kv(SETTINGS_KEY, data)
            os.rename(settings_json, settings_json + ".migrated")
            print("[DB] Migrated bot_settings.json to SQLite")
        except Exception as e:
            print(f"[DB] Migration of bot_settings.json failed: {e}")

def migrate_json_to_sqlite(json_path: str, table: str, key_col: str = "slug", value_col: str = "data") -> int:
    """One-time migration from JSON file to SQLite table."""
    if not os.path.exists(json_path):
        return 0
    with open(json_path) as f:
        data = json.load(f)
    conn = _get_conn()
    now = time.time()
    count = 0
    if isinstance(data, dict) and table == "positions":
        for slug, pos_data in data.items():
            conn.execute(
                f"INSERT OR REPLACE INTO {table} ({key_col}, {value_col}, updated_at) VALUES (?, ?, ?)",
                (slug, json.dumps(pos_data, default=str), now)
            )
            count += 1
    elif isinstance(data, dict) and table == "hypotheses":
        hyps = data.get("hypotheses", data)
        if isinstance(hyps, list):
            for h in hyps:
                slug = h.get("slug", "")
                if slug:
                    resolved = 1 if h.get("resolved") else 0
                    hyp_data = {k: v for k, v in h.items() if k != "slug"}
                    conn.execute(
                        f"INSERT OR REPLACE INTO {table} ({key_col}, {value_col}, resolved, updated_at) VALUES (?, ?, ?, ?)",
                        (slug, json.dumps(hyp_data, default=str), resolved, now)
                    )
                    count += 1
        elif isinstance(hyps, dict):
            for slug, hyp_data in hyps.items():
                resolved = 1 if hyp_data.get("resolved") else 0
                conn.execute(
                    f"INSERT OR REPLACE INTO {table} ({key_col}, {value_col}, resolved, updated_at) VALUES (?, ?, ?, ?)",
                    (slug, json.dumps(hyp_data, default=str), resolved, now)
                )
                count += 1
    conn.commit()
    return count
