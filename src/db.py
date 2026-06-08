#!/usr/bin/env python3
"""SQLite storage layer for DOTM Sniper. Replaces JSON files for high-contention data."""
import json
import os
import sqlite3
import threading
import time

DB_PATH = "/root/dotm-sniper/sniper.db"

_local = threading.local()

def _get_conn():
    """Get a thread-local connection."""
    if not hasattr(_local, 'conn') or _local.conn is None:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return _local.conn

def init_db():
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

def load_positions():
    """Load all positions as a dict (same format as positions.json)."""
    conn = _get_conn()
    rows = conn.execute("SELECT slug, data FROM positions").fetchall()
    return {row['slug']: json.loads(row['data']) for row in rows}

def save_positions(positions_dict):
    """Save all positions (full replace)."""
    conn = _get_conn()
    now = time.time()
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("DELETE FROM positions")
    for slug, data in positions_dict.items():
        conn.execute(
            "INSERT INTO positions (slug, data, updated_at) VALUES (?, ?, ?)",
            (slug, json.dumps(data, default=str), now)
        )
    conn.commit()

def update_position(slug, data):
    """Update or insert a single position (merge-safe)."""
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO positions (slug, data, updated_at) VALUES (?, ?, ?)",
        (slug, json.dumps(data, default=str), time.time())
    )
    conn.commit()

def delete_position(slug):
    """Delete a single position."""
    conn = _get_conn()
    conn.execute("DELETE FROM positions WHERE slug = ?", (slug,))
    conn.commit()

def merge_save_positions(updated_positions):
    """Merge updated positions into existing data (same as hermes _merge_save_positions)."""
    conn = _get_conn()
    now = time.time()
    for slug, data in updated_positions.items():
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

def load_hypotheses():
    """Load hypothesis database."""
    conn = _get_conn()
    rows = conn.execute("SELECT slug, data FROM hypotheses").fetchall()
    return {"hypotheses": {row['slug']: json.loads(row['data']) for row in rows}}

def save_hypotheses(db_dict):
    """Save hypothesis database."""
    conn = _get_conn()
    now = time.time()
    hypotheses = db_dict.get("hypotheses", {})
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("DELETE FROM hypotheses")
    for slug, data in hypotheses.items():
        resolved = 1 if data.get("resolved") else 0
        conn.execute(
            "INSERT INTO hypotheses (slug, data, resolved, updated_at) VALUES (?, ?, ?, ?)",
            (slug, json.dumps(data, default=str), resolved, now)
        )
    conn.commit()

def load_kv(key, default=None):
    """Load a key-value pair."""
    conn = _get_conn()
    row = conn.execute("SELECT value FROM kv_store WHERE key = ?", (key,)).fetchone()
    if row:
        return json.loads(row['value'])
    return default

def save_kv(key, value):
    """Save a key-value pair."""
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, ?, ?)",
        (key, json.dumps(value, default=str), time.time())
    )
    conn.commit()

def load_position(slug):
    """Load a single position by slug. Returns None if not found."""
    conn = _get_conn()
    row = conn.execute("SELECT data FROM positions WHERE slug = ?", (slug,)).fetchone()
    return json.loads(row['data']) if row else None

def load_hypothesis(slug):
    """Load a single hypothesis by slug. Returns None if not found."""
    conn = _get_conn()
    row = conn.execute("SELECT data FROM hypotheses WHERE slug = ?", (slug,)).fetchone()
    return json.loads(row['data']) if row else None

def update_hypothesis(slug, data):
    """Update or insert a single hypothesis."""
    conn = _get_conn()
    resolved = 1 if data.get("resolved") else 0
    conn.execute(
        "INSERT OR REPLACE INTO hypotheses (slug, data, resolved, updated_at) VALUES (?, ?, ?, ?)",
        (slug, json.dumps(data, default=str), resolved, time.time())
    )
    conn.commit()

def delete_hypothesis(slug):
    """Delete a single hypothesis."""
    conn = _get_conn()
    conn.execute("DELETE FROM hypotheses WHERE slug = ?", (slug,))
    conn.commit()

def count_positions():
    """Count active positions."""
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) as cnt FROM positions").fetchone()
    return row['cnt']

def count_resolved_hypotheses():
    """Count resolved hypotheses."""
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) as cnt FROM hypotheses WHERE resolved = 1").fetchone()
    return row['cnt']

def position_slugs():
    """Get all position slugs."""
    conn = _get_conn()
    rows = conn.execute("SELECT slug FROM positions").fetchall()
    return [row['slug'] for row in rows]

def hypothesis_slugs():
    """Get all hypothesis slugs."""
    conn = _get_conn()
    rows = conn.execute("SELECT slug FROM hypotheses").fetchall()
    return [row['slug'] for row in rows]

SETTINGS_KEY = "bot_settings"


def load_settings():
    return load_kv(SETTINGS_KEY, {})


def save_settings(settings):
    save_kv(SETTINGS_KEY, settings)


def auto_migrate():
    """One-time migration from JSON files to SQLite. Called at startup."""
    init_db()

    positions_json = "/root/dotm-sniper/positions.json"
    hypothesis_json = "/root/dotm-sniper/hypothesis_db.json"
    settings_json = "/root/dotm-sniper/bot_settings.json"

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

def migrate_json_to_sqlite(json_path, table, key_col="slug", value_col="data"):
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
