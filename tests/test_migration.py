"""Test JSON -> SQLite migration."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import db as db_module


def _setup_db(tmp_path):
    db_module.DB_PATH = str(tmp_path / "test.db")
    if hasattr(db_module._local, 'conn') and db_module._local.conn is not None:
        db_module._local.conn.close()
        db_module._local.conn = None
    db_module._initialized = False
    db_module.init_db()


class TestMigration:
    def test_positions_migration(self, tmp_path):
        _setup_db(tmp_path)
        json_path = str(tmp_path / "positions.json")
        test_data = {"slug1": {"shares": 100}, "slug2": {"shares": 50}}
        with open(json_path, 'w') as f:
            json.dump(test_data, f)

        count = db_module.migrate_json_to_sqlite(json_path, "positions")
        assert count == 2

        result = db_module.load_positions()
        assert len(result) == 2
        assert result["slug1"]["shares"] == 100
        assert result["slug2"]["shares"] == 50

    def test_hypotheses_migration_list_format(self, tmp_path):
        _setup_db(tmp_path)
        json_path = str(tmp_path / "hypothesis_db.json")
        test_data = {
            "hypotheses": [
                {"slug": "s1", "p_model": 0.15, "resolved": False},
                {"slug": "s2", "p_model": 0.20, "resolved": True, "outcome": "yes"},
            ],
            "resolved": [
                {"slug": "s2", "p_model": 0.20, "resolved": True, "outcome": "yes"},
            ]
        }
        with open(json_path, 'w') as f:
            json.dump(test_data, f)

        count = db_module.migrate_json_to_sqlite(json_path, "hypotheses")
        assert count == 2

        result = db_module.load_hypotheses()
        assert "hypotheses" in result
        assert len(result["hypotheses"]) == 2
        assert result["hypotheses"]["s2"]["resolved"] is True

    def test_hypotheses_migration_dict_format(self, tmp_path):
        _setup_db(tmp_path)
        json_path = str(tmp_path / "hypothesis_db.json")
        test_data = {
            "hypotheses": {
                "s1": {"p_model": 0.15, "resolved": False},
                "s2": {"p_model": 0.20, "resolved": True, "outcome": "yes"},
            }
        }
        with open(json_path, 'w') as f:
            json.dump(test_data, f)

        count = db_module.migrate_json_to_sqlite(json_path, "hypotheses")
        assert count == 2

        result = db_module.load_hypotheses()
        assert len(result["hypotheses"]) == 2

    def test_empty_migration_skipped(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_positions({"existing": {"shares": 1}})

        json_path = str(tmp_path / "positions.json")
        with open(json_path, 'w') as f:
            json.dump({"new": {"shares": 99}}, f)

        count = db_module.migrate_json_to_sqlite(json_path, "positions")
        assert count == 1

        result = db_module.load_positions()
        assert "new" in result

    def test_settings_migration(self, tmp_path):
        _setup_db(tmp_path)
        settings_path = str(tmp_path / "bot_settings.json")
        test_settings = {"signal_threshold": 55, "min_p_model": 0.03}
        with open(settings_path, 'w') as f:
            json.dump(test_settings, f)

        with open(settings_path) as f:
            data = json.load(f)
        db_module.save_kv("bot_settings", data)

        result = db_module.load_settings()
        assert result["signal_threshold"] == 55
        assert result["min_p_model"] == 0.03

    def test_hypotheses_db_load_all_format(self, tmp_path):
        _setup_db(tmp_path)
        import hypotheses_db
        hypotheses_db._initialized = False

        db_module.update_hypothesis("h1", {"p_model": 0.15, "resolved": False})
        db_module.update_hypothesis("h2", {"p_model": 0.20, "resolved": True, "outcome": "YES"})

        result = hypotheses_db.load_all()
        assert "hypotheses" in result
        assert "resolved" in result

        active = [h for h in result["hypotheses"] if not h.get("resolved")]
        resolved = result["resolved"]
        assert len(active) == 1
        assert active[0]["slug"] == "h1"
        assert len(resolved) == 1
        assert resolved[0]["slug"] == "h2"

    def test_hypotheses_db_roundtrip(self, tmp_path):
        _setup_db(tmp_path)
        import hypotheses_db
        hypotheses_db._initialized = False

        db = {
            "hypotheses": [
                {"slug": "s1", "p_model": 0.15, "resolved": False},
            ],
            "resolved": [
                {"slug": "s2", "p_model": 0.20, "resolved": True, "outcome": "YES"},
            ]
        }
        hypotheses_db.save_all(db)

        result = hypotheses_db.load_all()
        assert len(result["hypotheses"]) == 2
        active = [h for h in result["hypotheses"] if not h.get("resolved")]
        assert len(active) == 1
        assert active[0]["slug"] == "s1"
        assert len(result["resolved"]) == 1
        assert result["resolved"][0]["slug"] == "s2"
        assert result["resolved"][0]["outcome"] == "YES"

    def test_count_resolved(self, tmp_path):
        _setup_db(tmp_path)
        import hypotheses_db
        hypotheses_db._initialized = False

        db_module.update_hypothesis("h1", {"resolved": True})
        db_module.update_hypothesis("h2", {"resolved": False})
        db_module.update_hypothesis("h3", {"resolved": True})

        assert hypotheses_db.count_resolved() == 2


class TestSchemaMigration:
    def test_migration_idempotency(self, tmp_path):
        _setup_db(tmp_path)
        original_migrations = db_module.MIGRATIONS
        db_module.MIGRATIONS = [
            (1, "add_foo_column", "ALTER TABLE positions ADD COLUMN foo TEXT DEFAULT NULL"),
        ]
        try:
            db_module.run_migrations()
            conn = db_module._get_conn()
            rows = conn.execute("SELECT id, name FROM _migrations ORDER BY id").fetchall()
            assert len(rows) == 1
            assert rows[0]["id"] == 1
            assert rows[0]["name"] == "add_foo_column"

            db_module.run_migrations()
            rows = conn.execute("SELECT id, name FROM _migrations ORDER BY id").fetchall()
            assert len(rows) == 1
        finally:
            db_module.MIGRATIONS = original_migrations

    def test_multiple_migrations_applied_in_order(self, tmp_path):
        _setup_db(tmp_path)
        original_migrations = db_module.MIGRATIONS
        db_module.MIGRATIONS = [
            (2, "add_bar_column", "ALTER TABLE positions ADD COLUMN bar TEXT DEFAULT NULL"),
            (1, "add_baz_column", "ALTER TABLE positions ADD COLUMN baz TEXT DEFAULT NULL"),
        ]
        try:
            db_module.run_migrations()
            conn = db_module._get_conn()
            rows = conn.execute("SELECT id, name FROM _migrations ORDER BY id").fetchall()
            assert len(rows) == 2
            assert rows[0]["name"] == "add_baz_column"
            assert rows[1]["name"] == "add_bar_column"

            db_module.run_migrations()
            rows = conn.execute("SELECT id, name FROM _migrations ORDER BY id").fetchall()
            assert len(rows) == 2
        finally:
            db_module.MIGRATIONS = original_migrations

    def test_migrations_table_created(self, tmp_path):
        _setup_db(tmp_path)
        conn = db_module._get_conn()
        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_migrations'"
        ).fetchall()]
        assert "_migrations" in tables
