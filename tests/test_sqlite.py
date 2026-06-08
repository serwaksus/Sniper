"""Tests for SQLite storage layer."""
import json
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import db as db_module

def _setup_db(tmp_path):
    db_module.DB_PATH = str(tmp_path / "test.db")
    if hasattr(db_module._local, 'conn'):
        db_module._local.conn.close()
        db_module._local.conn = None
    db_module._initialized = False
    db_module.init_db()

class TestKVStore:
    def test_save_and_load(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_kv("test_key", {"foo": "bar"})
        result = db_module.load_kv("test_key")
        assert result == {"foo": "bar"}

    def test_load_missing_returns_default(self, tmp_path):
        _setup_db(tmp_path)
        assert db_module.load_kv("nonexistent", 42) == 42

    def test_overwrite(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_kv("key", "v1")
        db_module.save_kv("key", "v2")
        assert db_module.load_kv("key") == "v2"


class TestPositions:
    def test_save_and_load(self, tmp_path):
        _setup_db(tmp_path)
        positions = {"slug1": {"shares": 100, "entry_price": 0.1}}
        db_module.save_positions(positions)
        result = db_module.load_positions()
        assert "slug1" in result
        assert result["slug1"]["shares"] == 100

    def test_update_single(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_positions({"s1": {"shares": 100}})
        db_module.update_position("s2", {"shares": 50})
        result = db_module.load_positions()
        assert len(result) == 2

    def test_delete(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_positions({"s1": {"shares": 100}, "s2": {"shares": 50}})
        db_module.delete_position("s1")
        result = db_module.load_positions()
        assert "s1" not in result
        assert "s2" in result

    def test_merge_save(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_positions({"s1": {"shares": 100, "entry_price": 0.1}})
        db_module.merge_save_positions({"s1": {"shares": 90}})
        result = db_module.load_positions()
        assert result["s1"]["shares"] == 90
        assert result["s1"]["entry_price"] == 0.1

    def test_concurrent_writes(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_positions({})
        errors = []

        def writer(prefix, count):
            try:
                for i in range(count):
                    db_module.update_position(f"{prefix}_{i}", {"shares": i})
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer, args=("a", 20))
        t2 = threading.Thread(target=writer, args=("b", 20))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors
        result = db_module.load_positions()
        assert len(result) == 40


class TestHypotheses:
    def test_save_and_load(self, tmp_path):
        _setup_db(tmp_path)
        hyps = {"hypotheses": {"s1": {"p_model": 0.15, "resolved": False}}}
        db_module.save_hypotheses(hyps)
        result = db_module.load_hypotheses()
        assert "hypotheses" in result
        assert "s1" in result["hypotheses"]

    def test_resolved_flag(self, tmp_path):
        _setup_db(tmp_path)
        hyps = {"hypotheses": {
            "s1": {"p_model": 0.15, "resolved": False},
            "s2": {"p_model": 0.20, "resolved": True},
        }}
        db_module.save_hypotheses(hyps)
        conn = db_module._get_conn()
        resolved = conn.execute("SELECT slug FROM hypotheses WHERE resolved = 1").fetchall()
        assert len(resolved) == 1
        assert resolved[0]['slug'] == "s2"


class TestLoadSingle:
    def test_load_position_found(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_positions({"s1": {"shares": 100}, "s2": {"shares": 50}})
        result = db_module.load_position("s1")
        assert result == {"shares": 100}

    def test_load_position_missing(self, tmp_path):
        _setup_db(tmp_path)
        assert db_module.load_position("nonexistent") is None

    def test_load_hypothesis_found(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_hypotheses({"hypotheses": {"h1": {"p_model": 0.15}}})
        result = db_module.load_hypothesis("h1")
        assert result["p_model"] == 0.15

    def test_load_hypothesis_missing(self, tmp_path):
        _setup_db(tmp_path)
        assert db_module.load_hypothesis("nonexistent") is None


class TestHypothesisCRUD:
    def test_update_hypothesis_insert(self, tmp_path):
        _setup_db(tmp_path)
        db_module.update_hypothesis("h1", {"p_model": 0.15, "resolved": False})
        result = db_module.load_hypothesis("h1")
        assert result["p_model"] == 0.15

    def test_update_hypothesis_replace(self, tmp_path):
        _setup_db(tmp_path)
        db_module.update_hypothesis("h1", {"p_model": 0.15, "resolved": False})
        db_module.update_hypothesis("h1", {"p_model": 0.25, "resolved": True})
        result = db_module.load_hypothesis("h1")
        assert result["p_model"] == 0.25
        assert result["resolved"] is True

    def test_delete_hypothesis(self, tmp_path):
        _setup_db(tmp_path)
        db_module.update_hypothesis("h1", {"p_model": 0.15})
        db_module.delete_hypothesis("h1")
        assert db_module.load_hypothesis("h1") is None


class TestCountAndSlugs:
    def test_count_positions_empty(self, tmp_path):
        _setup_db(tmp_path)
        assert db_module.count_positions() == 0

    def test_count_positions(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_positions({"s1": {"shares": 100}, "s2": {"shares": 50}})
        assert db_module.count_positions() == 2

    def test_count_resolved_hypotheses(self, tmp_path):
        _setup_db(tmp_path)
        db_module.update_hypothesis("h1", {"resolved": True})
        db_module.update_hypothesis("h2", {"resolved": False})
        assert db_module.count_resolved_hypotheses() == 1

    def test_position_slugs(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_positions({"s1": {}, "s2": {}, "s3": {}})
        slugs = db_module.position_slugs()
        assert sorted(slugs) == ["s1", "s2", "s3"]

    def test_hypothesis_slugs(self, tmp_path):
        _setup_db(tmp_path)
        db_module.update_hypothesis("h1", {"resolved": False})
        db_module.update_hypothesis("h2", {"resolved": False})
        slugs = db_module.hypothesis_slugs()
        assert sorted(slugs) == ["h1", "h2"]


class TestAutoMigrate:
    def test_migrates_positions_json(self, tmp_path):
        _setup_db(tmp_path)
        json_path = str(tmp_path / "positions.json")
        with open(json_path, "w") as f:
            json.dump({"s1": {"shares": 100}, "s2": {"shares": 50}}, f)
        original_db_path = db_module.DB_PATH
        count = db_module.migrate_json_to_sqlite(json_path, "positions")
        assert count == 2
        assert db_module.count_positions() == 2

    def test_migrates_hypotheses_json(self, tmp_path):
        _setup_db(tmp_path)
        json_path = str(tmp_path / "hypothesis_db.json")
        with open(json_path, "w") as f:
            json.dump({"hypotheses": {"h1": {"p_model": 0.15, "resolved": True}}}, f)
        count = db_module.migrate_json_to_sqlite(json_path, "hypotheses")
        assert count == 1
        assert db_module.count_resolved_hypotheses() == 1

    def test_auto_migrate_skips_if_data_exists(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_positions({"existing": {"shares": 1}})
        assert db_module.count_positions() == 1
