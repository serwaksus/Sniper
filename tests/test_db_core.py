"""Tests for db.py core functionality — settings, positions, hypotheses, migration, validation."""
import json
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import db as db_module


def _setup_db(tmp_path):
    db_module.DB_PATH = str(tmp_path / "test.db")
    if hasattr(db_module._local, "conn") and db_module._local.conn is not None:
        db_module._local.conn.close()
        db_module._local.conn = None
    db_module.init_db()


class TestSettingsRoundtrip:
    def test_save_and_load_empty(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_settings({})
        result = db_module.load_settings()
        assert result == {}

    def test_save_and_load_with_values(self, tmp_path):
        _setup_db(tmp_path)
        settings = {"signal_threshold": 55, "min_p_model": 0.03, "max_concurrent_trades": 15}
        db_module.save_settings(settings)
        result = db_module.load_settings()
        assert result["signal_threshold"] == 55
        assert result["min_p_model"] == 0.03

    def test_overwrite_settings(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_settings({"a": 1})
        db_module.save_settings({"b": 2})
        result = db_module.load_settings()
        assert result == {"b": 2}

    def test_load_settings_empty_db(self, tmp_path):
        _setup_db(tmp_path)
        result = db_module.load_settings()
        assert result == {}


class TestPositionsCRUD:
    def test_save_and_load_roundtrip(self, tmp_path):
        _setup_db(tmp_path)
        positions = {
            "slug-a": {"shares": 100, "entry_price": 0.10},
            "slug-b": {"shares": 50, "entry_price": 0.20},
        }
        db_module.save_positions(positions)
        result = db_module.load_positions()
        assert len(result) == 2
        assert result["slug-a"]["shares"] == 100
        assert result["slug-b"]["entry_price"] == 0.20

    def test_save_removes_deleted_slugs(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_positions({"s1": {"shares": 10}, "s2": {"shares": 20}})
        db_module.save_positions({"s1": {"shares": 10}})
        result = db_module.load_positions()
        assert "s1" in result
        assert "s2" not in result

    def test_update_position_new_slug(self, tmp_path):
        _setup_db(tmp_path)
        db_module.update_position("new-slug", {"shares": 30, "entry_price": 0.15})
        result = db_module.load_position("new-slug")
        assert result["shares"] == 30

    def test_update_position_merges_existing(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_positions({"s1": {"shares": 100, "entry_price": 0.10}})
        db_module.update_position("s1", {"shares": 80})
        result = db_module.load_position("s1")
        assert result["shares"] == 80
        assert result["entry_price"] == 0.10

    def test_delete_position(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_positions({"s1": {"shares": 100}, "s2": {"shares": 50}})
        db_module.delete_position("s1")
        assert db_module.load_position("s1") is None
        assert db_module.load_position("s2") is not None

    def test_delete_nonexistent_no_error(self, tmp_path):
        _setup_db(tmp_path)
        db_module.delete_position("ghost")

    def test_load_position_missing_returns_none(self, tmp_path):
        _setup_db(tmp_path)
        assert db_module.load_position("nonexistent") is None

    def test_empty_positions_load(self, tmp_path):
        _setup_db(tmp_path)
        assert db_module.load_positions() == {}


class TestHypothesesCRUD:
    def test_save_and_load_roundtrip(self, tmp_path):
        _setup_db(tmp_path)
        hyps = {"hypotheses": {"h1": {"p_model": 0.15, "resolved": False}}}
        db_module.save_hypotheses(hyps)
        result = db_module.load_hypotheses()
        assert "hypotheses" in result
        assert "h1" in result["hypotheses"]

    def test_save_removes_deleted_slugs(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_hypotheses({"hypotheses": {"h1": {"resolved": False}, "h2": {"resolved": True}}})
        db_module.save_hypotheses({"hypotheses": {"h1": {"resolved": False}}})
        result = db_module.load_hypotheses()
        assert "h2" not in result["hypotheses"]

    def test_load_hypothesis_single(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_hypotheses({"hypotheses": {"h1": {"p_model": 0.25}}})
        result = db_module.load_hypothesis("h1")
        assert result["p_model"] == 0.25

    def test_load_hypothesis_missing(self, tmp_path):
        _setup_db(tmp_path)
        assert db_module.load_hypothesis("ghost") is None


class TestValidatePosition:
    def test_valid_position_passes(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_positions({"s1": {"entry_price": 0.1, "shares": 100, "stop_loss": 0.05, "high_price": 0.15}})

    def test_non_numeric_entry_price_raises(self, tmp_path):
        _setup_db(tmp_path)
        try:
            db_module.save_positions({"s1": {"entry_price": "not_a_number"}})
            raise AssertionError("Should have raised")
        except (ValueError, TypeError):
            pass

    def test_non_numeric_shares_raises(self, tmp_path):
        _setup_db(tmp_path)
        try:
            db_module.save_positions({"s1": {"shares": "abc"}})
            raise AssertionError("Should have raised")
        except (ValueError, TypeError):
            pass

    def test_negative_shares_raises(self, tmp_path):
        _setup_db(tmp_path)
        try:
            db_module.save_positions({"s1": {"shares": -5}})
            raise AssertionError("Should have raised")
        except ValueError:
            pass

    def test_none_fields_allowed(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_positions({"s1": {"entry_price": None, "shares": 10}})

    def test_non_dict_data_passes(self):
        db_module._validate_position("not a dict")


class TestAutoMigrate:
    def test_migrate_positions_json(self, tmp_path):
        _setup_db(tmp_path)
        json_path = str(tmp_path / "positions.json")
        with open(json_path, "w") as f:
            json.dump({"s1": {"shares": 100}}, f)
        original_positions = db_module.POSITIONS_FILE
        db_module.POSITIONS_FILE = json_path
        db_module.POSITIONS_FILE = original_positions
        count = db_module.migrate_json_to_sqlite(json_path, "positions")
        assert count == 1

    def test_migrate_hypotheses_dict_format(self, tmp_path):
        _setup_db(tmp_path)
        json_path = str(tmp_path / "hypothesis_db.json")
        with open(json_path, "w") as f:
            json.dump({"hypotheses": {"h1": {"p_model": 0.20, "resolved": True}}}, f)
        count = db_module.migrate_json_to_sqlite(json_path, "hypotheses")
        assert count == 1

    def test_migrate_nonexistent_file_returns_zero(self, tmp_path):
        _setup_db(tmp_path)
        count = db_module.migrate_json_to_sqlite("/tmp/nonexistent_file.json", "positions")
        assert count == 0


class TestKVStore:
    def test_save_and_load_string(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_kv("key1", "value1")
        assert db_module.load_kv("key1") == "value1"

    def test_load_missing_returns_default(self, tmp_path):
        _setup_db(tmp_path)
        assert db_module.load_kv("missing", "default") == "default"

    def test_save_and_load_complex(self, tmp_path):
        _setup_db(tmp_path)
        data = {"nested": {"key": [1, 2, 3]}}
        db_module.save_kv("complex", data)
        assert db_module.load_kv("complex") == data


class TestConcurrentWrites:
    def test_concurrent_position_updates(self, tmp_path):
        _setup_db(tmp_path)
        db_module.save_positions({})
        errors = []

        def writer(prefix, count):
            try:
                for i in range(count):
                    db_module.update_position(f"{prefix}_{i}", {"shares": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"t{j}", 15)) for j in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert db_module.count_positions() == 60
