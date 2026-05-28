#!/usr/bin/env python3
"""
Tests for utils.py — atomic JSON I/O, flock, key normalization, edge cases.
"""
import json
import os
import tempfile
import threading
import unittest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import load_json, save_json, load_json_versioned, save_json_versioned


class TestLoadJsonBasic(unittest.TestCase):
    def test_missing_file_returns_default(self):
        result = load_json("/tmp/nonexistent_test_file_12345.json", {"a": 1})
        self.assertEqual(result, {"a": 1})

    def test_load_valid_json(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"key": "value"}, f)
            path = f.name
        try:
            result = load_json(path, {})
            self.assertEqual(result, {"key": "value"})
        finally:
            os.unlink(path)

    def test_load_strips_keys(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({" key ": 1, " nested ": {" inner ": 2}}, f)
            path = f.name
        try:
            result = load_json(path, {})
            self.assertIn("key", result)
            self.assertNotIn(" key ", result)
            self.assertIn("inner", result["nested"])
        finally:
            os.unlink(path)

    def test_load_empty_file_returns_default(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("")
            path = f.name
        try:
            result = load_json(path, {"default": True})
            self.assertEqual(result, {"default": True})
        finally:
            os.unlink(path)

    def test_load_invalid_json_returns_default(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{invalid json content")
            path = f.name
        try:
            result = load_json(path, {"fallback": 42})
            self.assertEqual(result, {"fallback": 42})
        finally:
            os.unlink(path)

    def test_load_list(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([1, 2, 3], f)
            path = f.name
        try:
            result = load_json(path, [])
            self.assertEqual(result, [1, 2, 3])
        finally:
            os.unlink(path)

    def test_load_nested_lists_and_dicts(self):
        data = {"a": [{"b": 1}, {"c": 2}]}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            result = load_json(path, {})
            self.assertEqual(result, data)
        finally:
            os.unlink(path)


class TestSaveJsonBasic(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        data = {"hello": "world", "num": 42}
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            save_json(path, data)
            loaded = load_json(path, {})
            self.assertEqual(loaded, data)
        finally:
            os.unlink(path)

    def test_save_strips_string_values(self):
        data = {"key": "  value  "}
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            save_json(path, data)
            loaded = load_json(path, {})
            self.assertEqual(loaded["key"], "value")
        finally:
            os.unlink(path)

    def test_save_strips_keys(self):
        data = {" key ": 1}
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            save_json(path, data)
            loaded = load_json(path, {})
            self.assertIn("key", loaded)
        finally:
            os.unlink(path)

    def test_save_empty_dict(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            save_json(path, {})
            loaded = load_json(path, {"x": 1})
            self.assertEqual(loaded, {})
        finally:
            os.unlink(path)

    def test_save_overwrites_existing(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"old": True}, f)
            path = f.name
        try:
            save_json(path, {"new": True})
            loaded = load_json(path, {})
            self.assertNotIn("old", loaded)
            self.assertIn("new", loaded)
        finally:
            os.unlink(path)

    def test_save_list(self):
        data = [1, "two", 3.0]
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            save_json(path, data)
            loaded = load_json(path, [])
            self.assertEqual(loaded, [1, "two", 3.0])
        finally:
            os.unlink(path)


class TestAtomicWrite(unittest.TestCase):
    def test_concurrent_writes_no_corruption(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        errors = []
        results = {}

        def writer(thread_id, count):
            try:
                for i in range(count):
                    data = {"thread": thread_id, "iter": i}
                    save_json(path, data)
                    loaded = load_json(path, {})
                    if "thread" not in loaded:
                        errors.append(f"corrupt at thread={thread_id} iter={i}")
            except Exception as e:
                errors.append(f"thread={thread_id} error={e}")
            results[thread_id] = True

        threads = [threading.Thread(target=writer, args=(f"t{i}", 20)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if os.path.exists(path):
            os.unlink(path)

        self.assertEqual(len(errors), 0, f"Concurrent write errors: {errors[:5]}")


class TestVersionedJson(unittest.TestCase):
    def test_versioned_save_increments_version(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            save_json_versioned(path, {"data": 1}, expected_version=None)
            _, v1 = load_json_versioned(path, {})
            self.assertEqual(v1, 1)

            save_json_versioned(path, {"data": 2}, expected_version=v1)
            _, v2 = load_json_versioned(path, {})
            self.assertEqual(v2, 2)
        finally:
            os.unlink(path)

    def test_versioned_save_detects_conflict(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            save_json_versioned(path, {"data": 1}, expected_version=None)
            result = save_json_versioned(path, {"data": 2}, expected_version=999)
            self.assertFalse(result)
        finally:
            os.unlink(path)

    def test_load_versioned_missing_file(self):
        data, version = load_json_versioned("/tmp/nonexistent_vtest_12345.json", {"x": 1})
        self.assertEqual(data, {"x": 1})
        self.assertEqual(version, 0)


class TestEdgeCases(unittest.TestCase):
    def test_load_default_none(self):
        result = load_json("/tmp/nonexistent_12345.json", None)
        self.assertIsNone(result)

    def test_save_load_large_dict(self):
        data = {f"key_{i}": f"value_{i}" for i in range(1000)}
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            save_json(path, data)
            loaded = load_json(path, {})
            self.assertEqual(len(loaded), 1000)
        finally:
            os.unlink(path)

    def test_save_unicode(self):
        data = {"key": "значение", "emoji": "test"}
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            save_json(path, data)
            loaded = load_json(path, {})
            self.assertEqual(loaded["key"], "значение")
        finally:
            os.unlink(path)

    def test_load_deeply_nested(self):
        data = {"a": {"b": {"c": {"d": 1}}}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            result = load_json(path, {})
            self.assertEqual(result["a"]["b"]["c"]["d"], 1)
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
