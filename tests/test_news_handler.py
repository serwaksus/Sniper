#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import save_json
import news_handler as nh


class TestNewsCacheFreshness(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_cache = nh.CACHE_FILE
        nh.CACHE_FILE = os.path.join(self.tmpdir, "cache.json")

    def tearDown(self):
        nh.CACHE_FILE = self.orig_cache

    def test_fresh_cache_returns_true(self):
        entry = {"timestamp": datetime.now().isoformat(), "data": "x"}
        save_json(nh.CACHE_FILE, {"news": {"politics": entry}})
        self.assertTrue(nh._check_news_cache_freshness("politics"))

    def test_stale_cache_returns_false(self):
        old_time = (datetime.now() - timedelta(hours=3)).isoformat()
        entry = {"timestamp": old_time, "data": "x"}
        save_json(nh.CACHE_FILE, {"news": {"politics": entry}})
        self.assertFalse(nh._check_news_cache_freshness("politics"))

    def test_missing_entry_returns_false(self):
        save_json(nh.CACHE_FILE, {"news": {}})
        self.assertFalse(nh._check_news_cache_freshness("politics"))

    def test_slug_key_combined(self):
        entry = {"timestamp": datetime.now().isoformat(), "data": "x"}
        save_json(nh.CACHE_FILE, {"news": {"cluster:slug-1": entry}})
        self.assertTrue(nh._check_news_cache_freshness("cluster", slug="slug-1"))

    def test_invalid_timestamp_returns_false(self):
        entry = {"timestamp": "not-a-date", "data": "x"}
        save_json(nh.CACHE_FILE, {"news": {"politics": entry}})
        self.assertFalse(nh._check_news_cache_freshness("politics"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
