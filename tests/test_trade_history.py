"""Tests for dual history tracking: trade_history table, record_trade, update_trade_outcome,
load_trade_history, compare_modes."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import db as db_module


def _setup_db(tmp_path):
    db_module.DB_PATH = str(tmp_path / "test.db")
    if hasattr(db_module._local, "conn") and db_module._local.conn is not None:
        db_module._local.conn.close()
        db_module._local.conn = None
    db_module.init_db()


class TestTradeHistoryMigration:
    def test_trade_history_table_exists(self, tmp_path):
        _setup_db(tmp_path)
        conn = db_module._get_conn()
        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trade_history'"
        ).fetchall()]
        assert "trade_history" in tables

    def test_indexes_created(self, tmp_path):
        _setup_db(tmp_path)
        conn = db_module._get_conn()
        indexes = [row[1] for row in conn.execute(
            "SELECT * FROM sqlite_master WHERE type='index' AND tbl_name='trade_history'"
        ).fetchall()]
        assert "idx_trade_history_mode" in indexes
        assert "idx_trade_history_slug" in indexes
        assert "idx_trade_history_ts" in indexes

    def test_migration_idempotent(self, tmp_path):
        _setup_db(tmp_path)
        db_module.init_db()
        db_module.init_db()
        conn = db_module._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM trade_history").fetchone()[0]
        assert count == 0


class TestRecordTrade:
    def test_basic_buy_record(self, tmp_path):
        _setup_db(tmp_path)
        row_id = db_module.record_trade(
            mode="demo", slug="test-market", action="buy",
            price=0.10, size_usd=50.0, shares=500,
            p_model=0.25, confidence=0.80, signal_score=65,
            prob_ratio=2.5, reason="score=65/55",
            cluster="ai_tech", source="sniper",
        )
        assert row_id > 0
        history = db_module.load_trade_history(mode="demo")
        assert len(history) == 1
        t = history[0]
        assert t["mode"] == "demo"
        assert t["slug"] == "test-market"
        assert t["action"] == "buy"
        assert t["price"] == 0.10
        assert t["size_usd"] == 50.0
        assert t["shares"] == 500
        assert t["p_model"] == 0.25
        assert t["confidence"] == 0.80
        assert t["signal_score"] == 65
        assert t["prob_ratio"] == 2.5
        assert t["outcome"] == "pending"
        assert t["reason"] == "score=65/55"
        assert t["cluster"] == "ai_tech"
        assert t["source"] == "sniper"

    def test_skip_record(self, tmp_path):
        _setup_db(tmp_path)
        row_id = db_module.record_trade(
            mode="demo", slug="skip-market", action="skip",
            price=0.05, reason="slippage_guard",
            metadata={"slippage_blocked": True, "ask": 0.15},
        )
        assert row_id > 0
        history = db_module.load_trade_history(mode="demo")
        assert history[0]["action"] == "skip"
        assert history[0]["metadata"]["slippage_blocked"] is True

    def test_simulated_mode_record(self, tmp_path):
        _setup_db(tmp_path)
        db_module.record_trade(
            mode="simulated", slug="sim-market", action="buy",
            price=0.08, p_model=0.20, signal_score=70, prob_ratio=2.5,
            reason="score=70/55", cluster="ai_tech", source="metaculus",
            metadata={"factors": ["momentum"], "source_signal": "metaculus_override"},
        )
        history = db_module.load_trade_history(mode="simulated")
        assert len(history) == 1
        assert history[0]["mode"] == "simulated"
        assert history[0]["metadata"]["source_signal"] == "metaculus_override"

    def test_metadata_json_roundtrip(self, tmp_path):
        _setup_db(tmp_path)
        meta = {"factors": ["a", "b"], "source_signal": "default", "nested": {"k": 1}}
        db_module.record_trade(mode="demo", slug="meta-test", action="buy",
                               price=0.10, metadata=meta)
        history = db_module.load_trade_history(mode="demo")
        assert history[0]["metadata"] == meta

    def test_no_metadata_stores_null(self, tmp_path):
        _setup_db(tmp_path)
        db_module.record_trade(mode="demo", slug="no-meta", action="buy", price=0.10)
        history = db_module.load_trade_history(mode="demo")
        assert history[0]["metadata"] is None

    def test_default_source_is_sniper(self, tmp_path):
        _setup_db(tmp_path)
        db_module.record_trade(mode="demo", slug="src-test", action="buy", price=0.10)
        history = db_module.load_trade_history(mode="demo")
        assert history[0]["source"] == "sniper"


class TestUpdateTradeOutcome:
    def test_update_to_win(self, tmp_path):
        _setup_db(tmp_path)
        db_module.record_trade(mode="demo", slug="win-test", action="buy", price=0.10)
        db_module.update_trade_outcome("win-test", "demo", pnl_pct=85.0, pnl_usd=42.5, outcome="win")
        history = db_module.load_trade_history(mode="demo")
        assert history[0]["outcome"] == "win"
        assert history[0]["pnl_pct"] == 85.0
        assert history[0]["pnl_usd"] == 42.5

    def test_update_to_loss(self, tmp_path):
        _setup_db(tmp_path)
        db_module.record_trade(mode="demo", slug="loss-test", action="buy", price=0.10)
        db_module.update_trade_outcome("loss-test", "demo", pnl_pct=-90.0, pnl_usd=-45.0, outcome="loss")
        history = db_module.load_trade_history(mode="demo")
        assert history[0]["outcome"] == "loss"
        assert history[0]["pnl_pct"] == -90.0

    def test_only_updates_pending(self, tmp_path):
        _setup_db(tmp_path)
        db_module.record_trade(mode="demo", slug="already-resolved", action="buy", price=0.10)
        db_module.update_trade_outcome("already-resolved", "demo", pnl_pct=50.0, pnl_usd=25.0, outcome="win")
        db_module.update_trade_outcome("already-resolved", "demo", pnl_pct=99.0, pnl_usd=99.0, outcome="win")
        history = db_module.load_trade_history(mode="demo")
        assert history[0]["pnl_pct"] == 50.0

    def test_mode_scoped(self, tmp_path):
        _setup_db(tmp_path)
        db_module.record_trade(mode="demo", slug="scoped", action="buy", price=0.10)
        db_module.record_trade(mode="simulated", slug="scoped", action="buy", price=0.10)
        db_module.update_trade_outcome("scoped", "demo", pnl_pct=80.0, pnl_usd=40.0, outcome="win")
        demo = db_module.load_trade_history(mode="demo")
        sim = db_module.load_trade_history(mode="simulated")
        assert demo[0]["outcome"] == "win"
        assert sim[0]["outcome"] == "pending"


class TestLoadTradeHistory:
    def test_filter_by_mode(self, tmp_path):
        _setup_db(tmp_path)
        db_module.record_trade(mode="demo", slug="d1", action="buy", price=0.10)
        db_module.record_trade(mode="simulated", slug="s1", action="buy", price=0.10)
        demo = db_module.load_trade_history(mode="demo")
        sim = db_module.load_trade_history(mode="simulated")
        assert len(demo) == 1
        assert len(sim) == 1
        assert demo[0]["slug"] == "d1"
        assert sim[0]["slug"] == "s1"

    def test_all_modes(self, tmp_path):
        _setup_db(tmp_path)
        db_module.record_trade(mode="demo", slug="d1", action="buy", price=0.10)
        db_module.record_trade(mode="simulated", slug="s1", action="buy", price=0.10)
        all_trades = db_module.load_trade_history()
        assert len(all_trades) == 2

    def test_limit(self, tmp_path):
        _setup_db(tmp_path)
        for i in range(10):
            db_module.record_trade(mode="demo", slug=f"lim-{i}", action="buy", price=0.10)
        result = db_module.load_trade_history(mode="demo", limit=3)
        assert len(result) == 3

    def test_ordered_by_ts_desc(self, tmp_path):
        _setup_db(tmp_path)
        db_module.record_trade(mode="demo", slug="first", action="buy", price=0.10)
        db_module.record_trade(mode="demo", slug="second", action="buy", price=0.10)
        result = db_module.load_trade_history(mode="demo")
        assert result[0]["slug"] == "second"
        assert result[1]["slug"] == "first"


class TestCompareModes:
    def test_empty_db(self, tmp_path):
        _setup_db(tmp_path)
        result = db_module.compare_modes()
        assert result["demo"]["total_trades"] == 0
        assert result["simulated"]["total_trades"] == 0
        assert result["divergence"]["trade_count_diff"] == 0

    def test_compare_with_data(self, tmp_path):
        _setup_db(tmp_path)
        db_module.record_trade(mode="demo", slug="d1", action="buy", price=0.10)
        db_module.update_trade_outcome("d1", "demo", pnl_pct=80.0, pnl_usd=40.0, outcome="win")

        db_module.record_trade(mode="demo", slug="d2", action="buy", price=0.10)
        db_module.update_trade_outcome("d2", "demo", pnl_pct=-50.0, pnl_usd=-25.0, outcome="loss")

        db_module.record_trade(mode="simulated", slug="s1", action="buy", price=0.10)
        db_module.update_trade_outcome("s1", "simulated", pnl_pct=90.0, pnl_usd=45.0, outcome="win")

        result = db_module.compare_modes()
        d = result["demo"]
        s = result["simulated"]
        assert d["total_trades"] == 2
        assert d["wins"] == 1
        assert d["losses"] == 1
        assert d["pending"] == 0
        assert d["avg_pnl_pct"] == 15.0
        assert d["total_pnl_usd"] == 15.0
        assert d["winrate"] == 0.5

        assert s["total_trades"] == 1
        assert s["wins"] == 1
        assert s["winrate"] == 1.0

        div = result["divergence"]
        assert div["trade_count_diff"] == 1
        assert div["winrate_diff"] == -0.5
        assert div["pnl_diff_pct"] == round(15.0 - 90.0, 4)

    def test_pending_trades_excluded_from_winrate(self, tmp_path):
        _setup_db(tmp_path)
        db_module.record_trade(mode="demo", slug="p1", action="buy", price=0.10)
        result = db_module.compare_modes()
        assert result["demo"]["pending"] == 1
        assert result["demo"]["winrate"] == 0

    def test_divergence_with_only_simulated(self, tmp_path):
        _setup_db(tmp_path)
        db_module.record_trade(mode="simulated", slug="s1", action="buy", price=0.10)
        db_module.update_trade_outcome("s1", "simulated", pnl_pct=50.0, pnl_usd=25.0, outcome="win")
        result = db_module.compare_modes()
        assert result["divergence"]["trade_count_diff"] == -1
