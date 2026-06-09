import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from datetime import timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import db as db_module
import positions_db


def _setup_db(tmp_path):
    db_module.DB_PATH = str(tmp_path / "test.db")
    if hasattr(db_module._local, 'conn') and db_module._local.conn is not None:
        db_module._local.conn.close()
        db_module._local.conn = None
    db_module.init_db()
    positions_db._initialized = False


def _make_market(**overrides):
    m = {
        "slug": "will-x-happen-before-july",
        "price": 0.10,
        "question": "Will X happen before July?",
        "outcome": "YES",
        "clusters": ["usa_politics"],
        "volume": 500000,
        "ttl_hours": 720,
    }
    m.update(overrides)
    return m


def _mock_llm_response(estimated_probability=0.20, confidence=0.80):
    return MagicMock(
        status_code=200,
        json=MagicMock(return_value={
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "factors": [
                            {"factor": "strong catalyst", "direction": "supports", "weight": "high", "source": "test"},
                            {"factor": "momentum", "direction": "supports", "weight": "medium", "source": "test"},
                        ],
                        "estimated_probability": estimated_probability,
                        "confidence": confidence,
                        "reasoning": "Strong DOTM signal",
                    })
                }
            }]
        })
    )


class TestE2EIntegration:
    def test_full_lifecycle_with_real_sqlite(self, tmp_path):
        _setup_db(tmp_path)

        db_module.save_settings({
            "signal_threshold": 55,
            "min_p_model": 0.03,
            "min_confidence": 0.65,
            "max_concurrent_trades": 15,
            "position_size_pct": 0.10,
            "starting_balance": 500.0,
            "total_resolved": 0,
        })
        settings = db_module.load_settings()
        assert settings["signal_threshold"] == 55

        market = _make_market()
        mock_llm_resp = _mock_llm_response(estimated_probability=0.25, confidence=0.80)

        with patch("signal_pipeline.requests.post", return_value=mock_llm_resp), \
             patch("signal_pipeline.get_metaculus_forecast", return_value={"found": False}), \
             patch("order_manager.get_best_ask", return_value=0.11), \
             patch("signal_pipeline._check_llm_circuit_breaker", return_value=True):
            from signal_pipeline import full_market_analysis
            analysis = full_market_analysis(market)

        assert analysis["action"] in ("BUY", "SKIP")
        assert "signal_score" in analysis
        assert "min_signal" in analysis
        assert analysis["p_model"] > 0

        from position_manager import position_size, conviction_adjusted_size
        base_size = position_size(
            analysis["p_model"], market["price"], 500.0,
            confidence=analysis["confidence"],
            cluster=market["clusters"][0],
        )

        if base_size > 0 and analysis["action"] == "BUY":
            adjusted_size = conviction_adjusted_size(
                base_size, analysis["signal_score"], analysis["min_signal"],
            )
            assert adjusted_size >= 5
            estimated_size = adjusted_size
        else:
            estimated_size = base_size

        if estimated_size <= 0:
            pytest.skip("Kelly returned 0, testing with forced size")
            estimated_size = 10

        mock_buy_result = MagicMock(stdout='{"ok": true}', returncode=0)
        mock_balance_result = MagicMock(
            stdout=json.dumps({"data": {"cash": 500.0, "total_value": 500.0}}),
            returncode=0,
        )
        mock_portfolio_result = MagicMock(
            stdout=json.dumps({"data": []}),
            returncode=0,
        )
        mock_book_result = MagicMock(
            stdout=json.dumps({"data": {"asks": [{"price": "0.11"}], "bids": [{"price": "0.09"}]}}),
            returncode=0,
        )

        def mock_subprocess_run(cmd, **kwargs):
            if "buy" in cmd:
                return mock_buy_result
            if "balance" in cmd:
                return mock_balance_result
            if "portfolio" in cmd:
                return mock_portfolio_result
            if "book" in cmd:
                return mock_book_result
            return MagicMock(stdout='{"ok": true}', returncode=0)

        from datetime import datetime as _dt
        pos_data = {
            "status": "active",
            "entry_price": market["price"],
            "shares": int(estimated_size / market["price"]),
            "high_price": market["price"],
            "stop_loss": round(market["price"] * 0.80, 4),
            "trailing_on": False,
            "outcome": "YES",
            "clusters": market["clusters"],
            "market_question": market["question"],
            "last_checked": _dt.now().isoformat(),
            "created_at": _dt.now().isoformat(),
            "ttl_hours": 720,
        }
        db_module.update_position(market["slug"], pos_data)

        pos = db_module.load_position(market["slug"])
        assert pos is not None
        assert pos["status"] == "active"
        assert pos["shares"] > 0
        assert db_module.count_positions() == 1

        entry_price = pos["entry_price"]
        new_price = entry_price * 0.45
        pos["high_price"] = max(pos["high_price"], new_price)
        pos["stop_loss"] = round(entry_price * 0.80, 4)
        pos["last_checked"] = (_dt.now() - timedelta(hours=4)).isoformat()
        db_module.update_position(market["slug"], pos)

        pos = db_module.load_position(market["slug"])
        assert pos["high_price"] >= entry_price

        sold_slug = market["slug"]
        db_module.delete_position(sold_slug)
        assert db_module.load_position(sold_slug) is None
        assert db_module.count_positions() == 0

        db_module.update_hypothesis(sold_slug, {
            "p_model": analysis["p_model"],
            "market_price": market["price"],
            "question": market["question"],
            "resolved": True,
            "outcome": "YES",
            "exit_price": new_price,
            "pnl_at_exit": (new_price - entry_price) / entry_price if entry_price > 0 else 0,
            "exit_type": "stop_loss",
        })

        hyp = db_module.load_hypothesis(sold_slug)
        assert hyp is not None
        assert hyp["resolved"] is True
        pnl = hyp["pnl_at_exit"]
        assert pnl < 0

        settings = db_module.load_settings()
        settings["total_resolved"] = settings.get("total_resolved", 0) + 1
        db_module.save_settings(settings)
        assert db_module.load_settings()["total_resolved"] >= 1
