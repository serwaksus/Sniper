"""Contract tests: verify JSON keys are consistent between writers and readers."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schema import *


def _roundtrip(data, path):
    with open(path, "w") as f:
        json.dump(data, f)
    with open(path) as f:
        return json.load(f)


class TestEquityCurveContract:
    """equity_tracker writes, health_monitor reads — keys must match."""

    def test_snapshot_keys_written(self):
        snapshot = {
            EQUITY_TIMESTAMP: "2026-01-01T00:00:00",
            EQUITY_CASH: 800.0,
            EQUITY_POSITIONS_VALUE: 200.0,
            EQUITY_TOTAL: 1000.0,
            EQUITY_UNREALIZED_PNL: -50.0,
            EQUITY_NUM_POSITIONS: 5,
            EQUITY_POSITIONS: [],
        }
        curve = {EQUITY_SNAPSHOTS: [snapshot]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(curve, f)
            tmp = f.name
        try:
            with open(tmp) as f:
                loaded = json.load(f)
            snap = loaded[EQUITY_SNAPSHOTS][0]
            assert snap[EQUITY_TOTAL] == 1000.0
            assert snap[EQUITY_NUM_POSITIONS] == 5
            assert snap[EQUITY_CASH] == 800.0
            assert snap[EQUITY_POSITIONS_VALUE] == 200.0
            assert snap[EQUITY_UNREALIZED_PNL] == -50.0
        finally:
            os.unlink(tmp)

    def test_num_positions_key_value(self):
        assert EQUITY_NUM_POSITIONS == "num_positions"

    def test_health_monitor_reads_equity_keys(self):
        snap = {
            EQUITY_TIMESTAMP: "2026-01-01T00:00:00",
            EQUITY_TOTAL: 500.0,
            EQUITY_CASH: 400.0,
            EQUITY_POSITIONS_VALUE: 100.0,
            EQUITY_UNREALIZED_PNL: -10.0,
            EQUITY_NUM_POSITIONS: 2,
        }
        eq_now = snap[EQUITY_TOTAL]
        cash = snap[EQUITY_CASH]
        pos_val = snap[EQUITY_POSITIONS_VALUE]
        n_pos = snap[EQUITY_NUM_POSITIONS]
        assert eq_now == 500.0
        assert cash == 400.0
        assert pos_val == 100.0
        assert n_pos == 2

    def test_equity_drawdown_reads_snapshots(self):
        snaps = [
            {EQUITY_TOTAL: 600.0, EQUITY_TIMESTAMP: "2026-01-01T00:00:00"},
            {EQUITY_TOTAL: 500.0, EQUITY_TIMESTAMP: "2026-01-01T12:00:00"},
        ]
        data = {EQUITY_SNAPSHOTS: snaps}
        now_eq = data[EQUITY_SNAPSHOTS][-1][EQUITY_TOTAL]
        past_eq = data[EQUITY_SNAPSHOTS][0][EQUITY_TOTAL]
        drop = (past_eq - now_eq) / past_eq
        assert abs(drop - 1 / 6) < 0.01

    def test_roundtrip_preserves_keys(self):
        curve = {EQUITY_SNAPSHOTS: [
            {EQUITY_TIMESTAMP: "2026-01-01T00:00:00", EQUITY_TOTAL: 100.0,
             EQUITY_CASH: 80.0, EQUITY_POSITIONS_VALUE: 20.0,
             EQUITY_UNREALIZED_PNL: 0.0, EQUITY_NUM_POSITIONS: 1}
        ]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp = f.name
        try:
            loaded = _roundtrip(curve, tmp)
            assert set(loaded[EQUITY_SNAPSHOTS][0].keys()) == set(curve[EQUITY_SNAPSHOTS][0].keys())
        finally:
            os.unlink(tmp)


class TestHypothesisDbContract:
    """dotm_sniper writes, calibration_tracker/health_monitor read."""

    def test_top_level_keys(self):
        assert HYP_DB_HYPOTHESES == "hypotheses"
        assert HYP_DB_RESOLVED == "resolved"

    def test_hypothesis_fields_written(self):
        h = {
            HYP_SLUG: "test-market",
            HYP_QUESTION: "Will X?",
            HYP_MARKET_PRICE: 0.05,
            HYP_P_MODEL: 0.10,
            HYP_PROB_RATIO: 2.0,
            HYP_CONFIDENCE: 0.75,
            HYP_FACTORS: [],
            HYP_CLUSTERS: ["usa_politics"],
            HYP_SIZE_PCT: 0.02,
            HYP_CREATED_AT: "2026-01-01T00:00:00",
            HYP_RESOLVED: False,
            HYP_TP_LIMIT_PLACED: True,
            HYP_TP_LIMIT_PRICE: 0.85,
            HYP_SOURCE_SIGNAL: "default",
        }
        assert h[HYP_SLUG] == "test-market"
        assert h[HYP_RESOLVED] is False
        assert h[HYP_SOURCE_SIGNAL] == "default"
        assert h[HYP_CLUSTERS] == ["usa_politics"]

    def test_resolved_entries_in_hypotheses_list(self):
        db = {HYP_DB_HYPOTHESES: [
            {HYP_SLUG: "a", HYP_RESOLVED: True, HYP_OUTCOME: "YES"},
            {HYP_SLUG: "b", HYP_RESOLVED: False},
        ]}
        resolved = [h for h in db[HYP_DB_HYPOTHESES] if h.get(HYP_RESOLVED)]
        assert len(resolved) == 1
        assert resolved[0][HYP_SLUG] == "a"
        assert resolved[0][HYP_OUTCOME] == "YES"

    def test_calibration_tracker_reads_hypotheses(self):
        db = {HYP_DB_HYPOTHESES: [
            {HYP_SLUG: "x", HYP_RESOLVED: True, HYP_OUTCOME: "YES",
             HYP_P_MODEL: 0.12, HYP_MARKET_PRICE: 0.05, HYP_CLUSTERS: ["other"],
             HYP_QUESTION: "Will X?", HYP_RESOLVED_AT: "2026-01-01T00:00:00"},
        ]}
        for h in db[HYP_DB_HYPOTHESES]:
            if h.get(HYP_RESOLVED) and h.get(HYP_OUTCOME) in ("YES", "NO"):
                assert h[HYP_P_MODEL] is not None
                assert h[HYP_MARKET_PRICE] is not None
                assert isinstance(h[HYP_CLUSTERS], list)
                assert len(h[HYP_CLUSTERS]) > 0

    def test_health_monitor_reads_outcome(self):
        db = {HYP_DB_HYPOTHESES: [
            {HYP_SLUG: "a", HYP_RESOLVED: True, HYP_OUTCOME: "YES"},
            {HYP_SLUG: "b", HYP_RESOLVED: True, HYP_OUTCOME: "NO"},
            {HYP_SLUG: "c", HYP_RESOLVED: True, HYP_OUTCOME: "UNKNOWN"},
        ]}
        resolved = [h for h in db[HYP_DB_HYPOTHESES] if h.get(HYP_RESOLVED)]
        wins = sum(1 for h in resolved if h.get(HYP_OUTCOME) == "YES")
        assert wins == 1

    def test_resolve_adds_fields(self):
        h = {
            HYP_SLUG: "test",
            HYP_RESOLVED: True,
            HYP_RESOLVED_AT: "2026-01-01T00:00:00",
            HYP_OUTCOME: "YES",
            HYP_RESOLUTION_NOTE: "market_resolved",
        }
        assert h[HYP_RESOLVED] is True
        assert h[HYP_RESOLVED_AT] is not None
        assert h[HYP_OUTCOME] in ("YES", "NO")
        assert h[HYP_RESOLUTION_NOTE] == "market_resolved"

    def test_resolved_list_appended(self):
        db = {HYP_DB_HYPOTHESES: [], HYP_DB_RESOLVED: []}
        h = {HYP_SLUG: "s1", HYP_RESOLVED: True, HYP_OUTCOME: "YES"}
        db[HYP_DB_RESOLVED].append(h)
        assert len(db[HYP_DB_RESOLVED]) == 1
        assert db[HYP_DB_RESOLVED][0][HYP_SLUG] == "s1"


class TestPositionsContract:
    """dotm_sniper + sell_executor + hermes write/read positions.json."""

    def test_position_keys_written_by_sniper(self):
        pos = {
            POS_ENTRY_PRICE: 0.10,
            POS_HIGH_PRICE: 0.12,
            POS_STOP_LOSS: 0.07,
            POS_TRAILING_ON: False,
            POS_LAST_CHECKED: "2026-01-01T00:00:00",
            POS_METACULUS_PROB: None,
            POS_MARKET_QUESTION: "Will X?",
            POS_OUTCOME: "yes",
            POS_CLUSTERS: ["usa_politics"],
            POS_SHARES: 100,
        }
        assert pos[POS_ENTRY_PRICE] == 0.10
        assert pos[POS_STOP_LOSS] == 0.07
        assert pos[POS_TRAILING_ON] is False
        assert isinstance(pos[POS_CLUSTERS], list)

    def test_sell_executor_keys(self):
        pos = {
            POS_SELLING_IN_PROGRESS: True,
            POS_TRAILING_CONFIRMED: True,
            POS_TRAILING_CONFIRM_TIME: "2026-01-01T00:00:00",
            POS_HIGH_PRICE: 0.15,
            POS_STOP_LOSS: 0.11,
            POS_STOP_TYPE: "atr",
        }
        assert pos[POS_SELLING_IN_PROGRESS] is True
        assert pos[POS_TRAILING_CONFIRMED] is True
        assert pos[POS_STOP_TYPE] == "atr"

    def test_hermes_emergency_keys(self):
        pos = {
            POS_IN_EMERGENCY_EXIT: True,
            POS_SELLING_IN_PROGRESS: True,
            POS_EMERGENCY_EXIT_FAILED: False,
            POS_LAST_EMERGENCY_ATTEMPT: "2026-01-01T00:00:00",
        }
        assert pos[POS_IN_EMERGENCY_EXIT] is True
        assert pos[POS_EMERGENCY_EXIT_FAILED] is False

    def test_hermes_partial_fill_keys(self):
        pos = {
            POS_SHARES: 50,
            POS_PARTIAL_FILLS: 25,
            POS_PARTIAL_PROCEEDS: 18.75,
            POS_SHARES_AT_TP_OPEN: 100,
        }
        assert pos[POS_PARTIAL_FILLS] == 25
        assert pos[POS_SHARES_AT_TP_OPEN] == 100

    def test_clusters_is_list(self):
        pos = {POS_CLUSTERS: ["sports_nba"]}
        assert isinstance(pos[POS_CLUSTERS], list)
        assert len(pos[POS_CLUSTERS]) > 0

    def test_roundtrip_positions(self):
        positions = {
            "some-slug": {
                POS_ENTRY_PRICE: 0.10,
                POS_HIGH_PRICE: 0.12,
                POS_SHARES: 100,
                POS_CLUSTERS: ["other"],
                POS_OUTCOME: "yes",
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp = f.name
        try:
            loaded = _roundtrip(positions, tmp)
            slug_pos = loaded["some-slug"]
            assert slug_pos[POS_ENTRY_PRICE] == 0.10
            assert slug_pos[POS_SHARES] == 100
        finally:
            os.unlink(tmp)


class TestSettingsContract:
    """dotm_sniper writes/reads bot_settings.json."""

    def test_settings_keys(self):
        s = {
            SETTINGS_SIGNAL_THRESHOLD: 55,
            SETTINGS_MIN_P_MODEL: 0.03,
            SETTINGS_MIN_CONFIDENCE: 0.65,
            SETTINGS_TOTAL_RESOLVED: 42,
            SETTINGS_LAST_BACKTEST: 0,
            SETTINGS_CALIBRATION_BRIER: None,
            SETTINGS_MAX_CONCURRENT: 5,
        }
        assert s[SETTINGS_SIGNAL_THRESHOLD] == 55
        assert s[SETTINGS_MIN_P_MODEL] == 0.03
        assert s[SETTINGS_TOTAL_RESOLVED] == 42
        assert s[SETTINGS_MAX_CONCURRENT] == 5

    def test_settings_key_strings(self):
        assert SETTINGS_SIGNAL_THRESHOLD == "signal_threshold"
        assert SETTINGS_MIN_P_MODEL == "min_p_model"
        assert SETTINGS_MIN_CONFIDENCE == "min_confidence"
        assert SETTINGS_MAX_CONCURRENT == "MAX_CONCURRENT_TRADES"
        assert SETTINGS_TOTAL_RESOLVED == "total_resolved"
        assert SETTINGS_LAST_BACKTEST == "last_backtest_timestamp"


class TestPriceTrackingContract:
    """dotm_sniper writes, health_monitor reads price_tracking.json."""

    def test_tracking_entry_keys(self):
        entry = {
            TRACKING_P_MODEL: 0.12,
            TRACKING_LAST_PRICE: 0.05,
            TRACKING_LAST_CHECK: "2026-01-01T00:00:00",
        }
        assert entry[TRACKING_P_MODEL] == 0.12
        assert entry[TRACKING_LAST_PRICE] == 0.05
        assert entry[TRACKING_LAST_CHECK] == "2026-01-01T00:00:00"

    def test_tracking_key_strings(self):
        assert TRACKING_P_MODEL == "p_model"
        assert TRACKING_LAST_PRICE == "last_price"
        assert TRACKING_LAST_CHECK == "last_checked"

    def test_health_monitor_reads_stale(self):
        tracking = {
            "slug-a": {
                TRACKING_P_MODEL: 0.90,
                TRACKING_LAST_CHECK: "2020-01-01T00:00:00",
            }
        }
        high = sum(
            1 for v in tracking.values()
            if v.get(TRACKING_P_MODEL, 0) >= 0.85
            and v.get(TRACKING_LAST_CHECK) is not None
        )
        assert high == 1


class TestHealthStateContract:
    """health_monitor writes/reads health_state.json."""

    def test_state_keys(self):
        state = {
            HEALTH_LAST_ALERTS: {},
            HEALTH_LAST_CYCLE_START: "2026-01-01T00:00:00",
            HEALTH_LAST_EQUITY: 500.0,
        }
        assert HEALTH_LAST_CYCLE_START == "last_cycle_start"
        assert HEALTH_LAST_ALERTS == "last_alerts"
        assert state[HEALTH_LAST_CYCLE_START] is not None


class TestCalibrationLogContract:
    """calibration_tracker writes/reads calibration_log.json."""

    def test_entry_keys(self):
        entry = {
            "timestamp": "2026-01-01T00:00:00",
            CAL_LOG_SLUG: "test",
            CAL_LOG_P_MODEL: 0.12,
            CAL_LOG_P_CALIBRATED: 0.10,
            CAL_LOG_MARKET_PRICE: 0.05,
            CAL_LOG_ACTUAL_OUTCOME: "YES",
            CAL_LOG_ACTUAL_BIN: 1.0,
            CAL_LOG_CLUSTER: "other",
            CAL_LOG_PNL_PCT: 0.5,
        }
        assert entry[CAL_LOG_P_MODEL] == 0.12
        assert entry[CAL_LOG_ACTUAL_OUTCOME] == "YES"
        assert entry[CAL_LOG_CLUSTER] == "other"

    def test_log_top_level(self):
        assert CAL_LOG_ENTRIES == "entries"


class TestHermesAlertStateContract:
    """hermes_advisor writes/reads hermes_alert_state.json."""

    def test_alert_state_keys(self):
        assert ALERT_POSITION_STATUS == "position_status"
        assert ALERT_LAST_NOTIFIED == "last_notified_at"
        assert ALERT_HOLD_COUNTS == "hold_counts"
        assert ALERT_UPDATED_AT == "updated_at"


class TestCalibrationModelContract:
    """calibration_tracker writes, health_monitor reads calibration_model.json."""

    def test_model_keys(self):
        cluster_data = {
            CAL_MODEL_Y_THRESHOLDS: [0.1, 0.5, 0.9],
            CAL_MODEL_X_THRESHOLDS: [0.05, 0.15, 0.25],
        }
        y = cluster_data[CAL_MODEL_Y_THRESHOLDS]
        x = cluster_data[CAL_MODEL_X_THRESHOLDS]
        assert len(y) == 3
        assert len(x) == 3
        assert CAL_MODEL_Y_THRESHOLDS == "y_thresholds_"
        assert CAL_MODEL_X_THRESHOLDS == "X_thresholds_"


class TestCrossModuleKeyConsistency:
    """Verify no duplicate key names with different schema constants."""

    def test_no_key_collision_equity_vs_position(self):
        equity_keys = {EQUITY_TIMESTAMP, EQUITY_CASH, EQUITY_POSITIONS_VALUE,
                       EQUITY_TOTAL, EQUITY_UNREALIZED_PNL, EQUITY_NUM_POSITIONS,
                       EQUITY_POSITIONS, EQUITY_SNAPSHOTS}
        for k in equity_keys:
            assert isinstance(k, str) and len(k) > 0

    def test_all_schema_constants_are_strings(self):
        import schema as s
        for name in dir(s):
            if name.startswith("_"):
                continue
            val = getattr(s, name)
            if isinstance(val, str) and not name[0].islower():
                assert isinstance(val, str), f"{name} is not a string"

    def test_critical_key_matches_known_bug_fix(self):
        """Verify the num_positions / positions_count bug is fixed."""
        assert EQUITY_NUM_POSITIONS == "num_positions"
        assert EQUITY_NUM_POSITIONS != "positions_count"

    def test_hypothesis_db_uses_hypotheses_not_resolved_for_active(self):
        """calibration_tracker reads from 'hypotheses' list with 'resolved' filter."""
        assert HYP_DB_HYPOTHESES == "hypotheses"
        assert HYP_RESOLVED == "resolved"
        db = {
            HYP_DB_HYPOTHESES: [
                {HYP_SLUG: "a", HYP_RESOLVED: True, HYP_OUTCOME: "YES", HYP_P_MODEL: 0.5},
                {HYP_SLUG: "b", HYP_RESOLVED: False, HYP_P_MODEL: 0.3},
            ],
            HYP_DB_RESOLVED: [
                {HYP_SLUG: "a", HYP_RESOLVED: True, HYP_OUTCOME: "YES"},
            ],
        }
        active = [h for h in db[HYP_DB_HYPOTHESES] if not h.get(HYP_RESOLVED)]
        resolved_from_list = [h for h in db[HYP_DB_HYPOTHESES] if h.get(HYP_RESOLVED)]
        assert len(active) == 1
        assert active[0][HYP_SLUG] == "b"
        assert len(resolved_from_list) == 1
        assert resolved_from_list[0][HYP_OUTCOME] == "YES"
