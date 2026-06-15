#!/usr/bin/env python3
from __future__ import annotations
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.path.join(PROJECT_ROOT, "sniper.db")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
BACKUP_DIR = os.path.join(PROJECT_ROOT, "backups")
PID_FILE = os.path.join(PROJECT_ROOT, "sniper.pid")
HERMES_PID_FILE = os.path.join(PROJECT_ROOT, "hermes.pid")
HERMES_MEMORY_FILE = os.path.join(PROJECT_ROOT, "hermes_memory.json")
HERMES_SKILLS_FILE = os.path.join(PROJECT_ROOT, "hermes_skills.json")
BAYESIAN_STATE_FILE = os.path.join(PROJECT_ROOT, "bayesian_state.json")
EQUITY_CURVE_FILE = os.path.join(PROJECT_ROOT, "equity_curve.json")
EQUITY_HISTORY_FILE = os.path.join(PROJECT_ROOT, "equity_history.json")
HEALTH_STATE_FILE = os.path.join(PROJECT_ROOT, "health_state.json")
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")
TG_QUEUE_FILE = os.path.join(PROJECT_ROOT, "tg_queue.json")
CACHE_FILE = os.path.join(PROJECT_ROOT, "source_cache.json")
STATUS_FILE = os.path.join(PROJECT_ROOT, "status.json")
MIGRATED_DIR = os.path.join(PROJECT_ROOT, "migrated")

POSITIONS_FILE = os.path.join(PROJECT_ROOT, "positions.json")
HYPOTHESIS_DB_FILE = os.path.join(PROJECT_ROOT, "hypothesis_db.json")
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "bot_settings.json")
PRICE_TRACKING_FILE = os.path.join(PROJECT_ROOT, "price_tracking.json")
BACKTEST_STATS_FILE = os.path.join(PROJECT_ROOT, "backtest_stats.json")
DAILY_STATS_FILE = os.path.join(PROJECT_ROOT, "daily_stats.json")
CURRENT_STATUS_FILE = os.path.join(PROJECT_ROOT, "current_status.json")
TRADES_JOURNAL_FILE = os.path.join(PROJECT_ROOT, "trades_journal.json")
TRADES_HISTORY_FILE = os.path.join(PROJECT_ROOT, "trades_history.json")
PRICE_HISTORY_FILE = os.path.join(PROJECT_ROOT, "price_history.json")
CALIBRATION_MODEL_FILE = os.path.join(PROJECT_ROOT, "calibration_model.json")
CALIBRATION_LOG_FILE = os.path.join(PROJECT_ROOT, "calibration_log.json")
PLATT_MODEL_FILE = os.path.join(PROJECT_ROOT, "platt_model.json")
CALIBRATOR_MODEL_FILE = os.path.join(PROJECT_ROOT, "calibrator_model.json")
BACKTEST_BASELINE_FILE = os.path.join(PROJECT_ROOT, "backtest_stats_v533_baseline.json")
CORRELATION_FILE = os.path.join(PROJECT_ROOT, "correlation_matrix.json")
BUZZ_CACHE_FILE = os.path.join(PROJECT_ROOT, "buzz_cache.json")
OPTIMIZER_OUTPUT_FILE = os.path.join(PROJECT_ROOT, "optimizer_results.json")

SNIPER_LOG = os.path.join(LOG_DIR, "sniper.log")
HERMES_LOG = os.path.join(LOG_DIR, "hermes.log")
REPORT_LOG = os.path.join(LOG_DIR, "report.log")
EQUITY_TRACKER_LOG = os.path.join(LOG_DIR, "equity_tracker.log")
SLIPPAGE_LOG_FILE = os.path.join(LOG_DIR, "slippage.json")
EMERGENCY_LOG_FILE = os.path.join(LOG_DIR, "emergency_log.json")
ALERT_STATE_FILE = os.path.join(LOG_DIR, "hermes_alert_state.json")
SNIPER_SCREEN_LOG = os.path.join(LOG_DIR, "sniper_screen.log")
HERMES_SCREEN_LOG = os.path.join(LOG_DIR, "hermes_screen.log")
ADVISOR_CRON_LOG = os.path.join(LOG_DIR, "advisor_cron.log")
HEALTH_HOURLY_LOG = os.path.join(LOG_DIR, "health_hourly.log")

MIN_P_MODEL = 0.10
MIN_CONFIDENCE = 0.65
BURN_IN_TRADES = 50
BAYESIAN_PRIOR_STRENGTH = 10
MIN_TRADES_FOR_WEIGHT = 20
MAX_P_MODEL_RATIO = 3.0
SIGNAL_THRESHOLD_DEFAULT = 50
MAX_CONCURRENT_TRADES = 15
PORTFOLIO_DRAWDOWN_STOP = 0.10
PER_POSITION_MAX_LOSS = 0.50
TIME_DECAY_EXIT_THRESHOLD = 0.60
CONVERGENCE_TAKE_PROFIT = 0.60

_TOKEN_RE = re.compile(r'bot\d+:[A-Za-z0-9_-]{20,}')


def sanitize(text: str) -> str:
    return _TOKEN_RE.sub('bot***:***REDACTED***', text)
