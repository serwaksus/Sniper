#!/usr/bin/env python3
"""
Health Monitor — scans logs and state files for pipeline problems.
Sends Telegram alert ONLY if issues found. Runs once per cycle.
"""
import json
import os
import re
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

HEALTH_STATE_FILE = "/root/dotm-sniper/health_state.json"
SNIPER_LOG = "/root/dotm-sniper/sniper.log"
PRICE_TRACKING_FILE = "/root/dotm-sniper/price_tracking.json"
POSITIONS_FILE = "/root/dotm-sniper/positions.json"
CALIBRATION_MODEL_FILE = "/root/dotm-sniper/calibration_model.json"

ALERT_COOLDOWN_HOURS = 6


def _load_state():
    try:
        with open(HEALTH_STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"last_alerts": {}}


def _save_state(state):
    try:
        import tempfile
        with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(HEALTH_STATE_FILE), delete=False) as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
            os.replace(f.name, HEALTH_STATE_FILE)
    except Exception as e:
        logger.warning(f"[HEALTH] Failed to save state: {e}")


def _should_alert(state, alert_key):
    last = state.get("last_alerts", {}).get(alert_key, "")
    if not last:
        return True
    try:
        last_dt = datetime.fromisoformat(last)
        return datetime.now() - last_dt > timedelta(hours=ALERT_COOLDOWN_HOURS)
    except Exception:
        return True


def _mark_alerted(state, alert_key):
    if "last_alerts" not in state:
        state["last_alerts"] = {}
    state["last_alerts"][alert_key] = datetime.now().isoformat()


def _read_recent_log(hours=24):
    try:
        cutoff = datetime.now() - timedelta(hours=hours)
        lines = []
        with open(SNIPER_LOG) as f:
            for line in f:
                try:
                    ts = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
                    if ts >= cutoff:
                        lines.append(line)
                except (ValueError, IndexError):
                    if lines:
                        lines.append(line)
        return lines
    except Exception:
        return []


def _send_telegram(message):
    token = os.environ.get("TG_BOT_TOKEN", "")
    chat_id = os.environ.get("TG_CHAT_ID", "")
    if not token or not chat_id:
        logger.warning("[HEALTH] No TG credentials, skipping alert")
        return False
    try:
        import requests
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message[:4096], "parse_mode": "HTML"},
            timeout=15,
        )
        return resp.ok
    except Exception as e:
        logger.warning(f"[HEALTH] Telegram send failed: {e}")
        return False


def _check_no_trades(lines, state):
    buys = sum(1 for l in lines if "Bought: 0" in l)
    trade_signals = sum(1 for l in lines if "=> BUY" in l)
    blocked = sum(1 for l in lines if "TRADE-BLOCKED" in l)
    executed = sum(1 for l in lines if "execute_trade" in l and "True" in l)
    cycles = max(sum(1 for l in lines if "DOTM SNIPER" in l and "starting" in l.lower()), 1)

    issues = []
    if trade_signals > 0 and executed == 0 and buys == cycles:
        block_rate = blocked / max(trade_signals, 1)
        issues.append(
            f"Advisor blocks {blocked}/{trade_signals} signals ({block_rate:.0%}). "
            f"Last 24h: 0 trades executed"
        )
    elif trade_signals == 0 and executed == 0:
        p_models = re.findall(r"p_model=([0-9.]+)%", " ".join(lines[-500:]))
        if p_models:
            avg_pm = sum(float(p) for p in p_models[-20:]) / min(len(p_models), 20)
            issues.append(
                f"0 signals in 24h. Avg LLM p_model={avg_pm:.1f}% "
                f"(need >3%). LLM too conservative or markets stale"
            )
        else:
            issues.append("0 signals in 24h. No LLM analyses seen — check API key / connectivity")

    if not issues:
        return None
    return "SIG_NO_TRADES", "🚫 <b>No trades 24h</b>\n" + "\n".join(f"• {i}" for i in issues)


def _check_calibration_overfit(state):
    try:
        with open(CALIBRATION_MODEL_FILE) as f:
            model = json.load(f)
    except Exception:
        return None

    issues = []
    for cluster, data in model.items():
        y_thresh = data.get("y_thresholds_", [])
        max_y = max(y_thresh) if y_thresh else 0
        x_thresh = data.get("X_thresholds_", [])
        min_x_for_max = x_thresh[-1] if x_thresh else 0
        if max_y >= 0.90 and min_x_for_max < 0.30:
            issues.append(
                f"Cluster '{cluster}': maps p>{min_x_for_max:.1%} → {max_y:.0%}. "
                f"Training data insufficient ({len(y_thresh)} points)"
            )

    if not issues:
        return None
    return "CALIB_OVERFIT", "⚠️ <b>Calibration overfit</b>\n" + "\n".join(f"• {i}" for i in issues)


def _check_price_tracking_anomalies(state):
    try:
        with open(PRICE_TRACKING_FILE) as f:
            tracking = json.load(f)
    except Exception:
        return None

    high_count = sum(1 for v in tracking.values() if v.get("p_model", 0) >= 0.85)
    stale_older = 0
    cutoff = datetime.now() - timedelta(hours=48)
    for v in tracking.values():
        ts = v.get("last_seen", v.get("timestamp", ""))
        if ts:
            try:
                if datetime.fromisoformat(ts) < cutoff:
                    stale_older += 1
            except Exception:
                pass

    issues = []
    if high_count > 0:
        issues.append(f"{high_count} cache entries with p_model >= 85% (likely stale)")
    if stale_older > 10:
        issues.append(f"{stale_older} stale cache entries older than 48h")

    if not issues:
        return None
    return "CACHE_STALE", "📦 <b>Cache anomalies</b>\n" + "\n".join(f"• {i}" for i in issues)


def _check_positions_age(state):
    try:
        with open(POSITIONS_FILE) as f:
            positions = json.load(f)
    except Exception:
        return None

    issues = []
    now = datetime.now()
    for slug, pos in positions.items():
        checked = pos.get("last_checked", "")
        if checked:
            try:
                age_days = (now - datetime.fromisoformat(checked)).days
                if age_days > 7:
                    issues.append(f"{slug[:40]}... not checked for {age_days}d")
            except Exception:
                pass

    if not issues:
        return None
    return "POS_STALE", "🕐 <b>Stale positions</b>\n" + "\n".join(f"• {i}" for i in issues)


def run_health_check():
    state = _load_state()
    lines = _read_recent_log(hours=24)
    alerts = []

    checks = [
        lambda: _check_no_trades(lines, state),
        lambda: _check_calibration_overfit(state),
        lambda: _check_price_tracking_anomalies(state),
        lambda: _check_positions_age(state),
    ]

    for check in checks:
        result = check()
        if result is None:
            continue
        alert_key, message = result
        if _should_alert(state, alert_key):
            alerts.append((alert_key, message))
            _mark_alerted(state, alert_key)

    _save_state(state)

    if not alerts:
        logger.info("[HEALTH] All checks passed, no alerts")
        return

    header = f"🔬 <b>DOTM Sniper Health Alert</b> ({datetime.now().strftime('%H:%M')})\n"
    body = "\n\n".join(msg for _, msg in alerts)
    full_msg = header + "\n" + body

    ok = _send_telegram(full_msg)
    if ok:
        logger.info(f"[HEALTH] Sent {len(alerts)} alerts to Telegram")
    else:
        logger.warning(f"[HEALTH] Failed to send {len(alerts)} alerts")

    return alerts


if __name__ == "__main__":
    from utils import load_env_file
    load_env_file()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = run_health_check()
    if results:
        print(f"Found {len(results)} issues, alerts sent")
    else:
        print("All healthy")
