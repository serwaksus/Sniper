#!/usr/bin/env python3
"""
Health Monitor — 10 checks covering the full trading pipeline.
Sends Telegram alert ONLY if issues found. Runs once per sniper cycle.
"""
import json
import os
import re
import subprocess
import logging
import shutil
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

HEALTH_STATE_FILE = "/root/dotm-sniper/health_state.json"
SNIPER_LOG = "/root/dotm-sniper/sniper.log"
PRICE_TRACKING_FILE = "/root/dotm-sniper/price_tracking.json"
POSITIONS_FILE = "/root/dotm-sniper/positions.json"
CALIBRATION_MODEL_FILE = "/root/dotm-sniper/calibration_model.json"
EQUITY_FILE = "/root/dotm-sniper/equity_curve.json"
HYPOTHESIS_DB_FILE = "/root/dotm-sniper/hypothesis_db.json"

ALERT_COOLDOWN_HOURS = 6
EQUITY_DRAWDOWN_PCT = 0.10
CYCLE_MAX_MINUTES = 15
ERROR_SPIKE_PER_HOUR = 5
LLM_MAX_PER_HOUR = 60
WINRATE_MIN = 0.15
WINRATE_MIN_SAMPLE = 15


def _load_state():
    try:
        with open(HEALTH_STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"last_alerts": {}, "last_cycle_start": None, "last_equity": None}


def _save_state(state):
    try:
        import tempfile
        with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(HEALTH_STATE_FILE), delete=False) as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
            os.replace(f.name, HEALTH_STATE_FILE)
    except Exception as e:
        logger.warning(f"[HEALTH] save state failed: {e}")


def _should_alert(state, alert_key):
    last = state.get("last_alerts", {}).get(alert_key, "")
    if not last:
        return True
    try:
        return datetime.now() - datetime.fromisoformat(last) > timedelta(hours=ALERT_COOLDOWN_HOURS)
    except Exception:
        return True


def _mark_alerted(state, alert_key):
    state.setdefault("last_alerts", {})[alert_key] = datetime.now().isoformat()


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


def _read_last_hour_log():
    return _read_recent_log(hours=1)


def _send_telegram(message):
    try:
        from tg_sender import send_telegram
        return send_telegram(message, max_retries=2, queue_on_fail=True)
    except Exception as e:
        logger.warning(f"[HEALTH] TG send failed: {e}")
        return False


# ── Check 1: No trades ──────────────────────────────────────────
def _check_no_trades(lines, state):
    trade_signals = sum(1 for l in lines if "=> BUY" in l)
    blocked = sum(1 for l in lines if "TRADE-BLOCKED" in l)
    executed = sum(1 for l in lines if "Bought:" in l and "Bought: 0" not in l)
    cycles = max(sum(1 for l in lines if "DOTM SNIPER" in l and "starting" in l.lower()), 1)

    issues = []
    if trade_signals > 0 and executed == 0:
        block_rate = blocked / max(trade_signals, 1)
        issues.append(f"Advisor blocks {blocked}/{trade_signals} ({block_rate:.0%}). 0 trades executed")
    elif trade_signals == 0 and executed == 0:
        p_models = re.findall(r"p_model=([0-9.]+)%", " ".join(lines[-500:]))
        if p_models:
            avg_pm = sum(float(p) for p in p_models[-20:]) / min(len(p_models), 20)
            issues.append(f"0 signals/24h. Avg LLM p_model={avg_pm:.1f}% (need >3%)")
        else:
            issues.append("0 signals/24h. No LLM analyses — check API key")

    if not issues:
        return None
    return "NO_TRADES", "🚫 <b>No trades 24h</b>\n" + "\n".join(f"• {i}" for i in issues)


# ── Check 2: Equity drawdown ────────────────────────────────────
def _check_equity_drawdown(state):
    try:
        data = json.load(open(EQUITY_FILE))
        snaps = data.get("snapshots", [])
        if len(snaps) < 2:
            return None
        now_eq = snaps[-1].get("total_equity", 0)
        cutoff = datetime.now() - timedelta(hours=24)
        past_eq = None
        for s in snaps:
            try:
                if datetime.fromisoformat(s["timestamp"]) >= cutoff:
                    past_eq = s.get("total_equity", 0)
                    break
            except Exception:
                continue
        if past_eq is None or past_eq <= 0:
            return None
        drop = (past_eq - now_eq) / past_eq
        if drop >= EQUITY_DRAWDOWN_PCT:
            return ("EQUITY_DROP",
                    f"📉 <b>Equity drawdown -{drop:.0%}</b>\n"
                    f"• 24h ago: ${past_eq:.2f} → now: ${now_eq:.2f}\n"
                    f"• Drop: ${past_eq - now_eq:.2f}")
        state["last_equity"] = now_eq
    except Exception:
        pass
    return None


# ── Check 3: Order health ───────────────────────────────────────
def _check_order_health(state):
    try:
        res = subprocess.run(
            ["pm-trader", "orders", "list"],
            capture_output=True, text=True, timeout=15, start_new_session=True
        )
        data = json.loads(res.stdout) if res.stdout else {}
        orders = data.get("data", []) if isinstance(data.get("data"), list) else []
    except Exception:
        return "ORDERS_API", "⚠️ <b>pm-trader orders list failed</b>\n• Cannot check order health"

    issues = []
    slug_prices = {}
    for o in orders:
        if o.get("status") != "pending":
            continue
        key = (o.get("market_slug", ""), o.get("limit_price", 0))
        slug_prices[key] = slug_prices.get(key, 0) + 1

    duplicates = {k: v for k, v in slug_prices.items() if v > 1}
    if duplicates:
        for (slug, price), cnt in duplicates.items():
            issues.append(f"Duplicate: {slug[:35]}... @${price:.2f} x{cnt}")

    try:
        with open(POSITIONS_FILE) as f:
            positions = json.load(f)
        pos_slugs = set(positions.keys())
    except Exception:
        pos_slugs = set()

    orphaned = [o for o in orders if o.get("status") == "pending"
                and o.get("market_slug", "") not in pos_slugs]
    if len(orphaned) > 2:
        issues.append(f"{len(orphaned)} sell orders without matching position")

    stale_cutoff = datetime.now() - timedelta(days=30)
    stale = [o for o in orders if o.get("status") == "pending"
             and o.get("created_at", "2099")[:10] < stale_cutoff.strftime("%Y-%m-%d")]
    if stale:
        issues.append(f"{len(stale)} orders older than 30 days")

    if not issues:
        return None
    return "ORDERS", "📋 <b>Order anomalies</b>\n" + "\n".join(f"• {i}" for i in issues)


# ── Check 4: API health ─────────────────────────────────────────
def _check_api_health(lines, state):
    hour_lines = _read_last_hour_log()
    issues = []

    llm_errors = sum(1 for l in hour_lines if "LLM error" in l or "LLM unavailable" in l)
    if llm_errors >= 3:
        issues.append(f"DeepSeek: {llm_errors} errors in last hour")

    gamma_fails = sum(1 for l in hour_lines if "[GAMMA]" in l and "status=" in l and "200" not in l)
    if gamma_fails >= 2:
        issues.append(f"Gamma API: {gamma_fails} failures in last hour")

    pm_fails = sum(1 for l in hour_lines if "pm-trader" in l.lower() and ("failed" in l.lower() or "error" in l.lower()))
    if pm_fails >= 2:
        issues.append(f"pm-trader: {pm_fails} errors in last hour")

    if not issues:
        return None
    return "API_HEALTH", "🔌 <b>API failures</b>\n" + "\n".join(f"• {i}" for i in issues)


# ── Check 5: Cycle timing ───────────────────────────────────────
def _check_cycle_timing(state):
    now_iso = datetime.now().isoformat()
    last_start = state.get("last_cycle_start")
    state["last_cycle_start"] = now_iso

    if not last_start:
        return None
    try:
        elapsed = (datetime.now() - datetime.fromisoformat(last_start)).total_seconds() / 60
        if elapsed > CYCLE_MAX_MINUTES:
            return ("CYCLE_SLOW",
                    f"⏱️ <b>Slow cycle: {elapsed:.0f} min</b>\n"
                    f"• Expected <{CYCLE_MAX_MINUTES}min. Bot may be hanging on LLM/Telegram timeout")
    except Exception:
        pass
    return None


# ── Check 6: Error spike ────────────────────────────────────────
def _check_error_spike(state):
    hour_lines = _read_last_hour_log()
    errors = sum(1 for l in hour_lines if " ERROR " in l or "Traceback" in l)
    if errors >= ERROR_SPIKE_PER_HOUR:
        samples = [l.strip()[:80] for l in hour_lines if " ERROR " in l][-3:]
        return ("ERROR_SPIKE",
                f"🔴 <b>Error spike: {errors}/hour</b> (threshold {ERROR_SPIKE_PER_HOUR})\n"
                + "\n".join(f"• {s}" for s in samples))
    return None


# ── Check 7: LLM cost guard ─────────────────────────────────────
def _check_llm_usage(state):
    hour_lines = _read_last_hour_log()
    llm_calls = sum(1 for l in hour_lines if "[ANALYSIS]" in l or "[BATCH]" in l or "[ADVISOR]" in l)
    signals = sum(1 for l in hour_lines if "=> BUY" in l)
    if llm_calls >= LLM_MAX_PER_HOUR and signals == 0:
        return ("LLM_WASTE",
                f"💸 <b>LLM waste: {llm_calls} calls/hour, 0 signals</b>\n"
                f"• Bot analyzes but never buys — check thresholds/prompt")
    return None


# ── Check 8: Disk space ─────────────────────────────────────────
def _check_disk_space(state):
    issues = []
    try:
        usage = shutil.disk_usage("/")
        pct = usage.used / usage.total
        if pct > 0.90:
            issues.append(f"/: {pct:.0%} full ({usage.free // 1024**3}GB free)")
    except Exception:
        pass
    try:
        log_size = os.path.getsize(SNIPER_LOG) / 1024**2
        if log_size > 200:
            issues.append(f"sniper.log: {log_size:.0f}MB — consider rotation")
    except Exception:
        pass
    try:
        tmp_files = sum(1 for _ in os.listdir("/tmp") if _.startswith("dotm_telegram"))
        if tmp_files > 5:
            issues.append(f"{tmp_files} telegram session files in /tmp")
    except Exception:
        pass

    if not issues:
        return None
    return "DISK", "💾 <b>Disk issues</b>\n" + "\n".join(f"• {i}" for i in issues)


# ── Check 9: Hypothesis DB growth ───────────────────────────────
def _check_hypothesis_db(state):
    try:
        db = json.load(open(HYPOTHESIS_DB_FILE))
        hypotheses = db.get("hypotheses", [])
        unresolved = [h for h in hypotheses if not h.get("resolved")]
        if len(unresolved) > 50:
            return ("HYP_DB",
                    f"📚 <b>Hypothesis DB: {len(unresolved)} unresolved</b>\n"
                    f"• Consider cleanup — stale entries slowing lookups")
    except Exception:
        pass
    return None


# ── Check 10: Winrate tracker ───────────────────────────────────
def _check_winrate(state):
    try:
        db = json.load(open(HYPOTHESIS_DB_FILE))
        resolved = [h for h in db.get("hypotheses", []) if h.get("resolved")]
        if len(resolved) < WINRATE_MIN_SAMPLE:
            return None
        recent = resolved[-WINRATE_MIN_SAMPLE:]
        wins = sum(1 for h in recent if h.get("outcome") == "YES")
        wr = wins / len(recent)
        if wr < WINRATE_MIN:
            return ("LOW_WINRATE",
                    f"📊 <b>Low winrate: {wins}/{len(recent)}={wr:.0%}</b> (min {WINRATE_MIN:.0%})\n"
                    f"• Strategy may need adjustment — check recent losses")
    except Exception:
        pass
    return None


# ── Calibration overfit ─────────────────────────────────────────
def _check_calibration_overfit(state):
    try:
        model = json.load(open(CALIBRATION_MODEL_FILE))
    except Exception:
        return None
    issues = []
    for cluster, data in model.items():
        y_thresh = data.get("y_thresholds_", [])
        x_thresh = data.get("X_thresholds_", [])
        if y_thresh and x_thresh:
            max_y = max(y_thresh)
            min_x_for_max = x_thresh[-1]
            if max_y >= 0.90 and min_x_for_max < 0.30:
                issues.append(f"'{cluster}': p>{min_x_for_max:.1%} → {max_y:.0%} ({len(y_thresh)} pts)")
    if not issues:
        return None
    return "CALIB_OVERFIT", "⚠️ <b>Calibration overfit</b>\n" + "\n".join(f"• {i}" for i in issues)


# ── Cache anomalies ─────────────────────────────────────────────
def _check_cache(state):
    try:
        tracking = json.load(open(PRICE_TRACKING_FILE))
    except Exception:
        return None
    high = sum(1 for v in tracking.values() if v.get("p_model", 0) >= 0.85)
    if high > 0:
        return "CACHE", f"📦 <b>Cache: {high} entries p_model >= 85%</b>\n• Likely stale — clear price_tracking.json"
    return None


# ── Main ────────────────────────────────────────────────────────
def run_health_check():
    state = _load_state()
    lines = _read_recent_log(hours=24)
    alerts = []

    checks = [
        lambda: _check_no_trades(lines, state),
        lambda: _check_equity_drawdown(state),
        lambda: _check_order_health(state),
        lambda: _check_api_health(lines, state),
        lambda: _check_cycle_timing(state),
        lambda: _check_error_spike(state),
        lambda: _check_llm_usage(state),
        lambda: _check_disk_space(state),
        lambda: _check_hypothesis_db(state),
        lambda: _check_winrate(state),
        lambda: _check_calibration_overfit(state),
        lambda: _check_cache(state),
    ]

    check_names = [
        "no_trades", "equity_drawdown", "order_health", "api_health",
        "cycle_timing", "error_spike", "llm_usage", "disk_space",
        "hypothesis_db", "winrate", "calib_overfit", "cache",
    ]

    for i, check in enumerate(checks):
        name = check_names[i] if i < len(check_names) else f"check_{i}"
        try:
            result = check()
        except Exception as e:
            logger.warning(f"[HEALTH-CHECK] {name}: CRASH - {e}")
            continue
        if result is None:
            logger.debug(f"[HEALTH-CHECK] {name}: OK")
            continue
        alert_key, message = result
        logger.info(f"[HEALTH-CHECK] {name}: ISSUE [{alert_key}]")
        if _should_alert(state, alert_key):
            alerts.append((alert_key, message))
            _mark_alerted(state, alert_key)

    _save_state(state)

    if not alerts:
        logger.info("[HEALTH] All 12 checks passed")
        return

    header = f"🔬 DOTM Health ({datetime.now().strftime('%m/%d %H:%M')}) — {len(alerts)} issues\n"
    body = "\n\n".join(msg for _, msg in alerts)
    _send_telegram(header + body)
    logger.info(f"[HEALTH] {len(alerts)} alerts sent")
    return alerts


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from utils import load_env_file
    load_env_file()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = run_health_check()
    print(f"{len(results) or 0} issues" if results else "All healthy")
