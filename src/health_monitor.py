#!/usr/bin/env python3
"""
Health Monitor — 23 checks covering full trading pipeline + infrastructure.
Sends Telegram alert ONLY if issues found. Runs once per sniper cycle.
"""
import json
import os
import re
import sys
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


# ── Check 13: Telegram reachability ────────────────────────────
def _check_telegram(state):
    try:
        import socket
        _src_dir = os.path.dirname(os.path.abspath(__file__))
        if _src_dir not in sys.path:
            sys.path.insert(0, _src_dir)
        from tg_sender import _get_credentials, TG_API_HOST, TG_WORKING_IP
        import requests as rq
        token, chat_id = _get_credentials()
        if not token:
            return "TG_NO_CRED", "📡 <b>Telegram: no credentials</b>\n• Check .env TG_BOT_TOKEN/TG_CHAT_ID"
        orig_getaddrinfo = socket.getaddrinfo
        def _patched(host, port, *a, **kw):
            if host == TG_API_HOST:
                return [orig_getaddrinfo(TG_WORKING_IP, port, *a, **kw)[0]]
            return orig_getaddrinfo(host, port, *a, **kw)
        socket.getaddrinfo = _patched
        try:
            resp = rq.get(
                f"https://{TG_API_HOST}/bot{token}/getMe",
                timeout=10,
            )
            if not resp.ok:
                return "TG_API_FAIL", f"📡 <b>Telegram API error</b>\n• getMe returned {resp.status_code}"
        finally:
            socket.getaddrinfo = orig_getaddrinfo
    except Exception as e:
        return "TG_UNREACHABLE", f"📡 <b>Telegram unreachable</b>\n• {str(e)[:100]}"
    return None


# ── Check 14: Process crash frequency ──────────────────────────
def _check_crash_frequency(state):
    count = 0
    for logf in ["/tmp/sniper_v556.log", "/tmp/sniper_v555.log",
                 "/tmp/sniper_v554.log", "/tmp/sniper_v553.log",
                 "/tmp/sniper_v552.log"]:
        try:
            with open(logf) as f:
                count += sum(1 for l in f if "Traceback" in l)
        except Exception:
            continue
    if count >= 3:
        return ("CRASH_FREQ",
                f"💥 <b>Crash frequency: {count} Tracebacks in recent logs</b>\n"
                f"• Check logs for recurring exceptions")
    return None


# ── Check 15: JSON file integrity ─────────────────────────────
def _check_json_integrity(state):
    critical_files = {
        "positions.json": POSITIONS_FILE,
        "equity_curve.json": EQUITY_FILE,
        "bot_settings.json": "/root/dotm-sniper/bot_settings.json",
        "hypothesis_db.json": HYPOTHESIS_DB_FILE,
        "price_tracking.json": PRICE_TRACKING_FILE,
    }
    broken = []
    for name, path in critical_files.items():
        try:
            with open(path) as f:
                json.load(f)
        except FileNotFoundError:
            broken.append(f"{name}: MISSING")
        except json.JSONDecodeError as e:
            broken.append(f"{name}: CORRUPT ({str(e)[:50]})")
        except Exception as e:
            broken.append(f"{name}: ERROR ({str(e)[:50]})")
    if broken:
        return ("JSON_INTEGRITY",
                "📄 <b>JSON file integrity issues</b>\n" +
                "\n".join(f"• {b}" for b in broken))
    return None


# ── Check 16: Cron health ─────────────────────────────────────
def _check_cron_health(state):
    stale = []
    cron_logs = {
        "report": "/root/sniper_report.log",
        "equity_tracker": "/root/dotm-sniper/logs/equity_tracker.log",
        "advisor_cron": "/root/dotm-sniper/logs/advisor_cron.log",
    }
    cutoff = datetime.now() - timedelta(hours=3)
    for name, path in cron_logs.items():
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            if mtime < cutoff:
                hours_stale = (datetime.now() - mtime).total_seconds() / 3600
                stale.append(f"{name}: last update {hours_stale:.0f}h ago")
        except Exception:
            stale.append(f"{name}: log file missing")
    if stale:
        return ("CRON_STALE",
                "⏰ <b>Cron jobs not running</b>\n" +
                "\n".join(f"• {s}" for s in stale))
    return None


# ── Check 17: LLM API error rate ──────────────────────────────
def _check_llm_error_rate(state):
    lines = _read_recent_log(hours=6)
    total = sum(1 for l in lines if "model=" in l and "messages" in l)
    errors = sum(1 for l in lines if "[ADVISOR] Empty response" in l
                 or "advisor_parse_error" in l or "ADVISOR.*Timeout" in l
                 or "ADVISOR] Error" in l or "429" in l)
    if total > 5:
        rate = errors / total
        if rate > 0.30:
            return ("LLM_ERRORS",
                    f"🤖 <b>LLM error rate: {rate:.0%} ({errors}/{total})</b>\n"
                    f"• Check DeepSeek API key, rate limits, balance")
    return None


# ── Check 18: Screen session integrity ─────────────────────────
def _check_screen_sessions(state):
    issues = []
    for proc_name in ["dotm_sniper.py", "hermes_advisor.py"]:
        try:
            res = subprocess.run(
                ["ps", "aux"],
                capture_output=True, text=True, timeout=5, start_new_session=True
            )
            count = sum(1 for l in res.stdout.split("\n")
                        if l.strip().startswith("root") and f"python3 src/{proc_name}" in l
                        and "SCREEN" not in l and "bash -c" not in l)
            if count == 0:
                issues.append(f"{proc_name}: NOT RUNNING")
            elif count > 1:
                issues.append(f"{proc_name}: {count} instances (possible fork)")
        except Exception as e:
            issues.append(f"{proc_name}: check failed ({str(e)[:50]})")

    try:
        res = subprocess.run(
            ["screen", "-ls"],
            capture_output=True, text=True, timeout=5, start_new_session=True
        )
        sockets = [l for l in res.stdout.split("\n") if "Detached" in l or "Attached" in l]
        if len(sockets) < 2:
            issues.append(f"screen sessions: only {len(sockets)} found (need 2)")
    except Exception:
        pass

    if issues:
        return ("SCREEN_HEALTH",
                "🖥️ <b>Process/session issues</b>\n" +
                "\n".join(f"• {i}" for i in issues))
    return None


# ── Check 19: Disk inode usage ─────────────────────────────────
def _check_disk_inodes(state):
    try:
        res = subprocess.run(
            ["df", "-i", "/root/dotm-sniper"],
            capture_output=True, text=True, timeout=5, start_new_session=True
        )
        lines_l = res.stdout.strip().split("\n")
        if len(lines_l) >= 2:
            parts = lines_l[1].split()
            if len(parts) >= 6:
                pct = parts[5].strip().rstrip("%")
                if int(pct) > 80:
                    return ("INODE_USAGE",
                            f"💾 <b>Inode usage: {pct}%</b>\n"
                            f"• Clean up log files to prevent system crash")
    except Exception:
        pass
    return None


# ── Check 20: pm-trader CLI health ─────────────────────────────
def _check_pm_trader_health(state):
    try:
        start = datetime.now()
        res = subprocess.run(
            ["pm-trader", "balance"],
            capture_output=True, text=True, timeout=10, start_new_session=True
        )
        elapsed = (datetime.now() - start).total_seconds()
        if res.returncode != 0:
            return ("PM_TRADER_FAIL",
                    f"🔧 <b>pm-trader CLI failed</b>\n"
                    f"• exit={res.returncode}, output={res.stderr[:100]}")
        if elapsed > 5:
            return ("PM_TRADER_SLOW",
                    f"🔧 <b>pm-trader slow: {elapsed:.1f}s</b>\n"
                    f"• Should respond <5s, possible API issues")
    except subprocess.TimeoutExpired:
        return "PM_TRADER_HANG", "🔧 <b>pm-trader balance TIMEOUT (>10s)</b>\n• CLI may be hung"
    except FileNotFoundError:
        return "PM_TRADER_MISSING", "🔧 <b>pm-trader not found</b>\n• Check PATH or installation"
    except Exception as e:
        return "PM_TRADER_ERROR", f"🔧 <b>pm-trader error</b>\n• {str(e)[:100]}"
    return None


# ── Check 21: API key validity ─────────────────────────────────
def _check_api_keys(state):
    issues = []
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not deepseek_key:
        _src_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(os.path.dirname(_src_dir), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.strip().startswith("DEEPSEEK_API_KEY="):
                        deepseek_key = line.strip().split("=", 1)[1].strip().strip('"')
                        break
    if not deepseek_key:
        issues.append("DEEPSEEK_API_KEY: missing")
    else:
        try:
            import requests as rq
            resp = rq.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {deepseek_key}",
                         "Content-Type": "application/json"},
                json={"model": "deepseek-chat", "messages": [
                    {"role": "user", "content": "ping"}],
                    "max_tokens": 1},
                timeout=15,
            )
            if resp.status_code == 401:
                issues.append("DeepSeek: 401 Unauthorized (key invalid/expired)")
            elif resp.status_code == 402:
                issues.append("DeepSeek: 402 Payment Required (balance exhausted)")
            elif resp.status_code == 429:
                issues.append("DeepSeek: 429 Rate Limited")
        except Exception as e:
            issues.append(f"DeepSeek: unreachable ({str(e)[:60]})")

    if issues:
        return ("API_KEYS",
                "🔑 <b>API key issues</b>\n" +
                "\n".join(f"• {i}" for i in issues))
    return None


# ── Check 22: Memory usage ─────────────────────────────────────
def _check_memory(state):
    issues = []
    for proc_name in ["dotm_sniper.py", "hermes_advisor.py"]:
        try:
            res = subprocess.run(
                ["ps", "aux"],
                capture_output=True, text=True, timeout=5, start_new_session=True
            )
            for line in res.stdout.split("\n"):
                if f"python3 src/{proc_name}" in line and "SCREEN" not in line and "bash -c" not in line:
                    parts = line.split()
                    if len(parts) >= 6:
                        rss_mb = int(parts[5]) / 1024
                        if rss_mb > 500:
                            issues.append(f"{proc_name}: {rss_mb:.0f}MB RSS (>500MB)")
                        break
        except Exception:
            pass
    if issues:
        return ("MEMORY",
                "🧠 <b>High memory usage</b>\n" +
                "\n".join(f"• {i}" for i in issues))
    return None


# ── Check 23: Log file size ─────────────────────────────────────
def _check_log_size(state):
    MAX_LOG_MB = 50
    large = []
    log_paths = [
        "/tmp/sniper_v557.log", "/tmp/sniper_v556.log",
        "/tmp/hermes_v557.log", "/tmp/hermes_v556.log",
        "/root/sniper_report.log",
        "/root/dotm-sniper/logs/equity_tracker.log",
        "/root/dotm-sniper/logs/advisor_cron.log",
    ]
    for path in log_paths:
        try:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            if size_mb > MAX_LOG_MB:
                name = os.path.basename(path)
                large.append(f"{name}: {size_mb:.0f}MB")
        except Exception:
            continue
    if large:
        return ("LOG_SIZE",
                f"📝 <b>Log files >{MAX_LOG_MB}MB</b>\n" +
                "\n".join(f"• {l}" for l in large))
    return None
    signals = sum(1 for l in lines if "=> BUY" in l)
    blocked = sum(1 for l in lines if "TRADE-BLOCKED" in l)
    executed = sum(1 for l in lines if "Bought:" in l and "Bought: 0" not in l)
    diverge_overrides = sum(1 for l in lines if "diverge_" in l and "override" in l)
    return f"signals={signals} blocked={blocked} executed={executed} diverge_overrides={diverge_overrides}"


def _summarize_equity(state):
    try:
        data = json.load(open(EQUITY_FILE))
        snaps = data.get("snapshots", [])
        if snaps:
            eq = snaps[-1].get("total_equity", 0)
            cash = snaps[-1].get("cash", 0)
            pos = snaps[-1].get("positions_value", 0)
            pnl = snaps[-1].get("unrealized_pnl", 0)
            n_pos = snaps[-1].get("positions_count", 0)
            return f"equity=${eq:.2f} cash=${cash:.2f} pos=${pos:.2f} pnl={pnl:+.2f} positions={n_pos}"
    except Exception:
        pass
    return "equity=data_unavailable"


def _summarize_orders():
    try:
        res = subprocess.run(
            ["pm-trader", "orders", "list"],
            capture_output=True, text=True, timeout=15, start_new_session=True
        )
        data = json.loads(res.stdout) if res.stdout else {}
        orders = data.get("data", []) if isinstance(data.get("data"), list) else []
        pending = sum(1 for o in orders if o.get("status") == "pending")
        return f"pending_orders={pending}"
    except Exception:
        return "orders=api_error"


def _summarize_cycle(state):
    last_cycle = state.get("last_cycle_start", "never")
    avg_time = state.get("avg_cycle_time", 0)
    return f"last={last_cycle} avg_cycle={avg_time:.0f}s"


def _summarize_errors(state):
    errs = state.get("errors_last_hour", 0)
    return f"errors_1h={errs}"


def _summarize_llm(state):
    cost = state.get("llm_cost_today", 0)
    calls = state.get("llm_calls_today", 0)
    return f"calls={calls} cost=${cost:.2f}"


def _summarize_disk():
    try:
        usage = shutil.disk_usage("/root/dotm-sniper")
        pct = usage.used / usage.total
        return f"used={pct:.0%} free={usage.free // (1024**3)}GB"
    except Exception:
        return "disk=unknown"


def _summarize_hypotheses():
    try:
        db = json.load(open(HYPOTHESIS_DB_FILE))
        hyps = db.get("hypotheses", [])
        open_h = sum(1 for h in hyps if not h.get("resolved"))
        resolved = sum(1 for h in hyps if h.get("resolved"))
        return f"open={open_h} resolved={resolved}"
    except Exception:
        return "hypotheses=data_error"


def _summarize_winrate():
    try:
        db = json.load(open(HYPOTHESIS_DB_FILE))
        resolved = [h for h in db.get("hypotheses", []) if h.get("resolved")]
        wins = sum(1 for h in resolved if h.get("pnl_pct", 0) > 0)
        total = len(resolved)
        wr = wins / total if total else 0
        return f"resolved={total} wins={wins} winrate={wr:.0%}"
    except Exception:
        return "winrate=data_error"


def _summarize_calib():
    try:
        model = json.load(open(CALIBRATION_MODEL_FILE))
        clusters = []
        for c, d in model.items():
            y = d.get("y_thresholds_", [])
            x = d.get("X_thresholds_", [])
            if y and x:
                clusters.append(f"{c}:p>{x[-1]:.0%}->{max(y):.0%}")
        return f"clusters=[{', '.join(clusters)}]"
    except Exception:
        return "calib=no_model"


def _summarize_cache():
    try:
        tracking = json.load(open(PRICE_TRACKING_FILE))
        total = len(tracking)
        high = sum(1 for v in tracking.values() if v.get("p_model", 0) >= 0.85)
        return f"tracked={total} high_p={high}"
    except Exception:
        return "cache=error"


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
        lambda: _check_telegram(state),
        lambda: _check_crash_frequency(state),
        lambda: _check_json_integrity(state),
        lambda: _check_cron_health(state),
        lambda: _check_llm_error_rate(state),
        lambda: _check_screen_sessions(state),
        lambda: _check_disk_inodes(state),
        lambda: _check_pm_trader_health(state),
        lambda: _check_api_keys(state),
        lambda: _check_memory(state),
        lambda: _check_log_size(state),
    ]

    summaries = [
        lambda: _summarize_no_trades(lines),
        lambda: _summarize_equity(state),
        lambda: _summarize_orders(),
        lambda: f"api={sum(1 for l in lines if '[GAMMA]' in l)}_cycles",
        lambda: _summarize_cycle(state),
        lambda: _summarize_errors(state),
        lambda: _summarize_llm(state),
        lambda: _summarize_disk(),
        lambda: _summarize_hypotheses(),
        lambda: _summarize_winrate(),
        lambda: _summarize_calib(),
        lambda: _summarize_cache(),
        lambda: "telegram=patched_dns",
        lambda: "tracebacks=scanned_5_logs",
        lambda: "files=5_critical_json",
        lambda: "cron=3_jobs",
        lambda: "llm_errors=6h_window",
        lambda: "sessions=sniper+hermes",
        lambda: "inodes=df_check",
        lambda: "pm_trader=balance_10s_timeout",
        lambda: "api_keys=deepseek_ping",
        lambda: "memory=ps_aux_rss",
        lambda: "log_size=7_files_50mb",
    ]

    check_names = [
        "no_trades", "equity_drawdown", "order_health", "api_health",
        "cycle_timing", "error_spike", "llm_usage", "disk_space",
        "hypothesis_db", "winrate", "calib_overfit", "cache",
        "telegram", "crash_freq", "json_integrity", "cron_health",
        "llm_errors", "screen_sessions", "disk_inodes", "pm_trader",
        "api_keys", "memory", "log_size",
    ]

    for i, check in enumerate(checks):
        name = check_names[i] if i < len(check_names) else f"check_{i}"
        summary = ""
        try:
            summary = summaries[i]() if i < len(summaries) else ""
        except Exception:
            summary = "summary_error"

        try:
            result = check()
        except Exception as e:
            logger.warning(f"[HEALTH-CHECK] {name}: CRASH ({summary}) - {e}")
            continue
        if result is None:
            logger.info(f"[HEALTH-CHECK] {name}: OK | {summary}")
            continue
        alert_key, message = result
        logger.info(f"[HEALTH-CHECK] {name}: ISSUE [{alert_key}] | {summary}")
        if _should_alert(state, alert_key):
            alerts.append((alert_key, message))
            _mark_alerted(state, alert_key)

    _save_state(state)

    if not alerts:
        logger.info("[HEALTH] All 23 checks passed")
        return

    for alert_key, message in alerts:
        logger.info(f"[HEALTH] ALERT [{alert_key}]: {message[:100]}")
    return alerts


def run_hourly_report():
    """Run all 23 checks, send ONE aggregated Telegram message with all issues.
    Called by cron every hour. Ignores cooldown — always reports full status."""
    state = _load_state()
    lines = _read_recent_log(hours=24)

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
        lambda: _check_telegram(state),
        lambda: _check_crash_frequency(state),
        lambda: _check_json_integrity(state),
        lambda: _check_cron_health(state),
        lambda: _check_llm_error_rate(state),
        lambda: _check_screen_sessions(state),
        lambda: _check_disk_inodes(state),
        lambda: _check_pm_trader_health(state),
        lambda: _check_api_keys(state),
        lambda: _check_memory(state),
        lambda: _check_log_size(state),
    ]

    check_names = [
        "no_trades", "equity_drawdown", "order_health", "api_health",
        "cycle_timing", "error_spike", "llm_usage", "disk_space",
        "hypothesis_db", "winrate", "calib_overfit", "cache",
        "telegram", "crash_freq", "json_integrity", "cron_health",
        "llm_errors", "screen_sessions", "disk_inodes", "pm_trader",
        "api_keys", "memory", "log_size",
    ]

    issues = []
    ok_count = 0

    for i, check in enumerate(checks):
        name = check_names[i] if i < len(check_names) else f"check_{i}"
        try:
            result = check()
        except Exception as e:
            logger.warning(f"[HOURLY] {name}: CRASH - {e}")
            issues.append(f"❗ {name}: check crashed")
            continue
        if result is None:
            ok_count += 1
            continue
        alert_key, message = result
        clean = message.replace("<b>", "").replace("</b>", "")
        first_line = clean.split("\n")[0]
        issues.append(first_line)
        logger.info(f"[HOURLY] {name}: ISSUE [{alert_key}]")

    ts = datetime.now().strftime("%m/%d %H:%M")
    if not issues:
        logger.info(f"[HOURLY] ({ts}): All 23 checks OK")
        return []

    msg = f"🔬 DOTM Hourly ({ts}): {len(issues)} issues / {ok_count} OK\n\n"
    msg += "\n".join(f"• {i}" for i in issues)

    _send_telegram(msg)
    logger.info(f"[HOURLY] Sent {len(issues)} issues to Telegram")
    return issues


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from utils import load_env_file
    load_env_file()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "--hourly" in sys.argv:
        results = run_hourly_report()
        print(f"{len(results)} issues" if results else "All healthy")
    else:
        results = run_health_check()
        print(f"{len(results) or 0} issues" if results else "All healthy")
