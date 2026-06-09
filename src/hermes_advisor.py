#!/usr/bin/env python3
"""
Hermes Advisor v5.4.0 - Async Position Risk Manager with Self-Improvement
Runs parallel to dotm_sniper.py, handles reconciliation and emergency exits.
Alert throttling: Telegram only on trigger_exit or status change.
Anti-Fossil Filter: news limited to last 30 days, max 5 results.
Self-improvement: tracks predictions, generates skills, adapts to outcomes.
"""
import subprocess
import json
import time
import os
import sys
import logging
import re
import signal
import threading
import html
from datetime import datetime
from logging.handlers import RotatingFileHandler
from log_formatter import StructuredFormatter
from bayesian_updater import update_posterior, should_exit as bayesian_should_exit, classify_news_with_llm
from hermes_memory import log_prediction, resolve_prediction, generate_skills, load_skills_for_prompt
import positions_db

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotm_report import TelegramReporter
from news_scanner import fetch_recent_news
from utils import load_json, save_json, sanitize_for_prompt, load_env_file, check_and_write_pid, cleanup_pid_file, validate_env_vars
from schema import (
    ALERT_HOLD_COUNTS, ALERT_LAST_NOTIFIED, ALERT_POSITION_STATUS, ALERT_UPDATED_AT,
    CACHE_LAST_UPDATE, CACHE_METACULUS, CACHE_NEWS, CACHE_TIMESTAMP,
    POS_CLUSTERS, POS_EMERGENCY_EXIT_FAILED, POS_ENTRY_PRICE, POS_HIGH_PRICE,
    POS_IN_EMERGENCY_EXIT, POS_LAST_EMERGENCY_ATTEMPT, POS_MARKET_QUESTION,
    POS_METACULUS_PROB, POS_OUTCOME, POS_PARTIAL_FILLS, POS_PARTIAL_PROCEEDS,
    POS_SELLING_IN_PROGRESS, POS_SHARES, POS_SHARES_AT_TP_OPEN,
)

load_env_file()
validate_env_vars(["DEEPSEEK_API_KEY", "TG_BOT_TOKEN", "TG_CHAT_ID"])

HERMES_LOG = "/root/dotm-sniper/logs/hermes.log"
os.makedirs(os.path.dirname(HERMES_LOG), exist_ok=True)

class UnbufferedRotatingFileHandler(RotatingFileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_handler_file = UnbufferedRotatingFileHandler(HERMES_LOG, maxBytes=10*1024*1024, backupCount=3)
_handler_stream = logging.StreamHandler()
if os.environ.get("LOG_FORMAT") == "json":
    _formatter = StructuredFormatter(json_mode=True)
else:
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
_handler_file.setFormatter(_formatter)
_handler_stream.setFormatter(_formatter)
logging.basicConfig(
    level=logging.INFO,
    handlers=[_handler_file, _handler_stream]
)
logger = logging.getLogger(__name__)

_shutdown_requested = False

def _handle_shutdown(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("[HERMES] Shutdown signal received, finishing current cycle...")

signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)

POSITIONS_FILE = "/root/dotm-sniper/positions.json"
SETTINGS_FILE = "/root/dotm-sniper/bot_settings.json"
CACHE_FILE = "/root/dotm-sniper/source_cache.json"

RECONCILE_INTERVAL_SECONDS = 900
NEWS_CHECK_INTERVAL_SECONDS = 600
TP_LIMIT_PRICE = 0.85
TP_LADDER_PRICES = {0.75, 0.85}
MAX_EMERGENCY_RETRIES = 3
NOTIFICATION_COOLDOWN_SECONDS = 4 * 3600
TELEGRAM_REPORTER = TelegramReporter()

ALERT_STATE_FILE = "/root/dotm-sniper/logs/hermes_alert_state.json"
_last_alert_status = {}
_last_notified_at = {}
_status_hold_counts = {}

_alert_state_lock = threading.RLock()
_positions_file_lock = threading.RLock()

NOTIFY_SEVERITIES = {"DIVERGENCE", "RED"}
STATUS_HOLD_SECONDS = 1800
STATUS_HOLD_COUNT = 2

def _load_alert_state():
    global _last_alert_status, _last_notified_at, _status_hold_counts
    state = load_json(ALERT_STATE_FILE, {})
    _last_alert_status = state.get(ALERT_POSITION_STATUS, {})
    _last_notified_at = state.get(ALERT_LAST_NOTIFIED, {})
    _status_hold_counts = state.get(ALERT_HOLD_COUNTS, {})

def _save_alert_state():
    with _alert_state_lock:
        save_json(ALERT_STATE_FILE, {
            ALERT_POSITION_STATUS: _last_alert_status,
            ALERT_LAST_NOTIFIED: _last_notified_at,
            ALERT_HOLD_COUNTS: _status_hold_counts,
            ALERT_UPDATED_AT: datetime.now().isoformat()
        })

def _should_send_telegram(slug, trigger_exit, current_status):
    with _alert_state_lock:
        if trigger_exit:
            return True
        normalized = str(current_status).upper().strip()
        if normalized not in NOTIFY_SEVERITIES:
            return False
        last_notified = _last_notified_at.get(slug)
        if last_notified:
            try:
                elapsed = (datetime.now() - datetime.fromisoformat(last_notified)).total_seconds()
                if elapsed < NOTIFICATION_COOLDOWN_SECONDS:
                    return False
            except (ValueError, TypeError):
                pass
        last_status = _last_alert_status.get(slug)
        last_normalized = str(last_status).upper().strip() if last_status else None
        return normalized != last_normalized

def _update_and_check_status(slug, trigger_exit, current_status):
    global _last_alert_status, _last_notified_at, _status_hold_counts
    with _alert_state_lock:
        normalized = str(current_status).upper().strip()
        last_status = _last_alert_status.get(slug)
        last_normalized = str(last_status).upper().strip() if last_status else None

        if last_normalized and normalized != last_normalized:
            if last_normalized == "DIVERGENCE" and normalized in ("GREEN", "YELLOW"):
                hold_key = f"{slug}:{normalized}"
                count = _status_hold_counts.get(hold_key, 0) + 1
                _status_hold_counts[hold_key] = count
                if count < STATUS_HOLD_COUNT:
                    logger.info(f"[HERMES] Hysteresis: {slug[:40]}... {last_normalized}→{normalized} hold {count}/{STATUS_HOLD_COUNT}")
                    _save_alert_state()
                    return False
                else:
                    del _status_hold_counts[hold_key]
                    for k in list(_status_hold_counts.keys()):
                        if k.startswith(f"{slug}:") and k != f"{slug}:{normalized}":
                            del _status_hold_counts[k]
            else:
                hold_key = f"{slug}:{normalized}"
                _status_hold_counts.pop(hold_key, None)
        should_send = _should_send_telegram(slug, trigger_exit, normalized)
        _last_alert_status[slug] = normalized
        if should_send:
            _last_notified_at[slug] = datetime.now().isoformat()
            _save_alert_state()
        return should_send


DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}


_load_alert_state()

def get_settings():
    from db import load_settings
    return load_settings() or {}

def get_balance():
    try:
        res = subprocess.run(["pm-trader", "balance"], capture_output=True, text=True, timeout=15, start_new_session=True)
        if res.returncode != 0:
            logger.error(f"[HERMES] pm-trader balance failed: rc={res.returncode} stderr={res.stderr[:200]}")
            return None
        return json.loads(res.stdout).get("data", {})
    except Exception:
        return None

def get_portfolio():
    try:
        res = subprocess.run(["pm-trader", "portfolio"], capture_output=True, text=True, timeout=15, start_new_session=True)
        if res.returncode != 0:
            logger.error(f"[HERMES] pm-trader portfolio failed: rc={res.returncode} stderr={res.stderr[:200]}")
            return None
        data = json.loads(res.stdout).get("data", [])
        return [p for p in data if float(p.get("shares", 0)) > 0.001]
    except Exception:
        return None

def get_open_orders():
    try:
        res = subprocess.run(["pm-trader", "orders", "list"],
                           capture_output=True, text=True, timeout=30, start_new_session=True)
        data = json.loads(res.stdout) if res.stdout else {}
        orders = data.get("data", []) if isinstance(data.get("data"), list) else []
        result = []
        for o in orders:
            if o.get("status") != "pending":
                continue
            result.append({
                "id": o.get("id"),
                "slug": o.get("market_slug", ""),
                "outcome": o.get("outcome", "yes"),
                "side": o.get("side", ""),
                "price": float(o.get("limit_price", 0)),
                "shares": float(o.get("amount", 0)),
            })
        return result
    except Exception as e:
        logger.error(f"[HERMES] Failed to get open orders: {e}")
        return []

def cancel_order(slug, outcome="yes"):
    try:
        orders = get_open_orders()
        matching = [o for o in orders if o.get("slug") == slug and o.get("side") == "sell"]
        if not matching:
            return False
        order_id = matching[0].get("id")
        if not order_id:
            return False
        res = subprocess.run(["pm-trader", "orders", "cancel", str(order_id)],
                           capture_output=True, text=True, timeout=20, start_new_session=True)
        result = json.loads(res.stdout) if res.stdout else {}
        if result.get("ok"):
            logger.info(f"[HERMES] Canceled order {order_id} for {slug[:40]}...")
            return True
        logger.warning(f"[HERMES] Cancel failed for {slug}: {result}")
        return False
    except Exception as e:
        logger.error(f"[HERMES] Cancel exception for {slug}: {e}")
        return False

def market_sell(slug, outcome="yes", shares=None):
    try:
        if shares is None:
            portfolio = get_portfolio()
            if portfolio is None:
                logger.error(f"[HERMES] Cannot get portfolio for market_sell {slug}, using last known shares")
                return False
            pos = next((p for p in portfolio if p.get("market_slug") == slug), None)
            if pos:
                shares = pos.get("shares", 0)

        if not shares or shares <= 0:
            logger.warning(f"[HERMES] No shares to sell for {slug}")
            return False

        res = subprocess.run(["pm-trader", "sell", slug, outcome, str(shares)],
                           capture_output=True, text=True, timeout=30, start_new_session=True)
        if res.returncode != 0:
            logger.error(f"[HERMES] Market sell failed for {slug}: rc={res.returncode}")
            return False
        result = json.loads(res.stdout) if res.stdout else {}
        if result.get("ok"):
            logger.info(f"[HERMES] Market sell executed for {slug}: {shares} shares")
            return True
        logger.warning(f"[HERMES] Market sell failed for {slug}: {result}")
        return False
    except Exception as e:
        logger.error(f"[HERMES] Market sell exception for {slug}: {e}")
        return False

def _merge_save_positions(deleted_slugs=None, updated_positions=None):
    for s in (deleted_slugs or set()):
        positions_db.delete(s)
    if updated_positions:
        positions_db.merge(updated_positions)

def reconcile_positions():
    logger.info("[HERMES] Starting position reconciliation...")

    portfolio = get_portfolio()
    if portfolio is None:
        logger.error("[HERMES] Portfolio API failed, skipping reconciliation to avoid data loss")
        return
    open_orders = get_open_orders()

    positions = positions_db.load_all()
    if not positions:
        logger.info("[HERMES] No positions to reconcile")
        return

    if len(portfolio) == 0 and len(positions) > 0:
        logger.warning(f"[HERMES] Empty portfolio but {len(positions)} tracked positions — API may be down, skipping reconciliation")
        return

    portfolio_slugs = {p.get("market_slug", "") for p in portfolio if p.get("market_slug")}

    deleted_slugs = set()
    updated_positions = {}

    for slug, pos_data in list(positions.items()):
        pos_modified = False

        if slug not in portfolio_slugs:
            if pos_data.get(POS_IN_EMERGENCY_EXIT):
                if not any(o.get("slug") == slug for o in open_orders):
                    logger.info(f"[HERMES] Emergency-exit position {slug[:40]}... no open orders, checking portfolio again")
                    fresh_portfolio = get_portfolio()
                    if fresh_portfolio is not None and slug not in {p["market_slug"] for p in fresh_portfolio}:
                        deleted_slugs.add(slug)
                        _notify_position_closed(slug, pos_data)
                continue

            logger.info(f"[HERMES] Position {slug[:40]}... not in portfolio, checking if fully closed")

            tp_order = next((o for o in open_orders if o.get("slug") == slug and o.get("side") == "sell" and any(abs(o.get("price", 0) - p) < 0.01 for p in TP_LADDER_PRICES)), None)

            if not tp_order:
                logger.info(f"[HERMES] No open TP for {slug[:40]}..., marking as closed")
                deleted_slugs.add(slug)
                _notify_position_closed(slug, pos_data)
            continue

        pos = next((p for p in portfolio if p.get("market_slug") == slug), None)
        if not pos:
            continue

        current_shares = pos.get(POS_SHARES, 0)
        recorded_shares = pos_data.get(POS_SHARES, 0)

        if current_shares != recorded_shares:
            logger.info(f"[HERMES] Share mismatch for {slug[:40]}...: recorded={recorded_shares}, actual={current_shares}")
            pos_data[POS_SHARES] = current_shares
            pos_modified = True

        entry_price = pos.get("avg_entry_price", 0)
        if entry_price > 0 and pos_data.get(POS_ENTRY_PRICE, 0) != entry_price:
            pos_data[POS_ENTRY_PRICE] = entry_price
            pos_modified = True

        tp_order = next((o for o in open_orders if o.get("slug") == slug and o.get("side") == "sell" and any(abs(o.get("price", 0) - p) < 0.01 for p in TP_LADDER_PRICES)), None)

        if tp_order:
            order_shares = tp_order.get("shares", 0)
            recorded_shares_at_tp = pos_data.get(POS_SHARES_AT_TP_OPEN, order_shares)
            current_shares_at_check = current_shares

            if current_shares_at_check < recorded_shares_at_tp and order_shares > 0:
                filled = recorded_shares_at_tp - current_shares_at_check
                total = order_shares

                if filled > 0 and filled < total:
                    logger.warning(f"[HERMES] PARTIAL FILL for {slug[:40]}...: {filled}/{total} (inferred from shares)")

                    fill_price = tp_order.get("price", TP_LIMIT_PRICE)
                    sold_value = filled * fill_price
                    pos_data[POS_SHARES] = current_shares
                    pos_data[POS_PARTIAL_FILLS] = pos_data.get(POS_PARTIAL_FILLS, 0) + filled
                    pos_data[POS_PARTIAL_PROCEEDS] = pos_data.get(POS_PARTIAL_PROCEEDS, 0) + sold_value

                    logger.info(f"[HERMES] Updated shares to {current_shares} (portfolio reflects partial fill), proceeds ${sold_value:.2f}")
                    pos_modified = True

                    _notify_partial_fill(slug, pos_data, filled, fill_price)

        if pos_modified:
            updated_positions[slug] = pos_data

    if deleted_slugs or updated_positions:
        _merge_save_positions(deleted_slugs=deleted_slugs, updated_positions=updated_positions)
        logger.info(f"[HERMES] Positions updated: {len(deleted_slugs)} deleted, {len(updated_positions)} updated")

def _notify_position_closed(slug, pos_data):
    try:
        entry = pos_data.get(POS_ENTRY_PRICE, 0)
        high = pos_data.get(POS_HIGH_PRICE, entry)
        if entry > 0 and high > 0:
            computed_pnl_pct = (high - entry) / entry * 100
            computed_pnl_abs = (high - entry) * pos_data.get(POS_SHARES, 0)
        else:
            computed_pnl_pct = 0
            computed_pnl_abs = 0
        if TELEGRAM_REPORTER:
            TELEGRAM_REPORTER.alert_convergence(
                market_slug=slug,
                question=pos_data.get(POS_MARKET_QUESTION, "Unknown"),
                pnl_pct=computed_pnl_pct,
                pnl_abs=computed_pnl_abs,
                convergence_ratio=0
            )
    except Exception as e:
        logger.warning(f"[HERMES] Position closed notification failed: {e}")

def _notify_partial_fill(slug, pos_data, filled, fill_price=None):
    try:
        if TELEGRAM_REPORTER:
            fp = fill_price or TP_LIMIT_PRICE
            TELEGRAM_REPORTER.alert_take_profit(
                market_slug=slug,
                question=pos_data.get("question", "Unknown"),
                pnl_pct=((fp - pos_data.get(POS_ENTRY_PRICE, 0)) / pos_data.get(POS_ENTRY_PRICE, 1)) * 100 if pos_data.get(POS_ENTRY_PRICE, 0) > 0 else 0,
                pnl_abs=filled * (fp - pos_data.get(POS_ENTRY_PRICE, 0))
            )
    except Exception as e:
        logger.warning(f"[HERMES] Partial fill notification failed: {e}")

def fetch_news_for_market(slug, question):
    try:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', question.lower())
        stop = {"will", "the", "a", "an", "be", "by", "of", "in", "on", "at", "to", "for", "this", "that", "is", "are", "was", "were"}
        keywords = [w for w in words if w not in stop][:5]

        news_items = fetch_recent_news(keywords, max_results=5, max_age_days=30)
        if isinstance(news_items, dict):
            headlines = news_items.get("headlines", [])
            news_items["headlines"] = [sanitize_for_prompt(str(h)) for h in headlines]
        elif isinstance(news_items, list):
            news_items = [sanitize_for_prompt(str(h)) if isinstance(h, str) else h for h in news_items]
        return news_items
    except Exception as e:
        logger.error(f"[HERMES] News fetch failed for {slug}: {e}")
        return []

def _prune_stale_cache():
    try:
        cache = load_json(CACHE_FILE, {CACHE_METACULUS: {}, CACHE_NEWS: {}, CACHE_LAST_UPDATE: None})
        now = datetime.now()
        pruned = False
        for section in (CACHE_METACULUS, CACHE_NEWS):
            entries = cache.get(section, {})
            stale = [k for k, v in entries.items()
                     if isinstance(v, dict) and v.get(CACHE_TIMESTAMP)
                     and (now - datetime.fromisoformat(v[CACHE_TIMESTAMP])).total_seconds() > 86400]
            for k in stale:
                del entries[k]
                pruned = True
        if pruned:
            save_json(CACHE_FILE, cache)
    except Exception as e:
        logger.warning(f"[HERMES] Cache prune error: {e}")

def evaluate_emergency_exit():
    logger.info("[HERMES] Starting emergency exit evaluation...")

    os.makedirs("logs", exist_ok=True)
    _prune_stale_cache()

    positions = positions_db.load_all()
    if not positions:
        logger.info("[HERMES] No positions to evaluate")
        return

    portfolio = get_portfolio()
    if portfolio is None:
        logger.error("[HERMES] Portfolio API failed, skipping cycle")
        return
    portfolio_map = {p.get("market_slug"): p for p in portfolio if p.get("market_slug")}

    for slug, pos_data in positions.items():
        if pos_data.get(POS_IN_EMERGENCY_EXIT):
            if pos_data.get(POS_EMERGENCY_EXIT_FAILED):
                last_attempt = pos_data.get(POS_LAST_EMERGENCY_ATTEMPT)
                if last_attempt:
                    try:
                        elapsed = (datetime.now() - datetime.fromisoformat(last_attempt)).total_seconds()
                        if elapsed > 600:
                            logger.info(f"[HERMES] Retrying failed emergency exit for {slug[:40]}...")
                            pos_data[POS_EMERGENCY_EXIT_FAILED] = False
                            pos_data[POS_IN_EMERGENCY_EXIT] = False
                            _merge_save_positions(updated_positions={slug: pos_data})
                    except (ValueError, TypeError):
                        pass
            else:
                logger.info(f"[HERMES] Skipping {slug[:40]}... - already in emergency exit")
            continue

        if pos_data.get(POS_SELLING_IN_PROGRESS):
            logger.info(f"[HERMES] Skipping {slug[:40]}... - sniper is selling")
            continue

        question = pos_data.get(POS_MARKET_QUESTION, "")
        if not question:
            continue

        metaculus_prob_raw = pos_data.get(POS_METACULUS_PROB)
        if metaculus_prob_raw is not None and metaculus_prob_raw > 0:
            bot_prob = metaculus_prob_raw
        else:
            bot_prob = min(pos_data.get(POS_ENTRY_PRICE, 0) * 2, 0.20)
        bot_prob = min(max(bot_prob, 0.0), 1.0)

        entry_price = pos_data.get(POS_ENTRY_PRICE, 0)
        portfolio_pos = portfolio_map.get(slug, {})
        current_price = portfolio_pos.get("live_price", 0)
        if entry_price > 0:
            if current_price > 0:
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = -1.0
        else:
            pnl_pct = 0

        news_data = fetch_news_for_market(slug, question)

        headlines = []
        if isinstance(news_data, dict):
            headlines = news_data.get("headlines", [])
        elif isinstance(news_data, list):
            headlines = news_data

        if not headlines:
            logger.info(f"[NEWS] No fresh news for 30 days, skipping evaluation for {slug[:40]}...")
            continue

        news_text = "\n".join([f"- {sanitize_for_prompt(str(h))}" for h in headlines[:5]])

        skills_context = load_skills_for_prompt(max_skills=5)

        prompt = f"""You are a risk analysis bot. Determine if recent news makes the outcome YES mathematically impossible (0% probability).

Market Question: {sanitize_for_prompt(question)}

Bot's estimated probability (p_bot): {bot_prob:.1%}

Recent News:
{news_text}
{skills_context}
Instructions:
- Analyze if any news fundamentally invalidates the YES outcome
- Estimate the current probability (p_hermes) of YES outcome based on the news
- If YES outcome is now impossible (0%), set trigger_exit=true
- If p_hermes dropped by more than 50% relative to p_bot (p_hermes < p_bot * 0.5), set status="DIVERGENCE"
- If p_hermes dropped by more than 50% AND there are confirmed facts making the outcome impossible, set trigger_exit=true
- If there's any chance (>0%) and no major probability drop, set trigger_exit=false

Return ONLY JSON:
{{"trigger_exit": true/false, "p_hermes": 0.XX, "status": "GREEN/YELLOW/RED/DIVERGENCE", "reason": "brief explanation"}}"""

        content = ""
        try:
            for _attempt in range(3):
                try:
                    import requests
                    resp = requests.post(DEEPSEEK_URL, headers=HEADERS, json={
                        "model": "deepseek-chat",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 300
                    }, timeout=30)

                    content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                    break
                except requests.exceptions.Timeout:
                    if _attempt < 2:
                        _backoff = 2 ** (_attempt + 1)
                        logger.warning(f"[HERMES] LLM timeout for {slug}, retry in {_backoff}s")
                        time.sleep(_backoff)
                    else:
                        raise
                except requests.exceptions.ConnectionError:
                    if _attempt < 2:
                        _backoff = 2 ** (_attempt + 1)
                        logger.warning(f"[HERMES] LLM connection error for {slug}, retry in {_backoff}s")
                        time.sleep(_backoff)
                    else:
                        raise

            trigger = False
            p_hermes_val = bot_prob
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group(0))
                trigger = decision.get("trigger_exit", False)
                reason = decision.get("reason", "")
                status = str(decision.get("status", "GREEN")).upper().strip()
                p_hermes_raw = decision.get("p_hermes")

                divergence_locked = False

                if p_hermes_raw is not None:
                    try:
                        p_bot_val = float(str(bot_prob).replace('%', '').strip())
                        p_hermes_val = float(str(p_hermes_raw).replace('%', '').strip())

                        if p_hermes_val > 1.0:
                            p_hermes_val = p_hermes_val / 100.0
                        if p_bot_val > 1.0:
                            p_bot_val = p_bot_val / 100.0

                        if p_bot_val > 0 and p_hermes_val < (p_bot_val * 0.5):
                            status = "DIVERGENCE"
                            divergence_locked = True
                            logger.warning(
                                f"[HERMES] PROBABILITY DROP for {slug[:40]}...: "
                                f"bot={p_bot_val:.1%} hermes={p_hermes_val:.1%} (-{(1 - p_hermes_val / p_bot_val) * 100:.0f}%)"
                            )
                    except (ValueError, TypeError) as parse_err:
                        logger.error(f"[HERMES] Probability parse error for {slug}: {parse_err}")

                if trigger:
                    status = "DIVERGENCE"
                    divergence_locked = True

                if divergence_locked:
                    status = "DIVERGENCE"
                elif status == "DIVERGENCE":
                    status = "YELLOW"

                normalized_status = status.upper().strip()

                try:
                    cluster = pos_data.get(POS_CLUSTERS, ["unknown"])[0] if isinstance(pos_data.get(POS_CLUSTERS), list) else "unknown"
                except (IndexError, TypeError):
                    cluster = "unknown"
                log_prediction(
                    slug=slug, question=question, p_bot=bot_prob,
                    p_hermes=p_hermes_val if p_hermes_raw is not None else bot_prob,
                    verdict=reason[:50], status=normalized_status,
                    reason=reason, cluster=cluster,
                )

                should_send = _update_and_check_status(slug, trigger, normalized_status)

                if trigger:
                    logger.warning(f"[HERMES] EMERGENCY EXIT TRIGGERED for {slug[:40]}...: {reason}")
                    fresh_positions = positions_db.load_all()
                    pos_data = fresh_positions.get(slug, pos_data)
                    if pos_data.get(POS_SELLING_IN_PROGRESS) or pos_data.get(POS_IN_EMERGENCY_EXIT):
                        logger.info(f"[HERMES] Skipping {slug[:40]}... another process already handling")
                        continue
                    _execute_emergency_exit(slug, pos_data, reason)
                elif should_send:
                    if normalized_status == "DIVERGENCE" and pnl_pct >= 0.50:
                        logger.info(f"[HERMES] Profitable position {slug[:40]}... P&L={pnl_pct:.0%}, downgrading DIVERGENCE→YELLOW notification")
                        normalized_status = "YELLOW"
                        with _alert_state_lock:
                            _last_alert_status[slug] = "YELLOW"
                            hold_key_prefix = f"{slug}:"
                            for k in list(_status_hold_counts.keys()):
                                if k.startswith(hold_key_prefix):
                                    del _status_hold_counts[k]
                        _save_alert_state()
                    logger.info(f"[HERMES] Status changed to {normalized_status} for {slug[:40]}...: {reason}")
                    if TELEGRAM_REPORTER:
                        try:
                            msg = "⚠️ <b>HERMES STATUS CHANGE</b>\n\n"
                            msg += f"📌 {html.escape(question[:55])}...\n\n"
                            msg += f"🔄 Status: <b>{normalized_status}</b>\n"
                            msg += f"📊 Bot P: {bot_prob:.0%}"
                            if p_hermes_raw is not None:
                                try:
                                    p_display = float(str(p_hermes_raw).replace('%', '').strip())
                                    if p_display > 1.0:
                                        p_display = p_display / 100.0
                                    msg += f" | Hermes P: {p_display:.0%}"
                                except (ValueError, TypeError):
                                    pass
                            if pnl_pct != 0:
                                msg += f"\n💰 P&L: {pnl_pct:+.0%}"
                            msg += f"\n📝 {html.escape(reason)}"
                            TELEGRAM_REPORTER._send(msg)
                        except Exception as e:
                            logger.warning(f"[HERMES] Telegram send failed: {e}")
                else:
                    logger.info(f"[HERMES] Routine check {slug[:40]}...: status={normalized_status} reason={reason}")
            else:
                logger.warning(f"[HERMES] LLM returned non-JSON for {slug[:40]}...: {content[:100]}")

            if not trigger and headlines:
                try:
                    news_cat = classify_news_with_llm(question, headlines)
                    update_posterior(slug, news_cat)
                    bayes_exit, bayes_reason = bayesian_should_exit(slug)
                    if bayes_exit:
                        logger.warning(f"[HERMES] BAYESIAN EXIT for {slug[:40]}...: {bayes_reason}")
                        _update_and_check_status(slug, True, "DIVERGENCE")
                        fresh_positions_bayes = positions_db.load_all()
                        pos_data = fresh_positions_bayes.get(slug, pos_data)
                        if pos_data.get(POS_SELLING_IN_PROGRESS) or pos_data.get(POS_IN_EMERGENCY_EXIT):
                            logger.info(f"[HERMES] Skipping bayesian exit {slug[:40]}... another process already handling")
                            continue
                        _execute_emergency_exit(slug, pos_data, bayes_reason)
                except Exception as be:
                    logger.warning(f"[HERMES] Bayesian update failed for {slug}: {be}")
        except Exception as e:
            logger.error(f"[HERMES] LLM evaluation failed for {slug}: {e}")

def _execute_emergency_exit(slug, pos_data, reason):
    logger.info(f"[HERMES] Executing emergency exit for {slug[:40]}...")

    outcome = pos_data.get(POS_OUTCOME, "yes")
    shares = pos_data.get(POS_SHARES, 0)

    pos = positions_db.load(slug)
    if pos is None:
        logger.warning(f"[HERMES] {slug} not found in positions, aborting emergency")
        return
    pos[POS_IN_EMERGENCY_EXIT] = True
    pos[POS_SELLING_IN_PROGRESS] = True
    positions_db.update(slug, pos)

    for attempt in range(MAX_EMERGENCY_RETRIES):
        logger.info(f"[HERMES] Cancel attempt {attempt + 1}/{MAX_EMERGENCY_RETRIES} for {slug[:40]}...")

        if cancel_order(slug, outcome):
            logger.info(f"[HERMES] Order canceled for {slug[:40]}...")
            break

        if attempt == MAX_EMERGENCY_RETRIES - 1:
            logger.error(f"[HERMES] Cancel failed after {MAX_EMERGENCY_RETRIES} attempts, proceeding with market sell anyway")

    time.sleep(2)

    logger.info(f"[HERMES] Executing market sell for {slug[:40]}... ({shares} shares, outcome={outcome})")
    if market_sell(slug, outcome, shares=shares if shares > 0 else None):
        logger.info(f"[HERMES] Market sell successful for {slug[:40]}...")

        portfolio = get_portfolio()
        remaining_shares = 0
        if portfolio is not None:
            remaining_pos = next((p for p in portfolio if p.get("market_slug") == slug), None)
            if remaining_pos:
                remaining_shares = remaining_pos.get("shares", 0)

        if remaining_shares > 0:
            logger.warning(f"[HERMES] {remaining_shares} shares remain for {slug[:40]}..., updating position instead of deleting")
            positions_db.merge({slug: {
                POS_SHARES: remaining_shares,
                POS_IN_EMERGENCY_EXIT: False,
                POS_SELLING_IN_PROGRESS: False,
                POS_EMERGENCY_EXIT_FAILED: False,
            }})
        else:
            positions_db.delete(slug)
            logger.info(f"[HERMES] Removed position {slug[:40]}... from file")

        entry_price = pos_data.get(POS_ENTRY_PRICE, 0)
        fresh_portfolio = get_portfolio()
        fresh_pos = next((p for p in (fresh_portfolio or []) if p.get("market_slug") == slug), None)
        current_price = fresh_pos.get("live_price", 0) if fresh_pos else 0
        if current_price <= 0:
            current_price = pos_data.get("live_price", 0)
        actual_pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else -100
        actual_pnl_abs = (current_price - entry_price) * shares if entry_price > 0 and shares > 0 else 0

        if TELEGRAM_REPORTER:
            try:
                TELEGRAM_REPORTER.alert_stop_loss(
                    market_slug=slug,
                    question=pos_data.get(POS_MARKET_QUESTION, "Unknown"),
                    pnl_pct=actual_pnl_pct,
                    pnl_abs=actual_pnl_abs
                )
            except Exception as e:
                logger.warning(f"[HERMES] Emergency notification failed: {e}")

        _log_emergency_exit(slug, pos_data, reason)
    else:
        logger.error(f"[HERMES] Market sell FAILED for {slug[:40]}...")

        pos = positions_db.load(slug)
        if pos is not None:
            pos[POS_EMERGENCY_EXIT_FAILED] = True
            pos[POS_LAST_EMERGENCY_ATTEMPT] = datetime.now().isoformat()
            pos[POS_SELLING_IN_PROGRESS] = False
            pos[POS_IN_EMERGENCY_EXIT] = False
            positions_db.update(slug, pos)

def _log_emergency_exit(slug, pos_data, reason):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "slug": slug,
        "question": pos_data.get(POS_MARKET_QUESTION, ""),
        "entry_price": pos_data.get(POS_ENTRY_PRICE, 0),
        "reason": reason,
        "action": "emergency_exit"
    }

    log_file = "/root/dotm-sniper/logs/emergency_log.json"
    logs = load_json(log_file, [])
    logs.append(log_entry)
    logs = logs[-1000:]
    save_json(log_file, logs)

def run_reconciliation_loop():
    while True:
        try:
            reconcile_positions()
        except Exception as e:
            logger.error(f"[HERMES] Reconciliation loop error: {e}")

        time.sleep(RECONCILE_INTERVAL_SECONDS)

def run_emergency_evaluation_loop():
    while True:
        try:
            evaluate_emergency_exit()
        except Exception as e:
            logger.error(f"[HERMES] Emergency evaluation error: {e}")

        time.sleep(NEWS_CHECK_INTERVAL_SECONDS)

def _resolve_predictions_loop():
    while True:
        try:
            time.sleep(3600)
            _check_resolved_markets()
            from hermes_memory import _load_memory
            m = _load_memory()
            last_skill = m.get("last_skill_generation")
            should_gen = not last_skill
            if last_skill:
                try:
                    elapsed = (datetime.now() - datetime.fromisoformat(last_skill)).total_seconds()
                    should_gen = elapsed >= 6 * 3600
                except Exception:
                    should_gen = True
            if should_gen:
                skills = generate_skills()
                if skills:
                    logger.info(f"[HERMES-SKILLS] Generated {len(skills)} skills")
        except Exception as e:
            logger.error(f"[HERMES] Resolution loop error: {e}")


def _check_resolved_markets():
    from hermes_memory import _load_memory
    m = _load_memory()
    predictions = m.get("predictions", {})
    if not predictions:
        return

    slugs = list(predictions.keys())
    try:
        subprocess.run(
            ["pm-trader", "orders", "list"],
            capture_output=True, text=True, timeout=15, start_new_session=True
        )
    except Exception:
        return

    known_active = set()
    known_active.update(positions_db.slugs())

    try:
        import requests as _req
        gamma_url = "https://gamma-api.polymarket.com/markets"
        for slug in slugs:
            try:
                resp = _req.get(gamma_url, params={"slug": slug}, timeout=10)
                if resp.status_code == 200:
                    markets = resp.json()
                    if markets:
                        market = markets[0]
                        if market.get("closed") or market.get("resolved"):
                            outcome = "yes" if market.get("outcome", "").lower() == "yes" else "no"
                            if market.get("outcomePrices"):
                                try:
                                    prices = json.loads(market["outcomePrices"])
                                    if len(prices) >= 2 and float(prices[0]) > float(prices[1]):
                                        outcome = "yes"
                                    else:
                                        outcome = "no"
                                except (json.JSONDecodeError, ValueError, IndexError):
                                    pass
                            resolve_prediction(slug, outcome)
            except Exception as e:
                logger.warning(f"[resolve_markets] {type(e).__name__}: {e}")
    except Exception as e:
        logger.warning(f"[HERMES] Resolution check failed: {e}")


HERMES_PID_FILE = "/root/dotm-sniper/hermes.pid"

def main():
    logger.info("="*60)
    logger.info("  HERMES ADVISOR v5.4.0 - Starting (with self-improvement)")
    logger.info("="*60)

    if not check_and_write_pid(HERMES_PID_FILE):
        logger.error("[HERMES] Another instance is already running, exiting")
        return

    import atexit
    atexit.register(cleanup_pid_file, HERMES_PID_FILE)

    reconcile_thread = threading.Thread(target=run_reconciliation_loop, daemon=True)
    emergency_thread = threading.Thread(target=run_emergency_evaluation_loop, daemon=True)
    resolution_thread = threading.Thread(target=_resolve_predictions_loop, daemon=True)

    reconcile_thread.start()
    emergency_thread.start()
    resolution_thread.start()

    logger.info("[HERMES] All 3 loops started (reconcile, emergency, resolution+skills)")

    _restart_timestamps = []

    try:
        while not _shutdown_requested:
            time.sleep(60)

            restarted_any = False
            if not reconcile_thread.is_alive():
                logger.error("[HERMES] Reconciliation thread died, restarting")
                reconcile_thread = threading.Thread(target=run_reconciliation_loop, daemon=True)
                reconcile_thread.start()
                restarted_any = True
            if not emergency_thread.is_alive():
                logger.error("[HERMES] Emergency thread died, restarting")
                emergency_thread = threading.Thread(target=run_emergency_evaluation_loop, daemon=True)
                emergency_thread.start()
                restarted_any = True
            if not resolution_thread.is_alive():
                logger.error("[HERMES] Resolution thread died, restarting")
                resolution_thread = threading.Thread(target=_resolve_predictions_loop, daemon=True)
                resolution_thread.start()
                restarted_any = True

            if restarted_any:
                _restart_timestamps.append(time.time())
                cutoff = time.time() - 600
                _restart_timestamps = [t for t in _restart_timestamps if t > cutoff]
                if len(_restart_timestamps) > 5:
                    logger.critical(
                        f"[HERMES] Thread restart storm: {len(_restart_timestamps)} restarts in 10 minutes!"
                    )

            positions = positions_db.load_all()
            active_count = len([p for p in positions.values() if not p.get(POS_IN_EMERGENCY_EXIT)])

            logger.debug(f"[HERMES] Heartbeat: {active_count} active positions")

        logger.info("[HERMES] Graceful shutdown complete")
    except KeyboardInterrupt:
        logger.info("[HERMES] Shutting down...")
    except Exception as e:
        logger.error(f"[HERMES] Fatal error: {e}")

if __name__ == "__main__":
    main()
