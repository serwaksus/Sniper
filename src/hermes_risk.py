#!/usr/bin/env python3
"""
Hermes Risk Management - Alert state, emergency exit, news fetching, risk evaluation.
Extracted from hermes_advisor.py for modularity.
"""
from __future__ import annotations
import json
import time
import os
import logging
import re
import html
import threading
from datetime import datetime
from bayesian_updater import update_posterior, should_exit as bayesian_should_exit, classify_news_with_llm
from hermes_memory import log_prediction, load_skills_for_prompt
import positions_db
from news_scanner import fetch_recent_news
from utils import load_json, save_json, sanitize_for_prompt
from config import ALERT_STATE_FILE, EMERGENCY_LOG_FILE, CACHE_FILE
from schema import (
    ALERT_HOLD_COUNTS, ALERT_LAST_NOTIFIED, ALERT_POSITION_STATUS, ALERT_UPDATED_AT,
    CACHE_LAST_UPDATE, CACHE_METACULUS, CACHE_NEWS, CACHE_TIMESTAMP,
    POS_CLUSTERS, POS_EMERGENCY_EXIT_FAILED, POS_ENTRY_PRICE,
    POS_IN_EMERGENCY_EXIT, POS_LAST_EMERGENCY_ATTEMPT, POS_MARKET_QUESTION,
    POS_METACULUS_PROB, POS_OUTCOME,
    POS_SELLING_IN_PROGRESS, POS_SHARES,
)

logger = logging.getLogger(__name__)

NOTIFY_SEVERITIES = {"DIVERGENCE", "RED"}
STATUS_HOLD_SECONDS = 1800
STATUS_HOLD_COUNT = 2
NOTIFICATION_COOLDOWN_SECONDS = 4 * 3600
TP_LIMIT_PRICE = 0.85
TP_LADDER_PRICES = {0.75, 0.85}
MAX_EMERGENCY_RETRIES = 3

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}

_last_alert_status: dict[str, str] = {}
_last_notified_at: dict[str, str] = {}
_status_hold_counts: dict[str, int] = {}

_alert_state_lock = threading.RLock()


def _load_alert_state() -> None:
    global _last_alert_status, _last_notified_at, _status_hold_counts
    state = load_json(ALERT_STATE_FILE, {})
    _last_alert_status = state.get(ALERT_POSITION_STATUS, {})
    _last_notified_at = state.get(ALERT_LAST_NOTIFIED, {})
    _status_hold_counts = state.get(ALERT_HOLD_COUNTS, {})


def _save_alert_state() -> None:
    with _alert_state_lock:
        save_json(ALERT_STATE_FILE, {
            ALERT_POSITION_STATUS: _last_alert_status,
            ALERT_LAST_NOTIFIED: _last_notified_at,
            ALERT_HOLD_COUNTS: _status_hold_counts,
            ALERT_UPDATED_AT: datetime.now().isoformat()
        })


def _should_send_telegram(slug: str, trigger_exit: bool, current_status: str) -> bool:
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


def _update_and_check_status(slug: str, trigger_exit: bool, current_status: str) -> bool:
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


def fetch_news_for_market(slug: str, question: str) -> list | dict:
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


def _prune_stale_cache() -> None:
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


def _merge_save_positions(deleted_slugs: set[str] | None = None, updated_positions: dict | None = None) -> None:
    for s in (deleted_slugs or set()):
        positions_db.delete(s)
    if updated_positions:
        positions_db.merge(updated_positions)


def evaluate_emergency_exit() -> None:
    from hermes_advisor import TELEGRAM_REPORTER as _tg
    TELEGRAM_REPORTER = _tg
    logger.info("[HERMES] Starting emergency exit evaluation...")

    os.makedirs("logs", exist_ok=True)
    _prune_stale_cache()

    positions = positions_db.load_all()
    if not positions:
        logger.info("[HERMES] No positions to evaluate")
        return

    from order_manager import get_portfolio
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


def _execute_emergency_exit(slug: str, pos_data: dict, reason: str) -> None:
    from hermes_advisor import cancel_order, market_sell, get_portfolio, TELEGRAM_REPORTER
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


def _log_emergency_exit(slug: str, pos_data: dict, reason: str) -> None:
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "slug": slug,
        "question": pos_data.get(POS_MARKET_QUESTION, ""),
        "entry_price": pos_data.get(POS_ENTRY_PRICE, 0),
        "reason": reason,
        "action": "emergency_exit"
    }

    log_file = EMERGENCY_LOG_FILE
    logs = load_json(log_file, [])
    logs.append(log_entry)
    logs = logs[-1000:]
    save_json(log_file, logs)


_load_alert_state()
