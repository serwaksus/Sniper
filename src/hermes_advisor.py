#!/usr/bin/env python3
"""
Hermes Advisor v5.3.7 - Async Position Risk Manager
Runs parallel to dotm_sniper.py, handles reconciliation and emergency exits.
Alert throttling: Telegram only on trigger_exit or status change.
Anti-Fossil Filter: news limited to last 30 days, max 5 results.
"""
import subprocess, json, time, os, sys, logging, fcntl, re, threading, html
from datetime import datetime, timedelta
from collections import defaultdict
from bayesian_updater import update_posterior, should_exit as bayesian_should_exit, classify_news_with_llm, init_posterior

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotm_report import TelegramReporter
from news_scanner import fetch_recent_news
from utils import load_json, save_json, _lock_file, _unlock_file, _normalize_keys, _strip_dict_keys_recursive, sanitize_for_prompt

def _load_env_manual():
    env_path = "/root/dotm-sniper/.env"
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    val = val.strip().strip('"').strip("'")
                    os.environ.setdefault(key.strip(), val)

_load_env_manual()

HERMES_LOG = "/root/dotm-sniper/logs/hermes.log"
os.makedirs(os.path.dirname(HERMES_LOG), exist_ok=True)

class UnbufferedFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        UnbufferedFileHandler(HERMES_LOG),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
_positions_file_lock = threading.Lock()

NOTIFY_SEVERITIES = {"DIVERGENCE", "RED"}
STATUS_HOLD_SECONDS = 1800
STATUS_HOLD_COUNT = 2

def _load_alert_state():
    global _last_alert_status, _last_notified_at, _status_hold_counts
    state = load_json(ALERT_STATE_FILE, {})
    _last_alert_status = state.get("position_status", {})
    _last_notified_at = state.get("last_notified_at", {})
    _status_hold_counts = state.get("hold_counts", {})

def _save_alert_state():
    with _alert_state_lock:
        save_json(ALERT_STATE_FILE, {
            "position_status": _last_alert_status,
            "last_notified_at": _last_notified_at,
            "hold_counts": _status_hold_counts,
            "updated_at": datetime.now().isoformat()
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
        if normalized != last_normalized:
            return True
        return False

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
    return load_json(SETTINGS_FILE, {})

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
        res = subprocess.run(["pm-trader", "orders", "--status", "open"],
                           capture_output=True, text=True, timeout=30, start_new_session=True)
        data = res.stdout
        orders = []
        if not data:
            return orders
        
        lines = data.strip().split('\n')
        for line in lines[1:]:
            if not line.strip() or '---' in line:
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 5:
                try:
                    order = {
                        "slug": parts[0],
                        "outcome": parts[1],
                        "side": parts[2],
                        "price": float(parts[3]) if parts[3] else 0.0,
                        "shares": round(float(parts[4])) if parts[4] else 0,
                        "filled": round(float(parts[5])) if len(parts) > 5 and parts[5] else 0,
                    }
                    orders.append(order)
                except (ValueError, IndexError):
                    continue
        return orders
    except Exception as e:
        logger.error(f"[HERMES] Failed to get open orders: {e}")
        return []

def cancel_order(slug, outcome="yes"):
    try:
        res = subprocess.run(["pm-trader", "orders", "cancel", slug, outcome],
                           capture_output=True, text=True, timeout=20, start_new_session=True)
        if res.returncode != 0:
            logger.warning(f"[HERMES] Cancel failed for {slug}: rc={res.returncode}")
            return False
        result = json.loads(res.stdout) if res.stdout else {}
        if result.get("ok"):
            logger.info(f"[HERMES] Canceled order for {slug[:40]}...")
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
    with _positions_file_lock:
        fresh = load_json(POSITIONS_FILE, {})
        for s in (deleted_slugs or set()):
            fresh.pop(s, None)
        for s, p in (updated_positions or {}).items():
            if s in fresh:
                for k, v in p.items():
                    fresh[s][k] = v
            else:
                fresh[s] = p
        save_json(POSITIONS_FILE, fresh)

def reconcile_positions():
    logger.info("[HERMES] Starting position reconciliation...")
    
    with _positions_file_lock:
        positions = load_json(POSITIONS_FILE, {})
        if not positions:
            logger.info("[HERMES] No positions to reconcile")
            return
        
        portfolio = get_portfolio()
        if portfolio is None:
            logger.error("[HERMES] Portfolio API failed, skipping reconciliation to avoid data loss")
            return
        if len(portfolio) == 0 and len(positions) > 0:
            logger.warning(f"[HERMES] Empty portfolio but {len(positions)} tracked positions — API may be down, skipping reconciliation")
            return
        
        portfolio_slugs = {p["market_slug"] for p in portfolio}
        open_orders = get_open_orders()
        
        deleted_slugs = set()
        updated_positions = {}
        
        for slug, pos_data in list(positions.items()):
            pos_modified = False
            
            if slug not in portfolio_slugs:
                if pos_data.get("in_emergency_exit"):
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
            
            current_shares = pos.get("shares", 0)
            recorded_shares = pos_data.get("shares", 0)
            
            if current_shares != recorded_shares:
                logger.info(f"[HERMES] Share mismatch for {slug[:40]}...: recorded={recorded_shares}, actual={current_shares}")
                pos_data["shares"] = current_shares
                pos_modified = True
            
            entry_price = pos.get("avg_entry_price", 0)
            if entry_price > 0 and pos_data.get("entry_price", 0) != entry_price:
                pos_data["entry_price"] = entry_price
                pos_modified = True
            
            tp_order = next((o for o in open_orders if o.get("slug") == slug and o.get("side") == "sell" and any(abs(o.get("price", 0) - p) < 0.01 for p in TP_LADDER_PRICES)), None)
            
            if tp_order:
                filled = tp_order.get("filled", 0)
                total = tp_order.get("shares", 0)
                
                if 0 < filled < total:
                    logger.warning(f"[HERMES] PARTIAL FILL for {slug[:40]}...: {filled}/{total}")
                    
                    if filled > 0:
                        fill_price = tp_order.get("price", TP_LIMIT_PRICE)
                        sold_value = filled * fill_price
                        pos_data["shares"] = current_shares
                        pos_data["partial_fills"] = pos_data.get("partial_fills", 0) + filled
                        pos_data["partial_proceeds"] = pos_data.get("partial_proceeds", 0) + sold_value
                        
                        logger.info(f"[HERMES] Updated shares to {current_shares} (portfolio already reflects partial fill), partial proceeds ${sold_value:.2f}")
                        pos_modified = True
                        
                        _notify_partial_fill(slug, pos_data, filled, fill_price)
            
            if pos_modified:
                updated_positions[slug] = pos_data
        
        if deleted_slugs or updated_positions:
            _merge_save_positions(deleted_slugs=deleted_slugs, updated_positions=updated_positions)
            logger.info(f"[HERMES] Positions updated: {len(deleted_slugs)} deleted, {len(updated_positions)} updated")

def _notify_position_closed(slug, pos_data):
    try:
        if TELEGRAM_REPORTER:
            TELEGRAM_REPORTER.alert_convergence(
                slug=slug,
                question=pos_data.get("question", "Unknown"),
                pnl_pct=pos_data.get("pnl_pct", 0) * 100,
                pnl_abs=pos_data.get("pnl_abs", 0),
                convergence_ratio=0
            )
    except Exception as e:
        logger.warning(f"[HERMES] Position closed notification failed: {e}")

def _notify_partial_fill(slug, pos_data, filled, fill_price=None):
    try:
        if TELEGRAM_REPORTER:
            fp = fill_price or TP_LIMIT_PRICE
            TELEGRAM_REPORTER.alert_take_profit(
                slug=slug,
                question=pos_data.get("question", "Unknown"),
                pnl_pct=((fp - pos_data.get("entry_price", 0)) / pos_data.get("entry_price", 1)) * 100 if pos_data.get("entry_price", 0) > 0 else 0,
                pnl_abs=filled * (fp - pos_data.get("entry_price", 0))
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
        cache = load_json(CACHE_FILE, {"metaculus": {}, "news": {}, "last_update": None})
        now = datetime.now()
        pruned = False
        for section in ("metaculus", "news"):
            entries = cache.get(section, {})
            stale = [k for k, v in entries.items()
                     if isinstance(v, dict) and v.get("timestamp")
                     and (now - datetime.fromisoformat(v["timestamp"])).total_seconds() > 86400]
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

    positions = load_json(POSITIONS_FILE, {})
    if not positions:
        logger.info("[HERMES] No positions to evaluate")
        return

    for slug, pos_data in positions.items():
        if pos_data.get("in_emergency_exit"):
            if pos_data.get("emergency_exit_failed"):
                last_attempt = pos_data.get("last_emergency_attempt")
                if last_attempt:
                    try:
                        elapsed = (datetime.now() - datetime.fromisoformat(last_attempt)).total_seconds()
                        if elapsed > 600:
                            logger.info(f"[HERMES] Retrying failed emergency exit for {slug[:40]}...")
                            pos_data.pop("emergency_exit_failed", None)
                            pos_data.pop("in_emergency_exit", None)
                            with _positions_file_lock:
                                _merge_save_positions(updated_positions={slug: pos_data})
                    except (ValueError, TypeError):
                        pass
            else:
                logger.info(f"[HERMES] Skipping {slug[:40]}... - already in emergency exit")
            continue

        if pos_data.get("selling_in_progress"):
            logger.info(f"[HERMES] Skipping {slug[:40]}... - sniper is selling")
            continue

        question = pos_data.get("market_question", "")
        if not question:
            continue

        portfolio = get_portfolio()
        if portfolio is None:
            logger.error("[HERMES] Portfolio API failed during emergency evaluation, skipping")
            return
        portfolio_map = {p.get("market_slug"): p for p in portfolio if p.get("market_slug")}

        bot_prob = pos_data.get("metaculus_prob") or (pos_data.get("entry_price", 0) * 2)
        bot_prob = min(max(bot_prob, 0.0), 1.0)

        entry_price = pos_data.get("entry_price", 0)
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

        prompt = f"""You are a risk analysis bot. Determine if recent news makes the outcome YES mathematically impossible (0% probability).

Market Question: {sanitize_for_prompt(question)}

Bot's estimated probability (p_bot): {bot_prob:.1%}

Recent News:
{news_text}

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

                should_send = _update_and_check_status(slug, trigger, normalized_status)

                if trigger:
                    logger.warning(f"[HERMES] EMERGENCY EXIT TRIGGERED for {slug[:40]}...: {reason}")
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
                            msg = f"⚠️ <b>HERMES STATUS CHANGE</b>\n\n"
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
                    posterior = update_posterior(slug, news_cat)
                    bayes_exit, bayes_reason = bayesian_should_exit(slug)
                    if bayes_exit:
                        logger.warning(f"[HERMES] BAYESIAN EXIT for {slug[:40]}...: {bayes_reason}")
                        _execute_emergency_exit(slug, pos_data, bayes_reason)
                except Exception as be:
                    logger.warning(f"[HERMES] Bayesian update failed for {slug}: {be}")
        except Exception as e:
            logger.error(f"[HERMES] LLM evaluation failed for {slug}: {e}")

def _execute_emergency_exit(slug, pos_data, reason):
    logger.info(f"[HERMES] Executing emergency exit for {slug[:40]}...")
    
    outcome = pos_data.get("outcome", "yes")
    shares = pos_data.get("shares", 0)
    
    with _positions_file_lock:
        positions = load_json(POSITIONS_FILE, {})
        
        if slug not in positions:
            logger.warning(f"[HERMES] {slug} not found in positions, aborting emergency")
            return
        
        positions[slug]["in_emergency_exit"] = True
        positions[slug]["selling_in_progress"] = True
        save_json(POSITIONS_FILE, positions)
    
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
            with _positions_file_lock:
                _merge_save_positions(updated_positions={slug: {
                    "shares": remaining_shares,
                    "in_emergency_exit": False,
                    "selling_in_progress": False,
                    "emergency_exit_failed": False,
                }})
        else:
            with _positions_file_lock:
                _merge_save_positions(deleted_slugs={slug})
            logger.info(f"[HERMES] Removed position {slug[:40]}... from file")
        
        entry_price = pos_data.get("entry_price", 0)
        current_price = pos_data.get("live_price", 0)
        actual_pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else -100
        actual_pnl_abs = (current_price - entry_price) * shares if entry_price > 0 and shares > 0 else 0
        
        if TELEGRAM_REPORTER:
            try:
                TELEGRAM_REPORTER.alert_stop_loss(
                    slug=slug,
                    question=pos_data.get("market_question", "Unknown"),
                    pnl_pct=actual_pnl_pct,
                    pnl_abs=actual_pnl_abs
                )
            except Exception as e:
                logger.warning(f"[HERMES] Emergency notification failed: {e}")
        
        _log_emergency_exit(slug, pos_data, reason)
    else:
        logger.error(f"[HERMES] Market sell FAILED for {slug[:40]}...")
        
        with _positions_file_lock:
            positions = load_json(POSITIONS_FILE, {})
            if slug in positions:
                positions[slug]["emergency_exit_failed"] = True
                positions[slug]["last_emergency_attempt"] = datetime.now().isoformat()
                positions[slug]["selling_in_progress"] = False
                positions[slug]["in_emergency_exit"] = False
            save_json(POSITIONS_FILE, positions)

def _log_emergency_exit(slug, pos_data, reason):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "slug": slug,
        "question": pos_data.get("market_question", ""),
        "entry_price": pos_data.get("entry_price", 0),
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

def main():
    logger.info("="*60)
    logger.info("  HERMES ADVISOR v5.3.7 - Starting")
    logger.info("="*60)
    
    reconcile_thread = threading.Thread(target=run_reconciliation_loop, daemon=True)
    emergency_thread = threading.Thread(target=run_emergency_evaluation_loop, daemon=True)
    
    reconcile_thread.start()
    emergency_thread.start()
    
    logger.info("[HERMES] Both loops started")
    
    try:
        while True:
            time.sleep(60)
            
            if not reconcile_thread.is_alive():
                logger.error("[HERMES] Reconciliation thread died, restarting")
                reconcile_thread = threading.Thread(target=run_reconciliation_loop, daemon=True)
                reconcile_thread.start()
            if not emergency_thread.is_alive():
                logger.error("[HERMES] Emergency thread died, restarting")
                emergency_thread = threading.Thread(target=run_emergency_evaluation_loop, daemon=True)
                emergency_thread.start()
            
            positions = load_json(POSITIONS_FILE, {})
            active_count = len([p for p in positions.values() if not p.get("in_emergency_exit")])
            
            logger.debug(f"[HERMES] Heartbeat: {active_count} active positions")
            
    except KeyboardInterrupt:
        logger.info("[HERMES] Shutting down...")
    except Exception as e:
        logger.error(f"[HERMES] Fatal error: {e}")

if __name__ == "__main__":
    main()
