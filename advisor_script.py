#!/usr/bin/env python3
"""
DOTM Advisor - Independent analysis script
Adaptive scheduling:
  - Long-term positions only (>30 days horizon): every 4 hours
  - Position near stop-loss (within 15%): elevated frequency
  - Mixed or short-term: default every 30 min
"""
import subprocess
import json
import logging
import os
import re
import sys
import fcntl
import html
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from utils import load_json, save_json, _normalize_keys, _strip_dict_keys_recursive, sanitize_for_prompt

load_json_safe = load_json
save_json_safe = save_json

logger = logging.getLogger(__name__)


def _load_notify_state():
    return load_json_safe(ADVISOR_NOTIFY_STATE_FILE, {"last_notified": {}})


def _save_notify_state(state):
    os.makedirs(os.path.dirname(ADVISOR_NOTIFY_STATE_FILE), exist_ok=True)
    save_json_safe(ADVISOR_NOTIFY_STATE_FILE, state)


def _should_notify(slug, verdict, p_diff, pnl_pct=0.0):
    if verdict == "WARNING":
        pass  # always notify on WARNING
    elif verdict == "DIVERGE":
        if p_diff <= 0.05:
            return False
    elif verdict not in ("CONFIRM", "UNKNOWN"):
        return False  # only block truly unknown verdicts
    if pnl_pct >= PROFITABLE_PNL_THRESHOLD and verdict != "WARNING":
        logger.info(f"[ADVISOR] {slug[:40]}... P&L={pnl_pct:.0%} >= {PROFITABLE_PNL_THRESHOLD:.0%}, suppressing non-critical alert")
        return False
    state = _load_notify_state()
    last_notified = state.get("last_notified", {}).get(slug)
    if last_notified:
        try:
            elapsed = (datetime.now() - datetime.fromisoformat(last_notified)).total_seconds()
            if elapsed < NOTIFY_COOLDOWN_SECONDS:
                logger.info(f"[ADVISOR] {slug[:40]}... cooldown {elapsed/60:.0f}m < {NOTIFY_COOLDOWN_SECONDS/60:.0f}m, suppressing")
                return False
        except (ValueError, TypeError):
            pass
    state.setdefault("last_notified", {})[slug] = datetime.now().isoformat()
    _save_notify_state(state)
    return True


def parse_llm_advisor_response(raw_text, log_label="ADVISOR"):
    """
    Parse LLM response into advisor analysis dict.

    Supports:
      - Clean JSON object
      - JSON inside ```json fenced code block
      - JSON after arbitrary preamble text
      - Extraction of first valid JSON object via brace balancing

    Validates schema:
      - p_estimate: float in [0, 1]
      - confidence: float in [0, 1]
      - factors: list of str
      - verdict: one of CONFIRM, DIVERGE, WARNING, UNKNOWN

    Returns (result_dict, error_reason).
    On success error_reason is None; on failure result_dict is None.
    """
    if not raw_text or not raw_text.strip():
        return None, "empty response"

    text = raw_text.strip()

    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    start = text.find('{')
    if start == -1:
        return None, "no '{' found in response"

    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError as e:
                    logger.warning(f"{log_label} JSON decode failed at brace-balanced span: {e}")
                    break
                err = _validate_advisor_schema(obj, log_label)
                if err:
                    return None, err
                return obj, None

    fallback_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if fallback_match:
        try:
            obj = json.loads(fallback_match.group(0))
            err = _validate_advisor_schema(obj, log_label)
            if err:
                return None, err
            return obj, None
        except json.JSONDecodeError as e:
            logger.warning(f"{log_label} fallback regex JSON decode failed: {e}")

    return None, "no valid JSON object found after all extraction attempts"


VALID_VERDICTS = {"CONFIRM", "DIVERGE", "WARNING", "UNKNOWN"}


def _validate_advisor_schema(obj, log_label="ADVISOR"):
    """
    Validate advisor analysis schema. Returns error string or None.
    """
    if not isinstance(obj, dict):
        return f"parsed value is {type(obj).__name__}, expected dict"

    errors = []

    p = obj.get("p_estimate")
    if p is None:
        errors.append("missing p_estimate")
    else:
        try:
            p = float(p)
            if not (0.0 <= p <= 1.0):
                errors.append(f"p_estimate={p} out of [0,1]")
        except (ValueError, TypeError):
            errors.append(f"p_estimate={p!r} is not a number")

    c = obj.get("confidence")
    if c is None:
        errors.append("missing confidence")
    else:
        try:
            c = float(c)
            if not (0.0 <= c <= 1.0):
                errors.append(f"confidence={c} out of [0,1]")
        except (ValueError, TypeError):
            errors.append(f"confidence={c!r} is not a number")

    factors = obj.get("factors")
    if factors is None:
        errors.append("missing factors")
    elif not isinstance(factors, list) or not all(isinstance(f, str) for f in factors):
        errors.append(f"factors is {type(factors).__name__}, expected list[str]")

    verdict = obj.get("verdict")
    if verdict is None:
        errors.append("missing verdict")
    elif verdict not in VALID_VERDICTS:
        errors.append(f"verdict={verdict!r} not in {VALID_VERDICTS}")

    if errors:
        msg = "; ".join(errors)
        logger.warning(f"{log_label} schema validation failed: {msg}")
        return f"schema invalid: {msg}"
    return None

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL = "deepseek-reasoner"
URL = "https://api.deepseek.com/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def _load_env():
    env_path = "/root/dotm-sniper/.env"
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ.setdefault(key.strip(), val.strip())

_load_env()
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def get_bot_status():
    try:
        return load_json('/root/.openclaw/workspace/dotm_status.json', {"portfolio": [], "balance": {}})
    except:
        return {"portfolio": [], "balance": {}}

def get_positions_tracking():
    return load_json_safe('/root/dotm-sniper/positions.json', {})

def analyze_market(market_slug, market_question, current_price, entry_price):
    prompt = f"""You are DOTM Advisor - independent analyst comparing analysis with a trading bot.

Market: {sanitize_for_prompt(market_question)}
Current market price: ${current_price:.3f}
Bot entry price: ${entry_price:.3f}
P&L since entry: {(current_price - entry_price) / entry_price * 100:.1f}%

Your task:
1. Estimate TRUE probability (0.0 to 1.0) - be honest, not conservative
2. Identify 2 key factors that could change the thesis (be specific, not generic)
3. State confidence (0.5 to 0.95)

Return ONLY JSON:
{{"p_estimate": 0.XX, "confidence": 0.XX, "factors": ["factor1", "factor2"], "verdict": "CONFIRM/DIVERGE/WARNING"}}"""

    try:
        import requests
        resp = requests.post(URL, headers=HEADERS, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 2000
        }, timeout=120)

        data = resp.json()
        msg = data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "")
        if not content:
            content = msg.get("reasoning", "")

        if content:
            result, parse_err = parse_llm_advisor_response(content, log_label="ANALYSIS")
            if result is not None:
                if result.get("factors") == ["Fallback estimate"]:
                    result["factors"] = ["Market sentiment unchanged", "No material news events"]
                return result
            logger.warning(f"ANALYSIS JSON parse failed, using fallback: {parse_err}")
    except json.JSONDecodeError as e:
        logger.warning(f"ANALYSIS LLM response JSON error: {e}")
    except Exception as e:
        logger.error(f"ANALYSIS LLM error {e}")

    p_fallback = min(current_price * 1.5, 0.35)
    return {"p_estimate": p_fallback, "confidence": 0.55, "factors": ["Insufficient data for independent estimate", "Using market price as baseline"], "verdict": "UNKNOWN"}

def get_hypothesis_p_model(slug):
    try:
        db = load_json('/root/dotm-sniper/hypothesis_db.json', {})
        for h in db.get('hypotheses', []):
            if h['slug'] == slug and not h.get('resolved'):
                return h.get('p_model')
        for h in db.get('resolved', []):
            if h['slug'] == slug:
                return h.get('p_model')
    except:
        pass
    return None

def format_position_analysis(pos, hermes_data):
    slug = pos.get('market_slug', '')
    question = pos.get('market_question', '')
    current_price = pos.get('live_price', 0)
    entry_price = pos.get('avg_entry_price', 0)
    pnl_pct = pos.get('percent_pnl', 0)
    shares = pos.get('shares', 0)

    verdict = hermes_data.get('verdict', 'UNKNOWN')
    p_diff = abs(hermes_data.get('p_estimate', 0) - current_price)
    p_estimate = hermes_data.get('p_estimate', 0)

    if verdict == 'CONFIRM':
        emoji = "✅"
        alert_level = "GREEN"
    elif verdict == 'DIVERGE':
        emoji = "⚠️"
        alert_level = "YELLOW"
    elif verdict == 'WARNING':
        emoji = "🚨"
        alert_level = "RED"
    else:
        emoji = "➖"
        alert_level = "GRAY"

    msg = f"{emoji} <b>ADVISOR ANALYSIS</b>\n\n"
    msg += f"📌 {html.escape(question[:55])}...\n\n"
    msg += f"💵 Entry: ${entry_price:.3f} | Current: ${current_price:.3f} | P&L: {pnl_pct:+.1f}%\n"
    p_bot = get_hypothesis_p_model(slug) or (entry_price * 2)
    msg += f"🤖 Bot estimated P: {p_bot:.0%} | 🔍 Hermes P: {p_estimate:.0%}\n"

    if p_diff > 0.15:
        gap_direction = "HIGHER" if p_estimate > current_price else "LOWER"
        msg += f"\n⚠️ DIVERGENCE: Hermes sees {gap_direction} probability\n"
    elif p_diff <= 0.05:
        msg += f"\n✅ ALIGNED - thesis confirmed\n"

    if hermes_data.get('factors'):
        msg += f"\n📊 Factors:\n"
        for f in hermes_data['factors'][:2]:
            msg += f"   • {html.escape(f)}\n"

    msg += f"\n⚡ Status: {alert_level}"

    return msg

def send_telegram(message):
    token = os.environ.get("TG_BOT_TOKEN", "")
    chat_id = os.environ.get("TG_CHAT_ID", "")
    if not token or not chat_id:
        return False
    try:
        import requests
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10
        )
        return r.ok
    except Exception:
        return False

def _should_skip_long_term(positions_tracking):
    """
    TAZ-4: Check if all positions are long-term only.
    If so, advisor should run every 4 hours instead of 30 min.
    Returns (should_skip: bool, has_near_stop: bool)
    """
    state = load_json_safe(ADVISOR_STATE_FILE, {"last_run": {}, "schedule": "default"})

    portfolio = []
    try:
        res = subprocess.run(["pm-trader", "portfolio"], capture_output=True, text=True, timeout=15)
        portfolio = json.loads(res.stdout).get("data", [])
    except Exception:
        pass

    if not portfolio and not positions_tracking:
        return False, False

    now = datetime.now()
    all_long_term = True
    has_near_stop = False

    for pos in portfolio:
        slug = pos.get("market_slug", "")
        entry_price = pos.get("avg_entry_price", 0)
        live_price = pos.get("live_price", 0)

        if entry_price <= 0 or live_price <= 0:
            all_long_term = False
            continue

        pnl_pct = (live_price - entry_price) / entry_price

        pos_data = positions_tracking.get(slug, {})
        stop_loss = pos_data.get("stop_loss", 0)

        if stop_loss > 0:
            distance_to_stop = (live_price - stop_loss) / live_price if live_price > 0 else 1.0
            if distance_to_stop < STOP_LOSS_PROXIMITY_PCT:
                has_near_stop = True
                logger.info(
                    f"[ADVISOR-SCHED] {slug[:30]}... near stop-loss: "
                    f"price={live_price:.4f} stop={stop_loss:.4f} distance={distance_to_stop:.1%}"
                )

        end_date_str = pos.get("end_date", "")
        if end_date_str:
            try:
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                days_left = max(0, (end_dt - now).total_seconds() / 86400)
                if days_left <= LONG_TERM_THRESHOLD_DAYS:
                    all_long_term = False
            except (ValueError, TypeError):
                all_long_term = False
        else:
            all_long_term = False

    if has_near_stop:
        state["schedule"] = "elevated"
        required_interval = ELEVATED_INTERVAL_SECONDS
    elif all_long_term:
        state["schedule"] = "long_term"
        required_interval = LONG_TERM_INTERVAL_SECONDS
    else:
        state["schedule"] = "default"
        required_interval = DEFAULT_INTERVAL_SECONDS

    last_run_iso = state.get("last_run", {}).get("global")
    if last_run_iso:
        try:
            last_run_dt = datetime.fromisoformat(last_run_iso)
            elapsed = (now - last_run_dt).total_seconds()
            if elapsed < required_interval:
                logger.info(
                    f"[ADVISOR-SCHED] schedule={state['schedule']}, "
                    f"elapsed={elapsed/60:.0f}m < required={required_interval/60:.0f}m, skipping"
                )
                return True, has_near_stop
        except (ValueError, TypeError):
            pass

    return False, has_near_stop


def _update_advisor_state(analyzed_count, schedule):
    state = load_json_safe(ADVISOR_STATE_FILE, {"last_run": {}, "schedule": "default"})
    state["last_run"]["global"] = datetime.now().isoformat()
    state["schedule"] = schedule
    state["last_analyzed_count"] = analyzed_count
    save_json_safe(ADVISOR_STATE_FILE, state)


def run_advisory_cycle():
    positions_tracking = get_positions_tracking()

    should_skip, has_near_stop = _should_skip_long_term(positions_tracking)
    if should_skip:
        print("Advisory cycle skipped (schedule throttle)")
        return

    state = load_json_safe(ADVISOR_STATE_FILE, {"last_run": {}, "schedule": "default"})
    schedule = state.get("schedule", "default")

    status = get_bot_status()
    portfolio = status.get('portfolio', [])

    if not portfolio:
        print("No open positions to analyze")
        _update_advisor_state(0, schedule)
        return

    analyzed = 0
    alerts = []

    for pos in portfolio:
        slug = pos.get('market_slug', '')
        question = pos.get('market_question', '')
        current_price = pos.get('live_price', 0)
        entry_price = pos.get('avg_entry_price', 0)

        if current_price <= 0 or entry_price <= 0:
            continue

        hermes = analyze_market(slug, question, current_price, entry_price)
        alert = format_position_analysis(pos, hermes)

        verdict = hermes.get('verdict', 'UNKNOWN')
        p_diff = abs(hermes.get('p_estimate', 0) - current_price)
        pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
        if _should_notify(slug, verdict, p_diff, pnl_pct):
            send_telegram(alert)

        print(alert)
        print("---")
        alerts.append(alert)
        analyzed += 1

    print(f"\n✅ Analyzed {analyzed} positions")
    _update_advisor_state(analyzed, schedule)

if __name__ == "__main__":
    run_advisory_cycle()