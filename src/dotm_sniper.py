#!/usr/bin/env python3
"""
DOTM Sniper v5.3.0 - Adaptive Signal Thresholds
Based on the mathematical edge of Deep Out-The-Money trading

v5.3.0 Changelog:
- Per-horizon signal thresholds (short/medium/long) from bot_settings
- Lowered PRICE_DELTA_THRESHOLD $0.005 -> $0.002 for DOTM sensitivity
- Added [SIGNAL-BATCH] logging for batch analysis visibility
- Fixed Renan Santos missing from hypothesis_db
"""
import subprocess, json, requests, time, re, os, sys, logging, fcntl
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotm_report import TelegramReporter

from news_scanner import check_market_news, extract_keywords, fetch_recent_news
from utils import load_json, save_json, _lock_file, _unlock_file, _normalize_keys, _strip_dict_keys_recursive, sanitize_for_prompt, check_and_write_pid, cleanup_pid_file

PID_FILE = "/root/dotm-sniper/sniper.pid"

def _load_env_manual():
    env_path = "/root/dotm-sniper/.env"
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ.setdefault(key.strip(), val.strip())

_load_env_manual()

telegram_reporter = TelegramReporter()

LOG_FILE = "/root/dotm-sniper/sniper.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL_LIGHT = "deepseek-chat"
MODEL_MAIN = "deepseek-chat"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
URL = "https://api.deepseek.com/v1/chat/completions"
METACULUS_API_KEY = os.environ.get("METACULUS_TOKEN", "")
METACULUS_URL = "https://www.metaculus.com/api2/questions/"
METACULUS_HEADERS = {"Authorization": f"Token {METACULUS_API_KEY}"}

MIN_PROB_RATIO = 3.0
MIN_P_MODEL = 0.05
MAX_P_MODEL_RATIO = 5.0  # p_model cannot exceed market_price * this ratio
MIN_CONFIDENCE = 0.65
MAX_POS_PCT = 0.10
FRACTIONAL_KELLY_MULTIPLIER = 0.25
BASE_POS_PCT = 0.02
OTHER_BOOST_POS_PCT = 0.035
MAX_EXPOSURE_PER_CATEGORY = 0.20
DISPERSION_PENALTY_THRESHOLD = 0.25
METACULUS_GAP_THRESHOLD = 0.08
MIN_VOLUME = 25000
MIN_TTL_HOURS = 48
MAX_PRICE = 0.30
MAX_CLUSTER_PCT = 0.30
MAX_POSITIONS = 5
BURN_IN_TRADES = 50
HARD_STOP_LOSS = -0.30
TRAILING_ACTIVATION = 0.30
TRAILING_STOP = 0.25
TAKE_PROFIT = 2.00
CONVERGENCE_TAKE_PROFIT = 0.90
MIN_POSITION_CHECK_INTERVAL_HOURS = 3

# v5.1.0: Smart Exit - automatic TP limit orders at $0.85
SMART_EXIT_PRICE = 0.85
SMART_EXIT_SLIPPAGE = 0.015  # $0.015 slippage penalty for backtesting
ALLOWED_CLUSTERS = {"ai_tech", "russia_ukraine", "usa_politics", "fed_fomc", "sports_nba", "sports_ufc"}
BANNED_CLUSTERS = {"crypto"}

HYPOTHESIS_DB = "/root/dotm-sniper/hypothesis_db.json"
POSITIONS_FILE = "/root/dotm-sniper/positions.json"
SETTINGS_FILE = "/root/dotm-sniper/bot_settings.json"
CACHE_FILE = "/root/dotm-sniper/source_cache.json"
PRICE_TRACKING_FILE = "/root/dotm-sniper/price_tracking.json"
BACKTEST_STATS_FILE = "/root/dotm-sniper/backtest_stats.json"

DAILY_STATS_FILE = "/root/dotm-sniper/daily_stats.json"

BATCH_SIZE = 6
PRICE_DELTA_THRESHOLD = 0.002
CACHE_TTL_SECONDS = 21600
CALIBRATION_DAMPING_FACTOR = 0.65
CALIBRATION_DOTM_THRESHOLD = 0.10
CALIBRATION_AGGRESSIVE_PMODEL = 0.20
CALIBRATION_METACULUS_LOW = 0.10

MIN_TRADES_FOR_WEIGHT_ADJUSTMENT = 20
BAYESIAN_PRIOR_STRENGTH = 10
BACKTEST_COOLDOWN_SECONDS = 24 * 3600

CLUSTER_SCORE_ADJUSTMENTS = {
    "other": 15,
    "crypto": -25,
    "sports_nba": -15,
}


def calibrate_prediction(p_model, market_price, metaculus_prob=None, cluster=None):
    """
    Calibrate p_model using isotonic regression (if available) or fallback to dampening.

    Priority:
    1. Isotonic calibrator (if fitted on >= 20 samples)
    2. Legacy dampening (0.65x multiplier for DOTM markets)

    Returns (calibrated_p_model, was_calibrated: bool)
    """
    from calibration import get_calibrator

    calibrator = get_calibrator()
    if calibrator.is_fitted:
        p_calibrated = calibrator.predict(p_model, cluster or "other")
        if abs(p_calibrated - p_model) > 0.02:
            logger.info(
                f"[CALIBRATION] p_model={p_model:.1%} -> {p_calibrated:.1%} "
                f"(cluster={cluster}, isotonic)"
            )
        return p_calibrated, True

    if cluster == "other":
        return p_model, False

    if market_price > CALIBRATION_DOTM_THRESHOLD:
        return p_model, False

    if p_model <= CALIBRATION_AGGRESSIVE_PMODEL:
        return p_model, False

    meta_low = metaculus_prob is None or metaculus_prob < CALIBRATION_METACULUS_LOW
    if not meta_low:
        return p_model, False

    calibrated = p_model * CALIBRATION_DAMPING_FACTOR
    logger.info(
        f"[DAMPEN] p_model={p_model:.1%} -> {calibrated:.1%} "
        f"(price=${market_price:.3f}, metaculus={'none' if metaculus_prob is None else f'{metaculus_prob:.1%}'})"
    )
    return calibrated, True


def _cluster_score_adjustment(cluster):
    """
    Returns point adjustment to signal_score based on backtest-calibrated cluster weights.
    Reads from bot_settings.json cluster_score_adjustments, falls back to defaults.
    """
    settings = get_settings()
    adjustments = settings.get("cluster_score_adjustments", CLUSTER_SCORE_ADJUSTMENTS)
    adj = adjustments.get(cluster, 0)
    if adj != 0:
        logger.info(f"[CLUSTER-ADJ] {cluster}: {adj:+d} to signal_score")
    return adj


def update_daily_stats(balance, portfolio, trades_this_cycle):
    today = datetime.now().strftime("%Y-%m-%d")
    stats = load_json(DAILY_STATS_FILE, {"date": today, "trades": 0, "pnl": 0, "started": False})
    if stats.get("date") != today:
        stats = {"date": today, "trades": 0, "pnl": 0, "started": False}
    stats["started"] = True
    stats["trades"] = stats.get("trades", 0) + trades_this_cycle
    starting = get_settings().get("starting_balance", 500.0)
    stats["pnl"] = balance.get("total_value", 0) - starting
    save_json(DAILY_STATS_FILE, stats)

def load_cache():
    cache = load_json(CACHE_FILE, {"metaculus": {}, "news": {}, "last_update": None})
    now = datetime.now()
    for section in ("metaculus", "news"):
        entries = cache.get(section, {})
        stale = [k for k, v in entries.items()
                 if isinstance(v, dict) and v.get("timestamp")
                 and (now - datetime.fromisoformat(v["timestamp"])).total_seconds() > 86400]
        for k in stale:
            del entries[k]
    return cache

def normalize_probability(p):
    if p is None:
        return 0
    p = float(p)
    if p > 1.0:
        p = p / 100.0
    return max(0.0, min(1.0, p))

def parse_llm_json(response_text):
    start = response_text.find('{')
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(response_text)):
        c = response_text[i]
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
                try:
                    return json.loads(response_text[start:i + 1])
                except json.JSONDecodeError:
                    pass
                break
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None

def save_cache(cache):
    cache["last_update"] = datetime.now().isoformat()
    save_json(CACHE_FILE, cache)

def metaculus_search(query, limit=10):
    try:
        resp = requests.get(METACULUS_URL, headers=METACULUS_HEADERS,
                          params={"search": query, "limit": limit, "status": "open"},
                          timeout=15)
        if resp.status_code == 200:
            return resp.json().get("results", [])
    except:
        pass
    return []

def metaculus_get_question(qid):
    try:
        resp = requests.get(f"{METACULUS_URL}{qid}/", headers=METACULUS_HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None

def parse_resolve_date(date_str):
    if not date_str:
        return None
    date_str = date_str.replace("Z", "+00:00")
    try:
        from dateutil.parser import parse
        return parse(date_str)
    except:
        pass
    try:
        return datetime.fromisoformat(date_str)
    except:
        pass
    return None

DATE_WINDOW_DAYS = 7

def dates_match(date1, date2, window_days=DATE_WINDOW_DAYS):
    d1 = parse_resolve_date(date1)
    d2 = parse_resolve_date(date2)
    if d1 is None or d2 is None:
        return False
    try:
        diff = abs((d1 - d2).total_seconds() / 86400)
    except TypeError:
        d1 = d1.replace(tzinfo=None) if d1.tzinfo else d1
        d2 = d2.replace(tzinfo=None) if d2.tzinfo else d2
        diff = abs((d1 - d2).total_seconds() / 86400)
    return diff <= window_days

def get_metaculus_forecast(pm_question, pm_resolve_date=None):
    cache = load_cache()
    cache_key = pm_question[:60]

    if cache_key in cache.get("metaculus", {}):
        cached = cache["metaculus"][cache_key]
        if cached.get("timestamp"):
            cached_time = datetime.fromisoformat(cached["timestamp"])
            if (datetime.now() - cached_time).total_seconds() < 3600:
                return cached

    # Better search: use full question as query, also try key fragments
    search_queries = [pm_question] + _generate_search_queries(pm_question)

    best_match = None
    raw_best_score = 0
    for query in search_queries:
        results = metaculus_search(query, limit=10)
        if not results:
            continue
        for r in results:
            score = _calculate_metaculus_match(pm_question, r)
            if score > raw_best_score:
                raw_best_score = score
                best_match = r
        if best_match and raw_best_score >= 0.40:
            break

    if not best_match or raw_best_score < 0.30:
        return {"found": False, "probability": None, "reason": "no_title_match", "best_score": raw_best_score}

    # Skip slow NLP verify - our scoring is sufficient
    meta_title = best_match.get("title", "") or best_match.get("short_title", "")

    q_data = best_match.get("question", {})
    if not q_data:
        q_data = best_match

    if pm_resolve_date:
        meta_resolve = q_data.get("scheduled_resolve_time") or best_match.get("resolve_date")
        if meta_resolve and not dates_match(pm_resolve_date, meta_resolve):
            return {"found": False, "probability": None, "reason": "date_mismatch",
                    "meta_date": meta_resolve, "pm_date": pm_resolve_date}

    qid = q_data.get("id") or best_match.get("id")

    cp_reveal = q_data.get("cp_reveal_time")
    if cp_reveal:
        try:
            reveal_dt = datetime.fromisoformat(cp_reveal.replace("Z", "+00:00"))
            if datetime.now(reveal_dt.tzinfo) < reveal_dt:
                return {"found": False, "probability": None, "reason": "cp_not_revealed"}
        except:
            pass

    agg_data = q_data.get("aggregations") if q_data.get("aggregations") is not None else {}
    agg = agg_data.get("recency_weighted") if agg_data.get("recency_weighted") is not None else {}
    latest = agg.get("latest") if agg and isinstance(agg, dict) else None

    if not latest:
        full_q = metaculus_get_question(qid)
        if full_q:
            q_inner = full_q.get("question", {})
            inner_agg_data = q_inner.get("aggregations") if q_inner.get("aggregations") is not None else {}
            agg = inner_agg_data.get("recency_weighted") if inner_agg_data.get("recency_weighted") is not None else {}
            latest = agg.get("latest") if agg and isinstance(agg, dict) else None

    # Metaculus returns null aggregations for most questions - try alternative paths
    prob = None
    if latest:
        means = latest.get("means", [])
        if means:
            prob = float(means[0])

    # Fallback: check for prediction field in question data
    if prob is None:
        pred = q_data.get("prediction") or best_match.get("prediction")
        if pred and isinstance(pred, dict):
            prob = float(pred.get("number") or pred.get("p_above") or pred.get("p_below") or 0)

    # Fallback 2: check for community forecast in vote data
    if prob is None:
        vote = best_match.get("vote", {})
        if vote and isinstance(vote, dict):
            prob = float(vote.get("prediction", 0))

    if prob is None:
        return {"found": False, "probability": None, "reason": "no_aggregation", "best_match_title": meta_title}
    forecaster_count = latest.get("forecaster_count", 0)
    title = best_match.get("title", "") or best_match.get("short_title", "")

    q1 = latest.get("q1")
    q3 = latest.get("q3")
    std = latest.get("std")
    dispersion = None
    dispersion_penalty = 1.0

    if q1 is not None and q3 is not None:
        dispersion = q3 - q1
    elif std is not None:
        dispersion = std

    if dispersion is not None and dispersion > DISPERSION_PENALTY_THRESHOLD:
        dispersion_penalty = max(0.0, 1.0 - (dispersion - DISPERSION_PENALTY_THRESHOLD))
        logger.info(f"[DISPERSION] q1={q1:.3f}, q3={q3:.3f}, dispersion={dispersion:.3f}, penalty={dispersion_penalty:.2f}")

    result = {
        "found": True,
        "probability": prob,
        "question_title": title,
        "url": f"https://www.metaculus.com/questions/{qid}/",
        "forecaster_count": forecaster_count,
        "timestamp": datetime.now().isoformat(),
        "dispersion": dispersion,
        "dispersion_penalty": dispersion_penalty
    }

    cache.setdefault("metaculus", {})[cache_key] = result
    save_cache(cache)
    return result


def _generate_search_queries(question):
    """Generate multiple search queries from a question."""
    words = question.replace("?", " ").replace(",", " ").split()
    queries = []
    # Try 2-4 word combinations
    for i in range(len(words)):
        for j in range(i+1, min(i+5, len(words)+1)):
            phrase = " ".join(words[i:j])
            if len(phrase) >= 4:
                queries.append(phrase)
    return queries[:5]


def _calculate_metaculus_match(pm_question, result):
    """Calculate relevance score between PM question and Metaculus result."""
    from fuzzywuzzy import fuzz
    pm_lower = pm_question.lower()
    meta_title = (result.get("title", "") or result.get("short_title", "")).lower()
    
    # Score components
    pm_words = set(re.sub(r'[^\w\s]', ' ', pm_lower).split())
    meta_words = set(re.sub(r'[^\w\s]', ' ', meta_title).split())
    
    # 1. Word overlap (basic)
    stop = {"will", "the", "a", "an", "be", "by", "of", "in", "on", "at", "to", "for", "is", "are", "was", "were", "before", "after", "any", "this", "that", "before"}
    pm_clean = pm_words - stop
    meta_clean = meta_words - stop
    
    overlap = len(pm_clean & meta_clean)
    base_score = overlap / max(len(pm_clean), 1) if pm_clean else 0
    
    # 2. Substring bonus - important for multi-word concepts
    key_phrases = ["ai safety", "artificial intelligence", "anthropic", "ukraine", "nato", "nuclear", "china", "taiwan", "trump", "fed", "powell", "bitcoin"]
    substring_bonus = 0
    for phrase in key_phrases:
        if phrase in pm_lower and phrase in meta_title:
            substring_bonus += 0.15
        elif phrase in pm_lower and phrase.split()[0] in meta_title:
            substring_bonus += 0.05
    
    # 3. Number/date match bonus
    pm_nums = set(w for w in pm_clean if any(c.isdigit() for c in w))
    meta_nums = set(w for w in meta_clean if any(c.isdigit() for c in w))
    if pm_nums and meta_nums and pm_nums & meta_nums:
        base_score += 0.1

    # 4. Fuzzy matching bonus
    similarity = fuzz.partial_ratio(pm_lower, meta_title) / 100.0
    if similarity > 0.70:
        base_score += 0.15
    
    return min(base_score + substring_bonus, 1.0)

def get_time_decay_threshold(end_date_str):
    """
    Time-Decay: dynamic gap threshold based on days to resolution.

    Markets with longer time horizons should require a larger gap to compensate
    for uncertainty that accumulates over time. Short-term markets can be acted
    upon with smaller gaps since there's less time for the thesis to break down.

    Rules:
      - days_to_resolution > 30: threshold = 0.20 (generous window, high uncertainty)
      - 8 <= days <= 30: threshold = 0.15
      - 3 <= days <= 7: threshold = 0.10
      - 1 <= days <= 2: threshold = 0.05 (near-term, less uncertainty)
      - otherwise: use default METACULUS_GAP_THRESHOLD
    """
    if not end_date_str:
        return METACULUS_GAP_THRESHOLD

    try:
        end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        now = datetime.now(end_dt.tzinfo) if end_dt.tzinfo else datetime.now()
        days_to_res = max(0, (end_dt - now).total_seconds() / 86400)
    except Exception:
        return METACULUS_GAP_THRESHOLD

    if days_to_res > 30:
        threshold = 0.20
    elif days_to_res >= 8:
        threshold = 0.15
    elif days_to_res >= 3:
        threshold = 0.10
    elif days_to_res >= 1:
        threshold = 0.05
    else:
        threshold = 0.03 + (days_to_res * 0.02)

    logger.info(f"[TIME-DECAY] days_to_res={days_to_res:.1f}, threshold={threshold:.2f}")
    return threshold


def check_metaculus_gap(market, polymarket_prob=None):
    """
    Check if Metaculus probability significantly differs from Polymarket price.

    Uses two modifiers:
      1. Time-Decay: adjusts the required gap threshold based on days to resolution
      2. Dispersion Penalty: reduces signal strength when Metaculus forecasters are polarized

    Returns a gap signal dict if conditions are met, otherwise None.
    """
    meta = get_metaculus_forecast(market["question"], market.get("end_date"))

    if not meta.get("found"):
        return None

    metaculus_prob = meta.get("probability", 0)
    price_to_use = polymarket_prob if polymarket_prob is not None else market["price"]

    # Time-Decay: dynamic threshold
    required_gap = get_time_decay_threshold(market.get("end_date"))

    gap = metaculus_prob - price_to_use

    if gap <= required_gap:
        logger.info(f"[GAP-SKIP] gap={gap:.3f} <= required={required_gap:.3f} ({market['question'][:40]})")
        return None

    # Apply dispersion penalty from Metaculus forecaster consensus
    dispersion_penalty = meta.get("dispersion_penalty", 1.0)

    # raw signal strength based purely on gap magnitude
    raw_strength = min(gap / 0.15, 1.0)

    # Adjust for dispersion: high dispersion = weaker signal
    signal_strength = raw_strength * dispersion_penalty

    logger.info(
        f"[GAP-APPROVED] meta={metaculus_prob:.0%} vs pm={price_to_use:.0%} | "
        f"gap={gap:.3f} required={required_gap:.3f} | "
        f"disp_penalty={dispersion_penalty:.2f} => signal={signal_strength:.2f}"
    )

    return {
        "source": "metaculus",
        "metaculus_prob": metaculus_prob,
        "polymarket_prob": price_to_use,
        "gap": gap,
        "required_gap": required_gap,
        "signal_strength": signal_strength,
        "dispersion_penalty": dispersion_penalty,
        "reasoning": (
            f"Metaculus {metaculus_prob:.0%} vs Polymarket {price_to_use:.0%}: "
            f"gap={gap:.0%}, dispersion_penalty={dispersion_penalty:.2f}"
        )
    }

def extract_keywords(question):
    stop_words = {"will", "the", "a", "an", "be", "by", "of", "in", "on", "at", "to", "for", "this", "that", "is", "are", "was", "were"}
    words = re.findall(r'\b[a-zA-Z]{4,}\b', question.lower())
    return [w for w in words if w not in stop_words][:10]



def get_settings():
    s = load_json(SETTINGS_FILE, {
        "min_confidence": MIN_CONFIDENCE,
        "position_size_pct": MAX_POS_PCT,
        "calibration_brier": None,
        "total_resolved": 0,
        "signal_threshold": 55,
        "min_p_model": MIN_P_MODEL
    })
    return s

def save_settings(s):
    s["__version"] = s.get("__version", 0) + 1
    save_json(SETTINGS_FILE, s)

def load_hypothesis_db():
    db = load_json(HYPOTHESIS_DB, {"hypotheses": [], "resolved": []})
    dirty = False
    active = [h for h in db.get("hypotheses", []) if not h.get("resolved")]
    if len(active) != len(db.get("hypotheses", [])):
        db["hypotheses"] = active
        dirty = True
    resolved_slugs = {h["slug"] for h in db.get("resolved", [])}
    deduped = []
    seen = set()
    for h in db.get("resolved", []):
        if h["slug"] not in seen:
            deduped.append(h)
            seen.add(h["slug"])
    if len(deduped) != len(db.get("resolved", [])):
        db["resolved"] = deduped
        dirty = True
    if dirty:
        save_hypothesis_db(db)
    return db

def save_hypothesis_db(db):
    MAX_RESOLVED = 1000
    if len(db.get("resolved", [])) > MAX_RESOLVED:
        db["resolved"] = db["resolved"][-MAX_RESOLVED:]
    save_json(HYPOTHESIS_DB, db)

def detect_clusters(question):
    question_lower = question.lower()
    found = set()
    for cluster, keywords in CLUSTER_KEYWORDS.items():
        for kw in keywords:
            if ' ' in kw:
                if kw in question_lower:
                    found.add(cluster)
                    break
            else:
                if re.search(r'(?:^|\s)' + re.escape(kw) + r'(?:\s|$|[^a-z])', question_lower):
                    found.add(cluster)
                    break
    return list(found) if found else ["other"]

def check_cluster_limits(new_clusters, current_positions):
    cluster_exposure = defaultdict(float)
    for pos in current_positions:
        for c in pos.get("clusters", []):
            cluster_exposure[c] += pos.get("size_pct", 0)

    for cluster in new_clusters:
        if cluster_exposure.get(cluster, 0) >= MAX_CLUSTER_PCT:
            return False, f"Cluster {cluster} limit reached ({cluster_exposure[cluster]:.1%})"
    return True, "OK"

CLUSTER_KEYWORDS = {
    "venezuela": {"venezuela", "maduro", "caracas", "chavez", "bolivar"},
    "russia_ukraine": {"russia", "ukraine", "putin", "zelensky", "kremlin", "moscow", "kyiv", "nato", "war in ukraine", "russian invasion", "ceasefire", "peace deal", "peace talks", "territor", "donbas", "crimea", "donetsk"},
    "usa_politics": {"trump", "biden", "republican", "democratic", "congress", "senate", "house", "election", "president", "white house", "greenland", "tariff", "executive order", "governor", "primary", "nominee"},
    "fed_fomc": {"fed", "federal reserve", "fomc", "powell", "interest rate", "monetary", "s&p", "sp 500", "sp500", "recession", "inflation", "treasury", "stock market", "spy"},
    "sports_nba": {"nba", "basketball", "lakers", "warriors", "celtics"},
    "sports_ufc": {"ufc", "mma", "fight", "boxing", "fighter"},
    "crypto": {"bitcoin", "ethereum", "crypto", "btc", "eth", "blockchain", "solana", "monero"},
    "ai_tech": {"ai safety", "ai bill", "artificial intelligence", "openai", "google deepmind", "microsoft ai", "anthropic", "gpt", "llm", "bytedance", "ipo market cap"},
}

# ============================================================
# PORTFOLIO EXPOSURE: Track correlated risks by category
# ============================================================
def get_category_exposure(balance, portfolio=None):
    """
    Calculate current dollar exposure per category (tag) from open positions.

    Categories are derived from Polymarket market tags (e.g., "Politics", "Crypto",
    "Economics", "Sports"). This allows us to enforce MAX_EXPOSURE_PER_CATEGORY limit
    and avoid over-concentration in any single thematic area.

    Returns a dict: {category_name: dollar_exposure}
    """
    if portfolio is None:
        portfolio = get_portfolio()

    exposure = defaultdict(float)

    for pos in portfolio:
        shares_value = pos.get("current_value", 0) or pos.get("live_value", 0)
        slug = pos.get("market_slug", "")
        question = pos.get("market_question", "")

        slug_lower = slug.lower()
        question_lower = question.lower()
        detected_categories = set()
        for cluster, keywords in CLUSTER_KEYWORDS.items():
            for kw in keywords:
                if ' ' in kw:
                    if kw in slug_lower or kw in question_lower:
                        detected_categories.add(cluster)
                        break
                else:
                    if re.search(r'(?:^|\s)' + re.escape(kw) + r'(?:\s|$|[^a-z])', slug_lower) or re.search(r'(?:^|\s)' + re.escape(kw) + r'(?:\s|$|[^a-z])', question_lower):
                        detected_categories.add(cluster)
                        break

        if not detected_categories:
            detected_categories.add("other")

        for cat in detected_categories:
            exposure[cat] += shares_value

    # Log current exposure summary
    total_exposure = sum(exposure.values())
    for cat, val in sorted(exposure.items(), key=lambda x: -x[1]):
        pct = val / balance if balance > 0 else 0
        logger.info(f"[EXPOSURE] {cat}: ${val:.2f} ({pct:.1%} of balance)")

    return dict(exposure)


def check_category_limits(new_market, new_order_value, total_balance, portfolio=None):
    """
    Check if placing a new order would exceed MAX_EXPOSURE_PER_CATEGORY.

    Before any order is placed, we check each detected category in the new market
    against the MAX_EXPOSURE_PER_CATEGORY limit (e.g., 12% of total balance). If adding
    the new order's value would breach the limit for ANY category, the order is
    rejected to prevent correlated risk concentration.

    Returns (allowed: bool, reason: str)
    """
    exposure = get_category_exposure(total_balance, portfolio)

    # Detect categories for new market
    slug = new_market.get("slug", "").lower()
    clusters = new_market.get("clusters", [])

    new_categories = set(clusters)
    if not new_categories:
        new_categories.add("other")

    max_dollar = total_balance * MAX_EXPOSURE_PER_CATEGORY

    for cat in new_categories:
        current = exposure.get(cat, 0)
        projected = current + new_order_value
        if projected > max_dollar:
            logger.warning(
                f"[EXPOSURE-BLOCK] category={cat} current=${current:.2f} + "
                f"new=${new_order_value:.2f} > max=${max_dollar:.2f} ({MAX_EXPOSURE_PER_CATEGORY:.0%})"
            )
            return False, f"Category '{cat}' exposure limit reached ({projected/total_balance:.1%} > {MAX_EXPOSURE_PER_CATEGORY:.0%})"

    logger.info(f"[EXPOSURE-OK] new order ${new_order_value:.2f} passes category checks")
    return True, "OK"

def calculate_brier_score(db):
    resolved = [h for h in db.get("resolved", []) if h.get("outcome") in ("YES", "NO")]
    if len(resolved) < BURN_IN_TRADES:        return None

    brier_scores = []
    wins = 0
    losses = 0
    for h in resolved[-BURN_IN_TRADES:]:
        p = h.get("p_model", 0.5)
        o = 1 if h.get("outcome") == "YES" else 0
        brier_scores.append((p - o) ** 2)
        if o == 1:
            wins += 1
        else:
            losses += 1

    brier = sum(brier_scores) / len(brier_scores)
    winrate = wins / (wins + losses) if (wins + losses) > 0 else 0
    logger.info(f"Stats: Brier={brier:.3f}, Winrate={winrate:.1%} ({wins}W/{losses}L) [from {len(resolved)} resolved]")

    settings = get_settings()
    old_brier = settings.get("calibration_brier")

    if old_brier is not None and brier > 0:
        if brier > 0.08 and settings.get("signal_threshold", 55) < 65:
            settings["signal_threshold"] = settings.get("signal_threshold", 55) + 2
            save_settings(settings)
            logger.info(f"[CALIBRATE] Brier {brier:.3f} > 0.08, raising signal_threshold to {settings['signal_threshold']}")
        elif brier < 0.03 and winrate > 0.1 and settings.get("signal_threshold", 55) > 40:
            settings["signal_threshold"] = settings.get("signal_threshold", 55) - 2
            save_settings(settings)
            logger.info(f"[CALIBRATE] Brier {brier:.3f} < 0.03, winrate {winrate:.0%}, lowering signal_threshold to {settings['signal_threshold']}")

    if winrate == 0 and len(resolved) >= 10 and settings.get("signal_threshold", 55) < 80:
        settings["signal_threshold"] = min(80, settings.get("signal_threshold", 55) + 5)
        save_settings(settings)
        logger.warning(f"[CALIBRATE] 0% winrate ({len(resolved)} resolved), RAISING signal_threshold to {settings['signal_threshold']} (defensive mode)")
    elif winrate < 0.30 and len(resolved) >= 20 and settings.get("signal_threshold", 55) < 75:
        settings["signal_threshold"] = min(75, settings.get("signal_threshold", 55) + 3)
        save_settings(settings)
        logger.info(f"[CALIBRATE] Low winrate ({winrate:.0%}, {len(resolved)} resolved), raising signal_threshold to {settings['signal_threshold']}")

    settings["calibration_brier"] = brier
    save_settings(settings)

    if len(resolved) >= 50:
        recent_resolved = [h for h in resolved if h.get("resolved_at") and (datetime.now() - datetime.fromisoformat(h["resolved_at"])).days <= 90]
        if len(recent_resolved) >= 20:
            from calibration import get_calibrator
            calibrator = get_calibrator()
            calibrator.fit(recent_resolved)
            calibrator.save()
            logger.info(f"[CALIBRATION] Trained isotonic model on {len(recent_resolved)} recent markets (<=90 days)")
        else:
            logger.info(f"[CALIBRATION] Only {len(recent_resolved)} recent resolved, need >=20, skipping retrain")

    return brier

def learn_from_results(db):
    resolved = db.get("resolved", [])
    if len(resolved) < 10:
        return {}

    settings = get_settings()
    factor_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    cluster_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    source_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    signal_stats = defaultdict(lambda: {"wins": 0, "losses": 0})

    for h in resolved[-50:]:
        outcome = h.get("outcome")
        if outcome not in ("YES", "NO"):
            continue
        is_win = outcome == "YES"

        for factor in h.get("factors", []):
            key = f"{factor.get('direction')}:{factor.get('weight')}"
            if is_win:
                factor_stats[key]["wins"] += 1
            else:
                factor_stats[key]["losses"] += 1

        for cluster in h.get("clusters", []):
            if is_win:
                cluster_stats[cluster]["wins"] += 1
            else:
                cluster_stats[cluster]["losses"] += 1

        source_signal = h.get("source_signal", "default")
        if is_win:
            source_stats[source_signal]["wins"] += 1
        else:
            source_stats[source_signal]["losses"] += 1

    cluster_weights = {}
    for cluster, stats in cluster_stats.items():
        total = stats["wins"] + stats["losses"]
        if total >= MIN_TRADES_FOR_WEIGHT_ADJUSTMENT:
            winrate = stats["wins"] / total
            base_weight = {
                "venezuela": 0.30,
                "russia_ukraine": 0.25,
                "usa_politics": 0.20,
                "fed_fomc": 0.25,
                "ai_tech": 0.10,
                "sports_nba": 0.15,
                "sports_ufc": 0.15,
            }.get(cluster, 0.15)
            posterior_weight = (
                (base_weight * BAYESIAN_PRIOR_STRENGTH + winrate * total)
                / (BAYESIAN_PRIOR_STRENGTH + total)
            )
            posterior_weight = max(0.05, min(0.50, posterior_weight))
            cluster_weights[cluster] = posterior_weight
            logger.info(
                f"[LEARN] Cluster {cluster}: winrate={winrate:.1%} ({total} trades), "
                f"base={base_weight:.2f} → posterior={posterior_weight:.3f}"
            )
        else:
            logger.debug(
                f"[LEARN] Cluster {cluster}: insufficient data ({total}/{MIN_TRADES_FOR_WEIGHT_ADJUSTMENT} trades), "
                f"keeping base weight"
            )

    metaculus_bonus = 0.4
    geopol_bonus = 0.3
    sports_bonus = 0.2

    for source, stats in source_stats.items():
        total = stats["wins"] + stats["losses"]
        if total >= 3:
            winrate = stats["wins"] / total
            if source == "metaculus":
                metaculus_bonus = 0.4 * (1 + (winrate - 0.5))
                logger.info(f"📊 Source {source}: winrate={winrate:.1%}, adjusted bonus={metaculus_bonus:.3f}")
            elif source == "geopol":
                geopol_bonus = 0.3 * (1 + (winrate - 0.5))
                logger.info(f"📊 Source {source}: winrate={winrate:.1%}, adjusted bonus={geopol_bonus:.3f}")
            elif source == "sports":
                sports_bonus = 0.2 * (1 + (winrate - 0.5))
                logger.info(f"📊 Source {source}: winrate={winrate:.1%}, adjusted bonus={sports_bonus:.3f}")

    settings["cluster_weights"] = cluster_weights
    settings["source_bonus_metaculus"] = metaculus_bonus
    settings["source_bonus_geopol"] = geopol_bonus
    settings["source_bonus_sports"] = sports_bonus
    save_settings(settings)

    return {
        "cluster_weights": cluster_weights,
        "source_stats": dict(source_stats)
    }


def backtest_recent(n=20):
    """
    Backtest recent resolved hypotheses to evaluate strategy performance.
    Returns simulation results and recommendations.
    """
    db = load_hypothesis_db()
    resolved = db.get("resolved", [])
    
    if len(resolved) < 5:
        return {"error": f"Only {len(resolved)} resolved, need at least 5", "recommendation": "skip"}
    
    recent = [h for h in resolved[-n:] if h.get("outcome") in ("YES", "NO")]
    if len(recent) < 5:
        return {"error": f"Only {len(recent)} with YES/NO outcome", "recommendation": "skip"}
    
    wins = 0
    total_pnl = 0
    brier_sum = 0
    
    for h in recent:
        p_model = h.get("p_model", 0.5)
        market_price = h.get("market_price", 0.5)
        outcome = 1 if h.get("outcome") == "YES" else 0
        is_win = outcome == 1

        if is_win:
            wins += 1
            pnl = (1 - market_price) / market_price
        else:
            actual_pnl = h.get("sold_pnl_pct") or h.get("pnl_at_exit")
            if actual_pnl is not None and actual_pnl != 0:
                pnl = actual_pnl
            else:
                pnl = -1

        total_pnl += pnl
        brier_sum += (p_model - outcome) ** 2
    
    winrate = wins / len(recent)
    avg_brier = brier_sum / len(recent)
    avg_pnl = total_pnl / len(recent)
    
    # Evaluate thresholds
    current_signal_threshold = get_settings().get("signal_threshold", 55)
    current_min_p = get_settings().get("min_p_model", MIN_P_MODEL)
    
    # Calculate what would happen with different thresholds
    recommendations = []
    
    if winrate < 0.40:
        recommendations.append({
            "issue": "winrate_too_low",
            "current": winrate,
            "suggestion": "Raise MIN_PROB_RATIO or MIN_P_MODEL to be more selective"
        })
    
    if avg_brier > 0.20:
        recommendations.append({
            "issue": "poor_calibration",
            "current": avg_brier,
            "suggestion": "Improve p_model estimation or use market price as stronger prior"
        })
    
    if avg_pnl < 0:
        recommendations.append({
            "issue": "negative_avg_pnl",
            "current": avg_pnl,
            "suggestion": "Reduce position sizes or increase threshold"
        })
    
    # Winrate by cluster
    cluster_wins = defaultdict(lambda: {"wins": 0, "total": 0})
    for h in recent:
        for c in h.get("clusters", []):
            cluster_wins[c]["total"] += 1
            if h.get("outcome") == "YES":
                cluster_wins[c]["wins"] += 1
    
    cluster_performance = {}
    for c, stats in cluster_wins.items():
        if stats["total"] >= 3:
            cluster_performance[c] = stats["wins"] / stats["total"]
    
    result = {
        "n_analyzed": len(recent),
        "winrate": winrate,
        "avg_brier": avg_brier,
        "avg_pnl": avg_pnl,
        "cluster_performance": cluster_performance,
        "recommendations": recommendations,
        "recommendation": "use_current" if not recommendations else "adjust_thresholds"
    }
    
    logger.info(f"[BACKTEST] n={len(recent)}, winrate={winrate:.1%}, brier={avg_brier:.3f}, pnl={avg_pnl:.2f}")
    for r in recommendations:
        logger.info(f"[BACKTEST] REC: {r['issue']} -> {r['suggestion']}")
    
    return result


def get_order_book(slug):
    try:
        res = subprocess.run(["pm-trader", "book", slug, "--depth", "3"],
                           capture_output=True, text=True, timeout=15, start_new_session=True)
        data = json.loads(res.stdout)
        asks = data.get("data", {}).get("asks", [])
        bids = data.get("data", {}).get("bids", [])
        best_ask = float(asks[0].get("price", 0)) if asks and asks[0].get("price") is not None else None
        best_bid = float(bids[0].get("price", 0)) if bids and bids[0].get("price") is not None else None
        if best_bid and best_ask:
            mid_price = (best_bid + best_ask) / 2
        else:
            mid_price = best_ask or best_bid
        return {"best_bid": best_bid, "best_ask": best_ask, "mid_price": mid_price}
    except:
        return {"best_bid": None, "best_ask": None, "mid_price": None}

def get_best_ask(slug):
    book = get_order_book(slug)
    return book.get("best_ask")

def get_balance():
    try:
        res = subprocess.run(["pm-trader", "balance"], capture_output=True, text=True, timeout=15, start_new_session=True)
        if res.returncode != 0:
            logger.error(f"[SNIPER] pm-trader balance failed: rc={res.returncode}")
            return None
        return json.loads(res.stdout).get("data", {})
    except Exception:
        return None

def get_portfolio():
    try:
        res = subprocess.run(["pm-trader", "portfolio"], capture_output=True, text=True, timeout=15, start_new_session=True)
        if res.returncode != 0:
            logger.error(f"[SNIPER] pm-trader portfolio failed: rc={res.returncode}")
            return []
        return json.loads(res.stdout).get("data", [])
    except Exception:
        return []

def fetch_markets():
    try:
        res = subprocess.run(["pm-trader", "markets", "list", "--limit", "200"],
                           capture_output=True, text=True, timeout=30, start_new_session=True)
        if res.returncode != 0:
            logger.error(f"[MARKETS] pm-trader markets failed: rc={res.returncode}")
            return []
        data = json.loads(res.stdout)
        candidates = []
        now = datetime.now()

        for m in data.get("data", []):
            if not m.get("active") or m.get("closed") or m.get("status") in ("closing", "resolved", "invalid"):
                continue

            vol = float(m.get("volume", 0))
            if vol < MIN_VOLUME:
                continue

            liq = float(m.get("liquidity", 0))
            if liq < 100:
                continue

            end_date = m.get("end_date", "")
            ttl_hours = 999
            if end_date:
                try:
                    end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                    ttl_hours = max(0, (end - now).total_seconds() / 3600)
                except:
                    pass

            if ttl_hours < MIN_TTL_HOURS:
                continue

            for outcome, price in zip(m.get("outcomes", []), m.get("outcome_prices", [])):
                try:
                    price = float(price)
                except (ValueError, TypeError):
                    continue
                if price is None or price <= 0 or price > 1.0:
                    continue
                if price <= MAX_PRICE:
                    clusters = detect_clusters(m["question"])
                    if any(c in BANNED_CLUSTERS for c in clusters):
                        continue
                    is_allowed = any(c in ALLOWED_CLUSTERS for c in clusters)
                    if any(c in ("sports_nba", "sports_ufc") for c in clusters):
                        if vol < 250_000 or ttl_hours < 48: continue
                    is_other_high_vol = clusters == ["other"] and vol >= PRE_FILTER_OTHER_MIN_VOLUME
                    if not is_allowed and not is_other_high_vol:
                        continue
                    candidates.append({
                        "id": m["condition_id"],
                        "slug": m["slug"],
                        "question": m["question"],
                        "outcome": outcome,
                        "price": price,
                        "volume": vol,
                        "liquidity": float(m.get("liquidity", 0)),
                        "end_date": end_date,
                        "ttl_hours": ttl_hours,
                        "clusters": clusters,
                        "oracle_type": m.get("oracle_type", "unknown")
                    })

        candidates.sort(key=lambda x: -x["volume"])
        seen_slugs = set()
        unique = []
        for c in candidates:
            if c["slug"] not in seen_slugs:
                seen_slugs.add(c["slug"])
                unique.append(c)
        return unique[:10]
    except Exception as e:
        logger.error(f"[MARKETS] Fetch error: {e}")
        return []

def position_size(p_model, market_price, balance, confidence=1.0, best_ask=None, cluster=None):
    """
    Fractional Kelly position sizing with confidence weighting.

    Kelly Criterion formula: f = (p * b - q) / b
      p = p_model (our estimated true probability)
      q = 1 - p
      b = payout coefficient = (1 - price) / price
           (net odds received on winning YES bet)

    The Kelly fraction is reduced by two factors:
      1. FRACTIONAL_KELLY_MULTIPLIER (default 0.25 = "quarter Kelly")
         Quarter Kelly is a conservative approach that reduces volatility
         while still capturing most of the edge
      2. confidence score acts as an additional multiplier since high
         confidence in our probability estimate justifies a larger bet

    Hard limits enforced:
      - Minimum order: $5 (exchange fee protection)
      - Cluster-aware max cap:
          * "other" cluster: OTHER_BOOST_POS_PCT (3.5% of balance)
          * all others: BASE_POS_PCT (2% of balance)
      - Absolute ceiling: MAX_POS_PCT (10%)

    Args:
        p_model: our estimated probability (0.0 to 1.0)
        market_price: current Polymarket price (used if best_ask not provided)
        balance: current account balance in dollars
        confidence: our confidence in p_model estimate (0.0 to 1.0)
        best_ask: best ask price from order book (more accurate than midpoint)
        cluster: primary cluster name for position sizing boost

    Returns:
        Dollar amount to bet
    """
    if market_price <= 0:
        logger.warning("[KELLY] market_price <= 0, using minimum $5")
        return 5

    # Prefer best_ask for Kelly calculation (actual executable price)
    # vs market_price which might be midpoint with poor liquidity
    effective_price = best_ask if best_ask is not None else market_price

    if effective_price <= 0.001:
        logger.warning(f"[KELLY] effective_price={effective_price:.6f} too small, minimum $5")
        return 5
    b = (1 - effective_price) / effective_price

    p = p_model; q = 1 - p
    kelly_full = (b * p - q) / b

    logger.info(f"[KELLY] p={p:.3f}, b={b:.2f}, q={q:.3f}, kelly_full={kelly_full:.4f}")

    # Reject negative Kelly (no edge case)
    if kelly_full <= 0:
        logger.info(f"[KELLY] kelly_full={kelly_full:.4f} <= 0, no edge - skipping")
        return 0

    # Reject if our probability estimate is too low
    min_p_model = get_settings().get("min_p_model", MIN_P_MODEL)
    if p < min_p_model:
        logger.info(f"[KELLY] p_model={p:.1%} < MIN_P_MODEL={min_p_model:.1%}, skipping")
        return 0

    # Step 1: Fractional Kelly reduction (quarter Kelly by default)
    kelly_fraction = kelly_full * FRACTIONAL_KELLY_MULTIPLIER

    # Step 2: Confidence weighting (high confidence = bigger bet)
    kelly_with_confidence = kelly_fraction * confidence

    # Cluster-aware position cap
    effective_cap = OTHER_BOOST_POS_PCT if cluster == "other" else BASE_POS_PCT
    size_pct = min(kelly_with_confidence, effective_cap)

    kelly_dollars = round(balance * size_pct)
    if kelly_dollars <= 0:
        return 0
    kelly_dollars = max(kelly_dollars, 5)
    kelly_dollars = min(kelly_dollars, round(balance * MAX_POS_PCT))

    logger.info(
        f"[KELLY] kelly_full={kelly_full:.4f} * frac={FRACTIONAL_KELLY_MULTIPLIER:.2f} "
        f"* conf={confidence:.2f} = {kelly_with_confidence:.4f} "
        f"=> ${kelly_dollars} ({size_pct:.2%} of ${balance:.2f}) "
        f"[cap={effective_cap:.1%}, cluster={cluster}]"
    )

    return kelly_dollars

def buy(market, amount):
    try:
        book = get_order_book(market["slug"])
        best_ask = book.get("best_ask")
        best_bid = book.get("best_bid")

        if best_ask is None or best_ask <= 0:
            logger.warning(f"[SNIPER] No valid ask for {market['slug']}, aborting buy")
            return False

        market_price = market.get("price", 0)
        if market_price > 0 and best_ask > market_price * 1.15:
            logger.warning(
                f"[SNIPER] Slippage guard in buy(): ask={best_ask:.4f} > 15% above "
                f"price={market_price:.4f} for {market['slug']}, aborting"
            )
            return False

        if best_bid is not None and best_bid > 0 and best_ask > 0:
            spread = best_ask - best_bid
            if spread / best_ask > MAX_SPREAD_PCT:
                logger.warning(
                    f"[SNIPER] Spread too wide in buy(): spread={spread:.4f} "
                    f"({spread/best_ask:.1%}) for {market['slug']}, aborting"
                )
                return False

        res = subprocess.run(["pm-trader", "buy", market["slug"], market["outcome"], str(amount)],
                           capture_output=True, text=True, timeout=30, start_new_session=True)
        if res.returncode != 0:
            logger.error(f"[SNIPER] Buy failed for {market['slug']}: rc={res.returncode}")
            return False
        result = json.loads(res.stdout)
        if result.get("ok"):
            print(f"  ✅ {market['question'][:45]}... ${amount}")
            return True
        else:
            print(f"  ❌ {result}")
            return False
    except Exception as e:
        print(f"  ❌ {e}")
        return False

def resolve_hypothesis_immediately(slug, current_price, entry_price):
    _cancel_all_tp_orders(slug)
    db = load_hypothesis_db()
    for h in db["hypotheses"]:
        if h["slug"] == slug and not h.get("resolved"):
            h["resolved"] = True
            h["resolved_at"] = datetime.now().isoformat()
            h["exit_price"] = current_price
            h["pnl_at_exit"] = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            h["exit_type"] = "manual"
            h["outcome"] = "SOLD"
            h["sold_pnl_pct"] = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            db["resolved"].append(h)

            positions = load_json(POSITIONS_FILE, {})
            if slug in positions:
                del positions[slug]
                save_json(POSITIONS_FILE, positions)

            save_hypothesis_db(db)

            settings = get_settings()
            settings["total_resolved"] = settings.get("total_resolved", 0) + 1
            save_settings(settings)
            break

MAX_SPREAD_PCT = 0.15
MIN_BID_LIQUIDITY = 5.0
LIMIT_SPREAD_THRESHOLD = 0.03
LIMIT_PRICE_BUFFER = 0.005
LIMIT_MAX_ATTEMPTS = 3


def _place_limit_sell(slug, outcome, shares, limit_price):
    """
    Place a limit sell order via pm-trader CLI.
    Falls back to market order if --limit flag unsupported.
    """
    try:
        res = subprocess.run(
            ["pm-trader", "sell", slug, outcome, str(shares),
             "--limit", "--price", f"{limit_price:.4f}"],
            capture_output=True, text=True, timeout=20, start_new_session=True
        )
        result = json.loads(res.stdout) if res.stdout else {}
        if result.get("ok"):
            return True, "limit_placed"
    except Exception:
        pass
    return False, "limit_unsupported"


def _place_tp_limit_order_single(slug, outcome, shares, price):
    """
    v5.1.0: Place an unconditional Take-Profit limit sell order.
    This is called immediately after a successful BUY execution.
    The TP is placed regardless of time to expiration.

    Returns (ok: bool, method: str)
    """
    try:
        res = subprocess.run(
            ["pm-trader", "sell", slug, outcome, str(shares),
             "--limit", "--price", f"{price:.4f}"],
            capture_output=True, text=True, timeout=20, start_new_session=True
        )
        result = json.loads(res.stdout) if res.stdout else {}
        if result.get("ok"):
            logger.info(
                f"[SMART-EXIT] TP limit placed for {slug[:40]}... "
                f"@{price:.2f} (shares={shares})"
            )
            return True, "tp_limit_placed"
        else:
            logger.warning(
                f"[SMART-EXIT] TP limit failed for {slug[:40]}... "
                f"response={result}"
            )
    except Exception as e:
        logger.warning(f"[SMART-EXIT] Exception placing TP for {slug[:40]}...: {e}")
    return False, "tp_limit_failed"


def _place_tp_ladder(slug, outcome, total_shares):
    """v5.3.0 TP Ladder: 50% @$0.75, 30% @$0.85, 20% hold to expiry"""
    ladder = [(0.50, 0.75), (0.30, 0.85)]
    results = []; allocated = 0
    for pct, price in ladder:
        shares = max(round(5.0 / price), round(total_shares * pct), 1)
        if allocated + shares > total_shares: shares = total_shares - allocated
        if shares <= 0: continue
        ok, m = _place_tp_limit_order_single(slug, outcome, shares, price)
        results.append((price, shares, ok, m)); allocated += shares
    logger.info(f"[TP-LADDER] {slug[:40]}... placed {len(results)} rungs, {total_shares - allocated} held to expiry")
    return results


def _cancel_all_tp_orders(slug):
    """Cancel all open sell orders for a position on manual/stop exit."""
    try:
        res = subprocess.run(["pm-trader", "orders", "--status", "open"], capture_output=True, text=True, timeout=30, start_new_session=True)
        for line in res.stdout.strip().split('\n')[1:]:
            if not line.strip() or '---' in line: continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts)>=3 and parts[0]==slug and parts[2]=='sell':
                subprocess.run(["pm-trader", "orders", "cancel", slug, parts[1]], timeout=20, start_new_session=True)
                logger.info(f"[TP-CANCEL] Canceled sell order for {slug[:40]}...")
    except Exception as e: logger.warning(f"[TP-CANCEL] Failed for {slug}: {e}")


def _execute_sell(slug, outcome, shares, current_price, entry_price, force_market=False):
    """
    Smart sell: use limit order when spread is wide, market order when safe or forced.
    Returns (sold: bool, effective_price: float or None, method: str)
    """
    book = get_order_book(slug)
    best_bid = book.get("best_bid")
    best_ask = book.get("best_ask")

    if best_bid is None or best_bid <= 0:
        logger.warning(f"[SELL] {slug[:40]}... no bids at all")
        return False, None, "no_bids"

    spread = (best_ask - best_bid) if (best_bid and best_ask) else 0

    if not force_market and spread > LIMIT_SPREAD_THRESHOLD:
        positions = load_json(POSITIONS_FILE, {})
        pos = positions.get(slug, {})
        limit_attempts = pos.get("limit_sell_attempts", 0)

        if limit_attempts < LIMIT_MAX_ATTEMPTS:
            limit_price = best_bid + LIMIT_PRICE_BUFFER
            logger.info(
                f"[LIMIT-SELL] {slug[:40]}... spread=${spread:.4f} > ${LIMIT_SPREAD_THRESHOLD}, "
                f"placing limit at ${limit_price:.4f} (attempt {limit_attempts + 1}/{LIMIT_MAX_ATTEMPTS})"
            )
            ok, reason = _place_limit_sell(slug, outcome, shares, limit_price)
            if ok:
                pos["limit_sell_attempts"] = limit_attempts + 1
                pos["limit_sell_price"] = limit_price
                pos["limit_sell_since"] = datetime.now().isoformat()
                positions[slug] = pos
                save_json(POSITIONS_FILE, positions)
                return False, limit_price, "limit_pending"

        logger.warning(
            f"[FORCE-MARKET] {slug[:40]}... {limit_attempts} limit attempts exhausted, forcing market sell"
        )

    logger.info(f"[MARKET-SELL] {slug[:40]}... bid={best_bid:.4f} spread=${spread:.4f}")
    try:
        res = subprocess.run(["pm-trader", "sell", slug, outcome, str(shares)],
                             capture_output=True, text=True, timeout=20, start_new_session=True)
        result = json.loads(res.stdout) if res.stdout else {}
        if result.get("ok"):
            return True, best_bid, "market"
    except Exception:
        pass
    return False, best_bid, "market_failed"


def _check_sell_safety(slug, current_price, shares):
    """
    Verify order book has sufficient liquidity before placing a market sell.
    Returns (safe: bool, reason: str, effective_price: float or None)
    """
    book = get_order_book(slug)
    best_bid = book.get("best_bid")
    best_ask = book.get("best_ask")

    if best_bid is None or best_bid <= 0:
        logger.warning(f"[SLIPPAGE-GUARD] {slug[:40]}... no bids in order book, aborting sell")
        return False, "no_bids", None

    if best_ask and best_ask > 0:
        spread = (best_ask - best_bid) / best_ask
        if spread > MAX_SPREAD_PCT:
            logger.warning(
                f"[SLIPPAGE-GUARD] {slug[:40]}... spread={spread:.1%} > {MAX_SPREAD_PCT:.0%}, "
                f"bid={best_bid:.4f} ask={best_ask:.4f}, aborting sell"
            )
            return False, f"spread_too_wide:{spread:.1%}", best_bid

    if best_bid < current_price * 0.70:
        logger.warning(
            f"[SLIPPAGE-GUARD] {slug[:40]}... best_bid={best_bid:.4f} is >30% below mid={current_price:.4f}, "
            f"likely empty order book, aborting sell"
        )
        return False, f"bid_far_from_mid:{best_bid:.4f}_vs_{current_price:.4f}", best_bid

    logger.info(
        f"[SLIPPAGE-GUARD] {slug[:40]}... OK: bid={best_bid:.4f} ask={best_ask} mid={current_price:.4f}"
    )
    return True, "ok", best_bid

def trailing_stop_check():
    portfolio = get_portfolio()
    if not portfolio:
        return

    current_slugs = {p["market_slug"] for p in portfolio}
    now = datetime.now()

    db = load_hypothesis_db()
    resolved_slugs = {h["slug"] for h in db.get("resolved", [])}

    for pos in portfolio:
        slug = pos["market_slug"]
        shares = pos.get("shares", 0)
        entry_price = pos.get("avg_entry_price", 0)
        outcome = pos.get("outcome", "yes")

        if slug in resolved_slugs:
            positions = load_json(POSITIONS_FILE, {})
            if slug in positions:
                del positions[slug]
                save_json(POSITIONS_FILE, positions)
                logger.info(f"[SKIP-RESOLVED] {slug[:40]}... already resolved in hypothesis_db, removed from positions")
            continue

        book = get_order_book(slug)
        mid_price = book.get("mid_price")
        live_price = pos.get("live_price", 0)
        current_price = mid_price if mid_price is not None else live_price

        if current_price <= 0:
            continue

        pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0

        positions = load_json(POSITIONS_FILE, {})
        if slug not in positions:
            positions[slug] = {
                "entry_price": entry_price,
                "high_price": max(entry_price, current_price),
                "trailing_on": False,
                "stop_loss": entry_price * (1 + HARD_STOP_LOSS),
                "last_checked": now.isoformat(),
                "metaculus_prob": None,
                "market_question": pos.get("market_question", ""),
                "shares": shares
            }
            save_json(POSITIONS_FILE, positions)

        p = positions[slug]

        last_checked = None
        if p.get("last_checked"):
            try:
                last_checked = datetime.fromisoformat(p["last_checked"])
            except:
                last_checked = None

        check_interval = MIN_POSITION_CHECK_INTERVAL_HOURS * 3600
        if last_checked and (now - last_checked).total_seconds() < check_interval:
            logger.info(f"[POLLING] {slug[:40]}... skipping, checked {(now - last_checked).total_seconds()/3600:.1f}h ago")
            continue

        p["last_checked"] = now.isoformat()

        p["high_price"] = max(p.get("high_price", current_price), current_price, entry_price)

        if p["high_price"] > entry_price * (1 + TRAILING_ACTIVATION):
            p["trailing_on"] = True
            p["stop_loss"] = p["high_price"] * (1 - TRAILING_STOP)

        meta = get_metaculus_forecast(pos.get("market_question", ""), None)
        metaculus_prob = None
        if meta.get("found"):
            metaculus_prob = meta.get("probability")
            p["metaculus_prob"] = metaculus_prob

        positions[slug] = p
        save_json(POSITIONS_FILE, positions)

        sold = False
        sold_reason = ""

        convergence = None
        if metaculus_prob and metaculus_prob > 0:
            convergence = current_price / metaculus_prob
            logger.info(f"[CONVERGENCE] {slug[:40]}... mid={current_price:.3f}, meta={metaculus_prob:.0%}, ratio={convergence:.2f}")
            if convergence >= CONVERGENCE_TAKE_PROFIT:
                sold_reason = f"convergence={convergence:.2f} >= {CONVERGENCE_TAKE_PROFIT}"
                logger.info(f"[TAKE-PROFIT] Gap convergence reached: {sold_reason}")
                try:
                    res = subprocess.run(["pm-trader", "sell", slug, outcome, str(shares)],
                                         capture_output=True, text=True, timeout=20, start_new_session=True)
                    result = json.loads(res.stdout) if res.stdout else {}
                    if result.get("ok"):
                        logger.info(f"SOLD take-profit convergence: {slug} pnl={pnl_pct:.2%}")
                        sold = True
                        pnl_abs = shares * (current_price - entry_price)
                        if telegram_reporter:
                            telegram_reporter.alert_convergence(slug, pos.get("market_question", ""), pnl_pct * 100, pnl_abs, convergence)
                except:
                    pass

        if not sold and pnl_pct <= HARD_STOP_LOSS:
            sold_reason = f"hard_stop={pnl_pct:.0%}"
            logger.warning(f"[STOP-LOSS] Hard stop triggered: {slug[:40]}... pnl={pnl_pct:.0%}")
            try:
                pos_data = positions.get(slug, {})
                limit_attempts = pos_data.get("limit_sell_attempts", 0)
                force = limit_attempts >= LIMIT_MAX_ATTEMPTS
                if not force:
                    safe, safe_reason, sell_price = _check_sell_safety(slug, current_price, shares)
                    if not safe:
                        logger.warning(
                            f"[STOP-DELAYED] {slug[:40]}... sell unsafe: {safe_reason}. "
                            f"mid={current_price:.4f} entry={entry_price:.4f}"
                        )
                        if telegram_reporter:
                            telegram_reporter.alert_stop_loss(slug, pos.get("market_question", ""), pnl_pct * 100, shares * (current_price - entry_price))
                    else:
                        sold, eff_price, method = _execute_sell(slug, outcome, shares, current_price, entry_price)
                        if sold:
                            actual_pnl = (eff_price - entry_price) / entry_price if entry_price > 0 else pnl_pct
                            logger.info(f"SOLD hard stop ({method}): {slug} mid_pnl={pnl_pct:.2%} eff_pnl={actual_pnl:.2%}")
                            pnl_abs = shares * (eff_price - entry_price)
                            if telegram_reporter:
                                telegram_reporter.alert_stop_loss(slug, pos.get("market_question", ""), actual_pnl * 100, pnl_abs)
                else:
                    logger.warning(f"[EMERGENCY-SELL] {slug[:40]}... forcing market after {limit_attempts} limit attempts")
                    sold, eff_price, method = _execute_sell(slug, outcome, shares, current_price, entry_price, force_market=True)
                    if sold:
                        actual_pnl = (eff_price - entry_price) / entry_price if entry_price > 0 else pnl_pct
                        logger.info(f"SOLD emergency ({method}): {slug} pnl={actual_pnl:.2%}")
                        pnl_abs = shares * (eff_price - entry_price)
                        if telegram_reporter:
                            telegram_reporter.alert_stop_loss(slug, pos.get("market_question", ""), actual_pnl * 100, pnl_abs)
            except:
                pass

        if not sold and p.get("trailing_on") and current_price <= p.get("stop_loss", 0):
            if not p.get("trailing_confirmed"):
                p["trailing_confirmed"] = True
                p["trailing_confirm_time"] = now.isoformat()
                logger.info(f"[TRAILING-STOP] Confirming for {slug[:40]}... (1/2)")
            else:
                confirm_time = p.get("trailing_confirm_time")
                if confirm_time:
                    try:
                        elapsed = (now - datetime.fromisoformat(confirm_time)).total_seconds()
                        if elapsed < 300:
                            logger.info(f"[TRAILING-STOP] Waiting confirmation for {slug[:40]}... ({elapsed:.0f}s/300s)")
                            continue
                    except (ValueError, TypeError):
                        pass
            sold_reason = f"trailing={current_price:.3f} <= {p.get('stop_loss', 0):.3f}"
            logger.info(f"[TRAILING-STOP] Triggered for {slug[:40]}...")
            try:
                sold, eff_price, method = _execute_sell(slug, outcome, shares, current_price, entry_price)
                if sold:
                    logger.info(f"SOLD trailing stop ({method}): {slug}")
                    p.pop("trailing_confirmed", None)
                    p.pop("trailing_confirm_time", None)
                    pnl_abs = shares * (eff_price - entry_price)
                    actual_pnl = (eff_price - entry_price) / entry_price if entry_price > 0 else pnl_pct
                    if telegram_reporter:
                        if actual_pnl > 0:
                            telegram_reporter.alert_take_profit(slug, pos.get("market_question", ""), actual_pnl * 100, pnl_abs)
                        else:
                            telegram_reporter.alert_stop_loss(slug, pos.get("market_question", ""), actual_pnl * 100, pnl_abs)
            except:
                pass

        if not sold and pnl_pct >= TAKE_PROFIT:
            sold_reason = f"take_profit={pnl_pct:.0%}"
            logger.info(f"[TAKE-PROFIT] {slug[:40]}... +{pnl_pct:.0%}")
            try:
                sold, eff_price, method = _execute_sell(slug, outcome, shares, current_price, entry_price)
                if sold:
                    logger.info(f"SOLD take-profit ({method}): {slug}")
                    pnl_abs = shares * (eff_price - entry_price)
                    if telegram_reporter:
                        telegram_reporter.alert_take_profit(slug, pos.get("market_question", ""), pnl_pct * 100, pnl_abs)
            except:
                pass

        if sold:
            _cancel_all_tp_orders(slug)
            resolve_hypothesis_immediately(slug, current_price, entry_price)
            positions = load_json(POSITIONS_FILE, {})
            if slug in positions:
                del positions[slug]
                save_json(POSITIONS_FILE, positions)
        else:
            positions[slug] = p
            save_json(POSITIONS_FILE, positions)

    current_slugs = {p["market_slug"] for p in portfolio}
    positions = load_json(POSITIONS_FILE, {})
    stale = [s for s in list(positions.keys()) if s not in current_slugs or s in resolved_slugs]
    for s in stale:
        if s in positions:
            del positions[s]
            logger.info(f"[CLEANUP] Removed stale position: {s}")
    if stale:
        save_json(POSITIONS_FILE, positions)

def repair_positions_file():
    """Fix inconsistent data in positions.json (e.g. high_price < entry_price)."""
    positions = load_json(POSITIONS_FILE, {})
    dirty = False
    for slug, p in positions.items():
        entry = p.get("entry_price", 0)
        high = p.get("high_price", 0)
        if entry > 0 and high < entry:
            p["high_price"] = entry
            logger.info(f"[REPAIR] {slug[:40]}... high_price {high:.4f} < entry_price {entry:.4f}, fixed")
            dirty = True
    if dirty:
        save_json(POSITIONS_FILE, positions)

def resolve_hypotheses():
    db = load_hypothesis_db()
    portfolio = get_portfolio()
    portfolio_slugs = {p["market_slug"] for p in portfolio}

    all_hypotheses = db.get("hypotheses", [])
    unresolved = [h for h in all_hypotheses if not h.get("resolved") and h["slug"] not in portfolio_slugs]

    if not unresolved:
        return

    market_map = {}
    try:
        res = subprocess.run(["pm-trader", "markets", "list", "--limit", "200"],
                           capture_output=True, text=True, timeout=20, start_new_session=True)
        for m in json.loads(res.stdout).get("data", []):
            market_map[m["slug"]] = m
    except:
        pass

    new_resolved = 0
    for h in unresolved:
        slug = h["slug"]
        market_data = market_map.get(slug)

        if not market_data:
            h["resolved"] = True
            h["resolved_at"] = datetime.now().isoformat()
            h["outcome"] = "UNKNOWN"
            h["resolution_note"] = "market_not_found_in_api"
            db["resolved"].append(h)
            new_resolved += 1
            continue

        if not market_data.get("closed"):
            continue

        h["resolved"] = True
        h["resolved_at"] = datetime.now().isoformat()

        outcome = "UNKNOWN"
        if market_data.get("resolution") in ("YES", "NO"):
            outcome = market_data.get("resolution")
        elif market_data.get("outcome_prices"):
            yes_price = market_data.get("outcome_prices", [0.5])[0]
            outcome = "YES" if yes_price > 0.5 else "NO"

        h["outcome"] = outcome

        db["resolved"].append(h)
        new_resolved += 1

    save_hypothesis_db(db)

    if new_resolved > 0:
        settings = get_settings()
        settings["total_resolved"] = len([h for h in all_hypotheses if h.get("resolved")])
        save_settings(settings)

        if len(db.get("resolved", [])) >= BURN_IN_TRADES:
            calculate_brier_score(db)
            learn_from_results(db)


def full_market_analysis(market):
    """
    Single-step market analysis: combines factor generation + probability estimation.
    """
    cluster = market.get("clusters", ["other"])[0]
    is_geopol = cluster in ["venezuela", "russia_ukraine", "usa_politics"]

    best_ask = None
    polymarket_prob = market["price"]

    if market["price"] < 0.35:
        best_ask = get_best_ask(market["slug"])
        polymarket_prob = best_ask if best_ask is not None else market["price"]

    metaculus_gap = None
    if market["price"] < 0.35:
        metaculus_gap = check_metaculus_gap(market, polymarket_prob)

    source_signal = "default"
    if metaculus_gap:
        source_signal = "metaculus"

    confidence = 0.60
    if source_signal == "metaculus":
        confidence = 0.80
    elif is_geopol:
        confidence = 0.70

    confidence = min(confidence, 0.95)

    gap_info = ""
    if metaculus_gap:
        gap_info = f"- Metaculus forecast: {metaculus_gap['metaculus_prob']:.0%} vs Polymarket {metaculus_gap['polymarket_prob']:.0%}\n"

    prompt = f"""Prediction market analyst. Your job is to find DOTM (deep out-the-money) events where the crowd SIGNIFICANTLY underestimates probability.

Market: {sanitize_for_prompt(market['question'])}
Price: ${market['price']:.3f} ({market['price']*100:.1f}%) | Volume: ${market.get('volume', 0):,.0f} | Resolution: {market.get('ttl_hours', 999) / 24:.0f}d | Category: {cluster}
Best Ask: ${polymarket_prob:.3f}
{gap_info}
ANCHORING WARNING: Do NOT simply return a probability near the market price. The market price already reflects the crowd. You must independently assess the TRUE probability based on the underlying event. If you cannot find a strong reason the probability should be higher, return the market price, but DO NOT default to 2x the price without reasoning.

Task: Identify 2-3 SPECIFIC factors. Estimate TRUE probability. Rate confidence.

Return ONLY JSON:
{{"factors": [{{"factor": "description", "direction": "supports/opposes", "weight": "high/medium/low", "source": "source"}}], "estimated_probability": 0.XX, "confidence": 0.XX, "reasoning": "brief"}}

Rules:
- estimated_probability: decimal 0.0-1.0 (NOT percentage)
- For markets with >180d to resolution, uncertainty favors YES
- Conservative on low-volume (<$10K) markets
- If estimate >3x price, explain what crowd is missing
- IMPORTANT: For DOTM markets (price < 5%), even small probability increases are significant"""

    resp = None
    for _attempt in range(3):
        try:
            resp = requests.post(URL, headers=HEADERS, json={
                "model": MODEL_MAIN,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 500
            }, timeout=60)
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as _e:
            if _attempt < 2:
                _backoff = 2 ** (_attempt + 1)
                logger.warning(f"[ANALYSIS] LLM retry {_attempt+1}/3 in {_backoff}s: {_e}")
                time.sleep(_backoff)
            else:
                logger.error(f"[ANALYSIS] LLM error after 3 retries: {_e}")
                p_model_llm = market["price"] * 2
                factors = []
                resp = None
    try:
        if resp is None:
            raise Exception("LLM unavailable after retries")
        resp_data = resp.json()
        msg = resp_data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content") or ""
        if not content:
            content = msg.get("reasoning") or ""

        result = parse_llm_json(content)
        if result:
            p_model_llm = normalize_probability(result.get("estimated_probability", market["price"] * 2))
            confidence = min(max(float(result.get("confidence", confidence)), 0.1), 0.95)
            factors = result.get("factors", [])
        else:
            p_model_llm = market["price"] * 2
            factors = []
    except Exception as e:
        logger.error(f"[ANALYSIS] LLM error: {e}")
        p_model_llm = market["price"] * 2
        factors = []

    if metaculus_gap and metaculus_gap.get("signal_strength", 0) > 0.3:
        p_model_metaculus = metaculus_gap["metaculus_prob"]
        p_model = max(p_model_llm, p_model_metaculus)
        if p_model_metaculus > p_model_llm:
            logger.info(
                f"[METACULUS-OVERRIDE] LLM={p_model_llm:.1%} < Metaculus={p_model_metaculus:.1%}, "
                f"using Metaculus (gap={metaculus_gap['gap']:.1%}, signal={metaculus_gap['signal_strength']:.2f})"
            )
        source_signal = "metaculus_override"
        confidence = min(confidence + 0.10, 0.95)
    else:
        p_model = p_model_llm

    max_p_model = market["price"] * MAX_P_MODEL_RATIO
    if p_model > max_p_model:
        logger.info(f"[ANALYSIS] p_model={p_model:.1%} > max={max_p_model:.1%} (price={market['price']:.3f}), capping")
        p_model = max_p_model

    metaculus_prob_val = metaculus_gap.get("metaculus_prob") if metaculus_gap else None
    p_model_raw = p_model

    p_model, was_dampened = calibrate_prediction(p_model, market["price"], metaculus_prob_val, cluster=cluster)

    settings = get_settings()

    # Skip if our probability estimate is too low
    min_p_model = settings.get("min_p_model", MIN_P_MODEL)
    if p_model < min_p_model:
        logger.info(f"[ANALYSIS] p_model={p_model:.1%} < MIN_P_MODEL={min_p_model:.1%}, skipping")
        return {
            "question": market["question"],
            "slug": market["slug"],
            "market_price": market["price"],
            "p_model": p_model,
            "prob_ratio": 0,
            "confidence": confidence,
            "action": "SKIP",
            "factors": [],
            "source_signal": "default",
            "reasoning": f"p_model too low ({p_model:.1%})",
            "best_ask": best_ask
        }

    prob_ratio = p_model / market["price"] if market["price"] > 0 else 0

    supporting = [f for f in factors if f.get("direction") == "supports"]
    high_weight = [f for f in supporting if f.get("weight") == "high"]

    ratio_score = min(prob_ratio / 3.0, 1.0) * 30

    metaculus_alignment = 0
    metaculus_prob_val = metaculus_gap.get("metaculus_prob") if metaculus_gap else None
    if metaculus_prob_val is not None:
        diff_model_meta = abs(p_model - metaculus_prob_val)
        diff_meta_pm = abs(metaculus_prob_val - market["price"])
        if diff_model_meta < 0.05:
            metaculus_alignment = 10
            logger.info(
                f"[META-ALIGN] +10: p_model={p_model:.1%} ~ metaculus={metaculus_prob_val:.1%} (diff={diff_model_meta:.1%})"
            )
        elif p_model > metaculus_prob_val + 0.10 and diff_meta_pm < 0.03:
            metaculus_alignment = -20
            logger.info(
                f"[META-PENALTY] -20: p_model={p_model:.1%} >> metaculus={metaculus_prob_val:.1%} "
                f"and metaculus~price (diff={diff_meta_pm:.1%}), LLM hallucination suspected"
            )

    factor_score = min((len(supporting) + len(high_weight)) / 4, 1.0) * 20
    vol_score = min(market.get("volume", 0) / 1_000_000, 1.0) * 20
    ttl_days = market.get("ttl_hours", 0) / 24
    if ttl_days > 180:
        time_score = 20
    elif ttl_days > 90:
        time_score = 15
    elif ttl_days > 30:
        time_score = 10
    else:
        time_score = 0
    signal_score = ratio_score + factor_score + vol_score + time_score + metaculus_alignment + _cluster_score_adjustment(cluster)

    base_threshold = get_settings().get("signal_threshold", 55)
    if ttl_days > 90:
        min_signal = get_settings().get("signal_threshold_long_horizon", base_threshold + 10)
    elif ttl_days >= 31:
        min_signal = get_settings().get("signal_threshold_medium_horizon", base_threshold + 5)
    else:
        min_signal = base_threshold
    if source_signal == "metaculus_override":
        min_signal = max(min_signal - 10, 35)

    action = "BUY" if signal_score >= min_signal and confidence >= get_settings().get("min_confidence", MIN_CONFIDENCE) and prob_ratio >= MIN_PROB_RATIO else "SKIP"

    if supporting:
        print(f"   📊 Factors: {len(supporting)} supporting ({len(high_weight)} high)")

    logger.info(
        f"[SIGNAL] ratio={prob_ratio:.2f}x -> {ratio_score:.0f}, factors={len(supporting)} -> {factor_score:.0f}, "
        f"vol=${market.get('volume',0):,.0f} -> {vol_score:.0f}, ttl={ttl_days:.0f}d -> {time_score:.0f} "
        f"= {signal_score:.0f}/{min_signal} => {action}"
    )

    return {
        "question": market["question"],
        "slug": market["slug"],
        "market_price": market["price"],
        "p_model": p_model,
        "prob_ratio": prob_ratio,
        "confidence": confidence,
        "action": action,
        "factors": factors,
        "source_signal": source_signal,
        "signal_score": signal_score,
        "reasoning": f"score={signal_score:.0f}/{min_signal}(horizon), ratio={prob_ratio:.2f}x, conf={confidence:.2f}, src={source_signal}, meta_align={metaculus_alignment:+d}",
        "best_ask": best_ask
    }


def _load_price_tracking():
    tracking = load_json(PRICE_TRACKING_FILE, {})
    now = datetime.now()
    stale = [k for k, v in tracking.items()
             if isinstance(v, dict) and v.get("last_checked")
             and (now - datetime.fromisoformat(v["last_checked"])).total_seconds() > 86400]
    if stale:
        for k in stale:
            del tracking[k]
        _save_price_tracking(tracking)
    return tracking


def _save_price_tracking(tracking):
    save_json(PRICE_TRACKING_FILE, tracking)


def _check_price_delta(slug, current_price):
    """
    TAZ-3: Delta-scanning. Returns (should_analyze: bool, cached_p_model).
    If price changed < $0.005 since last check, reuse cached p_model.
    """
    tracking = _load_price_tracking()
    entry = tracking.get(slug)
    if entry:
        last_price = entry.get("last_price", 0)
        cached_p_model = entry.get("p_model")
        actual_threshold = max(PRICE_DELTA_THRESHOLD, current_price * 0.10)
        if cached_p_model is not None and abs(current_price - last_price) < actual_threshold:
            logger.info(
                f"[DELTA-SKIP] {slug[:40]}... price delta "
                f"${abs(current_price - last_price):.4f} < ${actual_threshold:.4f}, reusing p_model={cached_p_model:.1%}"
            )
            return False, cached_p_model
    return True, None


def _update_price_tracking(slug, current_price, p_model):
    tracking = _load_price_tracking()
    tracking[slug] = {
        "last_price": current_price,
        "p_model": p_model,
        "last_checked": datetime.now().isoformat(),
    }
    _save_price_tracking(tracking)


def _check_news_cache_freshness(cluster_key):
    """
    TAZ-3: Check source_cache.json freshness for a news cluster.
    Returns True if cache is fresh (< 6 hours), blocking new HTTP requests.
    """
    cache = load_json(CACHE_FILE, {"metaculus": {}, "news": {}, "last_update": None})
    news_section = cache.get("news", {})
    entry = news_section.get(cluster_key)
    if isinstance(entry, dict) and entry.get("timestamp"):
        try:
            cached_time = datetime.fromisoformat(entry["timestamp"])
            age_seconds = (datetime.now() - cached_time).total_seconds()
            if age_seconds < CACHE_TTL_SECONDS:
                logger.info(f"[CACHE-FRESH] news cluster '{cluster_key}' age={age_seconds/3600:.1f}h < 6h, using cache")
                return True
        except (ValueError, TypeError):
            pass
    return False


PRE_FILTER_OTHER_MIN_VOLUME = 100_000


def pre_filter_before_batching(markets):
    """
    TAZ-5: Pre-filter markets before batching to save LLM tokens.

    Markets classified as "other" cluster with volume < $100K are skipped
    instantly without LLM analysis. High-volume "other" markets are kept
    to avoid missing hyped events that lack keyword tags.

    Returns (kept: list[dict], skipped: list[dict])
    """
    kept = []
    skipped = []
    for m in markets:
        clusters = m.get("clusters", ["other"])
        if any(c in BANNED_CLUSTERS for c in clusters):
            skipped.append(m)
            continue
        is_other = clusters == ["other"] or (len(clusters) == 1 and clusters[0] == "other")
        if is_other:
            volume = m.get("volume", 0)
            if volume < PRE_FILTER_OTHER_MIN_VOLUME:
                slug = m.get("slug", "unknown")
                logger.info(f"[PRE-FILTER] Skipping low-volume 'other' market: {slug}")
                skipped.append(m)
                continue
        kept.append(m)
    return kept, skipped


def batch_analyze_markets(markets):
    """
    TAZ-2: Batch analysis. Groups 5-7 markets into a single LLM call.
    Returns list of analysis dicts (same schema as full_market_analysis).
    Falls back to individual analysis on parse failure.
    """
    if not markets:
        return []

    metaculus_cache = {}
    for m in markets:
        if m.get("price", 0) < 0.35:
            slug = m.get("slug", "")
            question = m.get("question", "")
            end_date = m.get("end_date")
            meta = get_metaculus_forecast(question, end_date)
            if meta.get("found"):
                metaculus_cache[slug] = meta.get("probability")
            else:
                metaculus_cache[slug] = None
        else:
            metaculus_cache[m.get("slug")] = None

    batch_items = []
    for m in markets:
        slug = m.get("slug", "")
        question = m.get("question", "")
        price = m.get("price", 0)
        volume = m.get("volume", 0)
        ttl_hours = m.get("ttl_hours", 999)
        cluster = m.get("clusters", ["other"])[0]
        batch_items.append({
            "slug": slug,
            "question": question,
            "market_price": round(price, 4),
            "volume": round(volume, 0),
            "ttl_hours": round(ttl_hours, 0),
            "cluster": cluster,
        })

    for item in batch_items:
        item["question"] = sanitize_for_prompt(item["question"])
    items_json = json.dumps(batch_items, indent=2)

    prompt = f"""Prediction market analyst. Analyze these DOTM (deep out-the-money) markets where the crowd may underestimates probability.

MARKETS (JSON array):
{items_json}

For EACH market, identify 2-3 specific factors and estimate the TRUE probability independently from the market price.

Return a JSON ARRAY with one object per market (same order as input):
[
  {{
    "slug": "market-slug",
    "factors": [{{"factor": "description", "direction": "supports/opposes", "weight": "high/medium/low", "source": "source"}}],
    "estimated_probability": 0.XX,
    "confidence": 0.XX,
    "reasoning": "brief explanation"
  }}
]

Rules:
- estimated_probability: decimal 0.0-1.0 (NOT percentage)
- Do NOT anchor on the market price - provide independent assessment
- Conservative on low-volume (<$10K) markets
- If estimate >3x price, explain what crowd is missing
- Return exactly {len(batch_items)} items matching the input slugs

CRITICAL REGULATION FOR CONFIDENCE SCORING: Do NOT default to a flat 0.65 confidence across multiple markets just to pass filters. You must utilize the full analytical spectrum from 0.65 to 1.0 based on the robustness of available data, market volume, and time to resolution. If a market has weak evidence, score it near 0.65. If the evidence is solid and aligned with predictive markets, score it between 0.80 and 0.95. Generating a flat 0.65 across the entire batch will break the downstream composite scoring and will be treated as an invalid evaluation."""

    resp = None
    for _attempt in range(3):
        try:
            resp = requests.post(URL, headers=HEADERS, json={
                "model": MODEL_MAIN,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2000
            }, timeout=90)
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as _e:
            if _attempt < 2:
                _backoff = 2 ** (_attempt + 1)
                logger.warning(f"[BATCH] LLM retry {_attempt+1}/3 in {_backoff}s: {_e}")
                time.sleep(_backoff)
            else:
                logger.error(f"[BATCH] LLM error after 3 retries: {_e}")
                resp = None
    try:
        if resp is None:
            raise Exception("LLM unavailable after retries")
        resp_data = resp.json()
        msg = resp_data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content") or ""
        if not content:
            content = msg.get("reasoning") or ""

        batch_results = _parse_batch_response(content, batch_items, metaculus_cache)
        if batch_results and len(batch_results) == len(markets):
            logger.info(f"[BATCH] Successfully parsed batch of {len(batch_results)} markets")
            return batch_results
        else:
            logger.warning(f"[BATCH] Batch parse returned {len(batch_results) if batch_results else 0}/{len(markets)} items, falling back to individual")
    except Exception as e:
        logger.error(f"[BATCH] LLM error: {e}, falling back to individual analysis")

    results = []
    for m in markets:
        results.append(full_market_analysis(m))
    return results


def _parse_batch_response(content, batch_items, metaculus_cache=None):
    """
    Parse batch LLM response. Expects a JSON array.
    Uses balanced-brace parsing to handle nested objects.
    """
    if metaculus_cache is None:
        metaculus_cache = {}

    start = content.find('[')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(content)):
        c = content[i]
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
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                try:
                    arr = json.loads(content[start:i + 1])
                    if isinstance(arr, list):
                        return _build_batch_results(arr, batch_items, metaculus_cache)
                except json.JSONDecodeError:
                    pass
                break

    fallback = re.search(r'\[.*\]', content, re.DOTALL)
    if fallback:
        try:
            arr = json.loads(fallback.group(0))
            if isinstance(arr, list):
                return _build_batch_results(arr, batch_items, metaculus_cache)
        except json.JSONDecodeError:
            pass

    return None


def _build_batch_results(parsed_array, batch_items, metaculus_cache=None):
    """
    Convert parsed batch array into list of analysis dicts
    matching the full_market_analysis schema.
    """
    slug_to_item = {it["slug"]: it for it in batch_items}
    slug_to_idx = {it["slug"]: i for i, it in enumerate(batch_items)}

    results_map = {}
    for item in parsed_array:
        if not isinstance(item, dict):
            continue

        slug = item.get("slug", "")
        if slug not in slug_to_item:
            if len(results_map) < len(batch_items):
                unmatched_idx = len(results_map)
                slug = batch_items[unmatched_idx]["slug"]
            else:
                continue

        bi = slug_to_item.get(slug)
        if not bi:
            continue

        market_price = bi["market_price"]
        cluster = bi["cluster"]

        p_model_llm = normalize_probability(item.get("estimated_probability", market_price * 2))
        confidence = min(max(float(item.get("confidence", 0.6)), 0.1), 0.95)
        factors = item.get("factors", [])

        max_p_model = market_price * MAX_P_MODEL_RATIO
        p_model = min(p_model_llm, max_p_model)

        metaculus_prob = None
        if metaculus_cache:
            metaculus_prob = metaculus_cache.get(slug)

        p_model_raw = p_model

        p_model, _ = calibrate_prediction(p_model, market_price, metaculus_prob, cluster=cluster)

        settings = get_settings()
        min_p_model = settings.get("min_p_model", MIN_P_MODEL)
        if p_model < min_p_model:
            results_map[slug] = {
                "question": bi["question"],
                "slug": slug,
                "market_price": market_price,
                "p_model": p_model,
                "prob_ratio": 0,
                "confidence": confidence,
                "action": "SKIP",
                "factors": [],
                "source_signal": "default",
                "reasoning": f"p_model too low ({p_model:.1%})",
                "best_ask": None,
            }
            continue

        prob_ratio = p_model / market_price if market_price > 0 else 0
        supporting = [f for f in factors if f.get("direction") == "supports"]
        high_weight = [f for f in supporting if f.get("weight") == "high"]

        ratio_score = min(prob_ratio / 3.0, 1.0) * 30

        metaculus_prob_val = metaculus_cache.get(slug) if metaculus_cache else None
        metaculus_alignment = 0
        if metaculus_prob_val is not None:
            diff_model_meta = abs(p_model - metaculus_prob_val)
            diff_meta_pm = abs(metaculus_prob_val - market_price)
            if diff_model_meta < 0.05:
                metaculus_alignment = 10
                logger.info(f"[META-ALIGN-BATCH] +10: p_model={p_model:.1%} ~ metaculus={metaculus_prob_val:.1%}")
            elif p_model > metaculus_prob_val + 0.10 and diff_meta_pm < 0.03:
                metaculus_alignment = -20
                logger.info(f"[META-PENALTY-BATCH] -20: p_model={p_model:.1%} >> metaculus={metaculus_prob_val:.1%}")

        factor_score = min((len(supporting) + len(high_weight)) / 4, 1.0) * 20
        vol_score = min(bi.get("volume", 0) / 1_000_000, 1.0) * 20
        ttl_hours = bi.get("ttl_hours", 999)
        ttl_days = ttl_hours / 24
        if ttl_days > 180:
            time_score = 20
        elif ttl_days > 90:
            time_score = 15
        elif ttl_days > 30:
            time_score = 10
        else:
            time_score = 0

        signal_score = ratio_score + factor_score + vol_score + time_score + metaculus_alignment + _cluster_score_adjustment(cluster)

        base_threshold = settings.get("signal_threshold", 55)
        if ttl_days > 90:
            min_signal = settings.get("signal_threshold_long_horizon", base_threshold + 10)
        elif ttl_days >= 31:
            min_signal = settings.get("signal_threshold_medium_horizon", base_threshold + 5)
        else:
            min_signal = base_threshold

        action = "BUY" if signal_score >= min_signal and confidence >= settings.get("min_confidence", MIN_CONFIDENCE) and prob_ratio >= MIN_PROB_RATIO else "SKIP"

        logger.info(
            f"[SIGNAL-BATCH] ratio={prob_ratio:.2f}x -> {ratio_score:.0f}, factors={len(supporting)}/{len(high_weight)} -> {factor_score:.0f}, "
            f"vol=${bi.get('volume',0):,.0f} -> {vol_score:.0f}, ttl={ttl_days:.0f}d -> {time_score:.0f} "
            f"= {signal_score:.0f}/{min_signal} => {action}"
        )

        results_map[slug] = {
            "question": bi["question"],
            "slug": slug,
            "market_price": market_price,
            "p_model": p_model,
            "prob_ratio": prob_ratio,
            "confidence": confidence,
            "action": action,
            "factors": factors,
            "source_signal": "default",
            "signal_score": signal_score,
            "reasoning": f"score={signal_score:.0f}/{min_signal}(batch), ratio={prob_ratio:.2f}x, conf={confidence:.2f}",
            "best_ask": None,
        }

    results = []
    for bi in batch_items:
        if bi["slug"] in results_map:
            results.append(results_map[bi["slug"]])
        else:
            results.append({
                "question": bi["question"],
                "slug": bi["slug"],
                "market_price": bi["market_price"],
                "p_model": bi["market_price"] * 2,
                "prob_ratio": 2.0,
                "confidence": 0.5,
                "action": "SKIP",
                "factors": [],
                "source_signal": "default",
                "reasoning": "batch_parse_fallback",
                "best_ask": None,
            })

    return results


ADVISOR_MODEL = "deepseek-reasoner"
ADVISOR_MIN_CONFIDENCE = 0.70


def advisor_pre_check(market, analysis):
    """
    Two-factor trade verification via independent Advisor (deepseek-reasoner).
    Uses Chain-of-Thought reasoning to validate or reject the bot's trade thesis.

    Returns (approved: bool, verdict: str, confidence: float, reason: str)
    """
    question = market.get("question", "")
    slug = market.get("slug", "")
    price = market.get("price", 0)
    p_model = analysis.get("p_model", 0)
    factors = analysis.get("factors", [])
    score = analysis.get("signal_score", 0)
    reasoning = analysis.get("reasoning", "")

    factors_text = "\n".join(
        f"  - [{f.get('weight', '?')}] {sanitize_for_prompt(f.get('factor', ''))} ({f.get('direction', '')})"
        for f in factors[:5]
    ) if factors else "  (none)"

    prompt = f"""You are DOTM Advisor - an independent risk analyst verifying a trade before execution.
Use Chain-of-Thought reasoning to evaluate the thesis.

MARKET: {sanitize_for_prompt(question)}
SLUG: {slug}
MARKET PRICE: ${price:.3f} ({price*100:.1f}%)
BOT P_MODEL (estimated true probability): {p_model:.1%}
PROBABILITY RATIO: {p_model/price:.2f}x vs market
COMPOSITE SIGNAL SCORE: {score:.0f}/100
BOT REASONING: {sanitize_for_prompt(reasoning)}

SUPPORTING FACTORS IDENTIFIED BY BOT:
{factors_text}

YOUR TASK:
1. Think step-by-step about whether the bot's thesis is sound.
2. Consider: Is the probability estimate realistic? Are the factors genuine or generic?
3. Check for hallucination patterns: p_model >> market price without concrete catalyst.
4. Is this a DOTM market where the crowd truly underestimates probability?

Return ONLY JSON:
{{"p_estimate": 0.XX, "confidence": 0.XX, "factors": ["factor1", "factor2"], "verdict": "CONFIRM/DIVERGE/WARNING/UNKNOWN"}}

Rules:
- CONFIRM: You agree the trade has positive expected value. confidence >= 0.70 required.
- DIVERGE: Your analysis contradicts the bot (e.g., you think probability is LOWER).
- WARNING: Significant risk factor the bot missed. Do NOT trade.
- UNKNOWN: Insufficient information to verify. Default safe choice."""

    try:
        resp = requests.post(URL, headers=HEADERS, json={
            "model": ADVISOR_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 2000
        }, timeout=120)

        resp_data = resp.json()
        msg = resp_data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""

        if not content and reasoning:
            json_match = re.search(r'\{[^{}]*\}', reasoning)
            if json_match:
                content = json_match.group(0)
                logger.info("[ADVISOR] Extracted JSON from reasoning_content (content was empty)")

        if not content:
            logger.warning("[ADVISOR] Empty response from reasoner, blocking trade")
            return False, "UNKNOWN", 0.0, "advisor_empty_response"

        from advisor_script import parse_llm_advisor_response
        result, parse_err = parse_llm_advisor_response(content, log_label="ADVISOR-PRE")
        if result is None:
            logger.warning(f"[ADVISOR] Parse failed: {parse_err}, blocking trade")
            return False, "UNKNOWN", 0.0, f"advisor_parse_error: {parse_err}"

        verdict = result.get("verdict", "UNKNOWN")
        confidence = result.get("confidence", 0.0)
        advisor_p = result.get("p_estimate", 0)
        advisor_factors = result.get("factors", [])

        logger.info(
            f"[ADVISOR] verdict={verdict} conf={confidence:.2f} "
            f"advisor_p={advisor_p:.1%} vs bot_p={p_model:.1%} | "
            f"factors: {advisor_factors[:2]}"
        )

        if verdict == "CONFIRM" and confidence >= ADVISOR_MIN_CONFIDENCE:
            logger.info(f"[ADVISOR] ✅ Trade APPROVED by advisor ({verdict}, conf={confidence:.2f})")
            return True, verdict, confidence, "approved"
        else:
            reason = f"advisor_veto: verdict={verdict}, conf={confidence:.2f}"
            logger.info(f"[ADVISOR] 🚫 Trade BLOCKED: {reason}")
            print(f"   🚫 Advisor veto: {verdict} (conf={confidence:.2f})")
            return False, verdict, confidence, reason

    except requests.exceptions.Timeout:
        logger.warning("[ADVISOR] Timeout, blocking trade")
        return False, "UNKNOWN", 0.0, "advisor_timeout"
    except Exception as e:
        logger.error(f"[ADVISOR] Error: {e}")
        return False, "UNKNOWN", 0.0, f"advisor_error: {str(e)[:80]}"


SLIPPAGE_LOG_FILE = "/root/dotm-sniper/logs/slippage.json"


def get_actual_fill_price(slug):
    try:
        res = subprocess.run(
            ["pm-trader", "history", "--limit", "5"],
            capture_output=True, text=True, timeout=15, start_new_session=True
        )
        data = json.loads(res.stdout)
        for trade in data.get("data", []):
            if trade.get("market_slug") == slug and trade.get("side") == "buy":
                return {
                    "avg_price": float(trade.get("avg_price", 0)),
                    "amount_usd": float(trade.get("amount_usd", 0)),
                    "shares": float(trade.get("shares", 0)),
                    "slippage": float(trade.get("slippage", 0)),
                    "levels_filled": int(trade.get("levels_filled", 0)),
                }
    except Exception as e:
        logger.warning(f"[SLIPPAGE] Failed to get fill price for {slug}: {e}")
    return None


def log_slippage(slug, expected_price, fill_data):
    if not fill_data:
        return
    actual_price = fill_data["avg_price"]
    slippage_pct = (actual_price - expected_price) / expected_price if expected_price > 0 else 0

    entry = {
        "slug": slug,
        "expected_price": expected_price,
        "actual_price": actual_price,
        "slippage_pct": round(slippage_pct, 4),
        "amount_usd": fill_data["amount_usd"],
        "shares": fill_data["shares"],
        "levels_filled": fill_data["levels_filled"],
        "timestamp": datetime.now().isoformat(),
    }

    try:
        logs = load_json(SLIPPAGE_LOG_FILE, [])
        logs.append(entry)
        logs = logs[-500:]
        os.makedirs(os.path.dirname(SLIPPAGE_LOG_FILE), exist_ok=True)
        save_json(SLIPPAGE_LOG_FILE, logs)
    except Exception as e:
        logger.warning(f"[SLIPPAGE] Failed to write log: {e}")

    if abs(slippage_pct) > 0.05:
        logger.warning(
            f"[SLIPPAGE-HIGH] {slug[:40]}... expected=${expected_price:.4f} "
            f"actual=${actual_price:.4f} slippage={slippage_pct:+.2%} "
            f"({fill_data['levels_filled']} levels, ${fill_data['amount_usd']:.2f})"
        )
    else:
        logger.info(
            f"[SLIPPAGE] {slug[:40]}... expected=${expected_price:.4f} "
            f"actual=${actual_price:.4f} slippage={slippage_pct:+.2%}"
        )


def execute_trade(market, estimated_size, factors, analysis, balance):
    """Execute trade with advisor pre-check. Returns True if successful."""
    approved, verdict, adv_conf, adv_reason = advisor_pre_check(market, analysis)
    if not approved:
        logger.info(f"[TRADE-BLOCKED] {market['slug']}: {adv_reason}")
        return False

    current_ask = get_best_ask(market["slug"])
    if current_ask is not None and current_ask > market["price"] * 1.15:
        logger.warning(f"[SNIPER] Slippage guard: ask={current_ask:.4f} > 15% above price={market['price']:.4f}, aborting")
        return False

    if not buy(market, estimated_size):
        print(f"   ❌ Buy failed for {market['slug']}")
        return False

    time.sleep(2)
    fill_data = get_actual_fill_price(market["slug"])
    if fill_data:
        log_slippage(market["slug"], market["price"], fill_data)

    shares = round(float(fill_data.get("shares", 0))) if fill_data and fill_data.get("shares", 0) > 0 else round(estimated_size / market["price"]) if market["price"] > 0 else 0
    if shares > 0:
        ladder_results = _place_tp_ladder(market["slug"], market["outcome"], shares)
        for price, shares_placed, ok, method in ladder_results:
            if ok:
                print(f"   🎯  TP rung placed @${price:.2f} ({shares_placed} shares)")
            else:
                print(f"   ⚠️  TP rung @{price:.2f} failed")
        if not ladder_results:
            print(f"   ⚠️  TP ladder placement failed, will rely on trailing_stop_check()")
    else:
        logger.warning(f"[SMART-EXIT] Zero shares for {market['slug']}, skipping TP")

    db = load_hypothesis_db()
    db["hypotheses"].append({
        "slug": market["slug"],
        "question": market["question"],
        "market_price": market["price"],
        "p_model": analysis["p_model"],
        "prob_ratio": analysis["prob_ratio"],
        "confidence": analysis["confidence"],
        "factors": factors,
        "clusters": market["clusters"],
        "size_pct": estimated_size / balance,
        "created_at": datetime.now().isoformat(),
        "resolved": False,
        "tp_limit_placed": True,
        "tp_limit_price": SMART_EXIT_PRICE,
    })
    save_hypothesis_db(db)

    if telegram_reporter:
        meta_prob = analysis.get("p_model")
        telegram_reporter.alert_new_position(
            market_slug=market["slug"],
            question=market["question"],
            entry_price=market["price"],
            amount=estimated_size,
            metaculus_prob=meta_prob,
            factors=factors,
            reasoning=analysis.get("reasoning", "")
        )

    return True


def _update_status_file():
    try:
        import shutil
        res = subprocess.run(["pm-trader", "balance"], capture_output=True, text=True, timeout=15, start_new_session=True)
        balance_data = json.loads(res.stdout).get("data", {})
        res = subprocess.run(["pm-trader", "portfolio"], capture_output=True, text=True, timeout=15, start_new_session=True)
        portfolio_data = json.loads(res.stdout).get("data", [])
        status = {"balance": balance_data, "portfolio": portfolio_data, "updated_at": datetime.now().isoformat()}
        save_json("/root/dotm-sniper/current_status.json", status)
        for dest in [
            "/root/.openclaw/workspace/dotm_status.json",
            "/root/.openclaw/agents/market_analyst/dotm_status.json",
            "/root/.openclaw/workspace/memory/portfolio-current.json",
        ]:
            try:
                shutil.copy("/root/dotm-sniper/current_status.json", dest)
            except Exception:
                pass
    except Exception:
        pass


def main():
    _main_inner()

def _main_inner():
    print("="*60)
    print("  DOTM SNIPER v5.3.0 - Batch Processing + Delta Scan + Advisor")
    print("="*60)

    repair_positions_file()

    settings = get_settings()
    last_backtest = settings.get("last_backtest_timestamp", 0)
    import time as _time
    now_ts = _time.time()
    if now_ts - last_backtest >= BACKTEST_COOLDOWN_SECONDS:
        bt = backtest_recent(n=20)
        if "error" not in bt:
            print(f"🧪 Backtest: winrate={bt['winrate']:.1%}, brier={bt['avg_brier']:.3f}, pnl={bt['avg_pnl']:.2f}")
            if bt.get("recommendations"):
                for r in bt["recommendations"]:
                    print(f"   ⚠️  {r['issue']}: {r['suggestion']}")
        settings["last_backtest_timestamp"] = now_ts
        save_settings(settings)
    else:
        hours_ago = (now_ts - last_backtest) / 3600
        logger.info(f"[BACKTEST-COOLDOWN] Skipping, last run {hours_ago:.1f}h ago (cooldown={BACKTEST_COOLDOWN_SECONDS/3600:.0f}h)")
    # Read max positions from settings, fallback to hardcoded default
    max_positions = settings.get("MAX_CONCURRENT_TRADES", MAX_POSITIONS)
    print(f"⚙️ Thresholds: signal={settings.get('signal_threshold', 55)}, min_p_model={settings.get('min_p_model', 0.05):.0%}, confidence={settings['min_confidence']:.2f}, max_pos={max_positions}")

    resolve_hypotheses()
    trailing_stop_check()

    balance_data = get_balance()
    if balance_data is None:
        print("⚠️ Could not fetch balance, skipping cycle")
        return
    balance = balance_data.get("cash", 100)
    total_balance = balance_data.get("total_value", balance)
    print(f"💰 Balance: ${balance:.2f} (total: ${total_balance:.2f})")

    portfolio = get_portfolio()
    print(f"📊 Open positions: {len(portfolio)}")

    _update_status_file()

    if len(portfolio) >= max_positions:
        print(f"⚠️ Max positions ({max_positions}) reached")
        return

    markets = fetch_markets()
    if not markets:
        print("No markets found")
        return

    print(f"📈 Candidates: {len(markets)}")

    candidates_bought = 0
    available_balance = balance

    db = load_hypothesis_db()
    current_positions_for_clusters = [
        {"clusters": h.get("clusters", []), "size_pct": h.get("size_pct", 0)}
        for h in db.get("hypotheses", []) if not h.get("resolved")
    ]

    existing_slugs = {h["slug"] for h in db.get("hypotheses", []) if not h.get("resolved")}
    market_analyses = {}

    candidates_to_analyze = []
    for m in markets:
        if len(portfolio) + candidates_bought >= max_positions:
            break
        if available_balance < 5:
            break
        can_pass, _ = check_cluster_limits(m["clusters"], current_positions_for_clusters)
        if not can_pass:
            continue
        if m["slug"] in existing_slugs:
            continue
        should_analyze, cached_p = _check_price_delta(m["slug"], m["price"])
        if not should_analyze and cached_p is not None:
            print(f"   ⏭️ {m['question'][:45]}... price unchanged, cached p={cached_p:.1%}")
            continue
        candidates_to_analyze.append(m)

    if candidates_to_analyze:
        candidates_to_analyze, pre_filtered = pre_filter_before_batching(candidates_to_analyze)
        if pre_filtered:
            print(f"   🔎 Pre-filtered {len(pre_filtered)} low-volume 'other' markets")
        print(f"\n📊 Batch-analyzing {len(candidates_to_analyze)} candidates (batch_size={BATCH_SIZE})...")
        for batch_start in range(0, len(candidates_to_analyze), BATCH_SIZE):
            batch = candidates_to_analyze[batch_start:batch_start + BATCH_SIZE]
            print(f"\n--- Batch {batch_start // BATCH_SIZE + 1} ({len(batch)} markets) ---")
            batch_results = batch_analyze_markets(batch)
            for m, analysis in zip(batch, batch_results):
                _update_price_tracking(m["slug"], m["price"], analysis["p_model"])
                market_analyses[m["slug"]] = (m, analysis)

    for m in markets:
        if len(portfolio) + candidates_bought >= max_positions:
            break

        if available_balance < 5:
            break

        can_pass, reason = check_cluster_limits(m["clusters"], current_positions_for_clusters)
        if not can_pass:
            continue

        if m["slug"] in existing_slugs:
            continue

        if m["slug"] in market_analyses:
            _, analysis = market_analyses[m["slug"]]
        else:
            should_analyze, cached_p = _check_price_delta(m["slug"], m["price"])
            if not should_analyze and cached_p is not None:
                continue
            analysis = full_market_analysis(m)
            _update_price_tracking(m["slug"], m["price"], analysis["p_model"])

        print(f"\n🔍 {m['question'][:55]}...")
        print(f"   Price: ${m['price']:.3f} | TTL: {m['ttl_hours']:.0f}h | Vol: ${m['volume']:,.0f}")
        print(f"   📈 P_model: {analysis['p_model']:.1%} | Ratio: {analysis.get('prob_ratio', 0):.2f}x | Conf: {analysis['confidence']:.2f}")

        if analysis["action"] == "SKIP":
            print(f"   ⏭️ Below threshold")
            continue

        factors = analysis.get("factors", [])

        estimated_size = position_size(
            analysis["p_model"],
            m["price"],
            available_balance,
            confidence=analysis["confidence"],
            best_ask=analysis.get("best_ask"),
            cluster=m.get("clusters", ["other"])[0]
        )

        if estimated_size <= 0:
            print(f"   ⏭️ Kelly edge negative, skipping")
            continue

        can_size, size_reason = check_category_limits(
            new_market=m,
            new_order_value=estimated_size,
            total_balance=total_balance,
            portfolio=portfolio
        )
        if not can_size:
            print(f"   ⏭️ Category limit: {size_reason}")
            continue

        print(f"   💵 Position size: ${estimated_size} ({estimated_size/available_balance:.1%} of balance)")

        market_for_news = {
            "question": m["question"],
            "slug": m["slug"],
            "price": m["price"],
            "metaculus_prob": analysis.get("p_model")
        }

        cluster_key = m.get("clusters", ["other"])[0]
        cache_fresh = _check_news_cache_freshness(cluster_key)
        if not cache_fresh:
            news_passed, news_reason = check_market_news(market_for_news)
            if not news_passed:
                print(f"   🚨 Trade blocked by news: {news_reason}")
                logger.info(f"[NEWS-BLOCK] {m['slug']}: {news_reason}")
                continue
        else:
            print(f"   📰 News cache fresh for cluster '{cluster_key}', skipping news check")

        if execute_trade(m, estimated_size, factors, analysis, balance):
            candidates_bought += 1
            available_balance -= estimated_size
            cluster = m.get("clusters", ["other"])[0]
            current_positions_for_clusters.append({
                "clusters": [cluster],
                "size_pct": estimated_size / total_balance
            })

    print(f"\n✅ Bought: {candidates_bought} | Available: ${available_balance:.2f}")

    update_daily_stats(balance_data, portfolio, candidates_bought)

    db = load_hypothesis_db()
    resolved = db.get("resolved", [])
    if len(resolved) >= BURN_IN_TRADES:
        recent = resolved[-BURN_IN_TRADES:]
        wins = sum(1 for h in recent if h.get("outcome") == "YES")
        logger.info(f"Cycle complete: bought={candidates_bought}, recent_winrate={wins/len(recent):.1%}")

if __name__ == "__main__":
    import sys
    single_run = len(sys.argv) > 1 and sys.argv[1] == "--once"

    if not check_and_write_pid(PID_FILE):
        sys.exit(1)
    try:
        if single_run:
            print("DOTM SNIPER v5.3.0 running single iteration...")
            _main_inner()
        else:
            print("DOTM SNIPER v5.3.0 starting...")
            while True:
                try:
                    _main_inner()
                except Exception as e:
                    print(f"Error: {e}")
                print("Sleeping 30 min...")
                time.sleep(1800)
    finally:
        cleanup_pid_file(PID_FILE)