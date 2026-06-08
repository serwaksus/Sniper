"""
signal_pipeline.py — Signal generation and market analysis pipeline.
Extracted from dotm_sniper.py v5.3.0.
"""
import json
import logging
import os
import re
import requests
import subprocess
import sys
import time
from datetime import datetime, UTC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_json, save_json, sanitize_for_prompt
from schema import *
from utils import load_env_file

load_env_file()

HYPOTHESIS_DB_FILE = "/root/dotm-sniper/hypothesis_db.json"

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

# ── DeepSeek API ─────────────────────────────────────────────
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL_MAIN = "deepseek-chat"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
URL = "https://api.deepseek.com/v1/chat/completions"

# ── Metaculus API ─────────────────────────────────────────────
METACULUS_API_KEY = os.environ.get("METACULUS_TOKEN", "")
METACULUS_URL = "https://www.metaculus.com/api2/questions/"
METACULUS_HEADERS = {"Authorization": f"Token {METACULUS_API_KEY}"}

# ── Signal thresholds ────────────────────────────────────────
MIN_PROB_RATIO = 2.0
MIN_P_MODEL = 0.03
MAX_P_MODEL_RATIO = 5.0
MIN_CONFIDENCE = 0.65
MIN_VOLUME = 25000
MIN_TTL_HOURS = 48
MAX_PRICE = 0.40
ALLOWED_CLUSTERS = {"ai_tech", "russia_ukraine", "usa_politics", "fed_fomc", "sports_nba", "sports_ufc"}
BANNED_CLUSTERS = {"crypto"}
PRE_FILTER_OTHER_MIN_VOLUME = 100_000
BATCH_SIZE = 6
CALIBRATION_DAMPING_FACTOR = 0.65
CALIBRATION_DOTM_THRESHOLD = 0.10
CALIBRATION_AGGRESSIVE_PMODEL = 0.20
CALIBRATION_METACULUS_LOW = 0.10
CLUSTER_SCORE_ADJUSTMENTS = {
    "other": 15,
    "crypto": -25,
    "sports_nba": -15,
}
DISPERSION_PENALTY_THRESHOLD = 0.25
METACULUS_GAP_THRESHOLD = 0.08
ADVISOR_MODEL = "deepseek-reasoner"
ADVISOR_MIN_CONFIDENCE = 0.70
DATE_WINDOW_DAYS = 7

CACHE_FILE = "/root/dotm-sniper/source_cache.json"


# ── Cache helpers ────────────────────────────────────────────

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


def save_cache(cache):
    cache["last_update"] = datetime.now().isoformat()
    save_json(CACHE_FILE, cache)


# ═══════════════════════════════════════════════════════════════
# GROUP D — Metaculus integration (no circular deps)
# ═══════════════════════════════════════════════════════════════

def metaculus_search(query, limit=10):
    try:
        resp = requests.get(METACULUS_URL, headers=METACULUS_HEADERS,
                          params={"search": query, "limit": limit, "status": "open"},
                          timeout=15)
        if resp.status_code == 200:
            return resp.json().get("results", [])
    except Exception as e:
        logger.warning(f"[metaculus_search] {type(e).__name__}: {e}")
    return []

def metaculus_get_question(qid):
    try:
        resp = requests.get(f"{METACULUS_URL}{qid}/", headers=METACULUS_HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning(f"[metaculus_get_question] {type(e).__name__}: {e}")
    return None

def parse_resolve_date(date_str):
    if not date_str:
        return None
    date_str = date_str.replace("Z", "+00:00")
    try:
        from dateutil.parser import parse
        return parse(date_str)
    except Exception as e:
        logger.debug(f"[parse_resolve_date] {type(e).__name__}: {e}")
    try:
        return datetime.fromisoformat(date_str)
    except Exception as e:
        logger.debug(f"[parse_resolve_date] {type(e).__name__}: {e}")
    return None

def dates_match(date1, date2, window_days=DATE_WINDOW_DAYS):
    d1 = parse_resolve_date(date1)
    d2 = parse_resolve_date(date2)
    if d1 is None or d2 is None:
        return False
    if d1.tzinfo is not None:
        d1 = d1.astimezone(UTC).replace(tzinfo=None)
    if d2.tzinfo is not None:
        d2 = d2.astimezone(UTC).replace(tzinfo=None)
    diff = abs((d1 - d2).total_seconds() / 86400)
    return diff <= window_days

def get_metaculus_forecast(pm_question, pm_resolve_date=None):
    cache = load_cache()
    cache_key = pm_question

    if cache_key in cache.get("metaculus", {}):
        cached = cache["metaculus"][cache_key]
        if cached.get("timestamp"):
            cached_time = datetime.fromisoformat(cached["timestamp"])
            if (datetime.now() - cached_time).total_seconds() < 3600:
                return cached

    search_queries = [pm_question, *_generate_search_queries(pm_question)]

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
        except Exception as e:
            logger.debug(f"[cp_reveal_parse] {type(e).__name__}: {e}")

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

    prob = None
    if latest:
        means = latest.get("means", [])
        if means:
            prob = float(means[0])

    if prob is None:
        pred = q_data.get("prediction") or best_match.get("prediction")
        if pred and isinstance(pred, dict):
            prob = pred.get("number")
            if prob is None:
                prob = pred.get("p_above")
            if prob is None:
                prob = pred.get("p_below")
            prob = float(prob) if prob is not None else 0.0

    if prob is None:
        vote = best_match.get("vote", {})
        if vote and isinstance(vote, dict):
            prob = float(vote.get("prediction", 0))

    if prob is None:
        return {"found": False, "probability": None, "reason": "no_aggregation", "best_match_title": meta_title}
    forecaster_count = latest.get("forecaster_count", 0) if latest else 0
    title = best_match.get("title", "") or best_match.get("short_title", "")

    q1 = (latest or {}).get("q1")
    q3 = (latest or {}).get("q3")
    std = (latest or {}).get("std")
    dispersion = None
    dispersion_penalty = 1.0

    if q1 is not None and q3 is not None:
        dispersion = q3 - q1
    elif std is not None:
        dispersion = std

    if dispersion is not None and dispersion > DISPERSION_PENALTY_THRESHOLD:
        dispersion_penalty = max(0.0, 1.0 - (dispersion - DISPERSION_PENALTY_THRESHOLD))
        logger.info(f"[DISPERSION] q1={q1!r}, q3={q3!r}, dispersion={dispersion:.3f}, penalty={dispersion_penalty:.2f}")

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
    for i in range(len(words)):
        for j in range(i+1, min(i+5, len(words)+1)):
            phrase = " ".join(words[i:j])
            if len(phrase) >= 4:
                queries.append(phrase)
                if len(queries) >= 5:
                    return queries
    return queries


def _calculate_metaculus_match(pm_question, result):
    """Calculate relevance score between PM question and Metaculus result."""
    from fuzzywuzzy import fuzz
    pm_lower = pm_question.lower()
    meta_title = (result.get("title", "") or result.get("short_title", "")).lower()

    pm_words = set(re.sub(r'[^\w\s]', ' ', pm_lower).split())
    meta_words = set(re.sub(r'[^\w\s]', ' ', meta_title).split())

    stop = {"will", "the", "a", "an", "be", "by", "of", "in", "on", "at", "to", "for", "is", "are", "was", "were", "before", "after", "any", "this", "that"}
    pm_clean = pm_words - stop
    meta_clean = meta_words - stop

    overlap = len(pm_clean & meta_clean)
    base_score = overlap / max(len(pm_clean), 1) if pm_clean else 0

    key_phrases = ["ai safety", "artificial intelligence", "anthropic", "ukraine", "nato", "nuclear", "china", "taiwan", "trump", "fed", "powell", "bitcoin"]
    substring_bonus = 0
    for phrase in key_phrases:
        if phrase in pm_lower and phrase in meta_title:
            substring_bonus += 0.15
        elif phrase in pm_lower and phrase.split()[0] in meta_title:
            substring_bonus += 0.05

    pm_nums = set(w for w in pm_clean if any(c.isdigit() for c in w))
    meta_nums = set(w for w in meta_clean if any(c.isdigit() for c in w))
    if pm_nums and meta_nums and pm_nums & meta_nums:
        base_score += 0.1

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

    required_gap = get_time_decay_threshold(market.get("end_date"))

    gap = metaculus_prob - price_to_use

    if gap <= required_gap:
        logger.info(f"[GAP-SKIP] gap={gap:.3f} <= required={required_gap:.3f} ({market['question'][:40]})")
        return None

    dispersion_penalty = meta.get("dispersion_penalty", 1.0)

    raw_strength = min(gap / 0.15, 1.0)

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


# ═══════════════════════════════════════════════════════════════
# GROUP E — Signal pipeline (calibration → analysis → batch)
# ═══════════════════════════════════════════════════════════════

def normalize_probability(p):
    if p is None:
        return 0
    p = float(p)
    if p > 5.0:
        p = p / 100.0
    return max(0.0, min(1.0, p))


def _count_resolved_hypotheses():
    try:
        db = load_json(HYPOTHESIS_DB_FILE, {"hypotheses": []})
        return sum(1 for h in db.get("hypotheses", []) if h.get("resolved"))
    except Exception:
        return 0


def calibrate_prediction(p_model, market_price, metaculus_prob=None, cluster=None):
    """
    Calibrate p_model using:
    1. Isotonic/Platt (only if >= 20 resolved hypotheses)
    2. Soft extremizing fallback: p * 1.1, capped at 50%
    2. Isotonic regression (if >= 20 samples)
    3. Legacy dampening fallback
    Then apply extremization to compensate crowd regression-to-mean.
    """
    if not isinstance(p_model, (int, float)) or p_model <= 0 or p_model >= 1:
        return p_model, False
    if not isinstance(market_price, (int, float)) or market_price < 0:
        return p_model, False

    from calibration import get_calibrator

    p_calibrated = p_model
    method = "raw"

    resolved_count = _count_resolved_hypotheses()
    if resolved_count >= 20:
        try:
            from calibration_tracker import get_platt_calibrated
            p_platt = get_platt_calibrated(p_model, cluster or "other")
            if p_platt is not None:
                p_calibrated = p_platt
                method = "platt"
        except Exception as e:
            logger.warning(f"[calibrate_platt] {type(e).__name__}: {e}")

    if method == "raw" and resolved_count >= 20:
        calibrator = get_calibrator()
        if calibrator.is_fitted:
            p_calibrated = calibrator.predict(p_model, cluster or "other")
            method = "isotonic"
            if p_calibrated >= 0.85 and (p_calibrated - p_model) > 0.30:
                p_calibrated = p_model
                method = "raw_overfit_guard"

    if method == "raw":
        p_calibrated = min(p_model * 1.1, 0.50)
        method = "soft_extremize"

    if market_price < MAX_PRICE:
        p_ext = min(p_calibrated, 0.85)
    else:
        p_ext = p_calibrated

    if method != "raw" or abs(p_ext - p_model) > 0.02:
        logger.info(
            f"[CALIBRATION] p_model={p_model:.1%} -> {p_calibrated:.1%} -> {p_ext:.1%} "
            f"(method={method}, cluster={cluster})"
        )

    return p_ext, method != "raw"


def _cluster_score_adjustment(cluster, settings=None):
    """
    Returns point adjustment to signal_score based on backtest-calibrated cluster weights.
    Reads from bot_settings.json cluster_score_adjustments, falls back to defaults.
    """
    if settings is None:
        from dotm_sniper import get_settings
        settings = get_settings()
    adjustments = settings.get("cluster_score_adjustments", CLUSTER_SCORE_ADJUSTMENTS)
    adj = adjustments.get(cluster, 0)
    if adj != 0:
        logger.info(f"[CLUSTER-ADJ] {cluster}: {adj:+d} to signal_score")
    return adj


def fetch_markets():
    from dotm_sniper import detect_clusters
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
                    end = datetime.fromisoformat(str(end_date).replace("Z", "+00:00")).replace(tzinfo=None)
                    ttl_hours = max(0, (end - now).total_seconds() / 3600)
                except Exception as e:
                    logger.debug(f"[ttl_parse] {type(e).__name__}: {e}")

            if ttl_hours < MIN_TTL_HOURS:
                continue

            for outcome, price in zip(m.get("outcomes", []), m.get("outcome_prices", []), strict=False):
                try:
                    price = float(price)
                except (ValueError, TypeError):
                    continue
                if price <= 0 or price > 1.0:
                    continue
                if price <= MAX_PRICE:
                    clusters = detect_clusters(m["question"])
                    if any(c in BANNED_CLUSTERS for c in clusters):
                        continue
                    is_allowed = any(c in ALLOWED_CLUSTERS for c in clusters)
                    if any(c in ("sports_nba", "sports_ufc") for c in clusters) and (vol < 250_000 or ttl_hours < 48):
                        continue
                    is_other_high_vol = clusters == ["other"] and vol >= PRE_FILTER_OTHER_MIN_VOLUME
                    if not is_allowed and not is_other_high_vol:
                        continue
                    candidates.append({
                        "id": m["condition_id"],
                        HYP_SLUG: m[HYP_SLUG],
                        "question": m["question"],
                        "outcome": outcome,
                        "price": price,
                        "volume": vol,
                        "liquidity": float(m.get("liquidity", 0)),
                        "end_date": end_date,
                        "ttl_hours": ttl_hours,
                        HYP_CLUSTERS: clusters,
                        "oracle_type": m.get("oracle_type", "unknown")
                    })

        candidates.sort(key=lambda x: -x["volume"])
        seen_slugs = set()
        unique = []
        for c in candidates:
            if c[HYP_SLUG] not in seen_slugs:
                seen_slugs.add(c[HYP_SLUG])
                unique.append(c)
        return unique[:30]
    except Exception as e:
        logger.error(f"[MARKETS] Fetch error: {e}")
        return []


def fetch_gamma_dotm_candidates(existing_slugs: set) -> list:
    from dotm_sniper import detect_clusters
    try:
        url = "https://gamma-api.polymarket.com/markets"
        all_markets = []
        for offset in (0, 100):
            params = {
                "closed": "false",
                "limit": 100,
                "offset": offset,
                "order": "volume",
                "ascending": "false",
            }
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code != 200:
                logger.warning(f"[GAMMA] status={resp.status_code} at offset={offset}")
                break
            batch = resp.json()
            if not batch:
                break
            all_markets.extend(batch)
            if len(batch) < 100:
                break
        markets = all_markets
        now = datetime.now()
        candidates = []
        for m in markets:
            if m.get("active") is False:
                continue
            slug = m.get(HYP_SLUG, "")
            if slug in existing_slugs:
                continue
            question = m.get("question", "")
            if not question:
                continue
            outcome_prices = m.get("outcomePrices", "[]")
            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except Exception:
                    continue
            outcomes = m.get("outcomes", "[]")
            if isinstance(outcomes, str):
                try:
                    outcomes = json.loads(outcomes)
                except Exception:
                    continue
            vol = float(m.get("volume", 0))
            if vol < MIN_VOLUME:
                continue
            end_date = m.get("endDate", m.get("end_date", ""))
            ttl_hours = 999
            if end_date:
                try:
                    end = datetime.fromisoformat(str(end_date).replace("Z", "+00:00")).replace(tzinfo=None)
                    ttl_hours = max(0, (end - now).total_seconds() / 3600)
                except Exception as e:
                    logger.debug(f"[ttl_parse] {type(e).__name__}: {e}")
            if ttl_hours < MIN_TTL_HOURS:
                continue
            for outcome, price_str in zip(outcomes, outcome_prices, strict=False):
                try:
                    price = float(price_str)
                except (ValueError, TypeError):
                    continue
                if price <= 0 or price > MAX_PRICE:
                    continue
                clusters = detect_clusters(question)
                if any(c in BANNED_CLUSTERS for c in clusters):
                    continue
                is_allowed = any(c in ALLOWED_CLUSTERS for c in clusters)
                is_other_high_vol = clusters == ["other"] and vol >= PRE_FILTER_OTHER_MIN_VOLUME
                if not is_allowed and not is_other_high_vol:
                    continue
                candidates.append({
                    "id": m.get("conditionId", m.get("condition_id", "")),
                    HYP_SLUG: slug,
                    "question": question,
                    "outcome": outcome,
                    "price": price,
                    "volume": vol,
                    "liquidity": float(m.get("liquidity", 0)),
                    "end_date": end_date,
                    "ttl_hours": ttl_hours,
                    HYP_CLUSTERS: clusters,
                    "oracle_type": m.get("oracleType", m.get("oracle_type", "unknown")),
                    "source": "gamma",
                })
        candidates.sort(key=lambda x: -x["volume"])
        seen = set()
        unique = []
        for c in candidates:
            if c[HYP_SLUG] not in seen and c[HYP_SLUG] not in existing_slugs:
                seen.add(c[HYP_SLUG])
                unique.append(c)
        logger.info(f"[GAMMA] Fetched {len(markets)} markets, {len(unique)} new DOTM candidates")
        return unique[:20]
    except Exception as e:
        logger.error(f"[GAMMA] Fetch error: {e}")
        return []


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
        clusters = m.get(HYP_CLUSTERS, ["other"])
        if any(c in BANNED_CLUSTERS for c in clusters):
            skipped.append(m)
            continue
        is_other = clusters == ["other"] or (len(clusters) == 1 and clusters[0] == "other")
        if is_other:
            volume = m.get("volume", 0)
            if volume < PRE_FILTER_OTHER_MIN_VOLUME:
                slug = m.get(HYP_SLUG, "unknown")
                logger.info(f"[PRE-FILTER] Skipping low-volume 'other' market: {slug}")
                skipped.append(m)
                continue
        kept.append(m)
    return kept, skipped


def full_market_analysis(market):
    """
    Single-step market analysis: combines factor generation + probability estimation.
    """
    from dotm_sniper import get_settings, parse_llm_json
    from order_manager import get_best_ask

    cluster = market.get(HYP_CLUSTERS, ["other"])[0]
    is_geopol = cluster in ["venezuela", "russia_ukraine", "usa_politics"]

    best_ask = None
    polymarket_prob = market["price"]

    if market["price"] < 0.35:
        best_ask = get_best_ask(market[HYP_SLUG])
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

CRITICAL DOTM METHODOLOGY: This market is priced at {market['price']*100:.1f} cents, meaning the crowd assigns only {market['price']*100:.1f}% chance. For DOTM markets, the crowd systematically underestimates tail risks because:
1. Recency bias - people overweight the status quo
2. Black swan blindness - people discount unprecedented but plausible events
3. Time premium - long-dated markets have more time for surprise developments

STEP 1: List 2-3 SPECIFIC plausible scenarios that could make this event happen. Think creatively about catalysts, tipping points, and nonlinear dynamics.
STEP 2: For each scenario, estimate its probability (even 1-5% each adds up).
STEP 3: Sum the scenario probabilities to get total TRUE probability.
STEP 4: If total probability > market price, the crowd is underestimating.

ANCHORING WARNING: Do NOT simply return a probability near the market price. The market price already reflects the crowd. You must independently assess the TRUE probability based on the underlying event scenarios.

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
            confidence = min(max(float(result.get(HYP_CONFIDENCE, confidence)), 0.1), 0.95)
            factors = result.get(HYP_FACTORS, [])
        else:
            p_model_llm = market["price"] * 2
            factors = []
    except Exception as e:
        logger.error(f"[ANALYSIS] LLM error: {e}")
        p_model_llm = market["price"] * 2
        factors = []

    if metaculus_gap and metaculus_gap.get("signal_strength", 0) > 0.3:
        p_model_metaculus = metaculus_gap["metaculus_prob"]
        if p_model_metaculus > p_model_llm:
            p_model = p_model_metaculus
            logger.info(
                f"[METACULUS-OVERRIDE] LLM={p_model_llm:.1%} < Metaculus={p_model_metaculus:.1%}, "
                f"using Metaculus (gap={metaculus_gap['gap']:.1%}, signal={metaculus_gap['signal_strength']:.2f})"
            )
        else:
            p_model = 0.6 * p_model_metaculus + 0.4 * p_model_llm
            logger.info(
                f"[METACULUS-BLEND] LLM={p_model_llm:.1%} > Metaculus={p_model_metaculus:.1%}, "
                f"blended to {p_model:.1%} (60/40 meta/llm)"
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
    p_model, _was_dampened = calibrate_prediction(p_model, market["price"], metaculus_prob_val, cluster=cluster)

    settings = get_settings()

    min_p_model = settings.get("min_p_model", MIN_P_MODEL)
    if p_model < min_p_model - 0.001:
        logger.info(f"[ANALYSIS] p_model={p_model:.1%} < MIN_P_MODEL={min_p_model:.1%}, skipping")
        return {
            "question": market["question"],
            HYP_SLUG: market[HYP_SLUG],
            "market_price": market["price"],
            HYP_P_MODEL: p_model,
            "prob_ratio": 0,
            HYP_CONFIDENCE: confidence,
            "action": "SKIP",
            HYP_FACTORS: [],
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
    vol_score = min(market.get("volume", 0) / 500_000, 1.0) * 20
    ttl_days = market.get("ttl_hours", 0) / 24
    if ttl_days > 180:
        time_score = 20
    elif ttl_days > 90:
        time_score = 15
    elif ttl_days > 30:
        time_score = 12
    elif ttl_days > 14:
        time_score = 8
    elif ttl_days >= 2:
        time_score = 5
    else:
        time_score = 0
    signal_score = ratio_score + factor_score + vol_score + time_score + metaculus_alignment + _cluster_score_adjustment(cluster, settings)

    try:
        from social_buzz import compute_buzz_score
        buzz = compute_buzz_score(market.get(HYP_SLUG, ""), market.get("question", ""))
        buzz_score = buzz.get("buzz_score", 0)
        signal_score += buzz_score
    except Exception:
        buzz_score = 0

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
        f"vol=${market.get('volume',0):,.0f} -> {vol_score:.0f}, ttl={ttl_days:.0f}d -> {time_score:.0f}, "
        f"buzz={buzz_score:.1f} "
        f"= {signal_score:.0f}/{min_signal} => {action}"
    )

    return {
        "question": market["question"],
        HYP_SLUG: market[HYP_SLUG],
        "market_price": market["price"],
        HYP_P_MODEL: p_model,
        "prob_ratio": prob_ratio,
        HYP_CONFIDENCE: confidence,
        "action": action,
        HYP_FACTORS: factors,
        "source_signal": source_signal,
        "signal_score": signal_score,
        "reasoning": f"score={signal_score:.0f}/{min_signal}(horizon), ratio={prob_ratio:.2f}x, conf={confidence:.2f}, src={source_signal}, meta_align={metaculus_alignment:+d}",
        "best_ask": best_ask
    }


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
            slug = m.get(HYP_SLUG, "")
            question = m.get("question", "")
            end_date = m.get("end_date")
            meta = get_metaculus_forecast(question, end_date)
            if meta.get("found"):
                metaculus_cache[slug] = meta.get("probability")
            else:
                metaculus_cache[slug] = None
        else:
            metaculus_cache[m.get(HYP_SLUG)] = None

    batch_items = []
    for m in markets:
        slug = m.get(HYP_SLUG, "")
        question = m.get("question", "")
        price = m.get("price", 0)
        volume = m.get("volume", 0)
        ttl_hours = m.get("ttl_hours", 999)
        cluster = m.get(HYP_CLUSTERS, ["other"])[0]
        batch_items.append({
            HYP_SLUG: slug,
            "question": question,
            "market_price": round(price, 4),
            "volume": round(volume, 0),
            "ttl_hours": round(ttl_hours, 0),
            "cluster": cluster,
        })

    for item in batch_items:
        item["question"] = sanitize_for_prompt(item["question"])
    items_json = json.dumps(batch_items, indent=2)

    prompt = f"""Prediction market analyst. Analyze these DOTM (deep out-the-money) markets where the crowd may underestimate probability.

CRITICAL DOTM METHODOLOGY: These markets are priced very low (often 1-30 cents). The crowd systematically underestimates tail risks due to recency bias, black swan blindness, and time premium. For each market:
1. List 2-3 SPECIFIC plausible scenarios that could make the event happen
2. Estimate probability for each scenario (even 1-5% each adds up)
3. Sum scenario probabilities to get TRUE probability
4. If total > market price, the crowd is underestimating

MARKETS (JSON array):
{items_json}

For EACH market, identify 2-3 specific scenarios/factors and estimate the TRUE probability independently from the market price.

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
- Do NOT anchor on the market price - provide independent assessment based on scenarios
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
    if metaculus_cache is None:
        metaculus_cache = {}

    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)

    for prefix in ("```json", "```JSON", "```"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]

    cleaned = cleaned.strip()
    if cleaned.startswith(":"):
        cleaned = cleaned[1:].strip()

    start = cleaned.find('[')
    if start == -1:
        logger.warning(f"[BATCH-PARSE] No '[' found in LLM response: {content[:200]}")
        return None

    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(cleaned)):
        c = cleaned[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                candidate = cleaned[start:i + 1]
                try:
                    arr = json.loads(candidate)
                    if isinstance(arr, list):
                        return _build_batch_results(arr, batch_items, metaculus_cache)
                except json.JSONDecodeError:
                    pass
                for end in range(i, len(cleaned)):
                    if cleaned[end] == ']':
                        try:
                            arr = json.loads(cleaned[start:end + 1])
                            if isinstance(arr, list):
                                return _build_batch_results(arr, batch_items, metaculus_cache)
                        except json.JSONDecodeError:
                            continue
                break

    fallback = re.search(r'\[[\s\S]*\]', cleaned)
    if fallback:
        try:
            arr = json.loads(fallback.group(0))
            if isinstance(arr, list):
                return _build_batch_results(arr, batch_items, metaculus_cache)
        except json.JSONDecodeError:
            pass

    individual_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned)
    if individual_objects:
        try:
            arr = [json.loads(obj) for obj in individual_objects]
            arr = [a for a in arr if isinstance(a, dict)]
            if arr:
                logger.info(f"[BATCH-PARSE] Recovered {len(arr)} individual objects from failed batch")
                return _build_batch_results(arr, batch_items, metaculus_cache)
        except json.JSONDecodeError:
            pass

    logger.warning(f"[BATCH-PARSE] All parsing methods failed. Response preview: {content[:300]}")

    return None


def _build_batch_results(parsed_array, batch_items, metaculus_cache=None):
    """
    Convert parsed batch array into list of analysis dicts
    matching the full_market_analysis schema.
    """
    from dotm_sniper import get_settings

    slug_to_item = {it[HYP_SLUG]: it for it in batch_items}
    {it[HYP_SLUG]: i for i, it in enumerate(batch_items)}

    results_map = {}
    for item in parsed_array:
        if not isinstance(item, dict):
            continue

        slug = item.get(HYP_SLUG, "")
        if slug not in slug_to_item:
            if len(results_map) < len(batch_items):
                unmatched_idx = len(results_map)
                slug = batch_items[unmatched_idx][HYP_SLUG]
            else:
                continue

        bi = slug_to_item.get(slug)
        if not bi:
            continue

        market_price = bi["market_price"]
        cluster = bi["cluster"]

        p_model_llm = normalize_probability(item.get("estimated_probability", market_price * 2))
        confidence = min(max(float(item.get(HYP_CONFIDENCE, 0.6)), 0.1), 0.95)
        factors = item.get(HYP_FACTORS, [])

        max_p_model = market_price * MAX_P_MODEL_RATIO
        p_model = min(p_model_llm, max_p_model)

        metaculus_prob = None
        if metaculus_cache:
            metaculus_prob = metaculus_cache.get(slug)

        p_model, _ = calibrate_prediction(p_model, market_price, metaculus_prob, cluster=cluster)

        settings = get_settings()
        min_p_model = settings.get("min_p_model", MIN_P_MODEL)
        if p_model < min_p_model - 0.001:
            results_map[slug] = {
                "question": bi["question"],
                HYP_SLUG: slug,
                "market_price": market_price,
                HYP_P_MODEL: p_model,
                "prob_ratio": 0,
                HYP_CONFIDENCE: confidence,
                "action": "SKIP",
                HYP_FACTORS: [],
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
        vol_score = min(bi.get("volume", 0) / 500_000, 1.0) * 20
        ttl_hours = bi.get("ttl_hours", 999)
        ttl_days = ttl_hours / 24
        if ttl_days > 180:
            time_score = 20
        elif ttl_days > 90:
            time_score = 15
        elif ttl_days > 30:
            time_score = 12
        elif ttl_days > 14:
            time_score = 8
        elif ttl_days >= 2:
            time_score = 5
        else:
            time_score = 0

        signal_score = ratio_score + factor_score + vol_score + time_score + metaculus_alignment + _cluster_score_adjustment(cluster, settings)

        try:
            from social_buzz import compute_buzz_score
            buzz = compute_buzz_score(slug, bi.get("question", ""))
            batch_buzz = buzz.get("buzz_score", 0)
        except Exception:
            batch_buzz = 0

        base_threshold = settings.get("signal_threshold", 55)
        if ttl_days > 90:
            min_signal = settings.get("signal_threshold_long_horizon", base_threshold + 10)
        elif ttl_days >= 31:
            min_signal = settings.get("signal_threshold_medium_horizon", base_threshold + 5)
        else:
            min_signal = base_threshold

        source_signal = "default"
        metaculus_prob_val = metaculus_cache.get(slug) if metaculus_cache else None
        if metaculus_prob_val is not None and metaculus_prob_val > market_price:
            gap = metaculus_prob_val - market_price
            signal_strength = gap / market_price if market_price > 0 else 0
            if signal_strength > 0.3:
                if metaculus_prob_val > p_model:
                    p_model = metaculus_prob_val
                else:
                    p_model = 0.6 * metaculus_prob_val + 0.4 * p_model
                source_signal = "metaculus_override"
                confidence = min(confidence + 0.10, 0.95)
                min_signal = max(min_signal - 10, 35)
                logger.info(f"[META-OVERRIDE-BATCH] {slug[:30]}... p_model={p_model:.1%} from metaculus={metaculus_prob_val:.1%}")

        prob_ratio = p_model / market_price if market_price > 0 else 0
        ratio_score = min(prob_ratio / 3.0, 1.0) * 30
        signal_score = ratio_score + factor_score + vol_score + time_score + metaculus_alignment + _cluster_score_adjustment(cluster, settings) + batch_buzz

        action = "BUY" if signal_score >= min_signal and confidence >= settings.get("min_confidence", MIN_CONFIDENCE) and prob_ratio >= MIN_PROB_RATIO else "SKIP"

        logger.info(
            f"[SIGNAL-BATCH] ratio={prob_ratio:.2f}x -> {ratio_score:.0f}, factors={len(supporting)}/{len(high_weight)} -> {factor_score:.0f}, "
            f"vol=${bi.get('volume',0):,.0f} -> {vol_score:.0f}, ttl={ttl_days:.0f}d -> {time_score:.0f}, "
            f"buzz={batch_buzz:.1f} "
            f"= {signal_score:.0f}/{min_signal} => {action}"
        )

        results_map[slug] = {
            "question": bi["question"],
            HYP_SLUG: slug,
            "market_price": market_price,
            HYP_P_MODEL: p_model,
            "prob_ratio": prob_ratio,
            HYP_CONFIDENCE: confidence,
            "action": action,
            HYP_FACTORS: factors,
            "source_signal": source_signal,
            "signal_score": signal_score,
            "reasoning": f"score={signal_score:.0f}/{min_signal}(batch), ratio={prob_ratio:.2f}x, conf={confidence:.2f}, src={source_signal}",
            "best_ask": None,
        }

    results = []
    for bi in batch_items:
        if bi[HYP_SLUG] in results_map:
            results.append(results_map[bi[HYP_SLUG]])
        else:
            results.append({
                "question": bi["question"],
                HYP_SLUG: bi[HYP_SLUG],
                "market_price": bi["market_price"],
                HYP_P_MODEL: bi["market_price"] * 2,
                "prob_ratio": 2.0,
                HYP_CONFIDENCE: 0.5,
                "action": "SKIP",
                HYP_FACTORS: [],
                "source_signal": "default",
                "reasoning": "batch_parse_fallback",
                "best_ask": None,
            })

    return results


def advisor_pre_check(market, analysis, estimated_size=0, balance=1):
    """
    Two-factor trade verification via independent Advisor (deepseek-reasoner).
    Uses Chain-of-Thought reasoning to validate or reject the bot's trade thesis.

    Returns (approved: bool, verdict: str, confidence: float, reason: str)
    """
    question = market.get("question", "")
    slug = market.get(HYP_SLUG, "")
    price = market.get("price", 0)
    p_model = analysis.get(HYP_P_MODEL, 0)
    factors = analysis.get(HYP_FACTORS, [])
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
PROBABILITY RATIO: {f"{p_model/price:.2f}" if price > 0 else "N/A"}x vs market
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
            size_pct = estimated_size / balance if balance > 0 else 1
            if size_pct <= 0.02:
                logger.info("[ADVISOR] Empty response, but micro-position <=2%, allowing")
                return True, "UNKNOWN", 0.0, "advisor_empty_micro_override"
            logger.warning("[ADVISOR] Empty response from reasoner, blocking trade")
            return False, "UNKNOWN", 0.0, "advisor_empty_response"

        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)
        from advisor_script import parse_llm_advisor_response
        result, parse_err = parse_llm_advisor_response(content, log_label="ADVISOR-PRE")
        if result is None:
            size_pct = estimated_size / balance if balance > 0 else 1
            if size_pct <= 0.02:
                logger.info(f"[ADVISOR] Parse failed but micro-position, allowing: {parse_err}")
                return True, "UNKNOWN", 0.0, "advisor_parse_micro_override"
            logger.warning(f"[ADVISOR] Parse failed: {parse_err}, blocking trade")
            return False, "UNKNOWN", 0.0, f"advisor_parse_error: {parse_err}"

        verdict = result.get("verdict", "UNKNOWN")
        confidence = result.get(HYP_CONFIDENCE, 0.0)
        advisor_p = result.get("p_estimate", 0)
        advisor_factors = result.get(HYP_FACTORS, [])

        logger.info(
            f"[ADVISOR] verdict={verdict} conf={confidence:.2f} "
            f"advisor_p={advisor_p:.1%} vs bot_p={p_model:.1%} | "
            f"factors: {advisor_factors[:2]}"
        )

        if verdict == "CONFIRM" and confidence >= ADVISOR_MIN_CONFIDENCE:
            logger.info(f"[ADVISOR] ✅ Trade APPROVED by advisor ({verdict}, conf={confidence:.2f})")
            return True, verdict, confidence, "approved"
        elif verdict == "WARNING" and confidence < ADVISOR_MIN_CONFIDENCE:
            logger.info("[ADVISOR] ⚠️ Trade WARNING (advisor uncertain), allowing small position")
            return True, verdict, confidence, "advisor_warning_allowed"
        elif verdict == "DIVERGE":
            size_pct = estimated_size / balance if balance > 0 else 1
            advisor_agrees_direction = advisor_p > price
            if size_pct <= 0.02:
                logger.info(f"[ADVISOR] 🔄 DIVERGE override: micro-position {size_pct:.1%} ≤ 2%, allowing")
                return True, verdict, confidence, "diverge_micro_override"
            if advisor_agrees_direction:
                logger.info(f"[ADVISOR] 🔄 DIVERGE override: advisor_p={advisor_p:.1%} > price={price:.1%}, direction agrees")
                return True, verdict, confidence, "diverge_direction_agrees"
            reason = f"advisor_veto: verdict={verdict}, conf={confidence:.2f}"
            logger.info(f"[ADVISOR] 🚫 Trade BLOCKED: {reason}")
            print(f"   🚫 Advisor veto: {verdict} (conf={confidence:.2f})")
            return False, verdict, confidence, reason
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
