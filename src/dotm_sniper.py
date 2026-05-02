#!/usr/bin/env python3
"""
DOTM Sniper v4.1 - Full Hypothesis Machine + Multi-Source Intelligence
Based on the mathematical edge of Deep Out-The-Money trading
"""
import subprocess, json, requests, time, re, os, logging
from datetime import datetime, timedelta
from collections import defaultdict

LOG_FILE = "/root/dotm-sniper/sniper.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("OPENROUTER_API_KEY", "REDACTED_OPENROUTER_KEY")
MODEL_MINIMAX25 = "minimax/minimax-m2.5:free"
MODEL_MINIMAX27 = "minimax/minimax-m2.7-20260318"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
URL = "https://openrouter.ai/api/v1/chat/completions"
METACULUS_API_KEY = "c200ca9c41c9866d781d5e517c2d6cd64e7f0432"
METACULUS_URL = "https://www.metaculus.com/api2/questions/"

def api_call_with_retry(url, headers, payload, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=payload.get("timeout", 60))
            if resp.status_code == 200:
                return resp.json()
            if attempt < max_retries:
                time.sleep(2 ** attempt)
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                raise
    return None

HYPOTHESIS_DB = "/root/dotm-sniper/hypothesis_db.json"
POSITIONS_FILE = "/root/dotm-sniper/positions.json"
SETTINGS_FILE = "/root/dotm-sniper/bot_settings.json"
CACHE_FILE = "/root/dotm-sniper/source_cache.json"

MAX_PRICE = 0.08
MIN_VOLUME = 5000
MIN_TTL_HOURS = 72
MAX_POSITIONS = 50
MAX_POS_PCT = 0.03
MAX_CLUSTER_PCT = 0.15
MIN_PROB_RATIO = 2.5
MIN_CONFIDENCE = 0.6
KELLY_FRACTION = 0.25
TAKE_PROFIT = 0.30
TRAILING_ACTIVATION = 0.15
TRAILING_STOP = 0.10
BURN_IN_TRADES = 50
METACULUS_GAP_THRESHOLD = 0.10

CLUSTER_KEYWORDS = {
    "venezuela": ["venezuela", "maduro", "chavez"],
    "usa_politics": ["trump", "biden", "republican", "democratic", "election", "congress", "senate"],
    "russia_ukraine": ["russia", "ukraine", "putin", "zelensky", "kiev", "moscow"],
    "sports_nba": ["nba", "basketball", "lakers", "celtics", "warriors", "finals"],
    "sports_ufc": ["ufc", "mma", "fight", "boxing"],
    "fed_fomc": ["fed", "federal reserve", "fomc", "interest rate", "powell"],
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto"],
    "ai_tech": ["ai", "openai", "anthropic", "google", "meta", "llm"],
}

SOURCE_TEMPLATES = {
    "metaculus": {
        "url": "https://www.metaculus.com/api2/questions/",
        "credibility": 0.95,
        "description": "Superforecasting community"
    },
    "cme_fedwatch": {
        "url": "https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html",
        "credibility": 0.90,
        "description": "Fed futures pricing"
    },
    "fred": {
        "url": "https://fred.stlouisfed.org/series/",
        "credibility": 0.85,
        "description": "Federal Reserve Economic Data"
    },
    "acled": {
        "url": "https://acleddata.com/api/acled/",
        "credibility": 0.80,
        "description": "Armed Conflict Location & Event Data"
    }
}

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"metaculus": {}, "news": {}, "last_update": None}

def save_cache(cache):
    cache["last_update"] = datetime.now().isoformat()
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def get_metaculus_forecast(question):
    cache = load_cache()
    cache_key = question[:50]

    if cache_key in cache.get("metaculus", {}):
        cached = cache["metaculus"][cache_key]
        if cached.get("timestamp"):
            cached_time = datetime.fromisoformat(cached["timestamp"])
            if (datetime.now() - cached_time).total_seconds() < 3600:
                return cached

    prompt = f"""Find this question on Metaculus and return the community prediction probability.

Question: {question}

If you find a similar Metaculus question, return JSON:
{{"found": true, "probability": 0.XX, "question_title": "...", "url": "..."}}

If not found or uncertain, return:
{{"found": false, "probability": null}}

Search for: {question[:100]}"""

    try:
        resp = requests.post(URL, headers=HEADERS, json={
            "model": MODEL_MINIMAX25, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2, "max_tokens": 300
        }, timeout=30)

        data = resp.json()
        content = (data.get("choices", [{}])[0].get("message", {}).get("content") or "")

        if content:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                result = json.loads(content[start:end])
                if result.get("found"):
                    result["timestamp"] = datetime.now().isoformat()
                    cache.setdefault("metaculus", {})[cache_key] = result
                    save_cache(cache)
                    return result
    except:
        pass

    return {"found": False, "probability": None}

def check_metaculus_gap(market):
    meta = get_metaculus_forecast(market["question"])

    if not meta.get("found"):
        return None

    metaculus_prob = meta.get("probability", 0)
    polymarket_prob = market["price"]

    gap = metaculus_prob - polymarket_prob

    if gap > METACULUS_GAP_THRESHOLD and polymarket_prob < 0.10:
        return {
            "source": "metaculus",
            "metaculus_prob": metaculus_prob,
            "polymarket_prob": polymarket_prob,
            "gap": gap,
            "signal_strength": min(gap / 0.15, 1.0),
            "reasoning": f"Metaculus {metaculus_prob:.0%} vs Polymarket {polymarket_prob:.0%}: gap {gap:.0%}"
        }

    return None

def search_geopolitical_sources(question):
    keywords = extract_keywords(question)

    sources = {
        "osint_telegram": [],
        "reddit_geopolitics": [],
        "news_sentiment": []
    }

    prompt = f"""Analyze geopolitical aspects of this question. Search your knowledge for relevant signals.

Question: {question}
Keywords: {', '.join(keywords)}

Return JSON with any relevant signals:
{{
  "regime_change_signals": ["list of specific indicators"],
  "military_movements": ["any relevant troop/conflict movements"],
  "diplomatic_events": ["embassy changes, sanctions, statements"],
  "sentiment_indicators": "bullish/bearish/mixed",
  "confidence": 0.XX,
  "sources_mentioned": ["specific sources that would confirm"]
}}

If no strong signals: {{"regime_change_signals": [], "confidence": 0.2}}"""

    try:
        resp = requests.post(URL, headers=HEADERS, json={
            "model": MODEL_MINIMAX25, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3, "max_tokens": 400
        }, timeout=45)

        data = resp.json()
        content = (data.get("choices", [{}])[0].get("message", {}).get("content") or "")

        if content:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                result = json.loads(content[start:end])
                return result
    except:
        pass

    return {"regime_change_signals": [], "confidence": 0.2}

def search_sports_sources(question):
    prompt = f"""Find relevant sports intelligence for this market.

Question: {question}

Return JSON with relevant signals:
{{
  "injury_mentions": ["any injury reports"],
  "weather_impact": "relevant/not_relevant/unknown",
  "line_movement": "significant/unusual/none",
  "insider_indicators": ["specific mentions of lineup changes, etc"],
  "confidence": 0.XX
}}

Focus on: injuries, weather, official announcements timing.
If no strong signals: {{"confidence": 0.2}}"""

    try:
        resp = requests.post(URL, headers=HEADERS, json={
            "model": MODEL_MINIMAX25, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3, "max_tokens": 300
        }, timeout=40)

        data = resp.json()
        content = (data.get("choices", [{}])[0].get("message", {}).get("content") or "")

        if content:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                result = json.loads(content[start:end])
                return result
    except:
        pass

    return {"confidence": 0.2}

def extract_keywords(question):
    stop_words = {"will", "the", "a", "an", "be", "by", "of", "in", "on", "at", "to", "for", "this", "that", "is", "are", "was", "were"}
    words = re.findall(r'\b[a-zA-Z]{4,}\b', question.lower())
    return [w for w in words if w not in stop_words][:10]

def score_sources(market, geopol=None, sports=None, metaculus_gap=None):
    score = 0.0
    factors = []
    weights = []

    if metaculus_gap:
        score += 0.4 * metaculus_gap["signal_strength"]
        factors.append(f"Metaculus gap: {metaculus_gap['gap']:.0%}")
        weights.append(("metaculus", 0.4))

    if geopol:
        signals = geopol.get("regime_change_signals", [])
        if signals:
            score += min(0.3, 0.1 * len(signals))
            factors.append(f"Geopolitical signals: {len(signals)}")
            weights.append(("geopol", 0.3))

        conf = geopol.get("confidence", 0)
        if conf > 0.6:
            score += 0.2
            factors.append(f"High geo confidence: {conf:.0%}")

    if sports:
        injury_signals = sports.get("injury_mentions", [])
        if injury_signals:
            score += 0.2
            factors.append(f"Sports injury intel: {len(injury_signals)}")
            weights.append(("sports", 0.2))

        weather = sports.get("weather_impact", "not_relevant")
        if weather == "relevant":
            score += 0.15
            factors.append("Weather impact relevant")

    cluster = market.get("clusters", ["other"])[0]
    cluster_weights = {
        "venezuela": 0.3,
        "russia_ukraine": 0.25,
        "usa_politics": 0.2,
        "fed_fomc": 0.25,
        "sports_nba": 0.15,
        "sports_ufc": 0.2,
        "crypto": 0.1,
        "ai_tech": 0.1,
        "other": 0.05
    }
    score += cluster_weights.get(cluster, 0.05)

    return {
        "total_score": min(score, 1.0),
        "factors": factors,
        "weights": weights,
        "source_signal": "metaculus" if metaculus_gap else ("geopol" if geopol and geopol.get("confidence", 0) > 0.6 else "default")
    }

def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            pass
    return default

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def get_settings():
    s = load_json(SETTINGS_FILE, {
        "prob_ratio_threshold": MIN_PROB_RATIO,
        "min_confidence": MIN_CONFIDENCE,
        "position_size_pct": MAX_POS_PCT,
        "calibration_brier": None,
        "total_resolved": 0
    })
    return s

def save_settings(s):
    save_json(SETTINGS_FILE, s)

def load_hypothesis_db():
    return load_json(HYPOTHESIS_DB, {"hypotheses": [], "resolved": []})

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
            if kw in question_lower:
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

def calculate_brier_score(db):
    resolved = db.get("resolved", [])
    if len(resolved) < BURN_IN_TRADES:
        return None

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
    logger.info(f"Stats: Brier={brier:.3f}, Winrate={winrate:.1%} ({wins}W/{losses}L)")

    settings = get_settings()
    old_brier = settings.get("calibration_brier")

    if old_brier is not None:
        if brier > 0.08 and settings["prob_ratio_threshold"] < 3.5:
            settings["prob_ratio_threshold"] += 0.2
            save_settings(settings)
            print(f"⚠️ Calibrating: Brier {brier:.3f} > 0.08, raising threshold to {settings['prob_ratio_threshold']}")
        elif brier < 0.03 and settings["prob_ratio_threshold"] > 1.8:
            settings["prob_ratio_threshold"] -= 0.15
            save_settings(settings)
            print(f"📈 Calibrating: Brier {brier:.3f} < 0.03, lowering threshold to {settings['prob_ratio_threshold']}")

    settings["calibration_brier"] = brier
    save_settings(settings)
    return brier

def get_balance():
    try:
        res = subprocess.run(["pm-trader", "balance"], capture_output=True, text=True, timeout=15)
        return json.loads(res.stdout).get("data", {})
    except:
        return {"cash": 100}

def get_portfolio():
    try:
        res = subprocess.run(["pm-trader", "portfolio"], capture_output=True, text=True, timeout=15)
        return json.loads(res.stdout).get("data", [])
    except:
        return []

def fetch_markets():
    try:
        res = subprocess.run(["pm-trader", "markets", "list", "--limit", "200"],
                           capture_output=True, text=True, timeout=30)
        data = json.loads(res.stdout)
        candidates = []
        now = datetime.now()

        for m in data.get("data", []):
            if not m.get("active") or m.get("closed"):
                continue

            vol = float(m.get("volume", 0))
            if vol < MIN_VOLUME:
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

            for outcome, price in zip(m["outcomes"], m["outcome_prices"]):
                if price <= MAX_PRICE and price > 0:
                    clusters = detect_clusters(m["question"])
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
        print(f"Ошибка рынков: {e}")
        return []

def research_market(market):
    cluster = market.get("clusters", ["other"])[0]
    is_geopol = cluster in ["venezuela", "russia_ukraine", "usa_politics"]
    is_sports = cluster in ["sports_nba", "sports_ufc", "sports"]
    is_fed = cluster in ["fed_fomc"]

    metaculus_gap = None
    geopol = None
    sports_intel = None

    print(f"   🔎 Checking sources...")
    if market["price"] < 0.10:
        metaculus_gap = check_metaculus_gap(market)
        if metaculus_gap:
            print(f"   📊 Metaculus gap: {metaculus_gap['metaculus_prob']:.0%} vs {metaculus_gap['polymarket_prob']:.0%}")

    if is_geopol:
        geopol = search_geopolitical_sources(market["question"])
        if geopol.get("regime_change_signals"):
            print(f"   🌍 Geopol signals: {len(geopol['regime_change_signals'])}")

    if is_sports:
        sports_intel = search_sports_sources(market["question"])
        if sports_intel.get("injury_mentions"):
            print(f"   ⚽ Sports intel: {len(sports_intel['injury_mentions'])} injuries mentioned")

    source_score = score_sources(market, geopol, sports_intel, metaculus_gap)

    base_prompt = f"""Ты - аналитик предсказательных рынков. Для рынка ниже найди 3 НЕЗАВИСИМЫХ фактора, которые могут привести к YES.

Рынок: {market['question']}
Цена: ${market['price']:.3f}
Объём: ${market['volume']:,.0f}
Дата закрытия: {market.get('end_date', 'unknown')}
Кластер: {cluster}

Источники сигналов:"""

    if source_score["factors"]:
        base_prompt += "\n" + "\n".join([f"- {f}" for f in source_score["factors"]])
    else:
        base_prompt += "\nВнешних источников сигналов не найдено."

    base_prompt += """

Верни JSON массив с 3 факторами:
[{{"factor": "описание", "direction": "supports/opposes", "weight": "high/medium/low", "source": "источник"}}]

Каждый фактор должен быть НЕЗАВИСИМЫМ от других (разные информационные источники).
Если факторов недостаточно - верни пустой массив [].

Учитывай данные из внешних источников если они релевантны."""

    try:
        resp = requests.post(URL, headers=HEADERS, json={
            "model": MODEL_MINIMAX27, "messages": [{"role": "user", "content": base_prompt}],
            "temperature": 0.3, "max_tokens": 1200
        }, timeout=120)

        resp_data = resp.json()
        msg = resp_data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content") or ""
        
        if not content:
            reasoning = msg.get("reasoning") or ""
            if reasoning:
                content = reasoning

        if not content:
            return [], source_score

        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        
        start = content.find('[')
        end = content.rfind(']') + 1
        if start != -1 and end > start:
            factors = json.loads(content[start:end])
            if isinstance(factors, list):
                return factors, source_score
    except:
        pass
    return [], source_score

def analyze_market(market, factors, source_score):
    supporting = [f for f in factors if f.get("direction") == "supports"]
    high_weight = [f for f in supporting if f.get("weight") == "high"]

    p_base = market["price"]

    source_signal = source_score.get("source_signal", "default")
    source_bonus = source_score.get("total_score", 0) * 0.2

    if source_signal == "metaculus":
        p_model = min(p_base * 3.5, 0.35)
        confidence = 0.80
    elif len(high_weight) >= 2:
        p_model = min(p_base * 4, 0.5)
        confidence = 0.75 + source_bonus
    elif len(supporting) >= 2:
        p_model = min(p_base * 3, 0.35)
        confidence = 0.65 + source_bonus
    elif len(supporting) == 1:
        p_model = min(p_base * 2.5, 0.20)
        confidence = 0.55 + source_bonus
    else:
        p_model = p_base * 1.5
        confidence = 0.40 + source_bonus

    confidence = min(confidence, 0.95)
    prob_ratio = p_model / p_base if p_base > 0 else 0

    prompt = f"""Оцени вероятность для этого рынка Polymarket.

Вопрос: {market['question']}
Рыночная цена: ${p_base:.3f} ({p_base*100:.1f}%)

Факторы в пользу YES:
{chr(10).join([f"- {f['factor']} ({f['weight']})" for f in supporting]) if supporting else "Нет сильных факторов"}

Твоя задача: дать СВОЮ оценку истинной вероятности (0-1), НЕ привязанную к рыночной цене.
Учитывай: качество факторов, их независимость, достоверность источников.

Верни JSON: {{"estimated_probability": 0.XX, "confidence": 0.XX, "reasoning": "кратко"}}
Ответь ТОЛЬКО JSON."""

    try:
        resp = requests.post(URL, headers=HEADERS, json={
            "model": MODEL_MINIMAX27, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5, "max_tokens": 500
        }, timeout=90)

        resp_data = resp.json()
        msg = resp_data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content") or ""
        
        if not content:
            reasoning = msg.get("reasoning") or ""
            if reasoning:
                content = reasoning

        if content:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                result = json.loads(content[start:end])
                p_model = result.get("estimated_probability", p_model)
                confidence = result.get("confidence", confidence)
                prob_ratio = p_model / p_base if p_base > 0 else 0
    except:
        pass

    action = "BUY" if prob_ratio >= get_settings().get("prob_ratio_threshold", MIN_PROB_RATIO) and confidence >= get_settings().get("min_confidence", MIN_CONFIDENCE) else "SKIP"

    return {
        "question": market["question"],
        "slug": market["slug"],
        "market_price": p_base,
        "p_model": p_model,
        "prob_ratio": prob_ratio,
        "confidence": confidence,
        "action": action,
        "factors": factors,
        "source_signal": source_signal,
        "reasoning": f"ratio={prob_ratio:.2f}x, conf={confidence:.2f}, src={source_signal}"
    }

def position_size(p_model, market_price, balance):
    if market_price <= 0:
        return 5
    b = (1 / market_price) - 1
    p = p_model
    q = 1 - p

    kelly = (b * p - q) / b if b > 0 else 0
    if kelly <= 0:
        kelly = 0.01

    f_star = kelly * KELLY_FRACTION
    size_pct = min(f_star, MAX_POS_PCT)

    kelly_dollars = int(balance * size_pct)
    kelly_dollars = max(kelly_dollars, 5)
    kelly_dollars = min(kelly_dollars, int(balance * MAX_POS_PCT))

    return kelly_dollars

def buy(market, amount):
    try:
        res = subprocess.run(["pm-trader", "buy", market["slug"], market["outcome"], str(amount)],
                           capture_output=True, text=True, timeout=30)
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

def trailing_stop_check():
    positions = load_json(POSITIONS_FILE, {})
    portfolio = get_portfolio()
    if not portfolio:
        return

    current_slugs = {p["market_slug"] for p in portfolio}

    for pos in portfolio:
        slug = pos["market_slug"]
        current_price = pos.get("live_price", 0)
        entry_price = pos.get("avg_entry_price", 0)
        pnl_pct = pos.get("percent_pnl", 0) / 100

        if slug not in positions:
            positions[slug] = {
                "entry_price": entry_price,
                "high_price": current_price,
                "trailing_on": False,
                "stop_loss": entry_price * 0.85
            }

        p = positions[slug]
        p["high_price"] = max(p.get("high_price", current_price), current_price)

        if p["high_price"] > entry_price * (1 + TRAILING_ACTIVATION):
            p["trailing_on"] = True
            p["stop_loss"] = p["high_price"] * (1 - TRAILING_STOP)

        if p.get("trailing_on") and current_price <= p.get("stop_loss", 0):
            outcome = pos.get("outcome", "Yes")
            print(f"  🎯 Trailing stop: {slug[:40]}...")
            try:
                res = subprocess.run(["pm-trader", "sell", slug, outcome, str(pos.get("shares", 0))],
                                     capture_output=True, text=True, timeout=20)
                result = json.loads(res.stdout) if res.stdout else {}
                if result.get("ok"):
                    logger.info(f"SOLD trailing stop: {slug}")
            except:
                pass
        elif pnl_pct >= TAKE_PROFIT:
            outcome = pos.get("outcome", "Yes")
            print(f"  🎯 Take-profit: {slug[:40]}... +{pnl_pct:.0%}")
            try:
                res = subprocess.run(["pm-trader", "sell", slug, outcome, str(pos.get("shares", 0))],
                                     capture_output=True, text=True, timeout=20)
                result = json.loads(res.stdout) if res.stdout else {}
                if result.get("ok"):
                    logger.info(f"SOLD take-profit: {slug}")
            except:
                pass

        positions[slug] = p

    save_json(POSITIONS_FILE, positions)

def resolve_hypotheses():
    db = load_hypothesis_db()
    portfolio = get_portfolio()
    portfolio_slugs = {p["market_slug"] for p in portfolio}
    db_slugs = {h["slug"] for h in db.get("hypotheses", [])}

    unclosed_slugs = db_slugs - portfolio_slugs
    if not unclosed_slugs:
        return

    market_map = {}
    try:
        res = subprocess.run(["pm-trader", "markets", "list", "--limit", "200"],
                           capture_output=True, text=True, timeout=20)
        for m in json.loads(res.stdout).get("data", []):
            market_map[m["slug"]] = m
    except:
        pass

    for slug in unclosed_slugs:
        for h in db["hypotheses"]:
            if h["slug"] == slug and h.get("resolved"):
                continue

            h["resolved"] = True
            h["resolved_at"] = datetime.now().isoformat()

            market_data = market_map.get(slug)

            outcome = "UNKNOWN"
            if market_data:
                if market_data.get("closed"):
                    yes_price = market_data.get("outcome_prices", [0.5])[0]
                    outcome = "YES" if yes_price > 0.5 else "NO"

            h["outcome"] = outcome
            db["resolved"].append(h)

            settings = get_settings()
            settings["total_resolved"] = settings.get("total_resolved", 0) + 1
            save_settings(settings)

    save_json(HYPOTHESIS_DB, db)
    calculate_brier_score(db)

def main():
    print("="*60)
    print("  DOTM SNIPER v4.1 - Hypothesis Machine + Multi-Source")
    print("="*60)

    settings = get_settings()
    print(f"⚙️ Thresholds: prob_ratio={settings['prob_ratio_threshold']:.1f}x, confidence={settings['min_confidence']:.2f}")

    resolve_hypotheses()
    trailing_stop_check()

    balance_data = get_balance()
    balance = balance_data.get("cash", 100)
    print(f"💰 Balance: ${balance:.2f}")

    portfolio = get_portfolio()
    print(f"📊 Open positions: {len(portfolio)}")

    if len(portfolio) >= MAX_POSITIONS:
        print(f"⚠️ Max positions ({MAX_POSITIONS}) reached")
        return

    markets = fetch_markets()
    if not markets:
        print("No markets found")
        return

    print(f"📈 Candidates: {len(markets)}")

    candidates_passed = 0
    candidates_bought = 0
    available_balance = balance

    db = load_hypothesis_db()
    current_positions_for_clusters = [
        {"clusters": h.get("clusters", []), "size_pct": h.get("size_pct", 0)}
        for h in db.get("hypotheses", []) if not h.get("resolved")
    ]

    for m in markets:
        if len(portfolio) + candidates_bought >= MAX_POSITIONS:
            break

        if available_balance < 5:
            break

        print(f"\n🔍 {m['question'][:55]}...")
        print(f"   Price: ${m['price']:.3f} | TTL: {m['ttl_hours']:.0f}h | Vol: ${m['volume']:,.0f}")

        can_pass, reason = check_cluster_limits(m["clusters"], current_positions_for_clusters)
        if not can_pass:
            print(f"   ⏭️ Cluster limit: {reason}")
            continue

        factors, source_score = research_market(m)

        min_factors = 2 if source_score["source_signal"] != "metaculus" else 1
        if len(factors) < min_factors:
            print(f"   ⏭️ Insufficient factors ({len(factors)})")
            continue

        supporting = [f for f in factors if f.get("direction") == "supports"]
        print(f"   📊 Factors: {len(supporting)} supporting, {len(factors) - len(supporting)} opposing")

        analysis = analyze_market(m, factors, source_score)

        print(f"   📈 P_model: {analysis['p_model']:.1%} | Ratio: {analysis['prob_ratio']:.2f}x | Conf: {analysis['confidence']:.2f}")

        if analysis["action"] == "SKIP":
            print(f"   ⏭️ Below threshold")
            continue

        db = load_hypothesis_db()
        db["hypotheses"].append({
            "slug": m["slug"],
            "question": m["question"],
            "market_price": m["price"],
            "p_model": analysis["p_model"],
            "prob_ratio": analysis["prob_ratio"],
            "confidence": analysis["confidence"],
            "factors": factors,
            "clusters": m["clusters"],
            "size_pct": 0,
            "created_at": datetime.now().isoformat(),
            "resolved": False
        })
        save_json(HYPOTHESIS_DB, db)

        size = position_size(analysis["p_model"], m["price"], available_balance)
        print(f"   💵 Position size: ${size} ({size/available_balance:.1%} of balance)")

        if buy(m, size):
            candidates_bought += 1
            available_balance -= size

            for h in db["hypotheses"]:
                if h["slug"] == m["slug"]:
                    h["size_pct"] = size / balance
                    break
            save_json(HYPOTHESIS_DB, db)

    print(f"\n✅ Bought: {candidates_bought} | Available: ${available_balance:.2f}")

    db = load_hypothesis_db()
    resolved = db.get("resolved", [])
    if len(resolved) >= BURN_IN_TRADES:
        recent = resolved[-BURN_IN_TRADES:]
        wins = sum(1 for h in recent if h.get("outcome") == "YES")
        logger.info(f"Cycle complete: bought={candidates_bought}, recent_winrate={wins/len(recent):.1%}")

    try:
        import shutil
        res = subprocess.run(["pm-trader", "balance"], capture_output=True, text=True, timeout=15)
        balance_data = json.loads(res.stdout).get("data", {})
        res = subprocess.run(["pm-trader", "portfolio"], capture_output=True, text=True, timeout=15)
        portfolio_data = json.loads(res.stdout).get("data", [])
        status = {"balance": balance_data, "portfolio": portfolio_data, "updated_at": datetime.now().isoformat()}
        with open("/root/dotm-sniper/current_status.json", "w") as f:
            json.dump(status, f, indent=2, default=str)
        shutil.copy("/root/dotm-sniper/current_status.json", "/root/.openclaw/workspace/dotm_status.json")
        shutil.copy("/root/dotm-sniper/current_status.json", "/root/.openclaw/agents/market_analyst/dotm_status.json")
        with open("/root/.openclaw/workspace/memory/portfolio-current.json", "w") as f:
            json.dump(status, f, indent=2, default=str)
    except:
        pass

if __name__ == "__main__":
    import sys
    single_run = len(sys.argv) > 1 and sys.argv[1] == "--once"

    if single_run:
        print("DOTM SNIPER v4.1 running single iteration...")
        main()
    else:
        print("DOTM SNIPER v4.1 starting...")
        while True:
            try:
                main()
            except Exception as e:
                print(f"Error: {e}")
            print("Sleeping 30 min...")
            time.sleep(1800)