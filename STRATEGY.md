# DOTM Sniper — Полная техническая документация

## 1. Что мы делаем (простыми словами)

Мы покупаем **дешёвые контракты** на Polymarket (рынок прогнозов), которые стоят **от 5 до 40 центов**, и продаём когда цена вырастает в 2-3 раза.

**Пример:**
- На Polymarket есть контракт «США выйдут из НАТО до 2027 года?»
- Цена контракта «ДА» — **8.5 центов** (толпа оценивает шанс в 8.5%)
- Наша система анализирует событие и решает: «Шанс больше — около 18%»
- Мы покупаем контракт за 8.5 центов
- Если цена вырастет до 75 центов — мы продаём и зарабатываем **+782%**

Нам не нужно, чтобы событие произошло. Достаточно, чтобы **рынок пересмотрел оценку** — цена может вырасти с 8 до 30 центов просто из-за одной новости.

### Математика

Допустим, мы делаем 10 покупок по 10 центов:

| Сценарий | Вероятность | Прибыль на $1 |
|----------|-------------|---------------|
| 7 сделок закрываются в минус (−30%) | 70% | −$0.30 × 7 = −$2.10 |
| 3 сделки дают средний рост в 3 раза | 30% | +$2.00 × 3 = +$6.00 |
| **Итого** | | **+$3.90 с $10 вложений (+39%)** |

---

## 2. Архитектура системы

### 2.1. Три сервиса systemd (24/7)

```
┌─────────────────────────────────────────────────────────────┐
│                    DOTM SNIPER (main loop, 30 мин)            │
│  Поиск рынков → Анализ → Сигнал → Размер → Покупка → TP      │
└──────────────────┬──────────────────────────┬────────────────┘
                   │                          │
    ┌──────────────▼──────────┐  ┌────────────▼──────────────┐
    │  HERMES ADVISOR (10мин)  │  │  METRICS SERVER (:8765)   │
    │  Риск-менеджер:          │  │  HTTP API:                 │
    │  • Сверка позиций        │  │  • /metrics (JSON)         │
    │  • Новости → экстренный  │  │  • /metrics/prometheus     │
    │    выход                 │  │  • /health (25 проверок)   │
    │  • Стоп-лоссы            │  │  • /dashboard (HTML)       │
    │  • Разрешение рынков     │  │                            │
    └──────────────────────────┘  └────────────────────────────┘
```

### 2.2. Главный цикл (Dotm Sniper)

Каждые 30 минут:

```
Шаг 1: ПОИСК КАНДИДАТОВ
┌─────────────────────────────────────────────┐
│ Источники:                                   │
│  • pm-trader CLI: до 200 рынков             │
│  • Gamma API: до 100 дополнительных DOTM   │
│                                              │
│ Фильтры:                                     │
│  • Цена: $0.05 – $0.40                      │
│  • Объём: ≥ $25,000                         │
│  • До закрытия: ≥ 48 часов                  │
│  • Кластер other: объём ≥ $100,000          │
│  • ALLOWED: ai_tech, russia_ukraine,        │
│    usa_politics, fed_fomc, sports_nba,      │
│    sports_ufc                                │
│  • BANNED: crypto                            │
│  • Без дубликатов (включая resolved!)        │
│                                              │
│ Результат: до 300 кандидатов                 │
└─────────────────────────────────────────────┘
                    ↓
Шаг 2: FORECAST CASCADE (3 источника, p_model override)
┌─────────────────────────────────────────────┐
│ Для каждого кандидата (price < $0.35):       │
│                                              │
│ 1. MANIFOLD MARKETS (src/manifold.py)        │
│    • search-markets API, fuzzy matching      │
│    • 5713 вопросов, бесплатный API           │
│    → Если найден матч и gap > порога:        │
│      p_model = manifold_prob (OVERRIDE)      │
│                                              │
│ 2. METACULUS (src/metaculus.py)              │
│    • Двухшаговый bridge:                     │
│      a) /api/posts/?search= (поиск вопроса)  │
│      b) Metaforecast GraphQL (probability)   │
│    • Metaculus /api/ aggregation = null      │
│      (by design, не баг)                     │
│    → Если Manifold не нашёл:                 │
│      проверяем Metaculus                     │
│                                              │
│ 3. METAFORECAST (src/metaforecast.py)        │
│    • GraphQL: metaforecast.org/api/graphql   │
│    • 6275 вопросов с 8 платформ:             │
│      Manifold(5713), Good Judgment(285),     │
│      GiveWell(163), Betfair(16), и др.       │
│    • Local cache: data/metaforecast_index    │
│      .json (24h TTL)                         │
│    → Fallback если оба выше не нашли         │
│                                              │
│Cascade logic: первый нашёл → override,       │
│остальные пропускаются                        │
└─────────────────────────────────────────────┘
                    ↓
Шаг 3: LIQUIDITY PRE-CHECK
┌─────────────────────────────────────────────┐
│ Если price < $0.35:                          │
│   get_best_ask(slug) → real ask price        │
│                                              │
│ Если price < $0.10 AND ask > 10× price:      │
│   → SKIP (нет ликвидности, экономим LLM)     │
└─────────────────────────────────────────────┘
                    ↓
Шаг 4: LLM ANALYSIS (DeepSeek)
┌─────────────────────────────────────────────┐
│ DeepSeek анализирует вопрос:                 │
│  • p_model — оценка вероятности (0-100%)     │
│  • confidence — уверенность (0-100%)         │
│  • factors — аргументы за/против             │
│                                              │
│ Retry: 3 попытки с backoff (2,4 секунды)     │
│ Circuit breaker: 60 calls/hour               │
│ Fallback: p_model = price × 2               │
└─────────────────────────────────────────────┘
                    ↓
Шаг 5: MODEL COUNCIL (9 советников + судья)
┌─────────────────────────────────────────────┐
│ Round 1: 9 моделей оценивают вопрос          │
│  • DeepSeek (основной)                       │
│  • 8 OVH моделей (Qwen, Mistral, и др.)      │
│  • Rate limit: 2 req/min → 31s между вызовами│
│  • Каждый вызов в try/except                 │
│                                              │
│ Round 2: Судья Qwen3.5-397B-A17B             │
│  • Синтезирует 9 оценок → финальное решение  │
│  • Анализирует расхождения между моделями    │
│                                              │
│ Fallback:                                    │
│  • Судья недоступен → confidence-weighted    │
│    average                                   │
│  • Все OVH недоступны → DeepSeek only        │
│                                              │
│ env: COUNCIL_DISABLED=1 отключает council    │
└─────────────────────────────────────────────┘
                    ↓
Шаг 6: P_MODEL BLENDING
┌─────────────────────────────────────────────┐
│ Если forecast cascade нашёл override:        │
│   p_model = cascade_prob (если > LLM)        │
│   ИЛИ blend: 0.6×cascade + 0.4×LLM           │
│   source_signal = "metaculus_override"       │
│   confidence += 0.10 (cap 0.95)              │
│                                              │
│ MAX_P_MODEL_RATIO = 2.0                      │
│   p_model не может быть > 2× цену            │
│                                              │
│ Калибровка:                                  │
│   1. Platt scaling (≥30 resolved per cluster)│
│   2. Isotonic regression (≥50 resolved)      │
│   3. soft_extremize: p × 1.05, cap 0.50      │
│                                              │
│ ML Blending (если обучено):                   │
│   • LightGBM (нужно 50+ resolved)            │
│   • SGD online learner (работает с 1 sample)  │
│   • Blend: 0.3×SGD + 0.3×LGBM + 0.4×LLM     │
└─────────────────────────────────────────────┘
                    ↓
Шаг 7: РАСЧЁТ СИГНАЛА (0 до ~150 баллов)
┌─────────────────────────────────────────────┐
│ БАЗОВЫЕ (до 85 б.):                          │
│                                              │
│ ratio_score    min(ratio/5, 1) × 25     25  │
│ factor_score   min((sup+high)/4, 1)×20  20  │
│ vol_score      min(volume/$500k, 1)×20  20  │
│ time_score     >180d=20, >90d=15, ...   20  │
│ metaculus_alignment  +10 / −20          ±30 │
│ cluster_adj    other=+15, sports=−15    ±15 │
│                                              │
│ ДОПОЛНИТЕЛЬНЫЕ (до +45 б.):                  │
│                                              │
│ buzz_score     GDELT(60%)+Google(40%)   20   │
│ orderbook      bid/ask imbalance         15  │
│ smart_money    Polygon on-chain          20  │
│ cascade_det    laggard markets           10  │
│                                              │
│ EXTERNAL ORACLES (до +45 б.):                │
│                                              │
│ fear_greed     cluster∈{crypto,ai_tech}  +5  │
│                AND FNG index < 30            │
│ manifold_arb   Manifold prob ≥ PM+15%   +15  │
│ dbnomics       Fed rate/CPI alignment   +10  │
│ yfinance       price within 10% target   +8  │
│ wikipedia      pageviews > 2× baseline   +7  │
│                                              │
│ ПОРОГ: signal_score ≥ min_signal (40-65)     │
│   base_threshold = 55 (settings)             │
│   long_horizon (>90d): +10                   │
│   medium (31-90d): +5                        │
│   metaculus_override: −10 (min 35)           │
│                                              │
│ BUY если: score ≥ threshold                  │
│           AND confidence ≥ 0.65              │
│           AND prob_ratio ≥ 1.5               │
└─────────────────────────────────────────────┘
                    ↓
Шаг 8: РАЗМЕР СТАВКИ (Bayesian Kelly)
┌─────────────────────────────────────────────┐
│ kelly_full = (b×p − q) / b                  │
│   b = (1−price)/price, p = p_model          │
│                                              │
│ Интегрирование по Beta(p_mean, p_std):       │
│   uncertainty_penalty = 1/(1 + aversion ×   │
│   std/mean)                                  │
│                                              │
│ kelly_fraction = kelly × tier_kelly ×       │
│   uncertainty × conviction                   │
│                                              │
│ Лимиты:                                      │
│  • max_pct на сделку (10-15%)                │
│  • max_cluster% в кластере (35-45%)          │
│  • Graph correlation r>0.4 → размер ×0.5    │
│  • Bid liquidity cap: bid_liquidity × 0.20  │
│  • Минимум $20 (DOTM ratio≥2x)              │
│                                              │
│ Conviction:                                  │
│  • ≥1.5× threshold → 100%                    │
│  • ≥1.2× threshold → 60%                     │
│  • Остальные → 30%                           │
└─────────────────────────────────────────────┘
                    ↓
Шаг 9: ФИНАЛЬНОЕ ОДОБРЕНИЕ + ПОКУПКА
┌─────────────────────────────────────────────┐
│ High-conviction skip:                        │
│   score ≥ 1.5× threshold → auto-approve      │
│   (без LLM вызова)                           │
│                                              │
│ Иначе: Advisor (DeepSeek Reasoner)           │
│   • Проверка новостей (DuckDuckGo + Tavily)  │
│   • advisor_p ≥ 0.5 × p_model → одобрить    │
│   • Иначе → вето                             │
│                                              │
│ Покупка → Bayesian init_posterior()          │
│                                              │
│ После покупки — TP ladder (лимитные ордера): │
│   • 50% акций по entry × 2                   │
│   • 50% акций по entry × 3                   │
└─────────────────────────────────────────────┘
```

### 2.3. Hermes Advisor — риск-менеджер

Параллельный сервис, проверяет позиции каждые 10-30 минут:

1. **Сверка позиций** — SQLite vs реальный портфель Polymarket, исправление расхождений
2. **Проверка новостей** — DuckDuckGo + Tavily для каждого рынка
3. **Экстренный выход** — новости доказывают невозможность → продажа немедленно
4. **Байесовское обновление** — posterior по категориям новостей (LR cap ±1.5)
5. **Стоп-лоссы** — hard −50%, ATR, trailing, time-decay
6. **Разрешение рынков** — проверка закрытия, фиксация P&L
7. **Self-learning memory** — гипотезы с p_model → resolved outcome → калибровка

---

## 3. Forecast Cascade — 3 источника внешних прогнозов

Каскадная система: **первый нашёл → override**, остальные пропускаются.

### 3.1. Manifold Markets (`src/manifold.py`, 302 строки)

| Параметр | Значение |
|----------|----------|
| API | `https://manifold.markets/api/v0/search-markets` |
| Ключ | Не требуется (бесплатно) |
| Вопросов | 5713 |
| Cache | 1 час в памяти |
| Timeout | 15 секунд |

**Логика:**
- Поиск по ключевым словам из вопроса Polymarket
- Fuzzy matching (токены, даты, числа)
- Если найден матч (score ≥ 0.25) и probability выше цены Polymarket
- Time-decay: дальние рынки требуют большего gap
- → **p_model override**

### 3.2. Metaculus (`src/metaculus.py`, 550 строк)

| Параметр | Значение |
|----------|----------|
| API | `/api/posts/?search=` + Metaforecast GraphQL bridge |
| Ключ | METACULUS_TOKEN (в .env) |
| Cache | В памяти + файловая |
| Timeout | 30 секунд, 3 retry |

**Двухшаговый bridge (обход сломанного API):**
1. **Поиск**: `GET /api/posts/?search=<query>` → находим ID вопроса
2. **Вероятность**: Metaforecast GraphQL `question(id: "metaculus-{N}")` → probability

Metaculus `/api/` и `/api2/` возвращают `aggregations.*.latest = null` для ВСЕХ вопросов (by design). Probability получается только через Metaforecast bridge.

### 3.3. Metaforecast (`src/metaforecast.py`, 466 строк)

| Параметр | Значение |
|----------|----------|
| API | `https://metaforecast.org/api/graphql` |
| Ключ | Не требуется |
| Вопросов | 6275 с 8 платформ |
| Cache | `data/metaforecast_index.json`, 24h TTL |

**Платформы в индексе:**

| Платформа | Вопросов | Тип |
|-----------|----------|-----|
| Manifold Markets | 5713 | Crowdsourced |
| **Good Judgment Open** | **285** | Проф. суперфоркастеры |
| GiveWell/OpenPhilanthropy | 163 | Благотворительность |
| Foretold | 40 | Crowdsourced |
| Insight Prediction | 27 | Crypto |
| FantasySCOTUS | 21 | Верховный суд |
| **Betfair** | **16** | Real money |
| Infer | 10 | Manifold affiliate |

**Логика:**
- Weighted consensus across platforms
- Fuzzy matching (`_fuzzy_score`)
- Dispersion penalty (высокий разброс между платформами → штраф)
- → **p_model override** (fallback если Manifold/Metaculus не нашли)

---

## 4. Model Council — 9 советников + судья

**Модуль:** `src/model_council.py` (742 строки)

### Архитектура

```
Round 1: 9 независимых оценок
┌──────────────────────────────────────┐
│ 1. DeepSeek (основной)               │
│ 2. Qwen-72B (OVH)                    │
│ 3. Mistral-Large (OVH)               │
│ 4. Mixtral-8x22B (OVH)               │
│ 5. Qwen-32B (OVH)                    │
│ 6. Codestral (OVH)                   │
│ 7. Aya-23 (OVH)                      │
│ 8. Llama-3-70B (OVH)                 │
│ 9. Qwen3.5-397B (OVH)                │
│                                      │
│ Rate limit: 2 req/min → 31s между    │
│ Batch: ~280 секунд на полный цикл    │
│ Каждый вызов в try/except            │
└──────────────┬───────────────────────┘
               ↓
Round 2: Судья синтезирует
┌──────────────────────────────────────┐
│ Qwen3.5-397B-A17B (судья)             │
│  • Анализирует 9 оценок              │
│  • Выявляет расхождения              │
│  • Взвешивает по уверенности         │
│  → Финальное p_model                 │
└──────────────────────────────────────┘
```

### Fallback chain

| Ситуация | Действие |
|----------|----------|
| Судья недоступен | Confidence-weighted average 9 моделей |
| Все OVH недоступны | DeepSeek только |
| Council полностью отключён | `COUNCIL_DISABLED=1` → пропускается |

### Интеграция

- **Single path** (`signal_scorer.py`): `council_single_consensus()` — merge с LLM оценкой
- **Batch path** (`signal_pipeline.py`): `council_batch_consensus()` — batch оценка для всех рынков

---

## 5. External Oracles — 5 бесплатных источников

**Модуль:** `src/external_oracles.py` (801 строка)

Каждый оракул добавляет **бонусные очки** к signal_score. Все вызовы в `try/except` — сбой одного не ломает остальные.

### 5.1. Alternative.me Fear & Greed Index (+5)

| Параметр | Значение |
|----------|----------|
| Endpoint | `GET https://api.alternative.me/fng/` |
| Ключ | Не требуется |
| Cache | 12 часов (file-based, atomic writes) |
| Trigger | cluster ∈ {crypto, ai_tech, tech} AND index < 30 |

**Логика:** Extreme Fear (индекс < 30) → толпа пессимистична → DOTM Yes контракты дешевле, чем должны быть → контрарианский buy сигнал.

### 5.2. Manifold Markets Arbitrage (+15)

| Параметр | Значение |
|----------|----------|
| Endpoint | `GET https://api.manifold.markets/v0/search-markets` |
| Ключ | Не требуется |
| Cache | 1 час per market slug (in-memory + RLock) |
| Trigger | Manifold prob ≥ Polymarket price + 15% (strict) |

**Логика:** Keyword extraction из вопроса (стоп-слова, месяцы, годы удаляются) → поиск на Manifold → если найден market с probability ≥ 15% выше цены Polymarket → кросс-платформенный арбитраж.

### 5.3. DBnomics Macroeconomic Data (+10)

| Параметр | Значение |
|----------|----------|
| Endpoint | `GET https://api.db.nomics.world/v22/series/FRED/...` |
| Ключ | Не требуется |
| Cache | 24 часа (file-based) |
| Trigger | cluster ∈ {fed_fomc, us_economic} AND macro alignment |

**Серии:** Fed Funds Rate (FRED/FEDFUNDS), CPI (FRED/CPIAUCSL)

**Alignment heuristics:**

| Вопрос содержит | Условие | Логика |
|-----------------|---------|--------|
| "rate cut", "rate decrease" | Fed rate ≥ 4.0% | Rate cuts likely |
| "rate hike", "rate increase" | Fed rate ≤ 2.0% | Rate hikes possible |
| "inflation", "CPI" | CPI > 300 | High inflation |
| "recession" | Fed rate ≥ 4.5% | High rates → recession |

### 5.4. Yahoo Finance (+8)

| Параметр | Значение |
|----------|----------|
| Библиотека | `yfinance` (скрапит Yahoo Finance) |
| Ключ | Не требуется |
| Cache | 1 час per ticker (in-memory) |
| Trigger | Тикер найден в вопросе AND цена в пределах 10% от target |

**Тикер маппинг (25+ тикеров):**

| Ключевые слова | Тикер | Что отслеживает |
|----------------|-------|-----------------|
| "s&p 500", "s&p", "sp500" | SPY | S&P 500 ETF |
| "nasdaq" | QQQ | NASDAQ 100 ETF |
| "dow jones", "dow" | DIA | Dow Jones ETF |
| "bitcoin", "btc" | BTC-USD | Bitcoin |
| "ethereum", "eth" | ETH-USD | Ethereum |
| "tesla" | TSLA | Tesla stock |
| "nvidia" | NVDA | Nvidia stock |
| "apple" | AAPL | Apple stock |
| "gold" | GLD | Gold ETF |
| "oil", "crude" | CL=F | Crude oil futures |
| "vix" | ^VIX | Volatility index |

**Логика:** Regex извлекает целевую цену из вопроса ("below 5000", "under $200") → yfinance получает текущую цену → если proximity ≤ 10% → контракт "live" → +8.

### 5.5. Wikipedia Pageviews Spike (+7)

| Параметр | Значение |
|----------|----------|
| Endpoint | `https://wikimedia.org/api/rest_v1/metrics/pageviews/...` |
| Ключ | Не требуется |
| Cache | 6 часов per article (in-memory + RLock) |
| Trigger | Pageviews за 3 дня > 2× от 20-дневного baseline median |

**Логика:**
1. `_extract_entities()` — извлечение имён из вопроса (заглавные фразы)
2. `_search_wikipedia()` — поиск статьи через Wikipedia search API
3. `_fetch_pageviews()` — 23 дня метрик через Wikimedia REST API
4. `_detect_wiki_spike()` — recent median > 2× baseline median

**Пример:** Вопрос «Will Donald Trump win?» → article "Donald_Trump" → pageviews 5000/day baseline → spike to 15000/day → +7.

### Сводка оракулов

| # | Источник | Bonus | Бесплатно | Кэш |
|---|----------|-------|-----------|-----|
| 1 | Fear & Greed | +5 | ✅ | 12h file |
| 2 | Manifold Arbitrage | +15 | ✅ | 1h in-mem |
| 3 | DBnomics Macro | +10 | ✅ | 24h file |
| 4 | Yahoo Finance | +8 | ✅ | 1h in-mem |
| 5 | Wikipedia Spike | +7 | ✅ | 6h in-mem |
| | **Максимум** | **+45** | | |

---

## 6. Стратегия выхода (когда продаём)

### Лесенка продаж (TP Ladder)

```
Цена растёт...
    │
    ├── entry × 2 ── Продать 50% позиции
    │                 (гарантированная прибыль)
    │
    └── entry × 3 ── Продать остаток
                      (основная прибыль)
```

- Автообновление: если нет pending ордеров и цена < $0.70
- Пропуск ступеней: если стоимость < $5
- Отмена TP перед market sell (предотвращает race condition)

### Защитные механизмы

| Триггер | Действие | Зачем |
|---------|----------|-------|
| Цена ≤ entry × 0.50 | Продать всё | Hard stop-loss −50% |
| Цена ≤ entry − 2.5×ATR(7д) | Продать всё | ATR-стоп-лосс |
| Цена выросла >30%, откатилась | Трейлинг-стоп | max(ATR, high×0.75) |
| Трейлинг сработал → 5 мин → подтверждение | Продать всё | Двойное подтверждение |
| Age > 60% TTL, price < entry×1.5 | Продать всё | Time-decay exit |
| Новости опровергают ставку | Продать немедленно | Экстренный выход |
| Drawdown > 10% | Прекратить новые покупки | Drawdown stop |
| current_price/metaculus ≥ 0.60 | Продать | Конвергентный TP |

### Fallback для неликвидных позиций

| Ситуация | Действие |
|----------|----------|
| Market sell не удался | Aggressive limit at best_bid |
| Slippage guard 5 раз подряд | Force market sell |
| Failed sell | Recheck через 15 минут |
| Position missing из API | _miss_count ≥ 3 перед удалением |

---

## 7. Система уровней (Tiers)

| Уровень | Баланс | Kelly | Базовый % | Макс. % | Позиций | Кластер | Макс. цена |
|---------|--------|-------|-----------|---------|---------|---------|------------|
| Микро | до $2,000 | 40% | 5% | 10% | 15 | 35% | $0.40 |
| Рост | $2K–$10K | 30% | 3% | 12% | 20 | 35% | $0.40 |
| Установившийся | $10K–$50K | 35% | 3.5% | 15% | 25 | 40% | $0.50 |
| Масштаб | $50K+ | 40% | 4% | 15% | 30 | 45% | $0.50 |

### Bayesian Kelly

```
kelly_full = (b × p − q) / b
  где b = (1−price)/price, p = p_model, q = 1−p

Интегрирование по Beta(p_mean, p_std):
  uncertainty_penalty = 1/(1 + risk_aversion × std/mean)

kelly_fraction = kelly_full × tier_kelly × uncertainty × conviction
```

При conf=0.72: −25% размер ставки. При conf=0.55: −33%.

---

## 8. Дополнительные сигналы в signal_score

### Social Buzz (до +20 баллов)

**Модуль:** `src/social_buzz.py` (321 строка). Кэш 1 час.

| Источник | Вес | Статус |
|----------|-----|--------|
| GDELT DOC API v2 | **60%** | ✅ Работает (10мин кулдаун при rate limit) |
| Google News RSS | **40%** | ✅ Работает |

Reddit удалён: 403 Forbidden с 2023 (требует OAuth2). Заменён Wikipedia oracle.

### Order Book Depth (до +15 баллов)

**Модуль:** `src/orderbook_analyzer.py` — CLOB API, bid/ask imbalance. Кэш 5 мин.
- Imbalance > 0.4 → +15
- Bid wall ($5k+) → +12

### Smart Money Tracking (до +20 баллов)

**Модуль:** `src/smart_money.py` — Polygonscan CTF Exchange ERC1155 events. Кэш 10 мин.
- Требует POLYGONSCAN_API_KEY
- Wallet discovery + activity tracking

### Cascade Detector (до +10 баллов)

**Модуль:** `src/cascade_detector.py` — price movement tracking.
- ≥2 рынка одного кластера двинулись ≥15% за 60 мин = cascade
- Laggard markets → +10
- 2-часовой decay

### Cross-Market Graph

**Модуль:** `src/market_graph.py` — networkx граф корреляций.
- Shared clusters (0.5), entity overlap (0.3), relationships (0.4)
- Корреляция > 0.4 → размер позиции × 0.5
- Louvain community detection

---

## 9. Калибровка и экстремизация

Пайплайн `calibrate_prediction()` в `signal_scorer.py`:

```
1. Platt scaling (per-cluster, ≥30 сэмплов)
   └── calibration_tracker.py: get_platt_calibrated()

2. Isotonic regression (≥50 сэмплов)
   └── calibration.py: IsotonicCalibrator

3. soft_extremize (fallback, <50 resolved)
   └── p × 1.05, cap 0.50

4. MAX_P_MODEL_RATIO = 2.0
   └── p_model не может быть > 2× цену
```

**Прогресс:** 0 properly resolved predictions → ждёт закрытия рынков → Platt/isotonic начнут работать автоматически.

---

## 10. Машинное обучение

### LightGBM (`src/ml_predictor.py`, 256 строк)

- Walk-forward validation
- Нужно 50+ resolved samples для обучения
- Веса: 0.3 (SGD) + 0.3 (LGBM) + 0.4 (LLM)
- Автоматическое обучение при достижении порога

### SGD Online Learner (`src/online_learner.py`, 256 строк)

- Работает с 1-го sample
- Drift detection
- Incremental partial_fit
- Persistence: `data/ml_models/sgd_model.joblib`

**Текущий статус:** 1 resolved sample — LightGBM отключён, SGD инициализирован.

---

## 11. Категории рынков

### Разрешённые кластеры

| Кластер | Описание |
|---------|----------|
| ai_tech | ИИ, технологии |
| russia_ukraine | Геополитика |
| usa_politics | Политика США |
| fed_fomc | Экономика, ФРС |
| sports_nba | Баскетбол |
| sports_ufc | Единоборства |

**BANNED:** `crypto` (слишком волатильно, неэффективно оценивается)

### Корректировки кластеров

| Кластер | Корректировка | Причина |
|---------|---------------|---------|
| other | +15 | Бонус за нестандартность |
| sports_nba | −15 | Слишком эффективно оценивается |
| crypto | −25 | BANNED, но штраф остался |

### Коррелированные группы (MAX_CORRELATED_GROUP_PCT = 25%)

| Группа | Кластеры |
|--------|----------|
| trump_admin_politics | usa_politics, russia_ukraine, geopolitics, venezuela |
| us_economic | fed_fomc, usa_politics |
| sports | sports_nba, sports_ufc |
| tech_ai | ai_tech, tech |

---

## 12. Управление рисками

### Многоуровневая защита

| Уровень | Правило | Значение |
|---------|---------|----------|
| Market | Price-aware slippage (DOTM: ask до 10× price) | Блокирует экстремальный спред |
| Signal | Liquidity pre-check (ask > 10× price → SKIP) | Экономит LLM токены |
| Signal | DOTM_PRICE_FLOOR: $0.05 | Измеримая вероятность |
| Signal | MIN_P_MODEL: 10% | Значимый edge |
| Position | Hard stop-loss: −50% | Ограничение убытка |
| Position | Time-decay exit (age > 60% TTL) | Авто |
| Portfolio | Drawdown stop: −10% | Прекратить новые покупки |
| Portfolio | LLM circuit breaker: 60 calls/hour | Защита от перерасхода |
| Portfolio | Graph correlation: r > 0.4 → ×0.5 | Диверсификация |
| Advisor | Override: advisor_p ≥ 0.5 × p_model | Не выше реальной вероятности |
| Sizing | Minimum trade $20 (ratio≥2x) | Не копеечные сделки |
| Sizing | Bid liquidity cap: × 0.20 | Не более 20% книги |

### Байесовский обновитель (`src/bayesian_updater.py`)

7 категорий новостей с likelihood ratios (LR cap ±1.5):

| Категория | p_yes | LR |
|-----------|-------|-----|
| confirms_impossible | 0.02 | −1.5 (capped) |
| strongly_contradicts | 0.10 | −1.5 (capped) |
| moderately_contradicts | 0.40 | −0.41 |
| neutral | 0.50 | 0.0 |
| moderately_supports | 0.65 | +0.62 |
| strongly_supports | 0.85 | +1.5 (capped) |
| confirms_inevitable | 0.95 | +1.5 (capped) |

Log-odds posterior по каждой позиции, инициализируется при покупке, обновляется при новостях.

---

## 13. Инфраструктура и данные

### Systemd сервисы

| Сервис | Описание | Интервал |
|--------|----------|----------|
| sniper | Главная программа | 30 мин цикл |
| hermes | Риск-менеджер | 10-30 мин цикл |
| metrics | HTTP сервер :8765 | Постоянно |

### Cron задания

| Расписание | Задача |
|------------|--------|
| */5 | Watchdog (проверка живности сервисов) |
| */30 | Advisor cycle |
| */30 | Equity snapshot |
| Hourly | Health check + alerts |
| 08:00 UTC | Daily report → Telegram |
| 03:00 | SQLite backup (7-day retention) |

### Хранилища данных

| Хранилище | Формат | Назначение |
|-----------|--------|------------|
| `sniper.db` | SQLite WAL | Позиции, гипотезы, settings, trade_history |
| `equity_curve.json` | JSON | История equity snapshots |
| `price_tracking.json` | JSON | Кэш p_model по рынкам |
| `calibration_model.json` | JSON | Per-cluster calibration |
| `bayesian_state.json` | JSON | Log-odds posterior per position |
| `market_graph.json` | JSON | Граф корреляций рынков |
| `metaforecast_index.json` | JSON | 6275 вопросов (24h cache) |
| `oracle_*.json` | JSON | Оракул кэши (FNG, Fed rate, CPI) |
| `ml_models/` | Binary | LightGBM + SGD модели |
| `logs/` | Текст | RotatingFileHandler 10MB × 3 |

### Переменные окружения (.env)

| Переменная | Назначение |
|------------|------------|
| DEEPSEEK_API_KEY | LLM модель (основная) |
| OVH_API_KEY | 8 моделей OVH + судья |
| METACULUS_TOKEN | Metaculus API |
| ALL_PROXY | socks5h://127.0.0.1:1080 (VLESS+Reality) |
| NO_PROXY | localhost,127.0.0.1 |
| POLYGONSCAN_API_KEY | Smart money tracking |
| TAVILY_API_KEY | News search (оба ключа exhausted) |
| TG_BOT_TOKEN | Telegram уведомления |
| TG_CHAT_ID | Telegram чат |

---

## 14. Модули проекта (53 файла, ~17,400 строк)

### Ядро системы

| Модуль | Строк | Ответственность |
|--------|-------|-----------------|
| dotm_sniper.py | 631 | Оркестрация, main loop, shutdown |
| signal_pipeline.py | 708 | Batch анализ, re-exports |
| signal_scorer.py | 508 | _compute_signal_score, calibration, full analysis |
| market_fetcher.py | 259 | fetch_markets, Gamma API, фильтры |
| config.py | 73 | 35+ констант, путей |
| schema.py | 139 | 70+ JSON ключей |

### Forecast cascade

| Модуль | Строк | Ответственность |
|--------|-------|-----------------|
| manifold.py | 302 | Manifold Markets API, fuzzy match |
| metaculus.py | 550 | Metaculus two-step bridge |
| metaforecast.py | 466 | Cross-platform GraphQL, 6275 вопросов |

### Внешние оракулы

| Модуль | Строк | Ответственность |
|--------|-------|-----------------|
| external_oracles.py | 801 | 5 источников: FNG, Manifold arb, DBnomics, YFinance, Wiki |

### Model council

| Модуль | Строк | Ответственность |
|--------|-------|-----------------|
| model_council.py | 742 | 9 советников + судья Qwen3.5-397B |

### Исполнение и риски

| Модуль | Строк | Ответственность |
|--------|-------|-----------------|
| trade_executor.py | 211 | execute_trade, pending_fill |
| sell_executor.py | 670 | Продажи, стопы, ATR, trailing, fallback |
| position_manager.py | 352 | Bayesian Kelly, tiers, conviction |
| order_manager.py | 287 | pm-trader CLI, ордера, TP ladder |
| hermes_advisor.py | 414 | Hermes main loop, reconciliation |
| hermes_risk.py | 515 | Emergency exit, stop-loss, news |
| hermes_resolution.py | 85 | Market resolution |
| hermes_memory.py | 375 | Self-learning memory |

### Сигналы

| Модуль | Строк | Ответственность |
|--------|-------|-----------------|
| social_buzz.py | 321 | GDELT(60%)+Google(40%) |
| orderbook_analyzer.py | 104 | CLOB bid/ask imbalance |
| smart_money.py | 189 | Polygon on-chain tracking |
| cascade_detector.py | 282 | Price cascade detection |
| market_graph.py | 295 | networkx корреляции |
| news_scanner.py | 253 | Tavily + DuckDuckGo |

### ML / Калибровка

| Модуль | Строк | Ответственность |
|--------|-------|-----------------|
| ml_predictor.py | 256 | LightGBM, walk-forward |
| online_learner.py | 256 | SGD online learning |
| calibration.py | 131 | Isotonic regression |
| calibration_tracker.py | 384 | Platt scaling |
| bayesian_updater.py | 231 | Log-odds posterior |
| probability_calibrator.py | 223 | Probability calibration utilities |

### Данные и инфраструктура

| Модуль | Строк | Ответственность |
|--------|-------|-----------------|
| db.py | 499 | SQLite WAL, migrations |
| hypotheses_db.py | 83 | Hypothesis storage |
| positions_db.py | 64 | Position storage |
| contracts.py | 76 | TypedDict contracts |
| health_checks.py | 945 | 25 checks |
| health_monitor.py | 370 | Health orchestrator |
| metrics_server.py | 139 | HTTP :8765, Prometheus |
| dashboard.py | 131 | HTML dashboard |
| equity_tracker.py | 252 | Equity snapshots |
| cluster_report.py | 77 | Per-cluster PnL |
| log_formatter.py | 44 | JSON + human log format |
| tg_sender.py | 180 | Telegram queue + retry |
| utils.py | 321 | JSON load/save, env, LLM parse |

### Бэктестинг

| Модуль | Строк | Ответственность |
|--------|-------|-----------------|
| backtest_simulator.py | 730 | Core simulation engine |
| backtest_stats.py | 452 | Stats, Brier, win rate |
| dotm_backtester.py | 334 | CLI entry points |
| dotm_optimizer.py | 441 | Parameter optimization |
| dotm_report.py | 270 | Report generation |
| stress_test.py | 280 | Stress testing |
| resolution.py | 446 | Market resolution logic |

---

## 15. Качество кода

### Тестирование

**1192 теста** в 47 файлах, все проходящие:

| Категория | Тестов | Покрытие |
|-----------|--------|----------|
| external_oracles | 106 | 5 источников, caching, fault injection |
| signal_pipeline | ~130 | Scoring, calibration, batch, full analysis |
| backtester | ~217 | Simulation, stats, P&L |
| position_manager | ~49 | Kelly, tiers, cluster limits |
| sell_executor | ~35 | Trailing, hard stop, time-decay |
| bayesian_updater | ~29 | Log-odds, posterior, LR cap |
| health_monitor | ~33 | 25 checks, alerts |
| market_graph | ~34 | Graph, correlation, cascade |
| contracts | 14 | Dict shapes, key access, cascade order |
| fault_injection | 13 | Timeout, 500, 429, council crash |
| e2e integration | ~10 | Full lifecycle с real SQLite |
| invariants | 13 | Property-based boundary tests |
| Прочие | ~513 | db, utils, config, order_manager, etc. |

### Статический анализ

- **ruff**: 0 violations (правила E/F/W/B/SIM/UP)
- **Pre-commit hook**: ruff + 14 contract tests + smoke import
- **TypedDict contracts**: `src/contracts.py` — GapCheckResult, ForecastResult

### Контракт-тесты (`tests/test_contracts.py`)

14 тестов проверяют межмодульные контракты:
- Dict shapes (наличие всех ключей)
- Cascade order (Manifold → Metaculus → Metaforecast)
- try/except wrapping (council calls)
- Forecast source key access

---

## 16. Текущие результаты

### Бэктест

| Метрика | Значение |
|---------|----------|
| Период | Июнь 2024 — Май 2026 (24 месяца) |
| Рынков | 1,903 DOTM контракта |
| Сделок (walk-forward) | 246 |
| Winrate | 28.9% |
| Доходность | +403% |
| Комиссии | 2% за сделку |

### Живые результаты (июнь 2026)

| Метрика | Значение |
|---------|----------|
| Баланс | ~$1,430 equity, ~$1,311 cash |
| Открытые позиции | 11 |
| Разрешённые рынки | 19 resolved, 0% winrate |
| Коммитов сегодня | d4f55a8, 08a2c2c, 90fe0ae |

### Уроки и исправления

**0% winrate (0/19 resolved)** — исправлено:
- MIN_P_MODEL: 3% → 10%
- MAX_P_MODEL_RATIO: 3.0 → 2.0
- DOTM_PRICE_FLOOR: $0.05
- ratio_score: min(ratio/5, 1) × 25 (harder to max out)
- Prompt bias reduction: contrary-evidence search, pessimistic framing
- 4 audit bugs fixed (manifold found flag, metaculus_prob_val, council try/except, backtest imports)

---

## 17. План роста

### От $1,500 к $5,000/месяц

```
Месяц 6:     ~$5,800
Месяц 12:    ~$12,100 ← переход «Рост»
Месяц 18:    ~$21,200 ← «Установившийся»
Месяц 24:    ~$35,400
Месяц 33:    ~$78,600 ← $5,500/мес дохода (при 7%/мес)
```

### Условия успеха

1. 50+ resolved сделок → isotonic calibration
2. 50+ resolved → LightGBM начнёт обучаться
3. POLYGONSCAN_API_KEY → smart money tracking
4. Мониторинг health checks и dashboard
5. Пополнения $500/мес значительно ускоряют рост

---

## 18. Полная карта взаимосвязей

```
                              ┌──────────────────┐
                              │  market_fetcher   │
                              │  (Gamma API)      │
                              └────────┬──────────┘
                                       │
                                       ▼
┌──────────────┐    ┌──────────────────────────────────┐
│  manifold.py │───▶│     signal_scorer.py              │
│  (cascade 1) │    │     full_market_analysis()        │
└──────────────┘    │                                   │
                    │  ┌── liquidity check (order_mgr)  │
┌──────────────┐    │  ├── forecast cascade ────────────│──▶ manifold.py
│  metaculus   │───▶│  │   (Manifold→Metaculus→Meta)   │──▶ metaculus.py
│  (cascade 2) │    │  │                               │──▶ metaforecast.py
└──────────────┘    │  ├── DeepSeek LLM analysis        │
                    │  ├── Model Council (9+1) ─────────│──▶ model_council.py
┌──────────────┐    │  │                               │
│ metaforecast │───▶│  ├── p_model blending             │
│  (cascade 3) │    │  │   (cascade override / blend)   │
└──────────────┘    │  │                               │
                    │  ├── calibration ─────────────────│──▶ calibration.py
┌──────────────┐    │  │   (Platt/isotonic/extremize)  │──▶ calibration_tracker.py
│model_council │───▶│  │                               │
│  (9+judge)   │    │  ├── ML blend ───────────────────│──▶ ml_predictor.py
└──────────────┘    │  │   (LightGBM + SGD)            │──▶ online_learner.py
                    │  │                               │
                    │  ├── _compute_signal_score:       │
                    │  │   ratio + factors + vol + time │
                    │  │   + metaculus_align + cluster  │
                    │  │   + buzz ──────────────────────│──▶ social_buzz.py
                    │  │   + orderbook ─────────────────│──▶ orderbook_analyzer.py
                    │  │   + smart_money ───────────────│──▶ smart_money.py
                    │  │   + cascade ───────────────────│──▶ cascade_detector.py
                    │  │   + oracle_bonus ──────────────│──▶ external_oracles.py
                    │  │     (FNG/Manifold/DBnomics/    │
                    │  │      YFinance/Wikipedia)       │
                    │  │                               │
                    │  ├── BUY/SKIP decision            │
                    │  └── signal_score output          │
                    └──────────────┬────────────────────┘
                                   │
                    ┌──────────────▼────────────────────┐
                    │     position_manager.py            │
                    │  Bayesian Kelly, tiers, limits     │──▶ bayesian_updater.py
                    │  Graph correlation ────────────────│──▶ market_graph.py
                    └──────────────┬────────────────────┘
                                   │
                    ┌──────────────▼────────────────────┐
                    │  Advisor check (DeepSeek Reasoner) │──▶ news_scanner.py
                    │  High-conviction → auto-approve    │
                    └──────────────┬────────────────────┘
                                   │
                    ┌──────────────▼────────────────────┐
                    │     trade_executor.py              │
                    │  execute_trade → pm-trader CLI     │
                    │  Bayesian init_posterior()         │──▶ bayesian_updater.py
                    │  TP ladder → order_manager.py      │──▶ order_manager.py
                    └──────────────┬────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │      hermes_advisor.py             │
                    │  (параллельный сервис, 10 мин)     │
                    │                                   │
                    │  Сверка позиций ──────────────────│──▶ positions_db.py
                    │  Новости → экстренный выход ──────│──▶ hermes_risk.py
                    │  Стоп-лоссы ──────────────────────│──▶ sell_executor.py
                    │  Разрешение рынков ───────────────│──▶ hermes_resolution.py
                    │  Байесовское обновление ──────────│──▶ bayesian_updater.py
                    │  Self-learning ───────────────────│──▶ hermes_memory.py
                    └──────────────────────────────────┘
```

---

## 19. Резюме

**Стратегия:** Покупать дешёвые контракты ($0.05–$0.40) на маловероятные события, которые рынок недооценивает. Продавать по лесенке (entry×2, entry×3) при росте цены.

**Преимущество:** Для прибыли не нужно, чтобы событие произошло — достаточно, чтобы рынок пересмотрел оценку.

**Технологический стек:**
- **53 модуля**, ~17,400 строк Python
- **1192 теста**, все проходящие
- **3 forecast cascade** источника (Manifold + Metaculus + Metaforecast)
- **5 external oracles** (FNG + Manifold arb + DBnomics + YFinance + Wikipedia)
- **9+1 model council** (DeepSeek + 8 OVH + Qwen3.5 судья)
- **Bayesian Kelly** position sizing с uncertainty penalty
- **ML blending** (LightGBM + SGD online learner)
- **Graph correlation** для диверсификации
- **25 health checks** + Prometheus + dashboard
- **systemd** + cron + SQLite WAL + atomic writes
- **ruff** + TypedDict contracts + pre-commit hook
