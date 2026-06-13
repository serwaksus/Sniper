# DOTM Sniper — Торговая система на Polymarket

## 1. Простыми словами: что мы делаем

Мы покупаем **дешёвые контракты** на сайте Polymarket (рынок прогнозов), которые стоят **от 5 до 40 центов**, и ждём, пока их цена вырастет в несколько раз.

**Пример:**
- На Polymarket есть контракт «США выйдут из НАТО до 2027 года?»
- Рынок оценивает это событие как маловероятное — цена контракта «ДА» всего **8.5 центов**
- Это означает: толпа считает, что шанс выхода из НАТО — 8.5%
- Наша система анализирует событие и решает: «Нет, шанс больше — около 18%»
- Мы покупаем контракт за 8.5 центов
- Если цена вырастет до 75 центов — мы продаём и зарабатываем **+782%**

Мы не пытаемся угадать, произойдёт событие или нет. Нам достаточно, чтобы **рынок пересмотрел свою оценку** — цена может вырасти с 8 центов до 30 просто потому, что появились новые данные, даже если событие в итоге не произойдёт.

---

## 2. Что такое Polymarket

Polymarket — это биржа прогнозов. Пользователи торгуют контрактами на исходы реальных событий:
- Политика («Кто выиграет выборы?»)
- Экономика («Понизит ли ФРС ставку?»)
- Спорт («Выиграют ли Кливленд Браунс?»)
- Технологии («Будет ли принят закон об ИИ?»)

Каждый контракт стоит от $0.01 до $0.99 и отражает вероятность события по мнению рынка. Если контракт «ДА» стоит $0.08 — рынок считает шанс 8%.

Если событие происходит — контракт «ДА» выплачивает $1.00. Если нет — обесценивается до $0.

---

## 3. Что такое DOTM и почему это работает

**DOTM** = Deep Out-The-Money = «Глубоко вне денег». Это контракты с ценой **до 40 центов**, то есть рынки, где большинство не верит в наступление события.

### Почему мы покупаем то, во что никто не верит?

Потому что **рынок часто ошибается в оценке маловероятных событий**. Причины:

1. **Люди не любят делать ставки на маловероятное** — это психологически неприятно. Большинство предпочитает ставить на «верняки» за 70-90 центов
2. **Рынок недооценивает хвостовые риски** — редкие, но возможные события (выход из НАТО, переворот, неожиданное политическое решение)
3. **Для роста цены с 5 до 15 центов нужен небольшой сдвиг мнения** — достаточно одной новости, одного комментария политика
4. **Потолок роста огромный** — если мы купили за 5 центов, а событие произошло — мы получаем $1, то есть **+1900%**

### Жёсткий ценовой фильтр (обновлено)

Мы **не покупаем** контракты дешевле **$0.05**. На суб-5-центовых рынках истинная вероятность практически неизмерима, а LLM систематически завышает p_model. Это привело к 0% winrate (0 выигрышей из 19 разрешённых позиций) и потребовало пересмотра параметров.

### Математика на простом примере

Допустим, мы делаем 10 покупок по 10 центов:

| Сценарий | Вероятность | Прибыль на $1 |
|----------|-------------|---------------|
| 7 сделок закрываются в минус (−30%) | 70% | −$0.30 × 7 = −$2.10 |
| 3 сделки дают средний рост в 3 раза | 30% | +$2.00 × 3 = +$6.00 |
| **Итого** | | **+$3.90 с $10 вложений (+39%)** |

Нам не нужно угадывать каждый раз. Достаточно угадывать **3 из 10**, и мы в большом плюсе, потому что прибыль от удачных сделок в несколько раз больше убытка от неудачных.

---

## 4. Как работает наша система

Система состоит из **трёх сервисов**, работающих параллельно 24/7 под systemd.

### 4.1. Снайпер (Dotm Sniper) — основная программа

Работает циклами каждые 30 минут. Каждый цикл:

```
Шаг 1: ПОИСК КАНДИДАТОВ
┌─────────────────────────────────────────────┐
│ Источники:                                   │
│  • pm-trader CLI: до 200 рынков             │
│  • Gamma API (gamma-api.polymarket.com):    │
│    до 100 дополнительных DOTM рынков        │
│                                              │
│ Фильтры:                                     │
│  • Только активные (не закрытые)             │
│  • Цена ≥ $0.05 (PRICE_FLOOR) и ≤ $0.40    │
│  • Объём торгов ≥ $25,000                   │
│  • До закрытия рынка минимум 48 часов        │
│  • Не криптовалюта                           │
│  • Без дубликатов (включая resolved!)        │
│  • Кластер «other»: объём ≥ $100,000        │
│                                              │
│ Результат: до 300 кандидатов на анализ       │
└─────────────────────────────────────────────┘
                    ↓
Шаг 2: ОЦЕНКА КАЖДОГО КАНДИДАТА
┌─────────────────────────────────────────────┐
│ Pre-check: LIQUIDITY                         │
│  Если ask > 10× price (для DOTM < $0.10):   │
│  → SKIP (не тратим LLM на неликвид)         │
│                                              │
│ Искусственный интеллект (DeepSeek) анализи-  │
│ рует вопрос рынка и выдаёт:                  │
│  • p_model — наша оценка вероятности (0-100%)│
│  • confidence — уверенность в оценке (0-100%) │
│  • factors — список аргументов за и против    │
│                                              │
│ Также проверяется:                           │
│  • Прогноз Metaculus (30s timeout, 3 retry) │
│  • Категория рынка (политика, спорт и т.д.)  │
│                                              │
│ ML blending (если модели обучены):           │
│  • LightGBM (нужно 50+ resolved)             │
│  • SGD online (работает с 1-го sample)       │
│  • Weights: LGBM+SGD+LLM = 0.3+0.3+0.4     │
│                                              │
│ Metaculus override: если metaculus > price   │
│ и разрыв > 30%, то p_model = metaculus,     │
│ confidence +0.10, порог −10 (мин 35)        │
└─────────────────────────────────────────────┘
                    ↓
Шаг 3: РАСЧЁТ СИГНАЛА (от 0 до ~130 баллов)
┌─────────────────────────────────────────────┐
│ Соотношение цен (ratio_score)     до 25 б.  │
│   min(prob_ratio / 5.0, 1.0) × 25          │
│                                              │
│ Качество аргументов (factor_score) до 20 б. │
│   min((supporting + high_weight) / 4, 1.0)  │
│   × 20                                      │
│                                              │
│ Объём торгов (vol_score)          до 20 б.  │
│   min(volume / $500,000, 1.0) × 20          │
│                                              │
│ Время до закрытия (time_score)    до 20 б.  │
│   >180д=20, >90д=15, >30д=12, >14д=8,      │
│   >2д=5, else=0                              │
│                                              │
│ Совпадение с Metaculus         до ±30 б.    │
│   +10 если p_model ~ metaculus (diff<5%)    │
│   −20 если p_model>>metaculus но            │
│      metaculus~price                         │
│                                              │
│ Корректировка кластера        ±15 баллов    │
│   other=+15, sports_nba=−15                 │
│                                              │
│ Социальный Buzz                до +20 б.    │
│ Order Book Depth               до +15 б.    │
│ Smart Money                    до +20 б.    │
│ Cascade Detector               до +10 б.    │
│                                              │
│ Порог: signal=50                             │
│ MIN_PROB_RATIO = 2.0, MIN_CONFIDENCE = 0.65 │
│ min_p_model = 10%                            │
│ DOTM_PRICE_FLOOR = $0.05                     │
│ Если набрал Enough → СИГНАЛ «ПОКУПАТЬ»       │
└─────────────────────────────────────────────┘
                    ↓
Шаг 4: РАСЧЁТ РАЗМЕРА СТАВКИ
┌─────────────────────────────────────────────┐
│ Bayesian Kelly (uncertainty-aware):          │
│                                              │
│   kelly_full = (b × p − q) / b              │
│     где b = (1−price)/price, p = p_model,   │
│     q = 1 − p                               │
│                                              │
│   Интегрирование Kelly по Beta(p_mean,      │
│   p_std) вместо точечной оценки             │
│   uncertainty_penalty = 1/(1 + aversion ×   │
│   std/mean)                                  │
│                                              │
│   При conf=0.72: −25% размер ставки         │
│   При conf=0.55: −33%                       │
│                                              │
│   kelly_fraction = kelly_full × tier_kelly  │
│                    × uncertainty_penalty     │
│                    × conviction              │
│                                              │
│ Проверки:                                    │
│  • Не больше max_pct на одну сделку          │
│  • Не больше max_cluster% в кластере         │
│  • Граф корреляций: r>0.4 → размер ×0.5     │
│  • Bid liquidity cap: max bid_liquidity×0.20 │
│  • Минимум $20 для DOTM с ratio≥2x          │
└─────────────────────────────────────────────┘
                    ↓
Шаг 5: ПОКУПКА
┌─────────────────────────────────────────────┐
│ Проверка новостей на рынке                   │
│ (DuckDuckGo fallback если Tavily исчерпан)  │
│ Советник (DeepSeek Reasoner) даёт финальное  │
│ одобрение или вето                           │
│                                              │
│ Advisor override: advisor_p ≥ 0.5 × p_model │
│ High-conviction skip: score ≥ 1.5× threshold │
│ → auto-approve без LLM вызова               │
│                                              │
│ Если все проверки пройдены → ПОКУПКА         │
│                                              │
│ После покупки автоматически ставятся         │
│ лимитные ордера на продажу (TP ladder):      │
│  • 50% акций по entry_price × 2              │
│  • 50% акций по entry_price × 3              │
│  (Последняя ступень = остаток, не max 1)     │
│                                              │
│ Bayesian init_posterior() на каждой покупке  │
└─────────────────────────────────────────────┘
```

### 4.2. Гермес (Hermes Advisor) — наблюдатель

Работает параллельно, проверяет открытые позиции каждые 10-30 минут.

**Что делает Гермес:**

1. **Сверка позиций** — сравнивает SQLite с реальным портфелем Polymarket, исправляет расхождения
2. **Проверка новостей** — DuckDuckGo (duckduckgo-search library) + Tavily (если доступен) для каждого рынка
3. **Экстренный выход** — если новости однозначно доказывают, что событие невозможно — продаём немедленно
4. **Отслеживание частичных продаж** — если сработала только часть лимитного ордера, Гермес это заметит
5. **Байесовское обновление** — обновляет апостериорную вероятность по категории новости (LR cap ±1.5)
6. **Разрешение рынков** — проверяет, не закрылся ли рынок, фиксирует P&L
7. **Портфель запрашивается 1 раз за цикл** (не N раз)
8. **Использует `pm-trader` CLI** для ордеров (JSON responses)

### 4.3. Metrics сервер

HTTP на `127.0.0.1:8765`:
- `/metrics` — JSON метрики
- `/metrics/prometheus` — Prometheus format (6 gauges)
- `/health` — Health check (25 проверок)
- `/dashboard` — HTML dashboard (dark theme, auto-refresh 30s)

### 4.4. Отчётность

| Что | Когда | Куда |
|-----|-------|------|
| Снимок капитала | Каждые 30 минут | SQLite + equity_curve.json |
| Ежедневный отчёт | 08:00 UTC (1 раз/день) | Telegram |
| Уведомление о покупке | Моментально | Telegram |
| Уведомление о продаже | Моментально | Telegram |
| Health alerts | Каждый час (при наличии) | Telegram |
| Cluster PnL отчёт | В daily report | Telegram |

---

## 5. Стратегия выхода (когда продаём)

Это **самая важная часть** системы. Мы не держим до последнего.

### Лесенка продаж (TP Ladder)

Когда мы покупаем контракт за 10 центов, мы заранее планируем продажу:

```
Цена растёт...
    │
    ├── entry × 2 ($0.20) ── Продать 50% позиции
    │                         (получили гарантированную прибыль)
    │
    └── entry × 3 ($0.30) ── Продать остаток позиции
                              (зафиксировали основную прибыль)
```

- Автообновление лесенки: если нет pending ордеров и цена < $0.70
- Пропуск ступеней: если стоимость ступени < $5
- Отмена TP ордеров перед market sell (предотвращает race condition)
- Tracking allocated shares между ступенями (последняя = остаток)

### Конвергентный Take-Profit

Если `current_price / metaculus_prob ≥ 0.60` → продажа (лесенка не активна).

### Защитные механизмы

| Триггер | Действие | Зачем |
|---------|----------|-------|
| Цена упала до entry × 0.50 | Продать всё | Hard stop-loss −50% |
| Цена упала до entry − 2.5 × ATR(7д) | Продать всё | ATR-стоп-лосс |
| Цена выросла >30%, потом откатилась | Трейлинг-стоп | max(ATR_trail, high × 0.75) |
| Трейлинг сработал → ждём 5 мин → подтверждаем | Продать всё | Двойное подтверждение |
| Возраст > 60% TTL, цена < entry × 1.5 | Продать всё | Time-decay exit |
| Новости опровергают ставку | Продать немедленно | Экстренный выход |
| Портфельная просадка > 10% | Прекратить новые покупки | Drawdown stop |
| entry_price = 0 | Использовать current_price | Не отключает стопы |

### Fallback для неликвидных позиций (новое)

| Ситуация | Действие |
|----------|----------|
| Market sell не удался | Aggressive limit order по best_bid |
| Slippage guard блокирует 5 раз подряд | Force market sell (игнорировать spread) |
| Failed sell | Recheck через 15 минут (не 3 часа) |
| Position missing из portfolio API | _miss_count ≥ 3 перед удалением |

---

## 6. Система уровней (Tiers)

Размер ставок зависит от баланса. Чем больше денег — тем агрессивнее (но контролируемо):

| Уровень | Баланс | Доля Келли | Базовый % | Other % | Макс. % | Макс. позиций | Макс. кластер | Макс. цена |
|---------|--------|------------|-----------|---------|---------|---------------|---------------|------------|
| Микро | до $2,000 | 40% | 5% | 5% | 10% | 15 | 35% | $0.40 |
| Рост | $2,000–$10,000 | 30% | 3% | 4.5% | 12% | 20 | 35% | $0.40 |
| Установившийся | $10,000–$50,000 | 35% | 3.5% | 5% | 15% | 25 | 40% | $0.50 |
| Масштаб | $50,000+ | 40% | 4% | 6% | 15% | 30 | 45% | $0.50 |

**Зачем:** На маленьком балансе мы осторожны (40% от формулы Келли, но 5% базовый), чтобы не потерять всё. С ростом капитала увеличиваем ставки.

**Minimum trade:** $20 для DOTM с хорошим edge (ratio≥2x), иначе skip.

**Conviction adjustment:** позиция масштабируется по conviction сигнала: ≥1.5x threshold → 100%, ≥1.2x → 60%, rest → 30%.

**Bayesian Kelly** — `kelly_full = (b × p − q) / b` интегрируется по Beta(p_mean, p_std) распределению вместо точечной оценки. uncertainty_penalty = 1/(1 + risk_aversion × std/mean). Мы используем дробную долю (30-40%), умноженную на uncertainty × conviction, с потолком effective_cap и абсолютным максимумом max_pct.

---

## 7. Дополнительные сигналы

### Социальный Buzz (до +20 баллов)

Модуль `social_buzz.py`. Кэш 1 час (BUZZ_CACHE_TTL = 3600).

| Источник | Вес | Статус |
|----------|-----|--------|
| GDELT DOC API v2 | 40% | 10мин кулдаун при rate limit (reset на успех) |
| Google News RSS | 30% | Работает |
| Reddit JSON API | 10% | Работает |

### Order Book Depth (до +15 баллов)

Модуль `orderbook_analyzer.py` — CLOB API, bid/ask imbalance. Кэш 5 мин.
- Imbalance > 0.4 → +15
- Bid wall ($5k+) → +12

### Smart Money Tracking (до +20 баллов)

Модуль `smart_money.py` — Polygonscan CTF Exchange ERC1155 events. Кэш 10 мин.
- Требует POLYGONSCAN_API_KEY (без ключа — модуль отключён)
- Wallet discovery + activity tracking

### Cascade Detector (до +10 баллов)

Модуль `cascade_detector.py` — price movement tracking.
- ≥2 рынка одного кластера двинулись ≥15% за 60 мин = cascade
- Laggard markets (не двинулись) в каскаде → +10 score
- 2-часовой decay

### Cross-Market Graph

Модуль `market_graph.py` — networkx граф корреляций.
- Shared clusters (0.5), entity overlap (0.3), entity relationships (0.4)
- Корреляция > 0.4 → размер позиции × 0.5
- Information cascade detection (BFS, 0.7^depth decay)
- Louvain community detection
- Portfolio diversification score

---

## 8. Калибровка и экстремизация

Пайплайн `calibrate_prediction()` в `signal_scorer.py`:

```
1. Platt scaling
   ├── Иерархический per-cluster (если ≥30 сэмплов)
   │
2. Isotonic regression
   ├── Если ≥50 сэмплов (иначе skip)
   │
3. Calibration multiplier
   ├── multiplier = 1.05 (консервативный)
   ├── cap = 0.35 (не раздувать оптимизм)
   │
4. Extremizing (компенсация регрессии к среднему)
   ├── price < $0.03 → d = 1.8
   ├── price < $0.07 → d = 1.6
   ├── price < $0.15 → d = 1.4
   ├── else          → d = 1.2
   │
   Формула: p_ext = p^d / (p^d + (1-p)^d)
```

**Ограничения:** MAX_P_MODEL_RATIO = 2.0 (p_model не может быть > 2× цену).

---

## 9. Категории рынков и наши предпочтения

Корректировки кластеров задаются в settings (cluster_score_adjustments):

| Кластер | Корректировка | Комментарий |
|---------|---------------|-------------|
| other | +15 | Бонус за нестандартность |
| sports_nba | −15 | Слишком эффективно оценивается |

### Коррелированные группы (MAX_CORRELATED_GROUP_PCT = 25%)

| Группа | Кластеры |
|--------|----------|
| trump_admin_politics | usa_politics, russia_ukraine, geopolitics, venezuela |
| us_economic | fed_fomc, usa_politics |
| sports | sports_nba, sports_ufc |
| tech_ai | ai_tech, tech |

---

## 10. Управление рисками

### Многоуровневая защита

| Уровень | Правило | Значение |
|---------|---------|----------|
| Market | Price-aware slippage: DOTM (<$0.10) tolerate ask до 10× price | Блокирует 44× (BoJ) |
| Signal | Liquidity pre-check: ask > 10× price → SKIP до LLM | Экономит токены |
| Signal | DOTM_PRICE_FLOOR: не покупаем дешевле $0.05 | Измеримая вероятность |
| Signal | MIN_P_MODEL: p_model ≥ 10% | Только значимый edge |
| Signal | ratio_score: min(ratio/5, 1) × 25 | Harder to max out |
| Position | Hard stop-loss per position | −50% |
| Position | Time-decay exit: age > 60% TTL + price < entry×1.5 | Авто |
| Position | Stale cleanup: 3 consecutive misses before delete | Защита от API глюков |
| Portfolio | Drawdown stop | −10% |
| Portfolio | LLM circuit breaker | 60 calls/hour |
| Portfolio | Graph correlation: r > 0.4 → размер × 0.5 | Диверсификация |
| Advisor | Override: advisor_p ≥ 0.5 × p_model | Не > price |
| Advisor | High-conviction skip: ≥1.5× threshold → auto-approve | Без LLM |
| Sizing | Minimum trade $20 (DOTM с ratio≥2x) | Не $5 |
| Sizing | Bid liquidity cap: max bid_liquidity × 0.20 | Не более 20% книги |
| Sizing | Bayesian Kelly uncertainty penalty | Меньше при низкой confidence |
| Sell | Cancel TP before market sell | Нет race condition |
| Sell | Market sell fail → aggressive limit at bid | Fallback для неликвида |
| Sell | 5 safety failures → force market | Нет бесконечного цикла |

### Байесовский обновитель

7 категорий новостей с likelihood ratios (LR cap ±1.5 за шаг):

| Категория | p_yes | LR |
|-----------|-------|-----|
| confirms_impossible | 0.02 | −3.9 (capped −1.5) |
| strongly_contradicts | 0.10 | −2.2 (capped −1.5) |
| moderately_contradicts | 0.40 | −0.41 |
| neutral | 0.50 | 0.0 (no change) |
| moderately_supports | 0.65 | +0.62 |
| strongly_supports | 0.85 | +1.7 (capped +1.5) |
| confirms_inevitable | 0.95 | +2.9 (capped +1.5) |

Log-odds posterior по каждой позиции, инициализируется при покупке, обновляется при новостях.

### Техническая защита

- **SQLite WAL mode** — ACID, кросс-процессная безопасность, BEGIN IMMEDIATE на writes
- **Atomic file writes** — tempfile + os.replace + fcntl.flock + fsync
- **PID управление** — lock files, нет race conditions
- **Graceful shutdown** — SIGTERM/SIGINT, 180×10s loop, WAL checkpoint
- **Crash safety** — pending_fill → active после fill; cleanup pending_fill при неудаче
- **Oversell protection** — _execute_sell() перечитывает фактические shares; TP ladder tracks allocated
- **Thread safety** — threading.RLock() во всех модулях
- **NaN/Inf guards** — order book prices валидируются
- **No production asserts** — все assert заменены на explicit guards

---

## 11. Архитектура

### Модули (48 файлов, ~14,400 строк)

| Модуль | Строк | Ответственность |
|--------|-------|-----------------|
| dotm_sniper.py | 624 | Оркестрация, main loop, graceful shutdown, WAL checkpoint |
| signal_pipeline.py | 670 | Рынки, scoring, batch анализ (re-exports) |
| signal_scorer.py | 471 | `_compute_signal_score()`, calibration, full analysis, cascade |
| market_fetcher.py | 259 | fetch_markets, Gamma API, pre-filter |
| metaculus.py | 374 | Metaculus API (30s timeout, 3 retry), gap detection, cache |
| trade_executor.py | 210 | execute_trade, pending_fill cleanup, Bayesian init |
| sell_executor.py | 670 | Продажи, стопы, ATR, trailing, time-decay, aggressive limit fallback |
| position_manager.py | 352 | Bayesian Kelly, tier, кластерные лимиты, conviction sizing |
| order_manager.py | 278 | pm-trader, ордера, TP ladder (allocated tracking), get_balance |
| hermes_advisor.py | 414 | Hermes main loop, reconciliation, PID |
| hermes_risk.py | 515 | Emergency exit, stop-loss, convergence, news |
| hermes_resolution.py | 85 | Market resolution, resolved checks |
| health_monitor.py | 365 | Health orchestrator, alerts, Telegram |
| health_checks.py | 793 | 25 individual checks (system/trading/data) |
| dotm_backtester.py | 334 | Backtest entry points, CLI |
| backtest_simulator.py | 724 | Core simulation engine |
| backtest_stats.py | 452 | Statistics, Brier, win rate, PnL |
| db.py | 488 | SQLite WAL, BEGIN IMMEDIATE, migrations, WAL checkpoint |
| config.py | 73 | 35+ path constants, trading params |
| schema.py | 139 | 70+ JSON key constants |
| bayesian_updater.py | 229 | init_posterior, update_posterior (LR cap ±1.5), should_exit |
| cascade_detector.py | 282 | Price tracking, cascade detection, laggard opportunities |
| market_graph.py | 295 | networkx graph, correlation, cascade, diversification |
| orderbook_analyzer.py | 104 | CLOB API, bid/ask imbalance, bid wall detection |
| smart_money.py | 186 | Polygonscan wallet tracking (API key guard) |
| online_learner.py | 256 | SGD online learning, drift detection |
| ml_predictor.py | 256 | LightGBM predictor, walk-forward validation |
| news_scanner.py | 243 | Tavily + DuckDuckGo (duckduckgo-search library) |
| social_buzz.py | 328 | GDELT + Google News + Reddit sentiment |
| cluster_report.py | 77 | Per-cluster PnL stats, format_cluster_report |
| dashboard.py | 131 | HTML dashboard generation |
| metrics_server.py | 139 | HTTP :8765, Prometheus, health, dashboard |

### Данные

| Хранилище | Формат | Использование |
|-----------|--------|---------------|
| `sniper.db` | SQLite WAL | Позиции, гипотезы, settings, kv_store, trade_history, миграции |
| `equity_curve.json` | JSON | История equity snapshots |
| `price_tracking.json` | JSON | Кэш p_model по рынкам |
| `calibration_model.json` | JSON | Per-cluster calibration |
| `bayesian_state.json` | JSON | Log-odds posterior per position |
| `market_graph.json` | JSON | Граф корреляций рынков |
| `ml_models/` | Binary | LightGBM + SGD модели, cascade state |
| `logs/` | Текст | RotatingFileHandler 10MB × 3, logrotate daily |

### Инфраструктура

| Компонент | Описание |
|-----------|----------|
| systemd | 3 сервиса: sniper, hermes, metrics — автозапуск при ребуте |
| Cron | watchdog */5, reports, advisor */30, equity */30, health hourly, backup 3am |
| logrotate | daily rotation, 7-day retention, 50MB max |
| SQLite backups | Daily at 3am, 7-day retention, chmod 600 |
| CI/CD | GitHub Actions: ruff + mypy --strict + pytest |
| Telegram | Alert queue с file_lock, 429 Retry-After, token sanitization |

---

## 12. Качество кода

### Тестирование

**1035 тестов** в 34 файлах, все проходящие:

| Категория | Тестов | Покрытие |
|-----------|--------|----------|
| db (core, migration, contract) | ~80 | SQLite, WAL, migrations, key consistency |
| position_manager | ~49 | Bayesian Kelly sizing, tiers, cluster limits, conviction |
| sell_executor | ~35 | Trailing stop, hard stop, time-decay, convergence, stale cleanup |
| signal_pipeline | ~130 | Scoring, calibration, full analysis, batch, constants |
| bayesian_updater | ~29 | Log-odds, posterior, LR cap, exit thresholds |
| health_monitor | ~33 | 25 checks, error spike, false positives |
| cascade_detector | ~15 | Price tracking, cascade detection, laggard signals |
| market_graph | ~34 | Graph building, correlation, cascade, diversification |
| smart_money | ~20 | Wallet discovery, activity tracking, API key guard |
| e2e integration | ~15 | Full lifecycle с real SQLite |
| config | ~19 | Constants, paths, sanitize() |
| backtester | ~217 | Simulation, stats, P&L |
| Прочие | ~359 | utils, order_manager, equity_tracker, tg_sender, etc. |

### Статический анализ

- **ruff**: 0 violations (правила E/F/W/B/SIM/UP)
- **mypy --strict**: 0 errors в 46 source files
- **Type annotations**: 96% coverage (337/351 функций)

### Аудиты

| Проход | Багов | Ключевые находки |
|--------|-------|------------------|
| Аудит 1 (initial) | 76 | estimated_size NameError, look-ahead bias, TP ladder double-counting |
| Аудит 2 (deep) | 40+ | Сломанные стоп-лоссы, selling_in_progress зависает, inter-process race, двойной hermes |
| Стабилизация (7 коммитов) | ~160 | 274→867 тестов, schema.py, RLock везде |
| Log audit (15h) | 3 | signal.signal в non-main thread, duplicate log handlers, calibration false positive |
| Full audit (48 модулей) | 27 | 6 CRITICAL (money loss), 4 HIGH (security), 12 MEDIUM |
| Post-audit fixes | 13 | P0-P2: ghost pending_fill, TP oversell, SQLite race, entry_price=0, Bayesian LR, stale cleanup |
| Log audit (48h) | 1 | Stuck position exit — aggressive limit fallback + safety_failures counter |

---

## 13. Текущие результаты

### Бэктест (проверка на исторических данных)

| Метрика | Значение |
|---------|----------|
| Период | Июнь 2024 — Май 2026 (24 месяца) |
| Количество рынков | 1,903 DOTM контракта |
| Количество сделок (walk-forward) | 246 |
| Доля прибыльных сделок | 28.9% |
| Совокупная доходность | +403% |
| Комиссии | 2% за сделку |

### Живые результаты (июнь 2026)

| Метрика | Значение |
|---------|----------|
| Баланс | ~$1,430 equity, $1,280 cash |
| Открытые позиции | 10-11 |
| Разрешённые рынки | 19 resolved, 0% winrate |
| Нереализованный P&L | −$10 to −$15 unrealized |
| Сигналов за всё время | 800+ |
| Заблокировано advisor | 109+ |
| Выполнено сделок | Несколько (бот консервативен) |
| Коммитов | 15+ в production chain |

### Уроки первых 6 недель

**0% winrate (0/19 resolved)** — главная проблема. Причины:
1. Покупали sub-5-cent контракты с p_model 2-5%, которые почти никогда не разрешаются YES
2. p_model систематически завышался: prompt bias + soft extremize + пустая калибровка
3. ratio_score всегда maxed out для DOTM (min(ratio/3, 1) × 30)

**Исправления (коммит `1c379e6`):**
- MIN_P_MODEL: 3% → 10%
- MAX_P_MODEL_RATIO: 3.0 → 2.0
- DOTM_PRICE_FLOOR: $0.05
- ratio_score: min(ratio/3, 1) × 30 → min(ratio/5, 1) × 25

---

## 14. План роста

### От $1,500 к $5,000/месяц дохода

Основано на консервативной оценке 7% в месяц (из бэктеста):

```
Сейчас:      $1,500 + $500/мес → micro tier
Месяц 6:     $1,500 + $500/мес → ~$5,800
Месяц 12:    $1,500 + $500/мес → ~$12,100 ← переход на «Рост»
Месяц 18:    $1,500 + $500/мес → ~$21,200 ← переход на «Установившийся»
Месяц 24:    $1,500 + $500/мес → ~$35,400
Месяц 33:    $1,500 + $500/мес → ~$78,600

$78,600 × 7% = $5,500/месяц дохода
```

### Что нужно для успеха

1. **Дисциплина** — не менять параметры вручную, доверять системе
2. **Пополнения** — $500/месяц значительно ускоряют рост
3. **Время** — минимум 12-18 месяцев для значимого результата
4. **50+ resolved сделок** — включить isotonic calibration
5. **50+ resolved** — LightGBM начнёт обучаться автоматически
6. **Мониторинг** — следить за health checks и dashboard
7. **POLYGONSCAN_API_KEY** — включить smart money tracking

---

## 15. Риски

| Риск | Вероятность | Последствия | Защита |
|------|-------------|-------------|--------|
| Серия убытков | Высокая (70% сделок в минусе) | Временное снижение баланса | Дробная ставка, стоп-лосс, drawdown stop |
| Ошибка ИИ в оценке | Средняя | Убыток на отдельной сделке | Двойная проверка, advisor veto |
| Polymarket закроется | Низкая | Потеря всех позиций | Диверсификация вне системы |
| Баг в коде | Низкая | Неправильная покупка/продажа | 1035 тестов, mypy --strict, ruff, 25 health checks |
| Нет ликвидности | Высокая | Бот не делает сделки | Liquidity pre-check, slippage guard, aggressive limit fallback |
| Слишком оптимистичный бэктест | Средняя | Реальные результаты ниже | Walk-forward, консервативные параметры, calibration cap |
| Stuck position (неликвид) | Средняя | Position не продаётся | Force market после 5 неудач, aggressive limit at bid |

---

## 16. Краткое резюме

**Стратегия:** Покупать дешёвые контракты ($0.05–$0.40) на маловероятные события, которые рынок недооценивает. Продавать по лестнице (entry×2, entry×3) при росте цены.

**Преимущество:** Для прибыли не нужно, чтобы событие произошло — достаточно, чтобы рынок пересмотрел оценку вероятности.

**Ожидание:** При 28.9% прибыльных сделок и среднем выигрыше в 5-10 раз больше среднего убытка, система математически прибыльна на дистанции.

**Цель:** Вырастить портфель с $1,500 до $70,000+ за 2.5-3 года с пополнениями $500/мес, что даст $5,000/мес пассивного дохода.

**Техническая зрелость:** 48 модулей, 1035 тестов, mypy --strict 0 errors, 25 health checks, systemd, Prometheus, CI/CD, Bayesian Kelly, ML blending, graph correlation, cascade detection.
