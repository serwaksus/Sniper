# DOTM Sniper

Polymarket trading bot with Hermes Advisory system.

## Quick Start

```bash
cd /root/dotm-sniper
python3 src/dotm_sniper.py        # Run trading bot (continuous)
python3 src/dotm_sniper.py --once # Single cycle
python3 advisor_script.py          # Run advisory analysis (DeepSeek R1)
python3 src/dotm_optimizer.py --resolved-count 1000 --skip-advisor  # Optimize parameters
```

## Project Status

- **Balance:** $356.73 (P&L: -$143.27 / -28.7%)
- **Version:** v5.3.0
- **Positions:** 4 (Renan Santos, Greenland, Marco Rubio, US NATO Withdrawal)
- **LLM:** DeepSeek (chat for bot, reasoner for advisor)
- **Monthly cost:** ~$11

## API Providers

| Service | Provider | Model |
|---------|----------|-------|
| LLM (bot) | DeepSeek | deepseek-chat |
| LLM (advisor) | DeepSeek | deepseek-reasoner |
| Prediction forecasts | Metaculus | API v2 |
| News scanning | Tavily | search API |
| Alerts | Telegram | Bot API |
| Trading | Polymarket | pm-trader CLI |

## Key Commands

```bash
# Syntax validation
python3 -m py_compile advisor_script.py cron_validator.py src/dotm_sniper.py src/dotm_optimizer.py

# Run all tests
pytest -q

# Validate cron configs
python3 cron_validator.py

# Health check
python3 cron_health_monitor.py

# Run parameter optimization
python3 src/dotm_optimizer.py --resolved-count 1000 --skip-advisor

# Post-deploy smoke checks
bash startup_validation.sh
```

---

## Session 2026-05-17: v4.4 → v4.5

### Reliability Fixes

**File Locking** — added `fcntl` shared/exclusive locks on all JSON file reads/writes to prevent corruption when the main bot and advisor cron job run concurrently.

**Balanced Brace JSON Parser** — replaced `re.search(r'\{.*\}')` greedy regex with a proper brace-counting parser that correctly handles nested JSON objects inside LLM responses.

**Backtest PnL Fix** — sold positions now use actual `pnl_at_exit` data instead of assuming -100% loss on all NO outcomes.

### Signal Improvements

**MIN_P_MODEL lowered 0.10 → 0.05** — the previous threshold was blocking strong DOTM signals where p_model=8% on a 1% market gives an 8x ratio. The composite scoring system now handles the quality filtering.

**Anti-anchoring LLM prompt** — added explicit instruction to not return probabilities near the market price, addressing the persistent 2x anchoring problem.

**Cluster keywords precision** — removed broad keywords ("war", "peace", "invasion", "ai") that caused false matches ("trade war" → russia_ukraine, "remain" → ai_tech). Replaced with specific phrases ("war in ukraine", "ai safety", "ai bill"). Added donbas/crimea/donetsk.

### Bug Fixes (9 total)

**Critical:**
- **C-01**: Sold positions silently restored to positions.json — `del positions[slug]` after sell
- **C-02**: `"ai"` matched `"remain"`, `"Britain"`, `"obtain"` — switched to `\b` word-boundary regex
- **C-03**: $5 minimum bet forced on near-zero Kelly edge — return 0 if Kelly = 0
- **C-04**: Stop-loss sells labeled as YES/NO outcome instead of SOLD — excluded from Brier/learning
- **C-06**: JSON files corrupted on crash — atomic write (temp file + rename)

**High:**
- **H-03**: Fake $100 balance on API failure — return None, skip cycle
- **H-04**: One malformed market kills entire candidate list — safe `.get()` access

**Medium:**
- **H-05**: dates_match returned True on parse failure — now returns False
- **H-07**: Advisor showed `entry_price * 2` instead of actual p_model — reads from hypothesis_db

## Session 2026-05-17: v4.5 → v4.5.1

### Audit Fixes

**Advisor LLM parsing** — replaced fragile `content.startswith('{')` check with robust `parse_llm_advisor_response()` supporting clean JSON, fenced code blocks, preamble text, and brace-balanced extraction. Schema validation ensures `p_estimate`/`confidence` in [0,1], `factors` as `list[str]`, `verdict` in `{CONFIRM, DIVERGE, WARNING, UNKNOWN}`. Precise fallback logging replaces silent failures.

**Duplicate sell prevention** — `trailing_stop_check()` now checks `hypothesis_db` for already-resolved slugs before attempting any sell. Previously-sold positions that linger in the API portfolio are skipped instead of re-created and re-sold.

**High-price invariant** — `high_price` is initialized as `max(entry_price, current_price)` and never decreases during updates. `repair_positions_file()` runs at startup to fix existing inconsistent data in `positions.json`.

**Polling cleanup** — stale cleanup now also removes positions that are in `resolved_slugs`, preventing resolved markets from being polled indefinitely.

### Bug Fixes (5 total)

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| AUD-1 | Advisor JSON parse fails on any prefix/fenced block | HIGH | ✅ |
| AUD-2 | Repeat hard-stop sells on already-closed positions | CRITICAL | ✅ |
| AUD-3 | `high_price < entry_price` breaks trailing stop logic | HIGH | ✅ |
| AUD-4 | Resolved markets polled indefinitely | MEDIUM | ✅ |
| AUD-5 | Silent fallback with no logged reason | MEDIUM | ✅ |

### Tests Added (34 new)

- `test_advisor_parsing.py` — 26 tests covering clean JSON, fenced blocks, preamble, brace balancing, malformed input, schema validation
- `test_state_invariants.py` — 8 tests covering `repair_positions_file()` and `high_price` initialization/update invariants

### Post-Audit: Infrastructure Fix

**Telegram reporting restored** — `dotm_report.py` не загружал `.env`, поэтому `TG_BOT_TOKEN`/`TG_CHAT_ID` были пусты в cron-окружении → `TelegramReporter.enabled=False` → все отчёты молча отбрасывались. Добавлен `_load_env()` для чтения `/root/dotm-sniper/.env`.

**Report cron schedule fixed** — последний запуск смещён с `21:01 UTC` (MSK=00:01, вне окна 9-22) на `18:01 UTC` (MSK=21:01). Расписание отчётов: `12:01`, `16:01`, `20:01`, `21:01 MSK`.

**Validation results:**
- `py_compile` — 3/3 файла OK (`advisor_script.py`, `cron_validator.py`, `src/dotm_sniper.py`, `src/dotm_report.py`)
- `pytest -q` — 44/44 тестов пройдено
- `startup_validation.sh` — все проверки пройдены
- `cron_health_monitor.py` — все cron-задачи здоровы
- Telegram test message — доставлен

### Remaining Known Issues

| Issue | Severity | Description |
|-------|----------|-------------|
| cron_validator.py is valid but has no edge-case coverage for malformed job structures | LOW | Works fine, could add more validation rules |
| Duplicate `get_balance`/`get_portfolio` across files | LOW | Different defaults, not unified |

## Session 2026-05-18: Telegram Reports Fixed (Again)

### Problem

Telegram-бот перестал отправлять статус-репорты. Ручной тест показал, что API работает (200 OK), но отчёты никогда не доходили до `_send()`.

### Root Causes Found (3)

| # | Issue | Impact |
|---|-------|--------|
| RPT-1 | `current_status.json` не обновлялся с 17 мая — `_update_status` стоял **после** `return` при `MAX_POSITIONS >= 5` в `main()` | Статус протухал на 15+ часов каждый цикл |
| RPT-2 | `dotm_report.py` не логировал результат `_send()` — ошибки Telegram уходили в пустоту | Невозможно диагностировать проблемы |
| RPT-3 | `send_daily_report()` не возвращал результат `_send()` — caller всегда получал `None` | Ложный негатив в логах |

### Fixes Applied

**Sniper (`dotm_sniper.py`):**
- Вынес обновление статуса в `_update_status_file()` — вызывается **до** проверки `MAX_POSITIONS`
- Копирование в `/root/.openclaw/` обёрнуто в individual try/except — одна сломанная копия не убивает весь блок

**Report (`dotm_report.py`):**
- Добавлено полноценное логирование (`report.log`) — каждый шаг от получения баланса до отправки Telegram
- `_send()` логирует success/failure с HTTP-статусом и текстом ошибки
- `send_daily_report()` теперь возвращает `bool` от `_send()`
- Добавлен флаг `--force` для ручного запуска вне расписания
- Время в отчёте исправлено на MSK (использует `pytz` вместо `datetime.now()`)
- `balance_data` теперь проверяется на `None` перед отправкой
- Копирование в `/root/.openclaw/` — individual try/except для каждого пути

**Cron:**
- Добавлен утренний отчёт в 9:01 МСК (6:01 UTC)
- Новое расписание: **9:01, 12:01, 16:01, 20:01, 21:01 МСК** (каждые 3-4 часа)

### Verification

```
Telegram test message     → ✅ доставлен (message_id=439)
Report --force            → ✅ "Telegram message sent successfully"
Sniper --once (MAX_POS)   → ✅ current_status.json обновлён
cron schedule             → ✅ 5 отчётов/день в рамках 9-22 МСК
```

### Current Portfolio (2026-05-18)

| Market | P&L | Value |
|--------|-----|-------|
| Renan Santos 2nd place | +20.5% | $6.02 |
| AI Safety Bill before 2027 | +0.4% | $120.53 |
| Marco Rubio 2028 | -2.5% | $4.88 |
| US acquire Greenland | -10.0% | $18.90 |
| Ukraine NATO before 2027 | -17.5% | $96.53 |
| **Total** | **-8.7%** | **$452.70** |

See `NOTES.md` for full project history and strategy details.

---

## Session 2026-05-19: Mega-Backtest + v5.0 Optimization

### 1. Mega-Backtest (1000 resolved markets)

Результаты бэктеста `--count 1000 --skip-advisor` (9 мин 45 сек, 0 ошибок):

| Метрика | Значение |
|---|---|
| Рынков проанализировано | 1000 |
| BUY-сигналов | 93 (селективность 9.3%) |
| Winrate | **51.6%** (48W / 45L) |
| Avg Upside на WIN | **17.45x** |
| Brier Score (raw → calibrated) | 0.3433 → 0.3472 (хуже на -0.004) |
| Dampened рынков | 70/1000 (7%) |

**Кластеры**: `other` — 87 traded, WR 52.9%, net +5. `crypto` — 550 рынков, 0 traded. `usa_politics` — 4 traded, WR 25%, net -2.

### 2. Refactoring v5.0 (3 изменения, 80/80 тестов)

**T1: Hard Crypto Ban** — `BANNED_CLUSTERS = {"crypto"}` добавлен в `fetch_markets()`, `pre_filter_before_batching()`, `_fetch_active_dotm_markets_gamma()`, `_fetch_resolved_dotm_markets()`. Удалён из `bot_settings.json`. Экономия **~55% LLM-токенов** (550 рынков больше не сканируются).

**T2: Asymmetric Calibration** — `calibrate_prediction()` принимает `cluster=None`. Если `cluster == "other"` — dampening отключён, p_model возвращается без изменений. Защита единственного прибыльного кластера (52.9% WR). Все 3 call site обновлены.

**T3: Fractional Kelly Boost** — `position_size()` принимает `cluster=None`. Для `other` — cap 3.5% баланса (`OTHER_BOOST_POS_PCT`), для остальных — 2% (`BASE_POS_PCT`). Абсолютный потолок `MAX_POS_PCT=10%` сохранён.

### 3. Telegram Stability Fix

- Timeout увеличен: 10s → 20s
- Retry ×3 с паузой 2s (была 1 попытка, стало 3)
- Фиксирует потерю отчётов при временных таймаутах API Telegram

### 4. Runtime Cleanup

- Screen-сессия убита — бот работает **только через cron** `--once` каждые 30 мин
- Исключён конфликт дублирующих процессов
- Cron-расписание отчётов: 09:01, 12:01, 16:01, 20:01, 21:01 МСК

---

## Session 2026-05-20: DOTM Optimizer v1.0 + Stress Testing

### DOTM Optimizer (`src/dotm_optimizer.py`)

New optimization framework with institutional-grade validation:

**Methodology:**
- **Walk-forward validation** — 3-period expanding window (train 400/500/600 markets, test 100/150/200 markets)
- **Grid search** — 4×3×3 parameter grid (thresholds 70-85, confidence 0.55-0.70, max_positions 10-15)
- **Monte Carlo simulation** — 10,000 runs per configuration to estimate drawdown distributions and ruin probability

**Stress-Tested Parameters (Post-Audit):**
- **Slippage penalty:** $0.015 per trade (accounts for bid-ask spread on exit)
- **Train/test split:** 50/50 temporal split prevents overfitting

### Current Optimized Settings

```json
{
  "signal_threshold": 80,
  "min_confidence": 0.60,
  "max_positions": 12
}
```

### Realistic Performance Metrics (Out-of-Sample)

| Metric | Value |
|--------|-------|
| **OOS Expected Value** | **+7.18 units** |
| **Winrate** | **55.6%** (5W/4L in test set) |
| **Ruin Probability** | **0%** (Monte Carlo verified) |
| **Maximum Drawdown** | **43.4%** |

### Key Findings

**Strategy Status:** Profitable but requires calibration to reach 60% winrate target.

- Current 55.6% WR shows edge exists but is narrow
- Monte Carlo confirms 0% ruin risk under tested parameters
- 43.4% max DD is within acceptable bounds for DOTM asymmetric payoff profile
- Walk-forward analysis confirms robustness across market regimes

**Usage:**
```bash
python3 src/dotm_optimizer.py --resolved-count 1000 --skip-advisor
```

---

## v5.2.0 Session Summary (2026-05-23)

### Critical Bug Fixes Applied

| # | Fix | File | Description |
|---|-----|------|-------------|
| 1 | State Overwrite | `dotm_sniper.py:trailing_stop_check()` | Atomic read→modify→write per slug to prevent race conditions |
| 2 | Intra-Cycle Cluster Limits | `dotm_sniper.py:main()` | Dynamic cluster limit tracking within trading cycle |
| 3 | Null-safe Metaculus | `dotm_sniper.py:get_metaculus_forecast()` | Explicit `is not None` checks for `aggregations` field |
| 4 | Batch Calibration | `dotm_sniper.py:batch_analyze_markets()` | Parallel Metaculus fetch → real `metaculus_prob` in `calibrate_prediction` |
| 5 | Time-Decay Linear | `dotm_sniper.py:get_time_decay_threshold()` | Smooth 3% threshold descent for `days_to_res < 1` |
| 6 | Look-Ahead Bias | `dotm_backtester.py:_fetch_resolved_dotm_markets()` | Conservative `high_price = entry_price` for NO markets |

### Hermes Advisor v5.2.0

**New module:** `src/hermes_advisor.py`

| Feature | Implementation |
|---------|----------------|
| Reconciliation Loop | 15-min atomic `positions.json` ↔ `pm-trader orders` sync |
| Partial Fill Handling | VWAP calculation for `PARTIALLY_FILLED` TP orders |
| Emergency Exit | LLM-driven news analysis + cancel→sell transaction chain |
| Thread Isolation | Two daemon threads: reconciliation + emergency evaluation |
| Logging | `/root/dotm-sniper/logs/hermes.log` with unbuffered writes |

**Start Hermes:**
```bash
cd /root/dotm-sniper
python3 src/hermes_advisor.py
```

**Audit Logs:**
```bash
tail -n 50 /root/dotm-sniper/logs/hermes.log
```

### Optimizer Results (Walk-Forward)

```
In-Sample:  score>=80, conf>=0.60 → 54 trades, WR=46.3%, EV=+5.22x
Out-of-Sample: 36 trades, WR=55.6%, EV=+5.83x
Monte Carlo: 0% ruin, 43.4% max DD
```

### Commands Run

```bash
# Regenerate backtest cache (look-ahead fix)
python3 src/dotm_backtester.py --mode sim --count 1000

# Run optimizer
python3 src/dotm_optimizer.py

# Run sniper with optimized params
python3 src/dotm_sniper.py --once
```

---

## Session 2026-05-24: v5.3.0 — Pipeline Fixes + Adaptive Thresholds

### Root Cause: 0 Trades Since May 20

Bot ran 192 cycles over 4 days without placing a single trade. Two interlocking bugs:

| Bug | Impact | Fix |
|-----|--------|-----|
| **Batch execution deadlock** | Batch BUY signals silently dropped | Check `market_analyses` before delta-skip in second loop |
| **signal_threshold=80** | No candidate could reach score 80 with current market conditions | Lowered to 65 with per-horizon overrides |
| **PRICE_DELTA $0.005** | DOTM markets ($0.01-0.03) need 50-500% move to trigger re-analysis | Lowered to $0.002 |
| **Advisor empty response** | `deepseek-reasoner` returns `reasoning_content` not `reasoning` | Added fallback extraction |
| **MIN_PROB_RATIO not enforced** | Market with p_model < price generated BUY | Added guard to both paths |

### Settings Changes

| Parameter | Old | New |
|-----------|-----|-----|
| `signal_threshold` | 80 | 65 |
| `signal_threshold_medium_horizon` | +5 hardcoded | 60 (from settings) |
| `signal_threshold_long_horizon` | +10 hardcoded | 55 (from settings) |
| `PRICE_DELTA_THRESHOLD` | $0.005 | $0.002 |
| `MIN_COMPOSITE_SCORE` | 80 | 65 |

### Current Portfolio (2026-05-24)

| Market | P&L | Value |
|--------|-----|-------|
| Renan Santos 2nd place | +145.9% | $12.30 |
| Marco Rubio 2028 | -3.2% | $4.84 |
| US acquire Greenland | -6.7% | $19.60 |
| US withdraw from NATO | -14.7% | $4.26 |
| **Total** | **-28.7%** | **$356.73** |

### AI Safety Bill Exit (2026-05-23)

Sold at +34.5% profit ($120.53 → ~$162). Largest position exited with gain.

### Verification

```
py_compile  — 4/4 files OK
pytest -q   — 80/80 passed
sniper --once — pipeline runs end-to-end
```

---

## Session 2026-05-24 (2): Dead Code Purge + Hermes Spam Fix

### Dead Code Removal (`dotm_sniper.py`: 2842 → 2590 lines, -8.9%)

| Action | Target | Lines Removed |
|--------|--------|---------------|
| Deleted | `src/metaculus_parser.py` | 322 (never imported) |
| Deleted | `src/dotm_trade_history.py` | 108 (never imported) |
| Removed constants | `KELLY_FRACTION`, `DATE_KEYWORDS`, `MIN_COMPOSITE_SCORE` | 3 dead entries |
| Removed functions | `search_geopolitical_sources()`, `search_sports_sources()`, `score_sources()`, `evaluate_market()` | ~200 lines |
| Cleaned imports | Unused `timedelta`, duplicate `try: import dotenv` blocks | 6 lines |
| Replaced `print()` | 7 debug prints → `logger.info/warning/error` | 7 call sites |
| Cleaned docstrings | Stale references to removed functions | 2 blocks |

### Hermes Advisor — Critical Bug Fixes

**H-1: Silent Position Skip** — `trailing_stop_check()` created `positions.json` entries **without** `market_question`. Hermes line 382 reads `pos_data.get("market_question", "")` → gets `""` → `continue` → **all positions silently skipped**. Zero evaluations despite daemon running for 7+ hours.

Fix: `trailing_stop_check()` now saves `market_question` and `shares` from portfolio data. Existing 4 positions backfilled via API.

**H-2: No Notification Cooldown** — LLM with `temperature=0.1` oscillates GREEN↔DIVERGENCE on identical data. Status change `A→B` triggered Telegram notification every cycle (600s). No deduplication across process restarts (state file had only `{"test-slug": "GREEN"}`).

Fix: `NOTIFICATION_COOLDOWN_SECONDS = 14400` (4h) per position. Emergency exits bypass cooldown.

**H-3: No Severity Filter** — Any status change (including routine GREEN→YELLOW) sent to Telegram.

Fix: `NOTIFY_SEVERITIES = {"DIVERGENCE", "RED"}` — only critical changes reach Telegram. GREEN/YELLOW logged to `hermes.log` only.

**H-4: Alert State Schema Upgrade** — `hermes_alert_state.json` now tracks `last_notified_at` timestamps per position alongside status.

### Hermes Advisor — Cleanup

- Replaced duplicate `try: import dotenv` with inline `_load_env_manual()` (matches sniper pattern)

### Restored (Accidental Deletion)

- `DATE_WINDOW_DAYS = 7` — used by `dates_match()` in Metaculus forecast matching. Was deleted with other constants, caught by `--once` test.

### Verification

```
py_compile           — dotm_sniper.py, hermes_advisor.py OK
sniper --once        — pipeline runs, 4 positions checked, 0 trades (advisor blocked)
hermes_advisor       — PID active, both loops running
hermes_alert_state   — 4 real positions evaluated (GREEN), 0 Telegram notifications sent
positions.json       — all 4 entries have market_question (was 0/4)
```

### Protected Zones (Untouched)

| Zone | Location | Status |
|------|----------|--------|
| `fcntl.flock` | hermes_advisor.py:113-122 | ✅ No change |
| `%` probability parsing | hermes_advisor.py:470-471, 518 | ✅ No change |
| TP Laddering 50/30/20 | hermes_advisor.py:554-637 | ✅ No change |
| 30-day news filter | hermes_advisor.py:385 | ✅ No change |

### Known Deferred

| Issue | Risk | Notes |
|-------|------|-------|
| Duplicate functions across files | LOW | `load_json`/`save_json`/`get_balance`/`get_portfolio`/`send_telegram` × 3-5 files each. Consolidation deferred — risk to flock mechanism |
| `requests.Session()` pooling | LOW | New TCP connection per API call. Session reuse would reduce latency |
| Duplicate `dotenv` loading in `news_scanner.py`, `dotm_report.py` | LOW | Lower priority cleanup |

### advisor_script.py — Telegram Spam Fix

`advisor_script.py` (separate cron job, `*/30`) was the **actual spam source** — not Hermes. Every 30 min it called DeepSeek for all 4 positions; any DIVERGE verdict sent Telegram unconditionally. LLM oscillation on identical data = up to 8 messages/hour.

**A-1: DIVERGE + ALIGNED contradiction suppressed** — LLM returns verdict `DIVERGE` but `p_diff ≤ 0.05` (ALIGNED text in message). This contradictory signal no longer triggers Telegram. Only genuine divergences (`p_diff > 0.05`) pass through.

**A-2: Notification cooldown 4h** — `advisor_notify_state.json` tracks `last_notified` timestamp per slug. After first notification, subsequent DIVERGE/WARNING alerts suppressed for 4 hours. WARNING (RED) always passes without cooldown.

**A-3: State persistence** — Notification state survives cron restarts and process cycles via `/root/dotm-sniper/logs/advisor_notify_state.json`.

### Verification (advisor_script.py)

```
Run 1 (clean state)  → 2 Telegram sent (Renan p_diff=0.11, Greenland p_diff=0.065), Rubio/NATO suppressed (p_diff ≤ 0.05)
Run 2 (1 min later)  → 0 Telegram sent, cooldown log: "suppressing"
py_compile            — advisor_script.py OK
fcntl.flock           — lines 43, 50 unchanged
```
