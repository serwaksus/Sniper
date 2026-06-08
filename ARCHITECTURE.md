# DOTM Sniper — Architecture

## Overview
DOTM Sniper is an automated trading bot for Polymarket DOTM (Deep Out of The Money) markets. It combines LLM analysis, Metaculus forecasts, and news sentiment to identify mispriced contracts.

## System Components

### Core Trading Loop (sniper)
- **Entry point**: `src/dotm_sniper.py` (orchestrator)
- **Signal generation**: `src/signal_pipeline.py` — scores markets using composite signals
- **Trade execution**: `src/trade_executor.py` — places limit buy orders
- **Position sizing**: `src/position_manager.py` — Kelly criterion with fee adjustment

### Risk Management (hermes)
- **Entry point**: `src/hermes_advisor.py` — 3 background threads
  - Position reconciliation (share count correction)
  - Emergency exit evaluation (probability drops, divergence)
  - Hypothesis resolution + skill generation
- **Sell execution**: `src/sell_executor.py` — stop-loss, trailing stop, take-profit ladder
- **Bayesian updates**: `src/bayesian_updater.py` — posterior probability updates
- **Self-improvement**: `src/hermes_memory.py` — tracks predictions, generates skills

### Data Layer
- **SQLite**: `src/db.py` — WAL mode, all persistent state
  - `positions` table — active positions
  - `hypotheses` table — trade hypotheses (for calibration)
  - `kv_store` table — settings, state
- **Facades**: `src/positions_db.py`, `src/hypotheses_db.py`

### External APIs
- **Polymarket**: `src/order_manager.py` — via `pm-trader` CLI subprocess
- **DeepSeek LLM**: `src/signal_pipeline.py`, `src/hermes_advisor.py` — market analysis
- **Metaculus**: `src/signal_pipeline.py` — crowd forecasts
- **GDELT**: `src/social_buzz.py` — news sentiment
- **DuckDuckGo**: `src/news_scanner.py` — news articles
- **Telegram**: `src/tg_sender.py` — notifications

### Monitoring
- **Health checks**: `src/health_monitor.py` — 25 checks (hourly cron)
- **Metrics API**: `src/metrics_server.py` — HTTP :8765/metrics
- **Equity tracking**: `src/equity_tracker.py` — equity curve (cron */30)
- **Watchdog**: `watchdog.sh` — process restart (cron */5)

## Data Flow
```
Polymarket API → fetch_markets() → signal_pipeline.analyze_market()
                                              ↓
                                    composite_score calculated
                                              ↓
                              score > threshold → execute_trade()
                                              ↓
                              SQLite: position + hypothesis created
                                              ↓
                              hermes monitors → sell_executor manages exits
                                              ↓
                              market resolves → hypothesis resolved → calibration
```

## Process Architecture
- **sniper** (screen:sniper) — main trading loop, 30-min cycles
- **hermes** (screen:hermes) — risk advisor, continuous monitoring
- **metrics** (screen:metrics) — HTTP metrics server
- **cron** — watchdog, health checks, equity tracking, reports

## Key Files
| File | Purpose |
|------|---------|
| `bot_settings.json` | Bot configuration (now in SQLite kv_store) |
| `sniper.db` | SQLite database (all persistent state) |
| `logs/` | All log files |
| `backups/` | Daily SQLite backups |

## Configuration
Settings are stored in SQLite `kv_store` table under key `bot_settings`. Key parameters:
- `signal_threshold` (50) — minimum composite score to trade
- `min_p_model` (0.03) — minimum model probability
- `min_confidence` (0.6) — minimum LLM confidence
- `max_concurrent_trades` (15) — position limit
