# DOTM Sniper - Project Notes

## Current Version: v5.3.2 Adaptive Signal Thresholds (2026-05-25)

---

## Architecture

```
dotm-sniper/
├── src/
│   ├── dotm_sniper.py      # Main trading bot
│   ├── dotm_report.py      # Portfolio reporting
│   ├── news_scanner.py     # News scanning
│   ├── metaculus_parser.py # Metaculus API integration
│   └── dotm_trade_history.py
├── advisor_script.py       # Hermes Advisory - portfolio analysis
├── cron_validator.py       # Cron config validation
├── cron_health_monitor.py  # Health monitoring + alerts
├── tests/
│   └── test_cron_validator.py  # Unit tests
├── docs/
│   └── CRON_CHECKLIST.md   # Cron job creation guide
└── startup_validation.sh   # Boot-time health check
```

---

## Current Portfolio (2026-05-16 22:00 UTC)

| Market | Position | P&L |
|--------|----------|-----|
| U.S. enacts AI safety bill before 2027? (YES) | $120.53 | +0.4% |
| Ukraine agrees not to join NATO before 2027? (YES) | $88.91 | -24.0% |
| Will the US acquire part of Greenland in 2026? (YES) | $6.77 | -3.3% |
| Will Renan Santos finish 2nd in Brazilian election? (YES) | $4.75 | -4.9% |

- **Total:** $452.64 | **Cash:** $231.69
- **P&L:** -$47.36 (-9.5% from $500 start)

---

## v4.5 Changes (2026-05-17)

### Reliability Fixes
| # | Change | Severity | Description |
|---|--------|----------|-------------|
| 1 | File locking on JSON reads/writes | HIGH | Added fcntl-based shared/exclusive locks to prevent race conditions between bot and advisor |
| 2 | Balanced brace JSON parser | MEDIUM | Replaced greedy regex `\{.*\}` with proper brace-counting parser that handles nested objects |
| 3 | Backtest PnL uses actual exit data | MEDIUM | Sold positions now use real `pnl_at_exit` instead of assuming -100% loss |

### Signal Improvements
| # | Change | Reason |
|---|--------|--------|
| 4 | MIN_P_MODEL lowered 0.10 → 0.05 | Was blocking DOTM markets with strong ratios (e.g., 8% estimate on 1% market = 8x ratio) |
| 5 | LLM prompt anti-anchoring | Added explicit warning against returning probability near market price |
| 6 | Cluster keywords: removed "war", "peace", "invasion" | Prevented "trade war" / "peace treaty" from matching russia_ukraine |
| 7 | Cluster keywords: added donbas, crimea, donetsk | Better russia_ukraine detection |
| 8 | Cluster keywords: "ai" → "ai safety", "ai bill" | Prevented "ai" matching "remain", "Britain" (regression from v4.4) |
| 9 | Multi-word keyword support | Compound phrases like "war in ukraine" now work with substring matching |

### Known Issues Fixed
| Issue | Status |
|-------|--------|
| Race condition on JSON files | ✅ fcntl locks |
| `"war"` matches "trade war" | ✅ replaced with specific phrases |
| parse_llm_json greedy regex | ✅ balanced brace parser |
| backtest_recent PnL assumes 100% loss | ✅ uses actual exit data |
| Duplicate get_balance/portfolio functions | ✅ consistent return types |
| Advisor JSON parse fails on prefix/fenced block | ✅ robust parse_llm_advisor_response() |
| Repeat hard-stop sells on closed positions | ✅ resolved-slugs check in trailing_stop_check |
| high_price < entry_price breaks trailing stop | ✅ max(entry_price, current_price) + repair |
| Resolved markets polled indefinitely | ✅ cleanup includes resolved_slugs |
| Silent JSON fallback with no logged reason | ✅ precise error logging |

---

## v4.5.1 Changes (2026-05-17)

### Audit Fixes
| # | Change | Severity | Description |
|---|--------|----------|-------------|
| 1 | Robust advisor LLM parsing | HIGH | New `parse_llm_advisor_response()` with fenced block, preamble, brace-balance, schema validation |
| 2 | Duplicate sell prevention | CRITICAL | `trailing_stop_check()` checks hypothesis_db for already-resolved slugs |
| 3 | High-price invariant enforcement | HIGH | `high_price = max(entry_price, current_price)` on init; `repair_positions_file()` at startup |
| 4 | Polling cleanup for resolved markets | MEDIUM | Stale cleanup includes `resolved_slugs` check |
| 5 | Precise fallback logging | MEDIUM | Advisor logs exact parse failure reason instead of silent fallback |

### New Tests
| File | Tests | Coverage |
|------|-------|----------|
| test_advisor_parsing.py | 26 | Clean JSON, fenced blocks, preamble, brace balance, schema validation, failure modes |
| test_state_invariants.py | 8 | repair_positions_file(), high_price init/update invariants |

---

## v4.4 Changes (2026-05-16)

### Composite Signal Scoring
Replaced fixed prob_ratio threshold (3.0x) with composite scoring (0-100):
- **Ratio component (40pts):** p_model/market_price / 3.0
- **Factor component (20pts):** number of supporting factors
- **Volume component (20pts):** market volume / $1M
- **Time component (20pts):** days to resolution
- **Threshold:** score >= 55 to trade (45 with Metaculus override)

### Other Changes
| # | Change | Reason |
|---|--------|--------|
| 1 | Added NATO, Greenland, S&P, recession, etc. to CLUSTER_KEYWORDS | Markets like "US withdraw from NATO" were classified as "other" and filtered out |
| 2 | MAX_PRICE 0.25 → 0.30 | Capture more opportunities in $0.25-0.30 range |
| 3 | MAX_EXPOSURE_PER_CATEGORY 0.12 → 0.20 | 12% was too tight, blocking all trades in exposed clusters |
| 4 | MIN_VOLUME 50000 → 25000 | Capture medium-liquidity markets |
| 5 | MIN_POSITION_CHECK_INTERVAL 6h → 3h | Faster response to bleeding positions |
| 6 | Metaculus probability override | When Metaculus gap is strong, use Metaculus p over LLM p |
| 7 | Improved LLM prompt | Less conservative, focused on mispricing detection |
| 8 | Timezone-aware datetime comparison fix | Prevented crash on offset-naive vs offset-aware comparison |
| 9 | Category exposure now checks question text (not just slug) | Better cluster detection for portfolio monitoring |

---

## Trading Strategy (v4.4)

### Thresholds
```python
MIN_PROB_RATIO = 3.0  # Used as ratio component in composite score
MIN_P_MODEL = 0.05    # Lowered from 0.10 to capture DOTM signals
MAX_P_MODEL_RATIO = 5.0
MIN_CONFIDENCE = 0.65
MAX_POS_PCT = 0.10
KELLY_FRACTION = 0.25
FRACTIONAL_KELLY_MULTIPLIER = 0.25
MAX_EXPOSURE_PER_CATEGORY = 0.20
MAX_PRICE = 0.30
MIN_VOLUME = 25000
ALLOWED_CLUSTERS = {"ai_tech", "russia_ukraine", "usa_politics", "fed_fomc"}
SIGNAL_THRESHOLD = 55  # Composite score out of 100
```

### Composite Signal Formula
```
score = (ratio/3 * 40) + (factors_score * 20) + (vol/$1M * 20) + time_bonus
```

### Key Fixes
1. **Metaculus Search** - Multiple search queries, substring matching, fallback to `prediction` field
2. **LLM Prompt** - Balanced: "assess whether the crowd is anchoring too low"
3. **Backtest Function** - Analyzes last 20 resolved markets for winrate/Brier score
4. **Composite Scoring** - Replaces fixed prob_ratio threshold with multi-factor signal

---

## Cron Jobs

| ID | Name | Type | Schedule | Status |
|----|------|------|----------|--------|
| advisor-001 | dotm-advisor | `shell` | every 30m | ✅ OK |
| b738b0ec-... | dotm-report | `agentTurn` | every 45m | ⚠️ Warning |

### Critical Lesson (2026-05-16)
**`shell` for scripts, `agentTurn` for messages to agents.**

`advisor-001` was broken 9 hours because it used `agentTurn` to run a python script. Fixed by changing to `shell`.

---

## Issues Fixed (v4.4)

| # | Problem | Severity | Status |
|---|---------|----------|--------|
| 18 | CLUSTER_KEYWORDS missing NATO, Greenland, S&P etc | HIGH | ✅ |
| 19 | LLM p_model always 2x market price (anchoring) | HIGH | ✅ composite scoring |
| 20 | MAX_EXPOSURE_PER_CATEGORY=12% too restrictive | HIGH | ✅ raised to 20% |
| 21 | MAX_PRICE=0.25 missing opportunities | MEDIUM | ✅ raised to 0.30 |
| 22 | Timezone comparison crash (offset-naive vs aware) | MEDIUM | ✅ |
| 23 | Metaculus override for strong gap signals | MEDIUM | ✅ |
| 24 | Category exposure only checked slug, not question | LOW | ✅ |

---

## Issues Fixed (v4.3)

| # | Problem | Severity | Status |
|---|---------|----------|--------|
| 10 | ALLOWED_CLUSTERS={ai_tech} only - missing opportunities | HIGH | ✅ |
| 11 | LLM p_model format mismatch (percentage vs decimal) | HIGH | ✅ |
| 12 | Duplicate print line in main() | LOW | ✅ |
| 13 | Version string mismatch (v4.1 vs v4.2) | LOW | ✅ |
| 14 | Resolved hypotheses in active list | MEDIUM | ✅ |
| 15 | API keys hardcoded in source files | MEDIUM | ✅ |
| 16 | daily_stats.json stale tracking | LOW | ✅ |
| 17 | Ukraine NATO position missing from hypothesis_db | MEDIUM | ✅ |

---

## Issues Fixed (v4.2)

| # | Problem | Severity | Status |
|---|---------|----------|--------|
| 1 | Metaculus search returned irrelevant results | HIGH | ✅ |
| 2 | Metaculus score threshold too high (0.45) | MEDIUM | ✅ |
| 3 | Metaculus probability extraction broken | HIGH | ✅ |
| 4 | LLM p_model diverged 20x from market | HIGH | ✅ |
| 5 | NLP verify slowed and blocked matches | MEDIUM | ✅ |
| 6 | MIN_PROB_RATIO=2.0 → 0% winrate | HIGH | ✅ |
| 7 | MIN_P_MODEL=0.05 too low | MEDIUM | ✅ |
| 8 | LLM prompt encouraged overestimation | HIGH | ✅ |
| 9 | No backtest function | MEDIUM | ✅ |

---

## Preventive Tools

### Validation
```bash
python3 /root/dotm-sniper/cron_validator.py          # Validate configs
python3 /root/dotm-sniper/tests/test_cron_validator.py  # Unit tests (10 tests)
```

### Monitoring
```bash
python3 /root/dotm-sniper/cron_health_monitor.py  # Alert on edge cases
/root/dotm-sniper/startup_validation.sh           # Boot-time check
```

### Cron Job Checklist
See `docs/CRON_CHECKLIST.md` - decision tree for `shell` vs `agentTurn`

---

## History

### 2026-05-17 - v4.5.1
- Audit fix: robust advisor LLM JSON parsing with schema validation
- Audit fix: duplicate sell prevention via resolved-slugs check
- Audit fix: high_price invariant (max of entry/current) + startup repair
- Audit fix: polling cleanup includes resolved markets
- Added 34 new tests (test_advisor_parsing.py, test_state_invariants.py)

### 2026-05-17 - v4.5
- Added file locking (fcntl) for JSON read/write safety
- Replaced greedy JSON regex with balanced brace parser
- Lowered MIN_P_MODEL 0.10 → 0.05 to capture DOTM signals
- Fixed cluster keywords: removed broad "war"/"peace"/"invasion", added specific phrases
- Improved LLM prompt with anti-anchoring warning
- Fixed backtest PnL to use actual exit data instead of -100% assumption
- Added multi-word keyword support in cluster detection

### 2026-05-16 (evening) - v4.4
- Added composite signal scoring (replaces fixed prob_ratio threshold)
- Expanded cluster keywords (NATO, Greenland, S&P, recession, etc.)
- Raised MAX_PRICE to 0.30, MAX_EXPOSURE_PER_CATEGORY to 0.20
- Added Metaculus probability override
- Fixed timezone crash in dates_match
- Bot found and executed 2 new trades (Greenland, Renan Santos)

### 2026-05-15 (evening)
- Fixed 10 critical bugs
- Refactored main() - clean architecture
- Cleared UNKNOWN resolved

### 2026-05-15 (morning)
- Fixed resolve_hypotheses, calibration, Brier score bugs
- Created Hermes Advisory agent
- Cleared 120k UNKNOWN entries
- Reset prob_ratio_threshold to 2.0

### 2026-05-24 - v5.3.0

**Bug Fixes (4 critical):**

| # | Fix | Severity | Description |
|---|-----|----------|-------------|
| B-01 | Batch execution deadlock | CRITICAL | `_update_price_tracking()` in batch analysis caused second loop's delta check to skip all batch-analyzed markets. Fixed: check `market_analyses` before delta. |
| B-02 | Advisor empty response | HIGH | `deepseek-reasoner` returns `reasoning_content` not `reasoning`. Added fallback JSON extraction from reasoning_content. Increased max_tokens 1000→2000. |
| B-03 | MIN_PROB_RATIO not enforced | HIGH | Batch scoring path had no MIN_PROB_RATIO check — market with ratio=0.60x (p_model < price) generated BUY signal. Added guard to both batch and individual paths. |
| B-04 | Renan Santos missing from hypothesis_db | MEDIUM | Position existed in positions.json but hypothesis_db had no entry. Manually reconstructed and added. |

**Signal Threshold Improvements:**

| Change | Old | New | Impact |
|--------|-----|-----|--------|
| signal_threshold | 80 | 65 | Base threshold for short horizon |
| signal_threshold_medium_horizon | hardcoded +5 | 60 (from settings) | Medium horizon (31-90d) |
| signal_threshold_long_horizon | hardcoded +10 | 55 (from settings) | Long horizon (>90d) |
| PRICE_DELTA_THRESHOLD | $0.005 | $0.002 | DOTM sensitivity (was blocking re-analysis) |

**New Features:**
- `[SIGNAL-BATCH]` logging — batch analysis scores now visible in logs (previously invisible)
- Per-horizon threshold support — `bot_settings.json` can override thresholds per time horizon
- Advisor reasoning_content fallback — extracts JSON from reasoner's thinking if content is empty

**Verification:**
- py_compile: 4/4 files OK
- pytest: 80/80 passed
- Sniper --once: pipeline runs end-to-end, BUY signals reach execution, advisor gate works

---

## v5.3.2 Changes (2026-05-25)

### Hermes Advisor Fixes (3 critical)

| # | Fix | Severity | Description |
|---|-----|----------|-------------|
| H-5 | LLM status override | HIGH | LLM returning `status="DIVERGENCE"` no longer overrides code's own probability check. If `p_hermes >= p_bot * 0.5`, LLM's DIVERGENCE is downgraded to YELLOW |
| H-6 | Status hysteresis | HIGH | DIVERGENCE→GREEN/YELLOW transitions require 2 consecutive evaluations confirming the downgrade. Prevents oscillation every 10 min |
| H-7 | P&L-aware DIVERGENCE | MEDIUM | Positions with P&L >= +50% have DIVERGENCE alerts downgraded to YELLOW in Telegram. Emergency exits always pass through |

### Advisor Script Fixes (1)

| # | Fix | Severity | Description |
|---|-----|----------|-------------|
| A-4 | P&L-aware notifications | MEDIUM | `PROFITABLE_PNL_THRESHOLD=50%` — non-WARNING alerts suppressed for positions with P&L >= 50%. Prevents unnecessary YELLOW alerts on winning positions (e.g., Renan Santos +135%) |

### Root Cause Analysis

**NATO oscillation** — Hermes evaluated NATO position every 10 min. LLM returned different `p_hermes` values (8-15%) each cycle due to `temperature=0.1`. When `p_hermes=12%` (vs `p_bot=17%`), code's own check says no divergence (`0.12 > 0.085 = p_bot*0.5`), but LLM set `status="DIVERGENCE"` in its JSON response. The code trusted the LLM's status field over its own math.

**Renan Santos false YELLOW** — Advisor compared Hermes P (8%) with current market price (14.3%), not with entry price (6.1%). The original thesis was "true P = 15% on a 6.1¢ market", and the market has since moved to 14.3¢, confirming the thesis. P&L of +135% makes the YELLOW alert noise.

### Verification

```
py_compile — hermes_advisor.py, advisor_script.py OK
pytest -q — 114/114 passed
hermes_advisor — PID active, both loops running
```

---

## Next Steps

1. Monitor first live trade with v5.3.0 thresholds — verify profitability
2. ✅ Hermes Advisor running as daemon (PID active, cron backup)
3. Evaluate if signal_threshold should be lowered further to 55-60
4. Track LLM p_model consistency (same market gets 35% vs 65% between runs)
5. Consider percentage-based PRICE_DELTA_THRESHOLD (2% of price vs fixed $0.002)
6. Consider adding Renan Santos partial take-profit at current level (+135%)