#!/bin/bash
# Daily live backtest: record predictions on active DOTM markets
# Runs WITH advisor (DeepSeek LLM) for real probability estimates
# Cost: ~$0.02/day (50 markets x 1 DeepSeek call each)
# Cron: 0 6 * * *  (09:00 MSK / 06:00 UTC)

set -euo pipefail

cd /root/dotm-sniper

# Source environment
set -a; source .env 2>/dev/null; set +a

DATE=$(date +%Y-%m-%d)
HISTORY_DIR="backtest_history"
OUTPUT_FILE="${HISTORY_DIR}/predictions_${DATE}.json"
LOG_FILE="logs/backtest_cron.log"

mkdir -p "$HISTORY_DIR"

echo "[BACKTEST-CRON] Starting daily live backtest for ${DATE}" >> "$LOG_FILE"

# Run live backtest with --skip-advisor for speed (~2min vs 30min)
# The initial LLM analysis still runs (DeepSeek), only the advisor verification is skipped
# p_model values are REAL estimates, not price*2 fallback
python3 src/dotm_backtester.py --mode live --count 50 --skip-advisor 2>> "$LOG_FILE"

# Copy to dated history file
if [ -f backtest_stats.json ]; then
    cp backtest_stats.json "$OUTPUT_FILE"
    echo "[BACKTEST-CRON] Saved predictions to ${OUTPUT_FILE}" >> "$LOG_FILE"

    # Quick summary
    python3 -c "
import json
with open('$OUTPUT_FILE') as f:
    data = json.load(f)
s = data.get('summary', {})
buys = s.get('buy_signals', 0)
skips = s.get('skip_signals', 0)
total = buys + skips
print(f'[BACKTEST-CRON] {total} markets: {buys} BUY, {skips} SKIP')
" >> "$LOG_FILE" 2>&1
else
    echo "[BACKTEST-CRON] ERROR: backtest_stats.json not created!" >> "$LOG_FILE"
fi

echo "[BACKTEST-CRON] Done for ${DATE}" >> "$LOG_FILE"
