#!/bin/bash
# OpenClaw startup validation
# Runs on system boot to verify cron configuration is healthy

echo "========================================="
echo "OpenClaw Startup Validation"
echo "========================================="
echo ""

CRON_JOBS_PATH="/root/.openclaw/cron/jobs.json"
VALIDATOR_SCRIPT="/root/dotm-sniper/cron_validator.py"
HEALTH_SCRIPT="/root/dotm-sniper/cron_health_monitor.py"
LOG_FILE="/root/openclaw_startup.log"

log() {
    echo "[$(date -u +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting OpenClaw validation..."

# 1. Validate cron configuration
log "Validating cron jobs..."
python3 "$VALIDATOR_SCRIPT" >> "$LOG_FILE" 2>&1
VALIDATOR_RESULT=$?

if [ $VALIDATOR_RESULT -eq 0 ]; then
    log "✅ Cron configuration validation passed"
else
    log "❌ Cron configuration validation failed"
    log "Running health check..."
    python3 "$HEALTH_SCRIPT" >> "$LOG_FILE" 2>&1
fi

# 2. Check for consecutive errors on any job
log "Checking for problematic jobs..."
if grep -q "consecutiveErrors" "$CRON_JOBS_PATH" 2>/dev/null; then
    CONSECUTIVE=$(grep -o '"consecutiveErrors":[0-9]*' "$CRON_JOBS_PATH" | grep -o '[0-9]*' | sort -n | tail -1)
    if [ "${CONSECUTIVE:-0}" -ge 3 ]; then
        log "⚠️  Found job with ${CONSECUTIVE} consecutive errors"
    fi
fi

# 3. Check last run status
log "Checking recent run history..."
if [ -f "/root/.openclaw/cron/runs/advisor-001.jsonl" ]; then
    LAST_RUN=$(tail -1 "/root/.openclaw/cron/runs/advisor-001.jsonl" 2>/dev/null)
    if echo "$LAST_RUN" | grep -q '"status":"error"'; then
        log "⚠️  advisor-001 last run was error"
    else
        log "✅ advisor-001 last run was ok"
    fi
fi

# 4. Verify critical scripts exist
log "Verifying critical scripts..."
for script in \
    "/root/dotm-sniper/advisor_script.py" \
    "/root/dotm-sniper/src/dotm_sniper.py" \
    "/root/dotm-sniper/src/dotm_report.py"
do
    if [ -f "$script" ]; then
        log "✅ $script exists"
    else
        log "❌ $script missing"
    fi
done

log "Startup validation complete"
echo ""
echo "Full log: $LOG_FILE"
echo "Run 'python3 $VALIDATOR_SCRIPT' for detailed validation"