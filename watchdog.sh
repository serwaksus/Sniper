#!/bin/bash
# Process watchdog — restarts sniper/hermes if crashed
# Run via cron: */5 * * * * /root/dotm-sniper/watchdog.sh >> /tmp/watchdog.log 2>&1

exec 200>/tmp/watchdog.lock
flock -n 200 || { echo "$(date '+%Y-%m-%d %H:%M') [WATCHDOG] Already running, skipping"; exit 0; }

cd /root/dotm-sniper
set -a
source /root/dotm-sniper/.env 2>/dev/null
set +a

RESTARTED=""

if ! pgrep -f 'python3 src/dotm_sniper\.py' > /dev/null 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M') [WATCHDOG] sniper DOWN, restarting..."
    screen -wipe 2>/dev/null
    screen -dmS sniper bash -c "cd /root/dotm-sniper && python3 src/dotm_sniper.py 2>&1 | tee /tmp/sniper_watchdog.log"
    RESTARTED="sniper"
fi

# Check if sniper log has recent activity (within 30 min)
if [ -f "/tmp/sniper_v565.log" ]; then
    last_line=$(tail -1 /tmp/sniper_v565.log 2>/dev/null)
    if [ -n "$last_line" ]; then
        log_time=$(echo "$last_line" | grep -oP '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' | head -1)
        if [ -n "$log_time" ]; then
            log_epoch=$(date -d "$log_time" +%s 2>/dev/null)
            now_epoch=$(date +%s)
            if [ -n "$log_epoch" ]; then
                age=$(( now_epoch - log_epoch ))
                if [ "$age" -gt 1800 ]; then
                    echo "$(date '+%Y-%m-%d %H:%M') [WATCHDOG] Sniper log stale (${age}s old), restarting sniper" >> /tmp/watchdog.log
                    pkill -f 'python3 src/dotm_sniper\.py' 2>/dev/null
                    sleep 2
                    screen -wipe 2>/dev/null
                    screen -dmS sniper bash -c "cd /root/dotm-sniper && python3 src/dotm_sniper.py 2>&1 | tee /tmp/sniper_watchdog.log"
                    RESTARTED="${RESTARTED:+$RESTARTED }sniper_stale"
                fi
            fi
        fi
    fi
fi

if ! pgrep -f 'python3 src/hermes_advisor\.py' > /dev/null 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M') [WATCHDOG] hermes DOWN, restarting..."
    screen -wipe 2>/dev/null
    screen -dmS hermes bash -c "cd /root/dotm-sniper && python3 src/hermes_advisor.py 2>&1 | tee /tmp/hermes_watchdog.log"
    RESTARTED="${RESTARTED:+$RESTARTED }hermes"
fi

if [ -n "$RESTARTED" ]; then
    cd /root/dotm-sniper && python3 src/tg_sender.py "🔄 Process watchdog: restarted $RESTARTED" 2>/dev/null || true
fi
