#!/bin/bash
# Process watchdog — restarts sniper/hermes if crashed
# Run via cron: */5 * * * * /root/dotm-sniper/watchdog.sh >> /tmp/watchdog.log 2>&1

cd /root/dotm-sniper
source /root/dotm-sniper/.env 2>/dev/null

RESTARTED=""

if ! pgrep -f "python3 src/dotm_sniper.py" > /dev/null; then
    echo "$(date '+%Y-%m-%d %H:%M') [WATCHDOG] sniper DOWN, restarting..."
    screen -wipe 2>/dev/null
    screen -dmS sniper bash -c "cd /root/dotm-sniper && python3 src/dotm_sniper.py 2>&1 | tee /tmp/sniper_watchdog.log"
    RESTARTED="sniper"
fi

if ! pgrep -f "python3 src/hermes_advisor.py" > /dev/null; then
    echo "$(date '+%Y-%m-%d %H:%M') [WATCHDOG] hermes DOWN, restarting..."
    screen -wipe 2>/dev/null
    screen -dmS hermes bash -c "cd /root/dotm-sniper && python3 src/hermes_advisor.py 2>&1 | tee /tmp/hermes_watchdog.log"
    RESTARTED="${RESTARTED:+$RESTARTED }hermes"
fi

if [ -n "$RESTARTED" ]; then
    cd /root/dotm-sniper && python3 src/tg_sender.py "🔄 Process watchdog: restarted $RESTARTED" 2>/dev/null || true
fi
