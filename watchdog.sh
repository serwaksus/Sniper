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
    TOKEN="${TG_BOT_TOKEN}"
    CHAT="${TG_CHAT_ID}"
    if [ -n "$TOKEN" ] && [ -n "$CHAT" ]; then
        curl -s --max-time 10 "https://api.telegram.org/bot${TOKEN}/sendMessage" \
            -d chat_id="$CHAT" \
            -d text="🔄 Process watchdog: restarted $RESTARTED" \
            > /dev/null 2>&1
    fi
fi
