# DOTM Sniper — Operations Runbook

## Startup
```bash
# Start all services
cd /root/dotm-sniper
screen -dmS sniper bash -c "python3 src/dotm_sniper.py 2>&1 | tee logs/sniper_screen.log"
screen -dmS hermes bash -c "python3 src/hermes_advisor.py 2>&1 | tee logs/hermes_screen.log"
screen -dmS metrics bash -c "python3 src/metrics_server.py 2>&1 | tee logs/metrics.log"
```

## Check Status
```bash
screen -ls                    # Running screens
curl localhost:8765/health    # Health endpoint
curl localhost:8765/metrics   # Full metrics
tail -20 logs/sniper.log      # Recent sniper logs
tail -20 logs/hermes.log      # Recent hermes logs
```

## Restart
```bash
# Kill all
pkill -f 'dotm_sniper.py'
pkill -f 'hermes_advisor.py'
pkill -f 'metrics_server'
sleep 2
screen -wipe
# Then start (see Startup above)
```

## Emergency Stop
```bash
pkill -f 'dotm_sniper.py'   # Stop new trades immediately
pkill -f 'hermes_advisor.py' # Stop emergency exits (optional)
# Positions remain in SQLite — hermes will resume on restart
```

## Common Issues

### Bot not trading
1. Check `curl localhost:8765/metrics` — are positions < max_concurrent?
2. Check `bot_settings` in SQLite: `signal_threshold`, `min_p_model`
3. Check logs for `[DELTA-SKIP]` — all markets skipped?
4. Check LLM API key: `grep LLM_ERROR logs/sniper.log | tail -5`

### Position stuck (selling_in_progress)
Auto-cleared after 1 hour. To force-clear:
```python
import sys; sys.path.insert(0, 'src')
import positions_db
pos = positions_db.get("STUCK-SLUG")
if pos:
    pos["selling_in_progress"] = False
    positions_db.update("STUCK-SLUG", pos)
```

### SQLite corruption
```bash
sqlite3 sniper.db "PRAGMA integrity_check"
# If corrupted, restore from backup:
cp backups/sniper_YYYYMMDD.db sniper.db
```

### Telegram not working
1. Check DNS: `curl -s http://149.154.167.220/bot<TOKEN>/getMe`
2. Check queue: `python3 -c "import sys; sys.path.insert(0,'src'); from tg_sender import flush_queue; print(flush_queue())"`

## Cron Jobs
```
*/5  *      * * *  watchdog.sh
*/30 *      * * *  advisor_script.py
*/30 *      * * *  equity_tracker.py
*/30 *      * * *  tg_sender.py --flush
0    *      * * *  health_monitor.py --hourly
0,4,8,12,16,20 * * *  dotm_report.py
0    3      * * *  sqlite3 backup
```

## Monitoring Alerts
- Health check runs hourly — alerts sent to Telegram on errors
- Watchdog restarts dead processes every 5 min
- Metrics server provides real-time status on :8765
