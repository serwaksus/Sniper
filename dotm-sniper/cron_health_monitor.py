#!/usr/bin/env python3
"""
Cron Health Monitor - checks for edge cases and sends alerts
Run via cron or periodically to ensure cron jobs are healthy
"""
import json
import os
import sys
import requests
from datetime import datetime, timezone
from pathlib import Path


def _load_env():
    env_path = "/root/dotm-sniper/.env"
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ.setdefault(key.strip(), val.strip())

_load_env()

TELEGRAM_BOT_TOKEN = os.environ.get('TG_BOT_TOKEN', '') or os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TG_CHAT_ID', '') or os.environ.get('TELEGRAM_CHAT_ID', '')
ALERT_THRESHOLD_CONSECUTIVE_ERRORS = 3
ALERT_THRESHOLD_AGE_HOURS = 24

EDGE_CASE_SIGNALS = {
    'exec_unavailable': {
        'patterns': ['no exec tool', 'exec tool: NOT AVAILABLE', 'no shell access', 'exec unavailable'],
        'severity': 'HIGH',
        'message': 'Cron job cannot execute - no exec tool available'
    },
    'path_escapes_sandbox': {
        'patterns': ['Path escapes sandbox', 'outside the workspace sandbox', 'workspaceOnly'],
        'severity': 'HIGH',
        'message': 'Cron job path is outside allowed workspace'
    },
    'timeout': {
        'patterns': ['timeout', 'timed out', 'execution timeout'],
        'severity': 'MEDIUM',
        'message': 'Cron job is timing out'
    },
    'telegram_failed': {
        'patterns': ['sendMessage failed', 'Message failed', 'not-delivered'],
        'severity': 'MEDIUM',
        'message': 'Telegram delivery failed for cron job'
    }
}

def send_telegram(message):
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    data = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
    try:
        requests.post(url, data=data, timeout=10)
        return True
    except Exception as e:
        print(f"Telegram send failed: {e}")
        return False

def check_job_health(job):
    issues = []
    job_id = job.get('id', 'unknown')
    name = job.get('name', 'unknown')
    state = job.get('state', {})

    last_error = state.get('lastError', '')
    last_status = state.get('lastStatus', '')
    consecutive_errors = state.get('consecutiveErrors', 0)
    last_run_at = state.get('lastRunAtMs', 0)

    if consecutive_errors >= ALERT_THRESHOLD_CONSECUTIVE_ERRORS:
        issues.append({
            'type': 'consecutive_errors',
            'severity': 'HIGH',
            'message': f"{name}: {consecutive_errors} consecutive errors",
            'details': last_error[:200] if last_error else None
        })

    for case_name, case_info in EDGE_CASE_SIGNALS.items():
        for pattern in case_info['patterns']:
            if pattern.lower() in last_error.lower():
                issues.append({
                    'type': case_name,
                    'severity': case_info['severity'],
                    'message': f"{name}: {case_info['message']}",
                    'details': last_error[:200] if last_error else None
                })

    if last_run_at:
        age_hours = (datetime.now(timezone.utc).timestamp() * 1000 - last_run_at) / 3600000
        if age_hours > ALERT_THRESHOLD_AGE_HOURS:
            issues.append({
                'type': 'stale_job',
                'severity': 'HIGH',
                'message': f"{name}: no run in {age_hours:.1f} hours (last status: {state.get('lastRunStatus', 'unknown')})",
                'details': None
            })

    return issues

def monitor_cron_jobs(jobs_path=None):
    jobs_path = jobs_path or '/root/.openclaw/cron/jobs.json'

    try:
        with open(jobs_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load jobs: {e}")
        return []

    all_issues = []
    for job in data.get('jobs', []):
        issues = check_job_health(job)
        all_issues.extend(issues)

    return all_issues

def format_alert(issues):
    if not issues:
        return None

    message = "🚨 <b>Cron Health Alert</b>\n\n"

    high_severity = [i for i in issues if i['severity'] == 'HIGH']
    medium_severity = [i for i in issues if i['severity'] == 'MEDIUM']

    if high_severity:
        message += "🔴 <b>HIGH PRIORITY:</b>\n"
        for issue in high_severity:
            message += f"   • {issue['message']}\n"
            if issue.get('details'):
                message += f"     └ {issue['details'][:100]}\n"
        message += "\n"

    if medium_severity:
        message += "🟡 <b>MEDIUM PRIORITY:</b>\n"
        for issue in medium_severity:
            message += f"   • {issue['message']}\n"

    message += f"\n<i>Generated: {datetime.utcnow().isoformat()} UTC</i>"
    return message

def main():
    issues = monitor_cron_jobs()

    if issues:
        alert = format_alert(issues)
        if alert:
            send_telegram(alert)
            print(alert)
        return 1
    else:
        print("✅ All cron jobs healthy")
        return 0

if __name__ == '__main__':
    sys.exit(main())