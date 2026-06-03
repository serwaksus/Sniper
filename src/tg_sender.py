#!/usr/bin/env python3
"""
Unified Telegram sender with persistent queue and retry logic.
All Telegram sends in the DOTM sniper ecosystem should go through this module.

Features:
- Persistent queue: failed messages saved to disk for later retry
- Exponential backoff: 5s, 15s, 45s between retries
- Queue flush: run `python3 tg_sender.py --flush` to retry all pending
- Thread-safe via file locking (uses utils.load_json/save_json)
"""
import os
import sys
import time
import json
import logging
import requests
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_json, save_json

QUEUE_FILE = "/root/dotm-sniper/tg_queue.json"
MAX_QUEUE_SIZE = 100
MAX_AGE_HOURS = 48
SEND_TIMEOUT = 15
TG_WORKING_IP = "149.154.167.220"
TG_API_HOST = "api.telegram.org"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _get_credentials():
    token = os.environ.get("TG_BOT_TOKEN", "")
    chat_id = os.environ.get("TG_CHAT_ID", "")
    if not token or not chat_id:
        env_path = "/root/dotm-sniper/.env"
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("TG_BOT_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("TG_CHAT_ID="):
                        chat_id = line.split("=", 1)[1].strip().strip('"').strip("'")
    return token, chat_id


def _send_once(token, chat_id, message, timeout=SEND_TIMEOUT):
    import socket
    import urllib3
    from urllib3.util.connection import allowed_gai_family
    
    orig_getaddrinfo = socket.getaddrinfo
    def _patched_getaddrinfo(host, port, *args, **kwargs):
        if host == TG_API_HOST:
            return [orig_getaddrinfo(TG_WORKING_IP, port, *args, **kwargs)[0]]
        return orig_getaddrinfo(host, port, *args, **kwargs)
    
    socket.getaddrinfo = _patched_getaddrinfo
    try:
        resp = requests.post(
            f"https://{TG_API_HOST}/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": message[:4096],
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=timeout,
        )
        return resp.ok
    finally:
        socket.getaddrinfo = orig_getaddrinfo


def send_telegram(message, max_retries=3, queue_on_fail=True):
    token, chat_id = _get_credentials()
    if not token or not chat_id:
        logger.warning("[TG] No credentials, skipping send")
        return False

    delays = [5, 15, 45]
    for attempt in range(max_retries):
        try:
            if _send_once(token, chat_id, message):
                logger.info(f"[TG] Sent successfully (attempt {attempt+1})")
                return True
        except Exception as e:
            logger.warning(f"[TG] Attempt {attempt+1}/{max_retries} failed: {e}")
        if attempt < max_retries - 1:
            time.sleep(delays[min(attempt, len(delays) - 1)])

    logger.warning(f"[TG] All {max_retries} attempts failed")
    if queue_on_fail:
        _enqueue(message)
    return False


def _enqueue(message):
    queue = load_json(QUEUE_FILE, [])
    now = datetime.now().isoformat()
    queue.append({"message": message, "queued_at": now, "attempts": 0})
    cutoff = datetime.now().timestamp() - MAX_AGE_HOURS * 3600
    queue = [
        m for m in queue
        if datetime.fromisoformat(m["queued_at"]).timestamp() > cutoff
    ]
    if len(queue) > MAX_QUEUE_SIZE:
        queue = queue[-MAX_QUEUE_SIZE:]
    save_json(QUEUE_FILE, queue)
    logger.info(f"[TG-QUEUE] Queued message (queue size: {len(queue)})")


def flush_queue(max_messages=10):
    queue = load_json(QUEUE_FILE, [])
    if not queue:
        logger.info("[TG-FLUSH] Queue empty")
        return 0

    token, chat_id = _get_credentials()
    if not token or not chat_id:
        logger.warning("[TG-FLUSH] No credentials")
        return 0

    sent = 0
    remaining = []
    cutoff = datetime.now().timestamp() - MAX_AGE_HOURS * 3600

    for msg in queue[:max_messages]:
        ts = datetime.fromisoformat(msg["queued_at"]).timestamp()
        if ts < cutoff:
            continue
        try:
            if _send_once(token, chat_id, msg["message"]):
                sent += 1
                logger.info(f"[TG-FLUSH] Delivered queued message")
            else:
                msg["attempts"] = msg.get("attempts", 0) + 1
                if msg["attempts"] < 10:
                    remaining.append(msg)
                else:
                    logger.warning("[TG-FLUSH] Dropping message after 10 failed attempts")
        except Exception as e:
            msg["attempts"] = msg.get("attempts", 0) + 1
            logger.warning(f"[TG-FLUSH] Send failed (attempt {msg['attempts']}): {e}")
            if msg["attempts"] < 10:
                remaining.append(msg)

    remaining.extend(queue[max_messages:])
    save_json(QUEUE_FILE, remaining)
    logger.info(f"[TG-FLUSH] Sent {sent}, remaining {len(remaining)}")
    return sent


def get_queue_size():
    queue = load_json(QUEUE_FILE, [])
    return len(queue)


if __name__ == "__main__":
    if "--flush" in sys.argv:
        flush_queue()
    elif "--status" in sys.argv:
        print(f"Queue size: {get_queue_size()}")
    else:
        msg = sys.argv[1] if len(sys.argv) > 1 else "Test message from tg_sender"
        result = send_telegram(msg)
        print(f"Send result: {result}")
