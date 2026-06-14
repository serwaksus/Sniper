"""
Smart Money Tracker — monitors profitable Polymarket wallets.
Uses Polygonscan API to detect when known profitable wallets
trade on markets we're analyzing.

Signal: +20 points when smart money buys the same DOTM market.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

POLYGONSCAN_API = "https://api.etherscan.io/v2/api"
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
POLYSCAN_KEY_ENV = "POLYGONSCAN_API_KEY"
POLYGON_CHAIN_ID = "137"

_smart_money_cache: dict[str, dict] = {}
CACHE_TTL = 600

WALLET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "smart_money_wallets.json")
ACTIVITY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "smart_money_activity.json")


def load_smart_money_wallets() -> list[str]:
    try:
        with open(WALLET_FILE) as f:
            data = json.load(f)
        return data.get("wallets", [])
    except FileNotFoundError:
        return []
    except Exception as e:
        logger.debug(f"[SMART_MONEY] load failed: {e}")
        return []


def save_smart_money_wallets(wallets: list[str]) -> None:
    with open(WALLET_FILE, "w") as f:
        json.dump({"wallets": wallets, "updated_at": datetime.now().isoformat()}, f, indent=2)


def discover_profitable_wallets(days: int = 180, min_profit: float = 1000, min_winrate: float = 0.35) -> list[str]:
    api_key = os.environ.get(POLYSCAN_KEY_ENV, "")
    if not api_key:
        logger.info("[SMART_MONEY] No Polygonscan API key, skipping wallet discovery")
        return []
    try:
        params = {
            "chainid": POLYGON_CHAIN_ID,
            "module": "account",
            "action": "txlist",
            "address": CTF_EXCHANGE,
            "startblock": 0,
            "endblock": 99999999,
            "sort": "desc",
            "page": 1,
            "offset": 100,
            "apikey": api_key,
        }

        resp = requests.get(POLYGONSCAN_API, params=params, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"[SMART_MONEY] Polygonscan HTTP {resp.status_code}, skipping discovery")
            return []
        try:
            data = resp.json()
        except ValueError:
            logger.warning("[SMART_MONEY] Polygonscan returned non-JSON response")
            return []

        if data.get("status") != "1":
            logger.info(f"[SMART_MONEY] Polygonscan returned: {data.get('message', 'unknown')}")
            return []

        txs = data.get("result", [])
        if not isinstance(txs, list):
            return []

        wallets = list({tx.get("from", "") for tx in txs if tx.get("from", "")})
        logger.info(f"[SMART_MONEY] Found {len(wallets)} active wallets from recent CTF Exchange txs")
        return wallets[:100]

    except Exception as e:
        logger.warning(f"[SMART_MONEY] Discovery failed: {type(e).__name__}: {e}")
        return []


def check_smart_money_activity(condition_token_id: str) -> dict[str, Any]:
    wallets = load_smart_money_wallets()
    if not wallets:
        return _empty_sm_result("no_wallets_tracked")

    now = time.time()
    cached = _smart_money_cache.get(condition_token_id)
    if cached and now - cached["detected_at"] < CACHE_TTL:
        return cached["result"]

    api_key = os.environ.get(POLYSCAN_KEY_ENV, "")
    if not api_key:
        return _empty_sm_result("no_api_key")

    active_wallets = []
    total_vol = 0.0

    try:
        params = {
            "chainid": POLYGON_CHAIN_ID,
            "module": "account",
            "action": "token1155tx",
            "address": CTF_EXCHANGE,
            "token_id": condition_token_id,
            "sort": "desc",
            "page": 1,
            "offset": 50,
        }
        if api_key:
            params["apikey"] = api_key

        resp = requests.get(POLYGONSCAN_API, params=params, timeout=10)
        if resp.status_code != 200:
            return []
        try:
            data = resp.json()
        except ValueError:
            return []

        if data.get("status") == "1" and isinstance(data.get("result"), list):
            tracked_lower = {w.lower() for w in wallets}
            for tx in data["result"]:
                addr = tx.get("from", "").lower()
                value = float(tx.get("value", 0))
                if addr in tracked_lower:
                    active_wallets.append(addr)
                    total_vol += value * 0.01
    except Exception as e:
        logger.debug(f"[SMART_MONEY] check failed: {e}")

    detected = len(active_wallets) > 0
    score = 20 if detected else 0

    result = {
        "detected": detected,
        "wallet_count": len(active_wallets),
        "total_volume_usd": round(total_vol, 2),
        "signal_score": score,
        "wallets": active_wallets[:5],
    }

    if detected:
        logger.info(
            f"[SMART_MONEY] Detected {len(active_wallets)} smart money wallets "
            f"(${total_vol:.0f} vol) for token {condition_token_id[:16]}..."
        )

    _smart_money_cache[condition_token_id] = {
        "detected_at": now,
        "result": result,
    }
    return result


def init_smart_money() -> None:
    wallets = load_smart_money_wallets()
    if not wallets:
        logger.info("[SMART_MONEY] No wallets tracked, discovering...")
        wallets = discover_profitable_wallets()
        if wallets:
            save_smart_money_wallets(wallets)
            logger.info(f"[SMART_MONEY] Saved {len(wallets)} wallets to track")
    else:
        logger.info(f"[SMART_MONEY] Loaded {len(wallets)} tracked wallets")


def _empty_sm_result(reason: str) -> dict[str, Any]:
    return {
        "detected": False,
        "wallet_count": 0,
        "total_volume_usd": 0.0,
        "signal_score": 0,
        "wallets": [],
    }
