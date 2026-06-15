#!/usr/bin/env python3
"""
Check resolutions for pending backtest predictions.

Scans backtest_history/ for pending predictions, checks Polymarket Gamma API
for resolved markets, records outcomes, and computes running metrics.

Usage:
    python3 scripts/check_resolutions.py
    python3 scripts/check_resolutions.py --verbose

Cron: 0 */6 * * *  (every 6 hours)
"""
from __future__ import annotations

import json
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import requests

PROJECT_ROOT = Path(__file__).parent.parent
HISTORY_DIR = PROJECT_ROOT / "backtest_history"
RESOLUTIONS_FILE = HISTORY_DIR / "resolutions.json"
LOG_FILE = PROJECT_ROOT / "logs" / "backtest_cron.log"

GAMMA_API = "https://gamma-api.polymarket.com/markets"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [RESOLVE-CHECK] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_resolutions() -> dict:
    if RESOLUTIONS_FILE.exists():
        with open(RESOLUTIONS_FILE) as f:
            return json.load(f)
    return {"resolved": {}, "metrics": {}, "last_updated": None}


def save_resolutions(data: dict) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    data["last_updated"] = datetime.now().isoformat()
    tmp = RESOLUTIONS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, RESOLUTIONS_FILE)


def load_all_predictions() -> list[dict]:
    """Load all predictions from history files."""
    predictions = []
    if not HISTORY_DIR.exists():
        return predictions

    for p in sorted(HISTORY_DIR.glob("predictions_*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            run_date = p.stem.replace("predictions_", "")
            for r in data.get("results", []):
                r["_run_date"] = run_date
                predictions.append(r)
        except Exception as e:
            logger.warning(f"Failed to load {p.name}: {e}")

    # Also check current backtest_stats.json
    bt_file = PROJECT_ROOT / "backtest_stats.json"
    if bt_file.exists():
        try:
            with open(bt_file) as f:
                data = json.load(f)
            run_date = data.get("timestamp", "unknown")[:10]
            existing_slugs = {p["slug"] for p in predictions}
            for r in data.get("results", []):
                if r.get("slug") not in existing_slugs:
                    r["_run_date"] = run_date
                    predictions.append(r)
        except Exception:
            pass

    return predictions


def check_market_resolution(slug: str) -> str | None:
    """Check if a market has resolved via Gamma API. Returns 'YES', 'NO', or None."""
    try:
        resp = requests.get(f"{GAMMA_API}?slug={slug}", timeout=15)
        if not resp.ok:
            return None
        data = resp.json()
        if not data or not isinstance(data, list) or len(data) == 0:
            return None

        market = data[0]
        # Check outcomePrices for resolution signal
        prices_raw = market.get("outcomePrices", "[]")
        if isinstance(prices_raw, str):
            prices = json.loads(prices_raw)
        else:
            prices = prices_raw

        if prices and len(prices) >= 2:
            p0 = float(prices[0])  # YES price
            p1 = float(prices[1])  # NO price

            if p0 >= 0.95 and p1 <= 0.05:
                return "YES"
            elif p1 >= 0.95 and p0 <= 0.05:
                return "NO"

        # Also check closed flag + resolutionSource
        if market.get("closed") and market.get("resolvedBy"):
            # Fallback: check if question has resolution
            outcomes = market.get("outcomes", "[]")
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            # If outcomePrices available, use them
            if prices and len(prices) >= 2:
                p0 = float(prices[0])
                if p0 > 0.5:
                    return "YES"
                elif p0 < 0.5:
                    return "NO"

        return None
    except Exception as e:
        logger.debug(f"Gamma lookup failed for {slug[:30]}: {type(e).__name__}: {e}")
        return None


def compute_metrics(resolved: dict) -> dict:
    """Compute running metrics from resolved predictions."""
    records = list(resolved.values())
    if not records:
        return {}

    total = len(records)
    buys = [r for r in records if r.get("action") == "BUY"]
    buy_yes = [r for r in buys if r.get("outcome") == "YES"]
    buy_no = [r for r in buys if r.get("outcome") == "NO"]

    # Win rate: predicted YES (BUY) and outcome was YES
    win_rate = len(buy_yes) / len(buys) if buys else 0

    # Expected vs actual: for BUY signals, we expected YES
    # Brier score: (p_model - outcome_binary)^2
    brier_scores = []
    calibration_buckets = {"0-5%": [], "5-10%": [], "10-20%": [], "20%+": []}

    for r in records:
        p = r.get("p_model", 0)
        outcome_binary = 1.0 if r.get("outcome") == "YES" else 0.0
        brier = (p - outcome_binary) ** 2
        brier_scores.append(brier)

        # Calibration buckets by market price (entry price)
        price = r.get("market_price", 0)
        if price < 0.05:
            calibration_buckets["0-5%"].append((p, outcome_binary))
        elif price < 0.10:
            calibration_buckets["5-10%"].append((p, outcome_binary))
        elif price < 0.20:
            calibration_buckets["10-20%"].append((p, outcome_binary))
        else:
            calibration_buckets["20%+"].append((p, outcome_binary))

    avg_brier = sum(brier_scores) / len(brier_scores) if brier_scores else 0

    # Calibration by price bucket
    cal_data = {}
    for bucket, items in calibration_buckets.items():
        if items:
            avg_p = sum(p for p, _ in items) / len(items)
            actual_rate = sum(o for _, o in items) / len(items)
            cal_data[bucket] = {
                "n": len(items),
                "avg_p_model": round(avg_p, 4),
                "actual_yes_rate": round(actual_rate, 4),
                "calibration_gap": round(avg_p - actual_rate, 4),
            }

    # Profitability simulation for BUY signals
    # If BUY (predicted YES) and outcome YES: profit = (1 - entry_price) * shares
    # If BUY and outcome NO: loss = entry_price * shares
    # Simplified: assume $10 per trade
    trade_size = 10.0
    pnl = 0
    for r in buys:
        price = r.get("market_price", 0)
        if r.get("outcome") == "YES":
            pnl += trade_size * (1.0 / price - 1)  # shares bought = trade_size / price
        else:
            pnl -= trade_size

    return {
        "total_resolved": total,
        "buy_signals": len(buys),
        "buy_wins": len(buy_yes),
        "buy_losses": len(buy_no),
        "win_rate": round(win_rate, 4),
        "avg_brier": round(avg_brier, 4),
        "simulated_pnl_usd": round(pnl, 2),
        "calibration_by_price": cal_data,
    }


def main() -> None:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    logger.info("Starting resolution check...")

    resolutions = load_resolutions()
    resolved = resolutions.get("resolved", {})

    predictions = load_all_predictions()
    pending = [p for p in predictions if p.get("slug") not in resolved]

    logger.info(f"Total predictions: {len(predictions)} | Already resolved: {len(resolved)} | Pending: {len(pending)}")

    if not pending:
        logger.info("No pending predictions to check.")
        metrics = compute_metrics(resolved)
        resolutions["metrics"] = metrics
        save_resolutions(resolutions)
        if verbose and metrics:
            print(json.dumps(metrics, indent=2))
        return

    # Check each pending market (with rate limiting)
    newly_resolved = 0
    api_calls = 0
    for p in pending:
        slug = p.get("slug", "")
        if not slug:
            continue

        # Rate limit: ~3 requests/second
        if api_calls > 0 and api_calls % 10 == 0:
            time.sleep(1)

        outcome = check_market_resolution(slug)
        api_calls += 1

        if outcome is not None:
            resolved[slug] = {
                "slug": slug,
                "question": p.get("question", ""),
                "outcome": outcome,
                "action": p.get("action", ""),
                "p_model": p.get("p_model", 0),
                "market_price": p.get("market_price", 0),
                "signal_score": p.get("signal_score", 0),
                "prob_ratio": p.get("prob_ratio", 0),
                "cluster": p.get("clusters", ["other"])[0] if isinstance(p.get("clusters"), list) else "other",
                "run_date": p.get("_run_date", ""),
                "resolved_date": datetime.now().strftime("%Y-%m-%d"),
            }
            newly_resolved += 1
            logger.info(f"  RESOLVED: {slug[:50]}... -> {outcome} (action={p.get('action')}, p_model={p.get('p_model', 0):.1%})")

    logger.info(f"API calls: {api_calls} | Newly resolved: {newly_resolved} | Total resolved: {len(resolved)}")

    # Compute and save metrics
    metrics = compute_metrics(resolved)
    resolutions["resolved"] = resolved
    resolutions["metrics"] = metrics
    save_resolutions(resolutions)

    if metrics:
        logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
    else:
        logger.info("No resolved predictions yet. Metrics will be available once markets resolve.")


if __name__ == "__main__":
    main()
