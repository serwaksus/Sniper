#!/usr/bin/env python3
"""
Hermes Market Resolution - Check resolved markets, resolve predictions, generate skills.
Extracted from hermes_advisor.py for modularity.
"""
from __future__ import annotations
import json
import logging
import subprocess
import time
from datetime import datetime
import positions_db
from hermes_memory import resolve_prediction, generate_skills, _load_memory

logger = logging.getLogger(__name__)


def _check_resolved_markets() -> None:
    m = _load_memory()
    predictions = m.get("predictions", {})
    if not predictions:
        return

    slugs = list(predictions.keys())
    try:
        subprocess.run(
            ["pm-trader", "orders", "list"],
            capture_output=True, text=True, timeout=15, start_new_session=True
        )
    except Exception as e:
        logger.debug(f"[hermes_resolution] {type(e).__name__}: {e}")
        return

    known_active = set()
    known_active.update(positions_db.slugs())

    try:
        import requests as _req
        gamma_url = "https://gamma-api.polymarket.com/markets"
        for slug in slugs:
            try:
                resp = _req.get(gamma_url, params={"slug": slug}, timeout=10)
                if resp.status_code == 200:
                    markets = resp.json()
                    if markets:
                        market = markets[0]
                        if market.get("closed") or market.get("resolved"):
                            outcome = "yes" if market.get("outcome", "").lower() == "yes" else "no"
                            if market.get("outcomePrices"):
                                try:
                                    prices = json.loads(market["outcomePrices"])
                                    if len(prices) >= 2 and float(prices[0]) > float(prices[1]):
                                        outcome = "yes"
                                    else:
                                        outcome = "no"
                                except (json.JSONDecodeError, ValueError, IndexError):
                                    pass
                            resolve_prediction(slug, outcome)
            except Exception as e:
                logger.warning(f"[resolve_markets] {type(e).__name__}: {e}")
    except Exception as e:
        logger.warning(f"[HERMES] Resolution check failed: {e}")


def _resolve_predictions_loop() -> None:
    while True:
        try:
            time.sleep(3600)
            _check_resolved_markets()
            m = _load_memory()
            last_skill = m.get("last_skill_generation")
            should_gen = not last_skill
            if last_skill:
                try:
                    elapsed = (datetime.now() - datetime.fromisoformat(last_skill)).total_seconds()
                    should_gen = elapsed >= 6 * 3600
                except Exception as e:
                    logger.debug(f"[hermes_resolution] {type(e).__name__}: {e}")
                    should_gen = True
            if should_gen:
                skills = generate_skills()
                if skills:
                    logger.info(f"[HERMES-SKILLS] Generated {len(skills)} skills")
        except Exception as e:
            logger.error(f"[HERMES] Resolution loop error: {e}")
