import subprocess
import json
import logging
import contextlib
from datetime import datetime
from collections import defaultdict

from utils import load_json, save_json
from calibration_tracker import log_calibration_entry, detect_model_drift
from schema import (
    HYP_CLUSTERS,
    HYP_DB_HYPOTHESES,
    HYP_DB_RESOLVED, HYP_EXIT_PRICE, HYP_EXIT_TYPE, HYP_FACTORS,
    HYP_MARKET_PRICE, HYP_OUTCOME, HYP_P_MODEL, HYP_PNL_AT_EXIT,
    HYP_QUESTION, HYP_RESOLUTION_NOTE, HYP_RESOLVED,
    HYP_RESOLVED_AT, HYP_SLUG, HYP_SOLD_PNL_PCT,
    HYP_SOURCE_SIGNAL,
    SETTINGS_CALIBRATION_BRIER, SETTINGS_CLUSTER_WEIGHTS,
    SETTINGS_MIN_P_MODEL,
    SETTINGS_SIGNAL_THRESHOLD, SETTINGS_TOTAL_RESOLVED,
)

logger = logging.getLogger(__name__)

BURN_IN_TRADES = 50
MIN_P_MODEL = 0.03
MIN_TRADES_FOR_WEIGHT_ADJUSTMENT = 20
BAYESIAN_PRIOR_STRENGTH = 10

POSITIONS_FILE = "/root/dotm-sniper/positions.json"


def _get_sniper_deps():
    from dotm_sniper import get_settings, save_settings, load_hypothesis_db, save_hypothesis_db
    return get_settings, save_settings, load_hypothesis_db, save_hypothesis_db


def calculate_brier_score(db):
    get_settings, save_settings, _, _ = _get_sniper_deps()

    resolved = [h for h in db.get(HYP_DB_RESOLVED, []) if h.get(HYP_OUTCOME) in ("YES", "NO")]
    if len(resolved) < BURN_IN_TRADES:
        return None

    brier_scores = []
    wins = 0
    losses = 0
    for h in resolved[-BURN_IN_TRADES:]:
        p = h.get(HYP_P_MODEL, 0.5)
        o = 1 if h.get(HYP_OUTCOME) == "YES" else 0
        brier_scores.append((p - o) ** 2)
        if o == 1:
            wins += 1
        else:
            losses += 1

    brier = sum(brier_scores) / len(brier_scores)
    winrate = wins / (wins + losses) if (wins + losses) > 0 else 0
    logger.info(f"Stats: Brier={brier:.3f}, Winrate={winrate:.1%} ({wins}W/{losses}L) [from {len(resolved)} resolved]")

    settings = get_settings()
    old_brier = settings.get(SETTINGS_CALIBRATION_BRIER)

    if old_brier is not None and brier > 0:
        if brier > 0.08 and settings.get(SETTINGS_SIGNAL_THRESHOLD, 55) < 65:
            settings[SETTINGS_SIGNAL_THRESHOLD] = settings.get(SETTINGS_SIGNAL_THRESHOLD, 55) + 2
            logger.info(f"[CALIBRATE] Brier {brier:.3f} > 0.08, raising signal_threshold to {settings[SETTINGS_SIGNAL_THRESHOLD]}")
        elif brier < 0.03 and winrate > 0.1 and settings.get(SETTINGS_SIGNAL_THRESHOLD, 55) > 40:
            settings[SETTINGS_SIGNAL_THRESHOLD] = settings.get(SETTINGS_SIGNAL_THRESHOLD, 55) - 2
            logger.info(f"[CALIBRATE] Brier {brier:.3f} < 0.03, winrate {winrate:.0%}, lowering signal_threshold to {settings[SETTINGS_SIGNAL_THRESHOLD]}")

    if winrate == 0 and len(resolved) >= 10 and settings.get(SETTINGS_SIGNAL_THRESHOLD, 55) < 80:
        settings[SETTINGS_SIGNAL_THRESHOLD] = min(80, settings.get(SETTINGS_SIGNAL_THRESHOLD, 55) + 5)
        logger.warning(f"[CALIBRATE] 0% winrate ({len(resolved)} resolved), RAISING signal_threshold to {settings[SETTINGS_SIGNAL_THRESHOLD]} (defensive mode)")
    elif winrate < 0.30 and len(resolved) >= 20 and settings.get(SETTINGS_SIGNAL_THRESHOLD, 55) < 75:
        settings[SETTINGS_SIGNAL_THRESHOLD] = min(75, settings.get(SETTINGS_SIGNAL_THRESHOLD, 55) + 3)
        logger.info(f"[CALIBRATE] Low winrate ({winrate:.0%}, {len(resolved)} resolved), raising signal_threshold to {settings[SETTINGS_SIGNAL_THRESHOLD]}")

    settings[SETTINGS_CALIBRATION_BRIER] = brier
    save_settings(settings)

    if len(resolved) >= 50:
        recent_resolved = [h for h in resolved if h.get(HYP_RESOLVED_AT) and (datetime.now() - datetime.fromisoformat(h[HYP_RESOLVED_AT])).days <= 90]
        if len(recent_resolved) >= 20:
            from calibration import get_calibrator
            calibrator = get_calibrator()
            calibrator.fit(recent_resolved)
            calibrator.save()
            logger.info(f"[CALIBRATION] Trained isotonic model on {len(recent_resolved)} recent markets (<=90 days)")
        else:
            logger.info(f"[CALIBRATION] Only {len(recent_resolved)} recent resolved, need >=20, skipping retrain")

    return brier


def learn_from_results(db):
    get_settings, save_settings, _, _ = _get_sniper_deps()
    resolved = db.get(HYP_DB_RESOLVED, [])
    if len(resolved) < 10:
        return {}

    settings = get_settings()
    factor_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    cluster_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    source_stats = defaultdict(lambda: {"wins": 0, "losses": 0})

    for h in resolved[-50:]:
        outcome = h.get(HYP_OUTCOME)
        if outcome not in ("YES", "NO"):
            continue
        is_win = outcome == "YES"

        for factor in h.get(HYP_FACTORS, []):
            key = f"{factor.get('direction')}:{factor.get('weight')}"
            if is_win:
                factor_stats[key]["wins"] += 1
            else:
                factor_stats[key]["losses"] += 1

        for cluster in h.get(HYP_CLUSTERS, []):
            if is_win:
                cluster_stats[cluster]["wins"] += 1
            else:
                cluster_stats[cluster]["losses"] += 1

        source_signal = h.get(HYP_SOURCE_SIGNAL, "default")
        if is_win:
            source_stats[source_signal]["wins"] += 1
        else:
            source_stats[source_signal]["losses"] += 1

    cluster_weights = {}
    for cluster, stats in cluster_stats.items():
        total = stats["wins"] + stats["losses"]
        if total >= MIN_TRADES_FOR_WEIGHT_ADJUSTMENT:
            winrate = stats["wins"] / total
            base_weight = {
                "venezuela": 0.30,
                "russia_ukraine": 0.25,
                "usa_politics": 0.20,
                "fed_fomc": 0.25,
                "ai_tech": 0.10,
                "sports_nba": 0.15,
                "sports_ufc": 0.15,
            }.get(cluster, 0.15)
            posterior_weight = (
                (base_weight * BAYESIAN_PRIOR_STRENGTH + winrate * total)
                / (BAYESIAN_PRIOR_STRENGTH + total)
            )
            posterior_weight = max(0.05, min(0.50, posterior_weight))
            cluster_weights[cluster] = posterior_weight
            logger.info(
                f"[LEARN] Cluster {cluster}: winrate={winrate:.1%} ({total} trades), "
                f"base={base_weight:.2f} → posterior={posterior_weight:.3f}"
            )
        else:
            logger.debug(
                f"[LEARN] Cluster {cluster}: insufficient data ({total}/{MIN_TRADES_FOR_WEIGHT_ADJUSTMENT} trades), "
                f"keeping base weight"
            )

    metaculus_bonus = 0.4
    geopol_bonus = 0.3
    sports_bonus = 0.2

    for source, stats in source_stats.items():
        total = stats["wins"] + stats["losses"]
        if total >= 3:
            winrate = stats["wins"] / total
            if source == "metaculus":
                metaculus_bonus = 0.4 * (1 + (winrate - 0.5))
                logger.info(f"Source {source}: winrate={winrate:.1%}, adjusted bonus={metaculus_bonus:.3f}")
            elif source == "geopol":
                geopol_bonus = 0.3 * (1 + (winrate - 0.5))
                logger.info(f"Source {source}: winrate={winrate:.1%}, adjusted bonus={geopol_bonus:.3f}")
            elif source == "sports":
                sports_bonus = 0.2 * (1 + (winrate - 0.5))
                logger.info(f"Source {source}: winrate={winrate:.1%}, adjusted bonus={sports_bonus:.3f}")

    settings[SETTINGS_CLUSTER_WEIGHTS] = cluster_weights
    settings["source_bonus_metaculus"] = metaculus_bonus
    settings["source_bonus_geopol"] = geopol_bonus
    settings["source_bonus_sports"] = sports_bonus
    save_settings(settings)

    return {
        SETTINGS_CLUSTER_WEIGHTS: cluster_weights,
        "source_stats": dict(source_stats)
    }


def backtest_recent(n=20):
    _, _, load_hypothesis_db, _ = _get_sniper_deps()
    db = load_hypothesis_db()
    resolved = db.get(HYP_DB_RESOLVED, [])

    if len(resolved) < 5:
        return {"error": f"Only {len(resolved)} resolved, need at least 5", "recommendation": "skip"}

    recent = [h for h in resolved[-n:] if h.get(HYP_OUTCOME) in ("YES", "NO")]
    if len(recent) < 5:
        return {"error": f"Only {len(recent)} with YES/NO outcome", "recommendation": "skip"}

    wins = 0
    total_pnl = 0
    brier_sum = 0

    for h in recent:
        p_model = h.get(HYP_P_MODEL, 0.5)
        market_price = h.get(HYP_MARKET_PRICE, 0.5)
        outcome = 1 if h.get(HYP_OUTCOME) == "YES" else 0
        is_win = outcome == 1

        if is_win:
            wins += 1
            pnl = (1 - market_price) / market_price
        else:
            actual_pnl = h.get(HYP_SOLD_PNL_PCT) or h.get(HYP_PNL_AT_EXIT)
            if actual_pnl is not None and actual_pnl != 0:
                pnl = actual_pnl
            else:
                pnl = -1

        total_pnl += pnl
        brier_sum += (p_model - outcome) ** 2

    winrate = wins / len(recent)
    avg_brier = brier_sum / len(recent)
    avg_pnl = total_pnl / len(recent)

    get_settings, _, _, _ = _get_sniper_deps()
    get_settings().get(SETTINGS_SIGNAL_THRESHOLD, 55)
    get_settings().get(SETTINGS_MIN_P_MODEL, MIN_P_MODEL)

    recommendations = []

    if winrate < 0.40:
        recommendations.append({
            "issue": "winrate_too_low",
            "current": winrate,
            "suggestion": "Raise MIN_PROB_RATIO or MIN_P_MODEL to be more selective"
        })

    if avg_brier > 0.20:
        recommendations.append({
            "issue": "poor_calibration",
            "current": avg_brier,
            "suggestion": "Improve p_model estimation or use market price as stronger prior"
        })

    if avg_pnl < 0:
        recommendations.append({
            "issue": "negative_avg_pnl",
            "current": avg_pnl,
            "suggestion": "Reduce position sizes or increase threshold"
        })

    cluster_wins = defaultdict(lambda: {"wins": 0, "total": 0})
    for h in recent:
        for c in h.get(HYP_CLUSTERS, []):
            cluster_wins[c]["total"] += 1
            if h.get(HYP_OUTCOME) == "YES":
                cluster_wins[c]["wins"] += 1

    cluster_performance = {}
    for c, stats in cluster_wins.items():
        if stats["total"] >= 3:
            cluster_performance[c] = stats["wins"] / stats["total"]

    result = {
        "n_analyzed": len(recent),
        "winrate": winrate,
        "avg_brier": avg_brier,
        "avg_pnl": avg_pnl,
        "cluster_performance": cluster_performance,
        "recommendations": recommendations,
        "recommendation": "use_current" if not recommendations else "adjust_thresholds"
    }

    logger.info(f"[BACKTEST] n={len(recent)}, winrate={winrate:.1%}, brier={avg_brier:.3f}, pnl={avg_pnl:.2f}")
    for r in recommendations:
        logger.info(f"[BACKTEST] REC: {r['issue']} -> {r['suggestion']}")

    return result


def resolve_hypothesis_immediately(slug, current_price, entry_price):
    get_settings, save_settings, load_hypothesis_db, save_hypothesis_db = _get_sniper_deps()
    from order_manager import _cancel_all_tp_orders

    _cancel_all_tp_orders(slug)
    db = load_hypothesis_db()
    for h in db[HYP_DB_HYPOTHESES]:
        if h[HYP_SLUG] == slug and not h.get(HYP_RESOLVED):
            h[HYP_RESOLVED] = True
            h[HYP_RESOLVED_AT] = datetime.now().isoformat()
            h[HYP_EXIT_PRICE] = current_price
            h[HYP_PNL_AT_EXIT] = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            h[HYP_EXIT_TYPE] = "manual"
            h[HYP_OUTCOME] = "SOLD"
            h[HYP_SOLD_PNL_PCT] = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            db[HYP_DB_RESOLVED].append(h)

            positions = load_json(POSITIONS_FILE, {})
            if slug in positions:
                del positions[slug]
                fresh = load_json(POSITIONS_FILE, {})
                fresh.pop(slug, None)
                save_json(POSITIONS_FILE, fresh)

                try:
                    from bayesian_updater import cleanup_slug
                    cleanup_slug(slug)
                except Exception as e:
                    logger.warning(f"[bayesian_cleanup] {type(e).__name__}: {e}")

            save_hypothesis_db(db)

            settings = get_settings()
            settings[SETTINGS_TOTAL_RESOLVED] = settings.get(SETTINGS_TOTAL_RESOLVED, 0) + 1
            save_settings(settings)
            break


def resolve_hypotheses():
    get_settings, save_settings, load_hypothesis_db, save_hypothesis_db = _get_sniper_deps()
    from order_manager import get_portfolio

    db = load_hypothesis_db()
    portfolio = get_portfolio()
    portfolio_slugs = {p["market_slug"] for p in portfolio}

    all_hypotheses = db.get(HYP_DB_HYPOTHESES, [])
    unresolved = [h for h in all_hypotheses if not h.get(HYP_RESOLVED) and h[HYP_SLUG] not in portfolio_slugs]

    if not unresolved:
        return

    market_map = {}
    try:
        res = subprocess.run(["pm-trader", "markets", "list", "--limit", "200"],
                           capture_output=True, text=True, timeout=20, start_new_session=True)
        for m in json.loads(res.stdout).get("data", []):
            market_map[m["slug"]] = m
    except Exception as e:
        logger.warning(f"[market_list] {type(e).__name__}: {e}")

    new_resolved = 0
    for h in unresolved:
        slug = h[HYP_SLUG]
        market_data = market_map.get(slug)

        if not market_data:
            h[HYP_RESOLVED] = True
            h[HYP_RESOLVED_AT] = datetime.now().isoformat()
            h[HYP_OUTCOME] = "UNKNOWN"
            h[HYP_RESOLUTION_NOTE] = "market_not_found_in_api"
            db[HYP_DB_RESOLVED].append(h)
            new_resolved += 1
            continue

        if not market_data.get("closed"):
            continue

        h[HYP_RESOLVED] = True
        h[HYP_RESOLVED_AT] = datetime.now().isoformat()

        outcome = "UNKNOWN"
        if market_data.get("resolution") in ("YES", "NO"):
            outcome = market_data.get("resolution")
        elif market_data.get("outcome_prices"):
            yes_price = market_data.get("outcome_prices", [0.5])[0]
            outcome = "YES" if yes_price > 0.5 else "NO"

        h[HYP_OUTCOME] = outcome

        db[HYP_DB_RESOLVED].append(h)
        new_resolved += 1

    save_hypothesis_db(db)

    if new_resolved > 0:
        settings = get_settings()
        settings[SETTINGS_TOTAL_RESOLVED] = len(db.get(HYP_DB_RESOLVED, []))
        save_settings(settings)

        if len(db.get(HYP_DB_RESOLVED, [])) >= BURN_IN_TRADES:
            calculate_brier_score(db)
            learn_from_results(db)

        for h in db.get(HYP_DB_RESOLVED, []):
            if h.get(HYP_OUTCOME) in ("YES", "NO") and h.get(HYP_P_MODEL) is not None:
                with contextlib.suppress(Exception):
                    log_calibration_entry(
                        slug=h[HYP_SLUG],
                        question=h.get(HYP_QUESTION, ""),
                        p_model=h[HYP_P_MODEL],
                        p_calibrated=0,
                        market_price=h.get(HYP_MARKET_PRICE, 0),
                        actual_outcome=h[HYP_OUTCOME],
                        cluster=h.get(HYP_CLUSTERS, ["other"])[0],
                        entry_price=h.get(HYP_MARKET_PRICE, 0),
                        exit_price=h.get(HYP_EXIT_PRICE, 0),
                        pnl_pct=h.get(HYP_PNL_AT_EXIT, 0),
                    )

        try:
            drift_alert = detect_model_drift()
            if drift_alert:
                logger.warning(f"[CALIBRATION] {drift_alert}")
        except Exception as e:
            logger.warning(f"[drift_detect] {type(e).__name__}: {e}")
