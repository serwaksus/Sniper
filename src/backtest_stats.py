#!/usr/bin/env python3
"""
Backtest Stats — Statistics computation, result formatting, and reporting.
Extracted from dotm_backtester.py for modularity.
"""
from __future__ import annotations
import logging
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


def apply_advisor(
    market: dict,
    analysis: dict,
    skip_advisor: bool,
    advisor_check_fn,
) -> tuple[str, bool, str]:
    """Apply advisor check. Returns (final_action, advisor_approved, advisor_verdict)."""
    advisor_approved = True
    advisor_verdict = "SKIPPED"
    if not skip_advisor and analysis["action"] == "BUY":
        advisor_approved, advisor_verdict = advisor_check_fn(market, analysis)
    final_action = analysis["action"]
    if final_action == "BUY" and not advisor_approved:
        final_action = "SKIP"
    return final_action, advisor_approved, advisor_verdict


def process_resolved_results(
    markets: list[dict],
    analyses: list[dict | None],
    skip_advisor: bool,
    advisor_check_fn,
    simulate_fn,
    print_progress: bool = True,
) -> tuple[list[dict], dict, dict, list[dict]]:
    """Process resolved/sim market results.
    Returns (results, summary, cluster_stats, dampened_markets)."""
    results = []
    wins = losses = skips = 0
    brier_scores = []
    brier_scores_raw = []
    dampened_count = 0
    upside_sum = 0.0
    upside_count = 0
    dampened_markets = []
    cluster_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "skips": 0, "total": 0})

    for i, (m, analysis) in enumerate(zip(markets, analyses, strict=False)):
        if analysis is None:
            continue

        final_action, advisor_approved, advisor_verdict = apply_advisor(
            m, analysis, skip_advisor, advisor_check_fn
        )

        if print_progress:
            _print_market_progress(i, len(markets), m, analysis, final_action,
                                   advisor_approved, advisor_verdict)

        actual_outcome = 1 if m["resolution"] == "YES" else 0
        p_model = analysis.get("p_model", 0)
        p_model_raw = analysis.get("p_model_raw", p_model)
        brier = (p_model - actual_outcome) ** 2
        brier_raw = (p_model_raw - actual_outcome) ** 2
        brier_scores.append(brier)
        brier_scores_raw.append(brier_raw)

        if analysis.get("was_dampened"):
            dampened_count += 1
            dampened_markets.append({
                "slug": m["slug"],
                "question": m["question"],
                "market_price": m["yes_price"],
                "p_model_raw": analysis.get("p_model_raw", 0),
                "p_model_calibrated": analysis.get("p_model", 0),
                "damping_delta": analysis.get("damping_delta", 0),
                "cluster": m.get("clusters", ["other"])[0],
            })

        high_price = m.get("high_price")
        if high_price is None:
            high_price = 1.0 if m["resolution"] == "YES" else m.get("yes_final", m["yes_price"])

        ladder_pnl = 0
        ladder_details = []
        if final_action == "SKIP":
            skips += 1
        elif final_action == "BUY":
            ladder_pnl, ladder_details = simulate_fn(
                entry_price=m["yes_price"],
                high_price=high_price,
                resolution=m["resolution"],
            )
            if ladder_pnl > 0:
                wins += 1
                upside_sum += ladder_pnl
                upside_count += 1
                logger.info(
                    f"[BACKTEST-LADDER] {m['slug'][:40]}... "
                    f"weighted_pnl={ladder_pnl:+.2%} "
                    f"rungs={[d['label'] for d in ladder_details]}"
                )
            else:
                losses += 1
                logger.info(
                    f"[BACKTEST-LADDER-LOSS] {m['slug'][:40]}... "
                    f"weighted_pnl={ladder_pnl:+.2%}"
                )

        for c in m.get("clusters", []):
            cluster_stats[c]["total"] += 1
            if final_action == "BUY" and ladder_pnl > 0:
                cluster_stats[c]["wins"] += 1
            elif final_action == "BUY":
                cluster_stats[c]["losses"] += 1
            else:
                cluster_stats[c]["skips"] += 1

        results.append({
            "slug": m["slug"],
            "question": m["question"],
            "market_price": m["yes_price"],
            "resolution": m["resolution"],
            "p_model": p_model,
            "p_model_raw": p_model_raw,
            "prob_ratio": analysis.get("prob_ratio", 0),
            "confidence": analysis.get("confidence", 0),
            "signal_score": analysis.get("signal_score", 0),
            "action": final_action,
            "advisor_verdict": advisor_verdict,
            "brier": brier,
            "brier_raw": brier_raw,
            "source_signal": analysis.get("source_signal", "default"),
            "clusters": m.get("clusters", []),
            "status": "resolved",
            "simulated_price": m.get("simulated_price", False),
            "was_dampened": analysis.get("was_dampened", False),
            "created_at": m.get("created_at", ""),
            "analyzed_at": datetime.now().isoformat(),
        })

    total_traded = wins + losses
    winrate = wins / total_traded if total_traded > 0 else 0
    avg_brier = sum(brier_scores) / len(brier_scores) if brier_scores else 0
    avg_brier_raw = sum(brier_scores_raw) / len(brier_scores_raw) if brier_scores_raw else 0
    brier_improvement = avg_brier_raw - avg_brier
    avg_upside = upside_sum / upside_count if upside_count > 0 else 0

    summary = {
        "total_markets": len(results),
        "traded": total_traded,
        "skipped": skips,
        "wins": wins,
        "losses": losses,
        "winrate": winrate,
        "brier_score": avg_brier,
        "brier_score_raw": avg_brier_raw,
        "brier_improvement": brier_improvement,
        "dampened_count": dampened_count,
        "avg_upside": avg_upside,
    }

    return results, summary, dict(cluster_stats), dampened_markets


def process_sim_results(
    markets: list[dict],
    analyses: list[dict | None],
    skip_advisor: bool,
    advisor_check_fn,
    simulate_fn,
    print_progress: bool = True,
) -> tuple[list[dict], dict, dict]:
    """Process sim market results. Returns (results, summary, cluster_stats)."""
    results = []
    wins = losses = skips = 0
    brier_scores = []
    upside_sum = 0.0
    upside_count = 0
    cluster_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "total": 0})

    for i, (m, analysis) in enumerate(zip(markets, analyses, strict=False)):
        if analysis is None:
            continue

        final_action, advisor_approved, advisor_verdict = apply_advisor(
            m, analysis, skip_advisor, advisor_check_fn
        )

        if print_progress:
            _print_market_progress(i, len(markets), m, analysis, final_action,
                                   advisor_approved, advisor_verdict, mode="sim")

        actual_outcome = 1 if m["resolution"] == "YES" else 0
        p_model = analysis.get("p_model", 0)
        brier = (p_model - actual_outcome) ** 2
        brier_scores.append(brier)

        high_price = m.get("high_price")
        if high_price is None:
            high_price = 1.0 if m["resolution"] == "YES" else m.get("yes_final", m["yes_price"])

        ladder_pnl = 0
        ladder_details = []
        if final_action == "SKIP":
            skips += 1
        elif final_action == "BUY":
            ladder_pnl, ladder_details = simulate_fn(
                entry_price=m["yes_price"],
                high_price=high_price,
                resolution=m["resolution"],
            )
            if ladder_pnl > 0:
                wins += 1
                upside_sum += ladder_pnl
                upside_count += 1
                logger.info(
                    f"[BACKTEST-LADDER] {m['slug'][:40]}... "
                    f"weighted_pnl={ladder_pnl:+.2%} "
                    f"rungs={[d['label'] for d in ladder_details]}"
                )
            else:
                losses += 1
                logger.info(
                    f"[BACKTEST-LADDER-LOSS] {m['slug'][:40]}... "
                    f"weighted_pnl={ladder_pnl:+.2%}"
                )

        for c in m.get("clusters", []):
            cluster_stats[c]["total"] += 1
            if final_action == "BUY" and ladder_pnl > 0:
                cluster_stats[c]["wins"] += 1
            elif final_action == "BUY":
                cluster_stats[c]["losses"] += 1

        results.append({
            "slug": m["slug"],
            "question": m["question"],
            "market_price": m["yes_price"],
            "resolution": m["resolution"],
            "p_model": p_model,
            "prob_ratio": analysis.get("prob_ratio", 0),
            "confidence": analysis.get("confidence", 0),
            "signal_score": analysis.get("signal_score", 0),
            "action": final_action,
            "advisor_verdict": advisor_verdict,
            "brier": brier,
            "source_signal": analysis.get("source_signal", "default"),
            "clusters": m.get("clusters", []),
            "simulated_price": m.get("simulated_price", False),
        })

    total_traded = wins + losses
    winrate = wins / total_traded if total_traded > 0 else 0
    avg_brier = sum(brier_scores) / len(brier_scores) if brier_scores else 0
    avg_upside = upside_sum / upside_count if upside_count > 0 else 0

    summary = {
        "total_markets": len(markets),
        "traded": total_traded,
        "skipped": skips,
        "wins": wins,
        "losses": losses,
        "winrate": winrate,
        "brier_score": avg_brier,
        "avg_upside": avg_upside,
    }

    return results, summary, dict(cluster_stats)


def process_live_results(
    markets: list[dict],
    analyses: list[dict | None],
    skip_advisor: bool,
    advisor_check_fn,
    print_progress: bool = True,
) -> tuple[list[dict], dict, dict]:
    """Process live market results. Returns (results, summary, cluster_stats)."""
    results = []
    buys = skips_count = 0
    cluster_stats = defaultdict(lambda: {"buys": 0, "skips": 0})

    for i, (m, analysis) in enumerate(zip(markets, analyses, strict=False)):
        if analysis is None:
            continue

        final_action, advisor_approved, advisor_verdict = apply_advisor(
            m, analysis, skip_advisor, advisor_check_fn
        )

        if print_progress:
            _print_market_progress(i, len(markets), m, analysis, final_action,
                                   advisor_approved, advisor_verdict, mode="live")

        if final_action == "BUY":
            buys += 1
        else:
            skips_count += 1

        for c in m.get("clusters", []):
            if final_action == "BUY":
                cluster_stats[c]["buys"] += 1
            else:
                cluster_stats[c]["skips"] += 1

        results.append({
            "slug": m["slug"],
            "question": m["question"],
            "market_price": m["yes_price"],
            "resolution": None,
            "p_model": analysis.get("p_model", 0),
            "prob_ratio": analysis.get("prob_ratio", 0),
            "confidence": analysis.get("confidence", 0),
            "signal_score": analysis.get("signal_score", 0),
            "action": final_action,
            "advisor_verdict": advisor_verdict,
            "source_signal": analysis.get("source_signal", "default"),
            "clusters": m.get("clusters", []),
            "status": "pending",
            "analyzed_at": datetime.now().isoformat(),
        })

    summary = {
        "total_markets": len(results),
        "buys": buys,
        "skips": skips_count,
        "pending_resolution": len(results),
    }

    return results, summary, dict(cluster_stats)


def print_resolved_report(summary, cluster_stats, dampened_markets, config, output_path):
    """Print formatted report for resolved mode."""
    total_traded = summary["traded"]
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS (resolved markets)")
    print("=" * 60)
    print(f"  Markets analyzed:  {summary['total_markets']}")
    print(f"  Traded:            {total_traded} (skipped: {summary['skipped']})")
    print(f"  Wins / Losses:     {summary['wins']} / {summary['losses']}")
    print(f"  Winrate:           {summary['winrate']:.1%}")
    print(f"  Brier Score (calibrated): {summary['brier_score']:.4f}")
    print(f"  Brier Score (raw):        {summary.get('brier_score_raw', 0):.4f}")
    print(f"  Brier Improvement:        {summary.get('brier_improvement', 0):+.4f}")
    print(f"  Dampened predictions:      {summary.get('dampened_count', 0)}/{summary['total_markets']}")
    if summary.get("avg_upside", 0) > 0:
        print(f"  Avg Upside (wins): {summary['avg_upside']:.2f}x")
    _print_cluster_breakdown(cluster_stats)
    print()
    print(f"  Results saved to: {output_path}")
    print()
    _print_dampened_markets(dampened_markets)
    print("=" * 60)


def print_sim_report(summary, cluster_stats, config, output_path):
    """Print formatted report for sim mode."""
    print("\n" + "=" * 60)
    print("  SIM BACKTEST RESULTS (simulated DOTM prices)")
    print("=" * 60)
    print(f"  Markets analyzed:  {summary['total_markets']}")
    print(f"  Traded:            {summary['traded']} (skipped: {summary['skipped']})")
    print(f"  Wins / Losses:     {summary['wins']} / {summary['losses']}")
    print(f"  Winrate:           {summary['winrate']:.1%}")
    print(f"  Brier Score:       {summary['brier_score']:.4f}")
    if summary.get("avg_upside", 0) > 0:
        print(f"  Avg Upside (wins): {summary['avg_upside']:.2f}x")
    print()
    print("  Cluster breakdown:")
    for cluster, cs in sorted(cluster_stats.items(), key=lambda x: -x[1]["total"]):
        wr = cs["wins"] / (cs["wins"] + cs["losses"]) if (cs["wins"] + cs["losses"]) > 0 else 0
        print(f"    {cluster:20s}: {cs['total']:3d} traded, winrate={wr:.1%}")
    print()
    print(f"  Results saved to: {output_path}")
    print("=" * 60)


def print_live_report(summary, cluster_stats, config, output_path):
    """Print formatted report for live mode."""
    print("\n" + "=" * 60)
    print("  LIVE BACKTEST RESULTS (predictions recorded)")
    print("=" * 60)
    print(f"  Markets analyzed:  {summary['total_markets']}")
    print(f"  BUY signals:       {summary['buys']}")
    print(f"  SKIP signals:      {summary['skips']}")
    print()
    print("  Cluster breakdown:")
    for cluster, cs in sorted(cluster_stats.items(), key=lambda x: -(x[1]["buys"] + x[1]["skips"])):
        total = cs["buys"] + cs["skips"]
        print(f"    {cluster:20s}: {total:3d} total, {cs['buys']:3d} buys")
    print()
    print("  Run '--check' later to resolve pending predictions")
    print(f"  Results saved to: {output_path}")
    print("=" * 60)


def _print_market_progress(i, total, market, analysis, final_action,
                           advisor_approved, advisor_verdict, mode="resolved"):
    """Print progress for a single market during processing."""
    m = market
    if mode == "sim":
        print(f"\n[{i+1}/{total}] {m['question'][:55]}...")
        print(f"  Sim Price: ${m['yes_price']:.3f} | Resolution: {m['resolution']} | Cluster: {m['clusters']}")
    elif mode == "live":
        print(f"\n[{i+1}/{total}] {m['question'][:55]}...")
        print(f"  Price: ${m['yes_price']:.3f} | Vol: ${m['volume']:,.0f} | Cluster: {m['clusters']}")
    else:
        print(f"\n[{i+1}/{total}] {m['question'][:55]}...")
        print(f"  Price: ${m['yes_price']:.3f} | Vol: ${m['volume']:,.0f} | Cluster: {m['clusters']}")
        print(f"  Resolution: {m['resolution']} | Created: {m.get('created_at', '?')[:10]}")
    print(f"  p_model={analysis['p_model']:.1%} | ratio={analysis.get('prob_ratio', 0):.2f}x | action={analysis['action']}")
    if analysis.get("was_dampened"):
        print(f"  [DAMPENED] raw={analysis.get('p_model_raw', 0):.1%} -> calibrated={analysis['p_model']:.1%}")
    if not advisor_approved:
        print(f"  Advisor: {advisor_verdict} (approved={advisor_approved})")
    if final_action != analysis["action"]:
        print("  -> VETOED by advisor")


def _print_cluster_breakdown(cluster_stats):
    """Print cluster breakdown for resolved mode."""
    print()
    print("  Cluster breakdown:")
    for cluster, cs in sorted(cluster_stats.items(), key=lambda x: -x[1]["total"]):
        traded_c = cs["wins"] + cs["losses"]
        wr = cs["wins"] / traded_c if traded_c > 0 else 0
        print(f"    {cluster:20s}: {cs['total']:3d} total, {traded_c:3d} traded, winrate={wr:.1%}")


def _print_dampened_markets(dampened_markets):
    """Print dampened markets breakdown."""
    if not dampened_markets:
        return
    print("  Dampened markets breakdown:")
    print(f"  {'Slug':<45s} {'Price':>7s} {'Raw':>7s} {'Calib':>7s} {'Delta':>7s} {'Cluster':>15s}")
    print(f"  {'-'*45} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*15}")
    for dm in dampened_markets:
        slug_display = dm['slug'][:44]
        print(
            f"  {slug_display:<45s} "
            f"${dm['market_price']:5.3f} "
            f"{dm['p_model_raw']:6.1%} "
            f"{dm['p_model_calibrated']:6.1%} "
            f"{dm['damping_delta']:6.1%} "
            f"{dm['cluster']:>15s}"
        )
    print()
