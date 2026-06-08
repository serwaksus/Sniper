#!/usr/bin/env python3
"""
Backtest v2 Portfolio Tracker — tracks positions, equity curve, drawdown, Sharpe.
"""
import logging

logger = logging.getLogger(__name__)

MAX_POSITIONS = 50
MAX_CLUSTER_PCT = 0.40
BASE_POS_PCT = 0.02
MAX_POS_PCT = 0.10
FRACTIONAL_KELLY = 0.25
TP_LADDER = [(0.50, 0.75), (0.30, 0.85)]
TRAILING_ACTIVATION = 0.30
TRAILING_STOP = 0.15
TAKE_PROFIT = 1.50
CONVERGENCE_TP = 0.85


def get_tier(balance: float) -> dict:
    if balance < 2000:
        return {"kelly": 0.25, "base_pct": 0.03, "other_pct": 0.045,
                "max_pct": 0.10, "max_positions": 12, "tier": "micro"}
    elif balance < 10000:
        return {"kelly": 0.30, "base_pct": 0.04, "other_pct": 0.05,
                "max_pct": 0.12, "max_positions": 20, "tier": "growth"}
    elif balance < 50000:
        return {"kelly": 0.35, "base_pct": 0.04, "other_pct": 0.06,
                "max_pct": 0.15, "max_positions": 25, "tier": "established"}
    else:
        return {"kelly": 0.40, "base_pct": 0.05, "other_pct": 0.07,
                "max_pct": 0.15, "max_positions": 30, "tier": "scale"}


class Position:
    def __init__(self, slug, question, outcome, entry_price, shares, cost,
                 liquidity, cluster="other", p_model=0.0, created_at=""):
        self.slug = slug
        self.question = question
        self.outcome = outcome
        self.entry_price = entry_price
        self.shares = shares
        self.cost = cost
        self.liquidity = liquidity
        self.cluster = cluster
        self.p_model = p_model
        self.created_at = created_at
        self.high_price = entry_price
        self.trailing_on = False
        self.stop_loss = 0.0
        self.trailing_confirmed = False
        self.tp_ladder_filled = False
        self.tp_ladder_results = None
        self.shares_after_tp = shares

    def current_value(self, market_price: float) -> float:
        return self.shares_after_tp * market_price

    def pnl_pct(self, market_price: float) -> float:
        if self.entry_price <= 0:
            return 0
        return (market_price - self.entry_price) / self.entry_price

    def pnl_abs(self, market_price: float) -> float:
        return self.current_value(market_price) - self.cost


class PortfolioTracker:
    def __init__(self, starting_balance: float = 500.0, profile: dict | None = None):
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.positions: dict[str, Position] = {}
        self.equity_curve: list[dict] = []
        self.trades: list[dict] = []
        self.rejected_trades: list[dict] = []
        self.cluster_exposure: dict[str, float] = {}
        self.step = 0
        self._profile = profile or {}

    def equity(self, prices: dict[str, float]) -> float:
        pos_value = sum(
            pos.current_value(prices.get(pos.slug, 0))
            for pos in self.positions.values()
        )
        return self.balance + pos_value

    def record_equity(self, prices: dict[str, float], timestamp: str = ""):
        eq = self.equity(prices)
        peak = max((e["equity"] for e in self.equity_curve), default=eq)
        drawdown = (eq - peak) / peak if peak > 0 else 0

        self.equity_curve.append({
            "step": self.step,
            "timestamp": timestamp,
            "equity": eq,
            "balance": self.balance,
            "positions": len(self.positions),
            "drawdown": drawdown,
        })

    def can_open_position(self, cluster: str, amount: float) -> tuple[bool, str]:
        tier = get_tier(self.balance + sum(p.cost for p in self.positions.values()))
        max_pos = self._profile.get("max_positions", tier["max_positions"])
        if len(self.positions) >= max_pos:
            return False, f"max_positions={max_pos}"

        max_cluster = self._profile.get("max_cluster_pct", MAX_CLUSTER_PCT)
        cluster_exposure = self.cluster_exposure.get(cluster, 0) + amount
        total_equity = self.balance + sum(
            p.cost for p in self.positions.values()
        )
        if total_equity > 0 and cluster_exposure / total_equity > max_cluster:
            return False, f"cluster {cluster} exposure={cluster_exposure/total_equity:.1%} > {max_cluster:.0%}"

        if amount > self.balance:
            return False, f"amount=${amount:.2f} > balance=${self.balance:.2f}"

        return True, "ok"

    def open_position(self, slug: str, question: str, outcome: str,
                      entry_price: float, shares: float, cost: float,
                      liquidity: float, cluster: str = "other",
                      p_model: float = 0.0, created_at: str = "",
                      fee: float = 0.0) -> bool:
        total_cost = cost + fee
        if total_cost > self.balance:
            return False
        self.balance -= total_cost
        pos = Position(slug, question, outcome, entry_price, shares, total_cost,
                       liquidity, cluster, p_model, created_at)
        self.positions[slug] = pos
        self.cluster_exposure[cluster] = self.cluster_exposure.get(cluster, 0) + total_cost
        return True

    def close_position(self, slug: str, proceeds: float, reason: str,
                       market_price: float = 0, fee: float = 0):
        if slug not in self.positions:
            return

        pos = self.positions[slug]
        net_proceeds = proceeds - fee
        pnl_abs = net_proceeds - pos.cost
        pnl_pct = (net_proceeds - pos.cost) / pos.cost if pos.cost > 0 else 0

        self.balance += net_proceeds
        cluster = pos.cluster
        self.cluster_exposure[cluster] = max(0, self.cluster_exposure.get(cluster, 0) - pos.cost)

        self.trades.append({
            "slug": slug,
            "outcome": pos.outcome,
            "entry_price": pos.entry_price,
            "shares": pos.shares,
            "cost": pos.cost,
            "proceeds": net_proceeds,
            "fee": fee,
            "pnl_abs": pnl_abs,
            "pnl_pct": pnl_pct,
            "reason": reason,
            "cluster": cluster,
        })

        del self.positions[slug]

    def position_size(self, p_model: float, market_price: float, cluster: str = "other",
                      best_ask: float | None = None) -> float:
        tier = get_tier(self.balance + sum(p.cost for p in self.positions.values()))
        effective_price = best_ask if best_ask is not None else market_price
        if effective_price <= 0.001:
            return 0

        b = (1 - effective_price) / effective_price
        p = p_model
        q = 1 - p
        kelly_full = (b * p - q) / b

        if kelly_full <= 0:
            return 0

        kelly_frac = self._profile.get("kelly_fraction", tier["kelly"])
        base_pct = self._profile.get("base_pct", tier["base_pct"])
        cap_pct = self._profile.get("max_pct", MAX_POS_PCT)

        kelly_with_conf = kelly_full * kelly_frac

        if cluster.startswith("other"):
            cap = base_pct
        else:
            cap = cap_pct

        size_pct = min(kelly_with_conf, cap)
        kelly_dollars = round(self.balance * size_pct)

        if kelly_dollars < 5:
            return 0

        kelly_dollars = min(kelly_dollars, round(self.balance * cap_pct))
        return kelly_dollars

    def update_trailing(self, slug: str, market_price: float):
        if slug not in self.positions:
            return
        pos = self.positions[slug]
        pos.high_price = max(pos.high_price, market_price)

        if pos.high_price > pos.entry_price * (1 + TRAILING_ACTIVATION):
            pos.trailing_on = True
            pos.stop_loss = pos.high_price * (1 - TRAILING_STOP)

    def summary(self) -> dict:
        if not self.trades:
            return {"total_trades": 0}

        wins = [t for t in self.trades if t["pnl_abs"] > 0]
        losses = [t for t in self.trades if t["pnl_abs"] <= 0]

        avg_win = sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0
        total_pnl = sum(t["pnl_abs"] for t in self.trades)

        max_dd = min((e["drawdown"] for e in self.equity_curve), default=0)

        returns = []
        for i in range(1, len(self.equity_curve)):
            prev = self.equity_curve[i-1]["equity"]
            curr = self.equity_curve[i]["equity"]
            if prev > 0:
                returns.append(curr / prev - 1)

        sharpe = 0
        if returns:
            avg_ret = sum(returns) / len(returns)
            var = sum((r - avg_ret) ** 2 for r in returns) / len(returns)
            std = var ** 0.5
            if std > 0:
                n_steps = len(self.equity_curve)
                n_days = max(1, n_steps)
                annualization = min((365 / n_days) ** 0.5, 252 ** 0.5)
                sharpe = avg_ret / std * annualization

        reasons = {}
        for t in self.trades:
            r = t["reason"]
            reasons[r] = reasons.get(r, 0) + 1

        rejected_reasons = {}
        for r in self.rejected_trades:
            reason = r.get("reason", "unknown")
            rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

        return {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.trades) if self.trades else 0,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl / self.starting_balance,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "final_equity": self.equity_curve[-1]["equity"] if self.equity_curve else self.starting_balance,
            "exit_reasons": reasons,
            "rejected_trades": len(self.rejected_trades),
            "rejected_reasons": rejected_reasons,
        }
