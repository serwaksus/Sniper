"""
LightGBM-based p_model predictor.
Trains on resolved markets to predict actual probability of YES resolution.

Features (10):
  1. llm_prob      — DeepSeek probability estimate
  2. llm_confidence — DeepSeek confidence
  3. metaculus_prob — Metaculus crowd forecast
  4. metaculus_n    — Number of Metaculus forecasters
  5. buzz_score     — Social buzz score (0-20)
  6. ob_imbalance   — Order book bid/ask imbalance (-1..1)
  7. smart_money    — Smart money detected (0/1)
  8. time_to_expiry — Days until market close
  9. price          — Current market price
 10. prob_ratio     — llm_prob / price

Target: 1 if resolved YES, 0 if resolved NO or SOLD at loss

Minimum samples for training: 50 (will use fallback before that)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MIN_SAMPLES = 50
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml_models")
MODEL_PATH = os.path.join(MODEL_DIR, "predictor.txt")
FEATURE_NAMES = [
    "llm_prob", "llm_confidence", "metaculus_prob", "metaculus_n",
    "buzz_score", "ob_imbalance", "smart_money", "time_to_expiry",
    "price", "prob_ratio",
]


def _ensure_dir() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)


def build_features(market: dict[str, Any]) -> np.ndarray:
    """Build feature vector from market data. Returns shape (10,) array."""
    llm_prob = float(market.get("p_model", 0))
    llm_conf = float(market.get("confidence", 0))
    meta_prob = float(market.get("metaculus_prob", 0) or 0)
    meta_n = float(market.get("metaculus_n", 0) or 0)
    buzz = float(market.get("buzz_score", 0) or 0)
    imbalance = float(market.get("ob_imbalance", 0) or 0)
    sm = 1.0 if market.get("smart_money_detected") else 0.0

    expiry_str = market.get("end_date_iso", "") or market.get("endDate", "")
    tte = 30.0
    if expiry_str:
        try:
            expiry = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
            delta = expiry - datetime.now(expiry.tzinfo)
            tte = max(delta.total_seconds() / 86400, 0)
        except Exception:
            pass

    price = float(market.get("price", 0) or market.get("market_price", 0))
    prob_ratio = llm_prob / max(price, 0.001)

    return np.array([
        llm_prob, llm_conf, meta_prob, meta_n,
        buzz, imbalance, sm, tte, price, prob_ratio,
    ], dtype=np.float32)


def train_model(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Train LightGBM on resolved market samples.
    Each sample must have: features (dict with keys from FEATURE_NAMES) + target (0/1).
    Returns metrics dict or None if insufficient data.
    """
    if len(samples) < MIN_SAMPLES:
        logger.info(f"[ML] Need {MIN_SAMPLES} samples, got {len(samples)} — skipping training")
        return None

    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, brier_score_loss

    _ensure_dir()

    X = np.array([build_features(s) for s in samples], dtype=np.float32)
    y = np.array([int(s.get("target", 0)) for s in samples], dtype=np.float32)

    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "n_jobs": 1,
        "learning_rate": 0.05,
        "num_leaves": 16,
        "max_depth": 4,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "seed": 42,
    }

    n = len(X)
    split = int(n * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )

    y_pred = model.predict(X_val)
    y_pred_binary = (y_pred >= 0.5).astype(int)

    brier = brier_score_loss(y_val, y_pred)
    acc = accuracy_score(y_val, y_pred_binary)

    importance = dict(zip(FEATURE_NAMES, model.feature_importance().tolist(), strict=True))

    model.save_model(MODEL_PATH)

    metrics = {
        "n_samples": n,
        "n_train": split,
        "n_val": n - split,
        "brier_score": round(brier, 4),
        "accuracy": round(acc, 4),
        "feature_importance": importance,
        "trained_at": datetime.now().isoformat(),
    }

    logger.info(f"[ML] Model trained: n={n}, brier={brier:.4f}, acc={acc:.4f}")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        logger.info(f"[ML]   {feat}: {imp}")

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def predict(market: dict[str, Any]) -> tuple[float, bool]:
    """
    Predict probability of YES resolution.
    Returns (p_predicted, used_model).
    If model not available, returns (0.0, False).
    """
    if not os.path.exists(MODEL_PATH):
        return 0.0, False

    try:
        import lightgbm as lgb

        model = lgb.Booster(model_file=MODEL_PATH)
        features = build_features(market)
        features = np.nan_to_num(features, nan=0.0).reshape(1, -1)
        pred = float(model.predict(features)[0])
        return max(0.0, min(1.0, pred)), True
    except Exception as e:
        logger.warning(f"[ML] Prediction failed: {e}")
        return 0.0, False


def collect_training_samples() -> list[dict[str, Any]]:
    """
    Collect resolved market data from hypotheses + positions for training.
    Returns list of dicts with features + target.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from db import _get_conn

    conn = _get_conn()

    rows = conn.execute("""
        SELECT slug, data FROM hypotheses
        WHERE json_extract(data, '$.resolved') = 1
    """).fetchall()

    samples = []
    for row in rows:
        d = json.loads(row["data"])
        actual = d.get("actual_outcome")
        if actual is None:
            outcome = d.get("outcome", "")
            if outcome == "YES":
                actual = "yes"
            elif outcome == "NO":
                actual = "no"
            elif outcome == "SOLD":
                pnl = d.get("pnl_pct", 0)
                actual = "yes" if pnl and pnl > 0 else "no"
            else:
                continue

        target = 1 if actual == "yes" else 0

        samples.append({
            "p_model": d.get("p_model", 0),
            "confidence": d.get("confidence", 0),
            "metaculus_prob": d.get("metaculus_prob", 0),
            "metaculus_n": d.get("metaculus_n", 0),
            "buzz_score": d.get("buzz_score", 0),
            "ob_imbalance": 0,
            "smart_money_detected": False,
            "end_date_iso": d.get("end_date_iso", ""),
            "price": d.get("market_price", d.get("entry_price", 0)),
            "target": target,
        })

    logger.info(f"[ML] Collected {len(samples)} training samples")
    return samples


def train_if_ready() -> dict[str, Any] | None:
    """Collect samples and train if enough data. Called periodically."""
    samples = collect_training_samples()
    if len(samples) < MIN_SAMPLES:
        return None
    return train_model(samples)


def get_model_info() -> dict[str, Any]:
    """Return model status info for health checks and dashboard."""
    model_exists = os.path.exists(MODEL_PATH)
    metrics: dict[str, Any] = {}
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
        except Exception:
            pass

    return {
        "model_available": model_exists,
        "min_samples": MIN_SAMPLES,
        "current_samples": len(collect_training_samples()),
        "metrics": metrics,
    }
