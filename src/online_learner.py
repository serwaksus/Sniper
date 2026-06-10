"""
Online Signal Learner — incremental SGD-based learning on resolved markets.

Complements the batch LightGBM predictor:
- LightGBM: retrained periodically, needs 50+ samples, high accuracy
- SGD Online: updates on every resolved market, works from sample 1,
  detects feature drift, adapts to changing market conditions

Blending: when both models are available, final p = 0.3*SGD + 0.3*LGBM + 0.4*LLM
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml_models")
SGD_MODEL_PATH = os.path.join(MODEL_DIR, "sgd_predictor.pkl")
SGD_SCALER_PATH = os.path.join(MODEL_DIR, "sgd_scaler.pkl")
SGD_STATE_PATH = os.path.join(MODEL_DIR, "sgd_state.json")
DRIFT_LOG_PATH = os.path.join(MODEL_DIR, "drift_log.json")

FEATURE_NAMES = [
    "llm_probability",
    "llm_confidence",
    "metaculus_gap",
    "social_buzz",
    "volume_ratio",
    "time_to_expiry",
    "price",
    "prob_ratio",
]

INITIAL_ETA = 0.01
DRIFT_THRESHOLD = 0.50
IMPORTANCE_WINDOW_RECENT = 30
IMPORTANCE_WINDOW_HISTORICAL = 90


class OnlineSignalLearner:
    def __init__(self) -> None:
        self.model = SGDRegressor(
            learning_rate="adaptive",
            eta0=INITIAL_ETA,
            penalty="l2",
            alpha=1e-4,
            max_iter=1,
            warm_start=True,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.is_initialized = False
        self.n_samples_seen = 0
        self._importance_history: list[dict[str, Any]] = []
        self._drift_log: list[dict[str, Any]] = []

    def _extract_features(self, market: dict[str, Any]) -> np.ndarray:
        llm_prob = float(market.get("p_model", 0))
        llm_conf = float(market.get("confidence", 0))
        meta_prob = float(market.get("metaculus_prob", 0) or 0)
        price = float(market.get("price", 0) or market.get("market_price", 0))
        meta_gap = llm_prob - meta_prob
        buzz = float(market.get("buzz_score", 0) or 0)
        vol_ratio = float(market.get("volume_ratio", 1.0) or 1.0)

        expiry_str = market.get("end_date_iso", "") or market.get("endDate", "")
        tte = 30.0
        if expiry_str:
            try:
                expiry = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
                delta = expiry - datetime.now(expiry.tzinfo)
                tte = max(delta.total_seconds() / 86400, 0)
            except Exception:
                pass

        prob_ratio = llm_prob / max(price, 0.001)

        return np.array([[
            llm_prob, llm_conf, meta_gap, buzz,
            vol_ratio, tte, price, prob_ratio,
        ]], dtype=np.float64)

    def partial_fit(self, market: dict[str, Any], target: float) -> None:
        X = self._extract_features(market)
        y = np.array([target], dtype=np.float64)

        if not self.is_initialized:
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.model.fit(X_scaled, y)
            self.is_initialized = True
        else:
            self.scaler.partial_fit(X)
            X_scaled = self.scaler.transform(X)
            self.model.partial_fit(X_scaled, y)

        self.n_samples_seen += 1

        importance = self.get_feature_importance()
        self._importance_history.append({
            "ts": datetime.now().isoformat(),
            "n_samples": self.n_samples_seen,
            **dict(importance),
        })

        if self.n_samples_seen % 5 == 0:
            self._log_importance(importance)

    def predict_probability(self, market: dict[str, Any]) -> float | None:
        if not self.is_initialized:
            return None

        try:
            X = self._extract_features(market)
            X_scaled = self.scaler.transform(X)
            raw = float(self.model.predict(X_scaled)[0])
            prob = 1.0 / (1.0 + np.exp(-raw))
            return max(0.0, min(1.0, float(prob)))
        except Exception as e:
            logger.debug(f"[ONLINE-ML] predict failed: {e}")
            return None

    def get_feature_importance(self) -> list[tuple[str, float]]:
        if not self.is_initialized:
            return [(name, 0.0) for name in FEATURE_NAMES]
        coef = self.model.coef_
        importance = sorted(
            zip(FEATURE_NAMES, np.abs(coef).tolist(), strict=True),
            key=lambda x: x[1],
            reverse=True,
        )
        return importance

    def detect_feature_drift(self) -> dict[str, float]:
        if len(self._importance_history) < 10:
            return {}

        now = datetime.now()
        recent_cutoff = (now - __import__("datetime").timedelta(days=IMPORTANCE_WINDOW_RECENT)).isoformat()
        hist_cutoff = (now - __import__("datetime").timedelta(days=IMPORTANCE_WINDOW_HISTORICAL)).isoformat()

        recent = [h for h in self._importance_history if h["ts"] >= recent_cutoff]
        historical = [h for h in self._importance_history if h["ts"] >= hist_cutoff and h["ts"] < recent_cutoff]

        if not recent or not historical:
            return {}

        drift: dict[str, float] = {}
        for feat in FEATURE_NAMES:
            recent_avg = np.mean([h.get(feat, 0) for h in recent])
            hist_avg = np.mean([h.get(feat, 0) for h in historical])
            if hist_avg > 0.001:
                change = (recent_avg - hist_avg) / hist_avg
                if abs(change) > DRIFT_THRESHOLD:
                    drift[feat] = round(change, 3)

        if drift:
            self._drift_log.append({
                "ts": datetime.now().isoformat(),
                "n_samples": self.n_samples_seen,
                "drift": drift,
            })
            logger.warning(f"[ONLINE-ML] Feature drift detected: {drift}")

        return drift

    def _log_importance(self, importance: list[tuple[str, float]]) -> None:
        top = importance[:3]
        parts = ", ".join(f"{name}={imp:.4f}" for name, imp in top)
        logger.info(f"[ONLINE-ML] Top features (n={self.n_samples_seen}): {parts}")

    def save(self) -> None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        if self.is_initialized:
            with open(SGD_MODEL_PATH, "wb") as f:
                pickle.dump(self.model, f)
            with open(SGD_SCALER_PATH, "wb") as f:
                pickle.dump(self.scaler, f)
        state = {
            "is_initialized": self.is_initialized,
            "n_samples_seen": self.n_samples_seen,
            "importance_history": self._importance_history[-100:],
            "drift_log": self._drift_log[-50:],
            "saved_at": datetime.now().isoformat(),
        }
        with open(SGD_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)

    def load(self) -> bool:
        try:
            if os.path.exists(SGD_MODEL_PATH) and os.path.exists(SGD_SCALER_PATH):
                with open(SGD_MODEL_PATH, "rb") as f:
                    self.model = pickle.load(f)
                with open(SGD_SCALER_PATH, "rb") as f:
                    self.scaler = pickle.load(f)
                self.is_initialized = True
            if os.path.exists(SGD_STATE_PATH):
                with open(SGD_STATE_PATH) as f:
                    state = json.load(f)
                self.n_samples_seen = state.get("n_samples_seen", 0)
                self._importance_history = state.get("importance_history", [])
                self._drift_log = state.get("drift_log", [])
            logger.info(f"[ONLINE-ML] Loaded: initialized={self.is_initialized}, n_samples={self.n_samples_seen}")
            return self.is_initialized
        except Exception as e:
            logger.warning(f"[ONLINE-ML] Load failed: {e}")
            return False


_learner: OnlineSignalLearner | None = None


def get_learner() -> OnlineSignalLearner:
    global _learner
    if _learner is None:
        _learner = OnlineSignalLearner()
        _learner.load()
    return _learner


def online_predict(market: dict[str, Any]) -> float | None:
    learner = get_learner()
    return learner.predict_probability(market)


def online_train_on_resolved(market: dict[str, Any], resolved_yes: bool) -> None:
    learner = get_learner()
    target = 1.0 if resolved_yes else 0.0
    learner.partial_fit(market, target)
    learner.save()
    logger.info(f"[ONLINE-ML] Trained on resolved: {str(market.get('slug', market.get('p_model', '?')))[:40]}... → {'YES' if resolved_yes else 'NO'} (n={learner.n_samples_seen})")


def online_detect_drift() -> dict[str, float]:
    learner = get_learner()
    return learner.detect_feature_drift()


def get_online_model_info() -> dict[str, Any]:
    learner = get_learner()
    importance = learner.get_feature_importance()
    return {
        "initialized": learner.is_initialized,
        "n_samples_seen": learner.n_samples_seen,
        "feature_importance": dict(importance),
        "drift_log": learner._drift_log[-5:],
    }
