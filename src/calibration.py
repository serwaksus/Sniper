"""
Isotonic Regression Calibrator for DOTM Sniper
Learns optimal p_model -> p_calibrated mapping from historical data
"""
import os
import sys
import logging
import threading
import numpy as np
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import save_json

logger = logging.getLogger(__name__)

CALIBRATION_MODEL_FILE = "/root/dotm-sniper/calibration_model.json"
MIN_SAMPLES_PER_CLUSTER = 20


class IsotonicCalibrator:
    def __init__(self):
        self.models = {}
        self.is_fitted = False

    def fit(self, hypotheses):
        self.models = {}
        self.is_fitted = False
        cluster_data = {}
        for h in hypotheses:
            if h.get("outcome") not in ("YES", "NO"):
                continue
            if h.get("p_model") is None:
                continue
            primary_cluster = h.get("clusters", ["other"])[0]
            if primary_cluster not in cluster_data:
                cluster_data[primary_cluster] = []
            cluster_data[primary_cluster].append({
                "p_model": h["p_model"],
                "outcome": 1 if h["outcome"] == "YES" else 0
            })

        for cluster, data in cluster_data.items():
            if len(data) < MIN_SAMPLES_PER_CLUSTER:
                logger.info(f"[CALIBRATION] Skipping {cluster}: only {len(data)} samples (need {MIN_SAMPLES_PER_CLUSTER})")
                continue
            X = np.array([d["p_model"] for d in data])
            y = np.array([d["outcome"] for d in data])
            iso = IsotonicRegression(out_of_bounds='clip', increasing=True)
            iso.fit(X, y)
            self.models[cluster] = iso
            logger.info(f"[CALIBRATION] Fitted isotonic model for {cluster}: {len(data)} samples")

        self.is_fitted = len(self.models) > 0
        if self.is_fitted:
            logger.info(f"[CALIBRATION] Total models fitted: {len(self.models)} clusters")

    def predict(self, p_model, cluster="other"):
        if not self.is_fitted:
            return p_model
        if cluster in self.models:
            model = self.models[cluster]
        elif "other" in self.models:
            model = self.models["other"]
        else:
            return p_model
        if isinstance(model, dict):
            X = model["X_thresholds_"]
            y = model["y_thresholds_"]
            p_calibrated = float(np.interp(p_model, X, y))
        else:
            p_calibrated = float(model.predict([p_model])[0])
        p_calibrated = max(0.0, min(1.0, p_calibrated))
        return p_calibrated

    def save(self, path=CALIBRATION_MODEL_FILE):
        if not self.is_fitted:
            logger.warning("[CALIBRATION] No models to save")
            return
        model_data = {}
        for cluster, iso in self.models.items():
            if isinstance(iso, dict):
                model_data[cluster] = iso
            else:
                model_data[cluster] = {
                    "X_thresholds_": iso.X_thresholds_.tolist(),
                    "y_thresholds_": iso.y_thresholds_.tolist(),
                }
        save_json(path, model_data)
        logger.info(f"[CALIBRATION] Saved {len(model_data)} models to {path}")

    def load(self, path=CALIBRATION_MODEL_FILE):
        if not os.path.exists(path):
            logger.info(f"[CALIBRATION] No calibration model found at {path}")
            return False
        try:
            from utils import load_json as _load_json
            model_data = _load_json(path, None)
            if model_data is None:
                logger.error(f"[CALIBRATION] Failed to load model from {path}")
                return False
            self.models = {}
            for cluster, params in model_data.items():
                X = np.array(params["X_thresholds_"])
                y = np.array(params["y_thresholds_"])
                self.models[cluster] = {"X_thresholds_": X, "y_thresholds_": y}
            self.is_fitted = len(self.models) > 0
            logger.info(f"[CALIBRATION] Loaded {len(self.models)} models from {path}")
            return True
        except Exception as e:
            logger.error(f"[CALIBRATION] Failed to load models: {e}")
            return False


_calibrator_lock = threading.RLock()
_calibrator_instance = None


def get_calibrator():
    global _calibrator_instance
    if _calibrator_instance is None:
        with _calibrator_lock:
            if _calibrator_instance is None:
                _calibrator_instance = IsotonicCalibrator()
                _calibrator_instance.load()
    return _calibrator_instance
