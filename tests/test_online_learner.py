from __future__ import annotations
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(autouse=True)
def _tmp_model_dir(tmp_path, monkeypatch):
    import online_learner as ol
    model_dir = str(tmp_path / "ml_models")
    monkeypatch.setattr(ol, "MODEL_DIR", model_dir)
    monkeypatch.setattr(ol, "SGD_MODEL_PATH", os.path.join(model_dir, "sgd_predictor.pkl"))
    monkeypatch.setattr(ol, "SGD_SCALER_PATH", os.path.join(model_dir, "sgd_scaler.pkl"))
    monkeypatch.setattr(ol, "SGD_STATE_PATH", os.path.join(model_dir, "sgd_state.json"))
    ol._learner = None
    yield
    ol._learner = None


def _make_market(**overrides):
    base = {
        "p_model": 0.10,
        "confidence": 0.72,
        "metaculus_prob": 0.08,
        "price": 0.05,
        "market_price": 0.05,
        "buzz_score": 5,
        "volume_ratio": 1.5,
        "end_date_iso": (datetime.now() + timedelta(days=30)).isoformat(),
    }
    base.update(overrides)
    return base


class TestFeatureExtraction:
    def test_all_features_present(self):
        from online_learner import OnlineSignalLearner
        learner = OnlineSignalLearner()
        X = learner._extract_features(_make_market())
        assert X.shape == (1, 8)
        assert not np.any(np.isnan(X))

    def test_missing_fields_default(self):
        from online_learner import OnlineSignalLearner
        learner = OnlineSignalLearner()
        X = learner._extract_features({})
        assert X.shape == (1, 8)
        assert float(X[0, 6]) == 0.0  # price defaults to 0
        assert float(X[0, 5]) == 30.0  # tte defaults to 30 days

    def test_metaculus_gap_calculation(self):
        from online_learner import OnlineSignalLearner
        learner = OnlineSignalLearner()
        X = learner._extract_features(_make_market(p_model=0.15, metaculus_prob=0.08))
        assert float(X[0, 2]) == pytest.approx(0.07, abs=0.001)


class TestPartialFit:
    def test_first_sample_initializes(self):
        from online_learner import OnlineSignalLearner
        learner = OnlineSignalLearner()
        assert not learner.is_initialized
        learner.partial_fit(_make_market(), target=1.0)
        assert learner.is_initialized
        assert learner.n_samples_seen == 1

    def test_incremental_update(self):
        from online_learner import OnlineSignalLearner
        learner = OnlineSignalLearner()
        learner.partial_fit(_make_market(p_model=0.10), target=1.0)
        learner.partial_fit(_make_market(p_model=0.03), target=0.0)
        assert learner.n_samples_seen == 2

    def test_save_and_load(self, tmp_path):
        from online_learner import OnlineSignalLearner
        learner = OnlineSignalLearner()
        learner.partial_fit(_make_market(), target=1.0)
        learner.partial_fit(_make_market(p_model=0.03), target=0.0)
        learner.save()

        learner2 = OnlineSignalLearner()
        loaded = learner2.load()
        assert loaded
        assert learner2.n_samples_seen == 2
        assert learner2.is_initialized


class TestPrediction:
    def test_not_initialized_returns_none(self):
        from online_learner import OnlineSignalLearner
        learner = OnlineSignalLearner()
        assert learner.predict_probability(_make_market()) is None

    def test_after_training_returns_probability(self):
        from online_learner import OnlineSignalLearner
        learner = OnlineSignalLearner()
        for i in range(10):
            target = 1.0 if i % 2 == 0 else 0.0
            learner.partial_fit(_make_market(p_model=0.10 + i * 0.01), target=target)
        prob = learner.predict_probability(_make_market(p_model=0.15))
        assert prob is not None
        assert 0.0 <= prob <= 1.0

    def test_high_p_model_predicts_higher(self):
        from online_learner import OnlineSignalLearner
        learner = OnlineSignalLearner()
        for _ in range(5):
            for _ in range(20):
                learner.partial_fit(_make_market(p_model=0.20), target=1.0)
            for _ in range(20):
                learner.partial_fit(_make_market(p_model=0.03), target=0.0)
        high = learner.predict_probability(_make_market(p_model=0.20))
        low = learner.predict_probability(_make_market(p_model=0.03))
        assert high is not None and low is not None
        assert high > low


class TestFeatureImportance:
    def test_importance_before_init(self):
        from online_learner import OnlineSignalLearner
        learner = OnlineSignalLearner()
        imp = learner.get_feature_importance()
        assert len(imp) == 8
        assert all(v == 0.0 for _, v in imp)

    def test_importance_after_training(self):
        from online_learner import OnlineSignalLearner
        learner = OnlineSignalLearner()
        for i in range(10):
            learner.partial_fit(_make_market(p_model=0.10 + i * 0.02), target=float(i % 2))
        imp = learner.get_feature_importance()
        assert len(imp) == 8
        assert all(v >= 0 for _, v in imp)

    def test_importance_sorted_descending(self):
        from online_learner import OnlineSignalLearner
        learner = OnlineSignalLearner()
        for i in range(10):
            learner.partial_fit(_make_market(p_model=0.10 + i * 0.02), target=float(i % 2))
        imp = learner.get_feature_importance()
        values = [v for _, v in imp]
        assert values == sorted(values, reverse=True)


class TestDriftDetection:
    def test_drift_insufficient_history(self):
        from online_learner import OnlineSignalLearner
        learner = OnlineSignalLearner()
        for _ in range(5):
            learner.partial_fit(_make_market(), target=1.0)
        drift = learner.detect_feature_drift()
        assert drift == {}

    def test_drift_logged(self):
        from online_learner import OnlineSignalLearner
        learner = OnlineSignalLearner()
        now = datetime.now()
        for _ in range(20):
            learner.partial_fit(_make_market(p_model=0.10), target=1.0)
        for j in range(15):
            learner._importance_history[-(j + 1)]["ts"] = (now - timedelta(days=60)).isoformat()
        drift = learner.detect_feature_drift()
        assert isinstance(drift, dict)


class TestModuleFunctions:
    def test_get_learner_returns_singleton(self):
        import online_learner as ol
        l1 = ol.get_learner()
        l2 = ol.get_learner()
        assert l1 is l2

    def test_online_predict_no_model(self):
        import online_learner as ol
        result = ol.online_predict(_make_market())
        assert result is None

    def test_online_train_and_predict(self):
        import online_learner as ol
        ol.online_train_on_resolved(_make_market(p_model=0.15), resolved_yes=True)
        ol.online_train_on_resolved(_make_market(p_model=0.03), resolved_yes=False)
        prob = ol.online_predict(_make_market(p_model=0.10))
        assert prob is not None

    def test_get_online_model_info(self):
        import online_learner as ol
        info = ol.get_online_model_info()
        assert "initialized" in info
        assert "n_samples_seen" in info
        assert "feature_importance" in info
