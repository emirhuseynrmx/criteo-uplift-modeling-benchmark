from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold

from criteo_uplift_benchmark.config import BenchmarkConfig
from criteo_uplift_benchmark.schemas import UpliftPrediction


def make_classifier(config: BenchmarkConfig) -> CalibratedClassifierCV:
    base = LGBMClassifier(**config.lgbm.classifier_params())
    return CalibratedClassifierCV(
        base,
        method="isotonic",
        cv=config.runtime.calibration_cv,
    )


def make_regressor(config: BenchmarkConfig) -> LGBMRegressor:
    return LGBMRegressor(**config.lgbm.regressor_params())


class SLearner:
    name = "S-Learner"

    def __init__(self, config: BenchmarkConfig) -> None:
        self.model = make_classifier(config)

    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> SLearner:
        self.model.fit(np.column_stack([X, t]), y)
        return self

    def predict(self, X: np.ndarray) -> UpliftPrediction:
        ones = np.ones((len(X), 1), dtype=np.float32)
        zeros = np.zeros((len(X), 1), dtype=np.float32)
        p1 = self.model.predict_proba(np.column_stack([X, ones]))[:, 1]
        p0 = self.model.predict_proba(np.column_stack([X, zeros]))[:, 1]
        return UpliftPrediction(uplift=p1 - p0, prob_t1=p1, prob_t0=p0)


class TLearner:
    name = "T-Learner"

    def __init__(self, config: BenchmarkConfig) -> None:
        self.m1 = make_classifier(config)
        self.m0 = make_classifier(config)

    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> TLearner:
        treated = t == 1
        control = t == 0
        self.m1.fit(X[treated], y[treated])
        self.m0.fit(X[control], y[control])
        return self

    def predict(self, X: np.ndarray) -> UpliftPrediction:
        p1 = self.m1.predict_proba(X)[:, 1]
        p0 = self.m0.predict_proba(X)[:, 1]
        return UpliftPrediction(uplift=p1 - p0, prob_t1=p1, prob_t0=p0)


class XLearner:
    name = "X-Learner"

    def __init__(self, config: BenchmarkConfig) -> None:
        self.m1 = make_classifier(config)
        self.m0 = make_classifier(config)
        self.tau1 = make_regressor(config)
        self.tau0 = make_regressor(config)
        self.propensity = make_classifier(config)

    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> XLearner:
        treated = t == 1
        control = t == 0
        X1, y1 = X[treated], y[treated].astype(float)
        X0, y0 = X[control], y[control].astype(float)

        self.m1.fit(X1, y1)
        self.m0.fit(X0, y0)

        d1 = y1 - self.m0.predict_proba(X1)[:, 1]
        d0 = self.m1.predict_proba(X0)[:, 1] - y0
        self.tau1.fit(X1, d1)
        self.tau0.fit(X0, d0)
        self.propensity.fit(X, t)
        return self

    def predict(self, X: np.ndarray) -> UpliftPrediction:
        tau1 = self.tau1.predict(X)
        tau0 = self.tau0.predict(X)
        g = np.clip(self.propensity.predict_proba(X)[:, 1], 0.01, 0.99)
        p1 = self.m1.predict_proba(X)[:, 1]
        p0 = self.m0.predict_proba(X)[:, 1]
        return UpliftPrediction(uplift=g * tau0 + (1 - g) * tau1, prob_t1=p1, prob_t0=p0)


class DRLearner:
    name = "DR-Learner"

    def __init__(self, config: BenchmarkConfig, n_folds: int = 5) -> None:
        self.config = config
        self.n_folds = n_folds
        self.tau_model = make_regressor(config)
        self._m1_final = make_classifier(config)
        self._m0_final = make_classifier(config)

    def _make_nuisance(self) -> dict[str, CalibratedClassifierCV]:
        return {
            "m1": make_classifier(self.config),
            "m0": make_classifier(self.config),
            "e": make_classifier(self.config),
        }

    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> DRLearner:
        n = len(X)
        mu1_oof = np.zeros(n)
        mu0_oof = np.zeros(n)
        e_oof = np.zeros(n)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.config.split.seed)
        for train_idx, val_idx in kf.split(X):
            nuisance = self._make_nuisance()
            Xtr, ytr, ttr = X[train_idx], y[train_idx], t[train_idx]
            treated = ttr == 1
            control = ttr == 0

            nuisance["m1"].fit(Xtr[treated], ytr[treated])
            nuisance["m0"].fit(Xtr[control], ytr[control])
            nuisance["e"].fit(Xtr, ttr)

            mu1_oof[val_idx] = nuisance["m1"].predict_proba(X[val_idx])[:, 1]
            mu0_oof[val_idx] = nuisance["m0"].predict_proba(X[val_idx])[:, 1]
            e_oof[val_idx] = nuisance["e"].predict_proba(X[val_idx])[:, 1]

        e_oof = np.clip(e_oof, 0.01, 0.99)
        pseudo_outcome = (
            mu1_oof
            - mu0_oof
            + t * (y - mu1_oof) / e_oof
            - (1 - t) * (y - mu0_oof) / (1 - e_oof)
        )
        self.tau_model.fit(X, pseudo_outcome)
        treated = t == 1
        control = t == 0
        self._m1_final.fit(X[treated], y[treated])
        self._m0_final.fit(X[control], y[control])
        return self

    def predict(self, X: np.ndarray) -> UpliftPrediction:
        tau = self.tau_model.predict(X)
        p1 = self._m1_final.predict_proba(X)[:, 1]
        p0 = self._m0_final.predict_proba(X)[:, 1]
        return UpliftPrediction(uplift=tau, prob_t1=p1, prob_t0=p0)


class CausalForestModel:
    name = "Causal Forest"

    def __init__(self, config: BenchmarkConfig) -> None:
        from econml.dml import CausalForestDML

        self.config = config
        self.model = CausalForestDML(
            model_y=LGBMRegressor(**config.lgbm.regressor_params(n_estimators=300)),
            model_t=LGBMClassifier(**config.lgbm.classifier_params()),
            n_estimators=200,
            min_samples_leaf=config.lgbm.min_child_samples,
            random_state=config.split.seed,
            discrete_treatment=True,
            verbose=0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> CausalForestModel:
        self.model.fit(Y=y, T=t, X=X)
        return self

    def predict(self, X: np.ndarray) -> UpliftPrediction:
        tau = self.model.effect(X).flatten()
        return UpliftPrediction(
            uplift=tau,
            prob_t1=np.full(len(X), np.nan),
            prob_t0=np.full(len(X), np.nan),
        )


class NaiveResponseRanker:
    name = "Naive Response Ranker"

    def __init__(self, config: BenchmarkConfig) -> None:
        self.model = make_classifier(config)

    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray | None = None) -> NaiveResponseRanker:
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> UpliftPrediction:
        p = self.model.predict_proba(X)[:, 1]
        return UpliftPrediction(
            uplift=p,
            prob_t1=p,
            prob_t0=np.full(len(X), np.nan),
        )


def build_models(config: BenchmarkConfig) -> dict[str, object]:
    """Create benchmark models based on runtime config."""
    models: dict[str, object] = {
        SLearner.name: SLearner(config),
        TLearner.name: TLearner(config),
        XLearner.name: XLearner(config),
        NaiveResponseRanker.name: NaiveResponseRanker(config),
    }
    if config.runtime.run_dr_learner:
        models[DRLearner.name] = DRLearner(config)
    if config.runtime.run_causal_forest:
        models[CausalForestModel.name] = CausalForestModel(config)
    return models
