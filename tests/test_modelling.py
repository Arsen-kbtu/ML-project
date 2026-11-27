"""Executable checks for the Week 3 modelling helpers (small data, runs in seconds)."""

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split

from airline_fe.config import ID_COL, RANDOM_STATE, TARGET
from airline_fe.modelling import (
    METRICS,
    EarlyStoppingXGBClassifier,
    comparison_table,
    make_model_pipeline,
    pick_shap_cases,
    predict_test,
    score_predictions,
    tune_model,
    tuning_report,
)

pytest.importorskip("xgboost")
DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "airline_passenger_satisfaction.csv"


@pytest.fixture(scope="module")
def small_split():
    df = pd.read_csv(DATA).sample(3_000, random_state=RANDOM_STATE)
    y = (df.pop(TARGET) == "Satisfied").astype(int)
    X = df.drop(columns=[ID_COL])
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)


def test_xgb_wrapper_is_clone_safe_and_early_stops():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.normal(size=(1500, 5)), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"] + 0.5 * rng.normal(size=1500) > 0).astype(int)
    est = EarlyStoppingXGBClassifier(n_estimators=500, learning_rate=0.3,
                                     early_stopping_rounds=10)
    assert clone(est).get_params() == est.get_params()
    est.fit(X, y)
    n_rounds = len(est.evals_result_["validation_1"]["logloss"])
    assert est.n_trees_ == est.best_iteration_ + 1 <= n_rounds < 500   # it stopped early
    proba = est.predict_proba(X)
    assert proba.shape == (1500, 2) and np.allclose(proba.sum(axis=1), 1)
    assert set(est.predict(X)) <= {0, 1}
    assert list(est.feature_names_in_) == list(X.columns)


def test_pipeline_tuning_report_and_joblib(small_split, tmp_path):
    X_tr, X_te, y_tr, y_te = small_split
    cv = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)
    pipe = make_model_pipeline(EarlyStoppingXGBClassifier(n_estimators=200, learning_rate=0.2,
                                                          early_stopping_rounds=10),
                               memory=str(tmp_path))
    res = tune_model("XGBoost", pipe, {"model__max_depth": [2, 4]}, X_tr, y_tr,
                     search="grid", cv=cv, n_jobs=1)
    assert res["n_candidates"] == 2 and "model__max_depth" in res["best_params"]
    assert set(res["cv"]) == set(res["pre_cv"]) == set(METRICS)

    fitted = res["estimator"]
    assert fitted[:-1].transform(X_te).shape[1] <= 20         # Week 2 cap holds
    rep = tuning_report({"XGBoost": res})
    assert rep.loc[0, "candidates"] == 2

    preds = predict_test({"XGBoost": res}, X_te)
    table = comparison_table({"XGBoost": res}, preds, y_te)
    assert table.shape == (1, 3 * len(METRICS))

    buf = io.BytesIO()
    joblib.dump(fitted, buf)
    buf.seek(0)
    assert np.allclose(joblib.load(buf).predict_proba(X_te), fitted.predict_proba(X_te))


def test_xgb_shap_values_add_up_to_predicted_log_odds(small_split):
    """SHAP must explain the early-stopped model, not the trees grown after the best round."""
    pytest.importorskip("shap")
    from airline_fe.modelling import shap_explanation

    X_tr, X_te, y_tr, _ = small_split
    pipe = make_model_pipeline(EarlyStoppingXGBClassifier(n_estimators=300, learning_rate=0.2,
                                                          early_stopping_rounds=10)).fit(X_tr, y_tr)
    xgb_step = pipe.named_steps["model"]
    assert xgb_step.n_trees_ < len(xgb_step.evals_result_["validation_1"]["logloss"])
    expl = shap_explanation(pipe, X_te.iloc[:200])
    p = pipe.predict_proba(X_te.iloc[:200])[:, 1]
    # compare in probability space: inverting float32 probabilities near 0/1 loses precision
    p_shap = 1 / (1 + np.exp(-(expl.base_values + expl.values.sum(axis=1))))
    assert np.allclose(p_shap, p, atol=1e-5)


def test_score_predictions_and_case_picking():
    y = np.array([0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.9, 0.8, 0.6, 0.3])
    pred = (proba >= 0.5).astype(int)
    s = score_predictions(y, pred, proba)
    assert s["accuracy"] == pytest.approx(0.6)
    assert s["recall_1"] == pytest.approx(2 / 3) and s["recall_0"] == pytest.approx(0.5)
    i_right, i_wrong = pick_shap_cases(y, pred, proba)
    assert pred[i_right] == y[i_right]
    assert i_wrong == 1                                       # most confident error (0.9 vs 0)
