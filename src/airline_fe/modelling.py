"""Week 3 modelling helpers: shared pipeline factory, metrics, tuning report,
XGBoost with early stopping, comparison table, confusion matrices and SHAP.

Every model is the Week 2 pipeline ``preprocess -> select -> model`` with only the
last step swapped, so all models see the same leak-free features. Everything that
ends up inside a fitted pipeline is defined at module level so ``joblib`` can
reload it in the Week 4 app.
"""

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.utils.validation import check_is_fitted

from airline_fe.config import RANDOM_STATE
from airline_fe.evaluation import CV, CappedRFECV
from airline_fe.pipeline import build_pipeline

CLASS_NAMES = ("Neutral or Dissatisfied", "Satisfied")   # 0, 1

# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------
def final_selector():
    """Unfitted twin of the Week 2 ``FINAL_SELECTOR`` (Method B won, 20 features)."""
    return CappedRFECV(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        max_features=20, step=1, min_features_to_select=5,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc", n_jobs=1,
    )


def make_model_pipeline(estimator=None, memory=None):
    """Week 2 pipeline with ``estimator`` as the ``model`` step.

    ``memory`` (a directory) caches the fitted ``preprocess`` and ``select`` steps.
    Their parameters and the fold rows are the same for every model and every
    search candidate, so ``CappedRFECV`` (~30 s per fit) runs once per fold
    instead of once per candidate. Caching changes no result, only the runtime.
    Output stays a DataFrame end to end, so the model sees feature names.
    """
    pipe = build_pipeline(selector=final_selector(), estimator=estimator)
    pipe.set_params(memory=None if memory is None else str(memory))   # Pipeline rejects Path
    return pipe.set_output(transform="pandas")


# ---------------------------------------------------------------------------
# Metrics (rubric: Accuracy, Precision, Recall, F1 per class, ROC-AUC, PR-AUC)
# ---------------------------------------------------------------------------
SCORING = {
    "accuracy": "accuracy",
    "precision_1": "precision",
    "recall_1": "recall",
    "f1_1": "f1",
    "precision_0": make_scorer(precision_score, pos_label=0, zero_division=0),
    "recall_0": make_scorer(recall_score, pos_label=0),
    "f1_0": make_scorer(f1_score, pos_label=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}
METRICS = tuple(SCORING)
PRIMARY = "roc_auc"


def score_predictions(y_true, y_pred, proba):
    """Same metrics as ``SCORING``, computed once on a held-out set."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_1": precision_score(y_true, y_pred, zero_division=0),
        "recall_1": recall_score(y_true, y_pred),
        "f1_1": f1_score(y_true, y_pred),
        "precision_0": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_0": recall_score(y_true, y_pred, pos_label=0),
        "f1_0": f1_score(y_true, y_pred, pos_label=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def _cv_summary(res):
    """{metric: (mean, std)} from a ``cross_validate`` result."""
    return {m: (float(np.mean(res[f"test_{m}"])), float(np.std(res[f"test_{m}"])))
            for m in METRICS}


def _search_summary(search):
    """{metric: (mean, std)} of the refitted (best) candidate of a search."""
    i = search.best_index_
    cvr = search.cv_results_
    return {m: (float(cvr[f"mean_test_{m}"][i]), float(cvr[f"std_test_{m}"][i]))
            for m in METRICS}


# ---------------------------------------------------------------------------
# Baseline and tuning (rubric: best params, pre-tuning CV, post-tuning CV, time)
# ---------------------------------------------------------------------------
def evaluate_untuned(name, pipe, X, y, cv=CV, n_jobs=-1):
    """5-fold CV of a pipeline as given, then a refit on all of ``X``."""
    t0 = time.perf_counter()
    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=n_jobs)
    cv_time = time.perf_counter() - t0
    fitted = clone(pipe).fit(X, y)
    return {"name": name, "estimator": fitted, "cv": _cv_summary(res),
            "pre_cv": _cv_summary(res), "best_params": None,
            "tuning_time_s": None, "cv_time_s": cv_time, "n_candidates": None,
            "search": None}


def tune_model(name, pipe, param_space, X, y, search="random", n_iter=20, cv=CV,
               n_jobs=-1, random_state=RANDOM_STATE):
    """Pre-tuning CV, then ``RandomizedSearchCV`` / ``GridSearchCV`` inside the pipeline.

    The search tunes ``model__*`` parameters on the shared ``CV`` folds, so the
    pre- and post-tuning scores are paired. ``refit="roc_auc"`` refits the best
    candidate on all of ``X``. The post-tuning score is the best candidate's
    mean over the same folds that picked it, so it is slightly optimistic; the
    test column of the comparison table is the unbiased check.
    """
    pre = evaluate_untuned(name, pipe, X, y, cv=cv, n_jobs=n_jobs)
    if search == "grid":
        searcher = GridSearchCV(pipe, param_space, cv=cv, scoring=SCORING,
                                refit=PRIMARY, n_jobs=n_jobs)
    elif search == "random":
        searcher = RandomizedSearchCV(pipe, param_space, n_iter=n_iter, cv=cv,
                                      scoring=SCORING, refit=PRIMARY, n_jobs=n_jobs,
                                      random_state=random_state)
    else:
        raise ValueError(f"search must be 'random' or 'grid', got {search!r}")
    t0 = time.perf_counter()
    searcher.fit(X, y)
    tuning_time = time.perf_counter() - t0
    return {"name": name, "estimator": searcher.best_estimator_,
            "cv": _search_summary(searcher), "pre_cv": pre["cv"],
            "best_params": searcher.best_params_, "tuning_time_s": tuning_time,
            "cv_time_s": pre["cv_time_s"], "n_candidates": len(searcher.cv_results_["params"]),
            "search": searcher}


def tuning_report(results):
    """One row per tuned model: best params, pre/post CV ROC-AUC, gain, time."""
    rows = []
    for r in results.values():
        if r["best_params"] is None:
            continue
        pre_m, pre_s = r["pre_cv"][PRIMARY]
        post_m, post_s = r["cv"][PRIMARY]
        rows.append({
            "model": r["name"],
            "pre-tuning CV ROC-AUC": f"{pre_m:.4f} ± {pre_s:.4f}",
            "post-tuning CV ROC-AUC": f"{post_m:.4f} ± {post_s:.4f}",
            "Δ ROC-AUC": round(post_m - pre_m, 5),
            "pre-tuning CV PR-AUC": round(r["pre_cv"]["pr_auc"][0], 4),
            "post-tuning CV PR-AUC": round(r["cv"]["pr_auc"][0], 4),
            "candidates": r["n_candidates"],
            "tuning time (s)": round(r["tuning_time_s"], 1),
            "best params": {k.removeprefix("model__"): v for k, v in r["best_params"].items()},
        })
    return pd.DataFrame(rows)


def cv_table(results):
    """CV mean ± std of every metric, one row per model (no test data involved)."""
    return pd.DataFrame({r["name"]: {m: f"{r['cv'][m][0]:.4f} ± {r['cv'][m][1]:.4f}"
                                     for m in METRICS}
                         for r in results.values()}).T


def best_model_name(results, metric=PRIMARY):
    """Best model by mean CV score (never by test score)."""
    return max(results, key=lambda k: results[k]["cv"][metric][0])


# ---------------------------------------------------------------------------
# Final comparison on the test set (the only place X_test is used)
# ---------------------------------------------------------------------------
def predict_test(results, X_test):
    """{name: (y_pred, proba_1)} for every fitted model."""
    out = {}
    for name, r in results.items():
        est = r["estimator"]
        out[name] = (est.predict(X_test), est.predict_proba(X_test)[:, 1])
    return out


def comparison_table(results, preds, y_test):
    """All models x all metrics, CV mean ± std next to the test score."""
    rows = []
    for name, r in results.items():
        test = score_predictions(y_test, *preds[name])
        row = {"model": name}
        for m in METRICS:
            mean, std = r["cv"][m]
            row[(m, "CV mean")] = mean
            row[(m, "CV std")] = std
            row[(m, "test")] = test[m]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model")
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["metric", ""])
    return df


# ---------------------------------------------------------------------------
# Advanced model: XGBoost with early stopping inside the pipeline
# ---------------------------------------------------------------------------
class EarlyStoppingXGBClassifier(ClassifierMixin, BaseEstimator):
    """``XGBClassifier`` that early-stops on a split of its own training data.

    As the last pipeline step it receives already-preprocessed rows. ``fit``
    holds out a stratified ``validation_fraction`` of them, grows up to
    ``n_estimators`` trees and stops when validation log-loss has not improved
    for ``early_stopping_rounds`` rounds. The validation rows come from the
    training fold only, so early stopping is leak-free inside CV and search
    (the same idea as ``HistGradientBoostingClassifier(early_stopping=True)``).
    ``predict_proba`` uses the trees up to ``best_iteration_``.
    """

    def __init__(self, n_estimators=2000, learning_rate=0.05, max_depth=6,
                 min_child_weight=1.0, subsample=0.8, colsample_bytree=0.8,
                 gamma=0.0, reg_alpha=0.0, reg_lambda=1.0,
                 early_stopping_rounds=50, validation_fraction=0.1,
                 tree_method="hist", n_jobs=1, random_state=RANDOM_STATE):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _xgb_params(self):
        keys = ("n_estimators", "learning_rate", "max_depth", "min_child_weight",
                "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda",
                "early_stopping_rounds", "tree_method", "n_jobs", "random_state")
        return {k: getattr(self, k) for k in keys}

    def fit(self, X, y):
        from xgboost import XGBClassifier   # imported lazily: only this model needs it

        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        y = np.asarray(y)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=self.validation_fraction, stratify=y,
            random_state=self.random_state)
        self.model_ = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                    **self._xgb_params())
        self.model_.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
        self.classes_ = self.model_.classes_
        self.best_iteration_ = int(self.model_.best_iteration)
        self.n_trees_ = self.best_iteration_ + 1
        self.evals_result_ = self.model_.evals_result()
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(X)     # uses best_iteration after early stopping

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @property
    def booster_(self):
        """The booster cut to the ``n_trees_`` trees that ``predict_proba`` uses.

        Early stopping keeps the trees grown after the best round in the model;
        prediction ignores them, so SHAP must explain this sliced booster, not
        the full one, or its values would not add up to the predicted log-odds.
        """
        check_is_fitted(self, "model_")
        return self.model_.get_booster()[: self.n_trees_]

    @property
    def feature_importances_(self):
        check_is_fitted(self, "model_")
        return self.model_.feature_importances_


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_early_stopping(xgb_step, ax=None):
    """Train / validation log-loss per boosting round, best iteration marked."""
    import matplotlib.pyplot as plt

    ev = xgb_step.evals_result_
    train, val = ev["validation_0"]["logloss"], ev["validation_1"]["logloss"]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    rounds = np.arange(1, len(val) + 1)
    ax.plot(rounds, train, color="#2a78d6", lw=1.8, label="train (90% of X_train)")
    ax.plot(rounds, val, color="#e34948", lw=1.8, label="validation (10% of X_train)")
    best = xgb_step.best_iteration_ + 1
    ax.axvline(best, color="#555555", ls="--", lw=1,
               label=f"best iteration = {best} (stopped at {len(val)})")
    ax.set(xlabel="boosting round", ylabel="log-loss",
           title="XGBoost early stopping (patience "
                 f"{xgb_step.early_stopping_rounds} rounds)")
    ax.legend(fontsize=9)
    return ax


def plot_confusion_matrices(preds, y_test, ncols=2):
    """Seaborn heatmaps of the test confusion matrix, one panel per model."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    names = list(preds)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5 * nrows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        cm = confusion_matrix(y_test, preds[name][0], labels=[0, 1])
        share = cm / cm.sum(axis=1, keepdims=True)
        labels = np.array([[f"{c:,}\n{s:.1%} of true row" for c, s in zip(rc, rs)]
                           for rc, rs in zip(cm, share)])
        sns.heatmap(share, annot=labels, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                    annot_kws={"fontsize": 10})
        ax.set(xlabel="predicted", ylabel="true", title=name)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def shap_explanation(fitted_pipe, X_explain, X_background=None, max_background=1000,
                     random_state=RANDOM_STATE):
    """SHAP values (log-odds of 'Satisfied') of the pipeline's model step.

    Rows are transformed by the fitted ``preprocess -> select`` head first, so
    the values belong to the 20 named model features. Tree models use
    ``TreeExplainer``; a linear model uses ``LinearExplainer`` with a background
    sample of ``X_background`` (training rows).
    """
    import shap

    head, model = fitted_pipe[:-1], fitted_pipe[-1]
    Xt = head.transform(X_explain)
    if isinstance(model, EarlyStoppingXGBClassifier):
        explainer = shap.TreeExplainer(model.booster_)
    elif hasattr(model, "coef_"):
        if X_background is None:
            raise ValueError("LinearExplainer needs X_background (training rows)")
        bg = head.transform(X_background)
        if len(bg) > max_background:
            bg = bg.sample(max_background, random_state=random_state)
        explainer = shap.LinearExplainer(model, shap.maskers.Independent(bg, max_samples=len(bg)))
    else:
        explainer = shap.TreeExplainer(model)
    expl = explainer(Xt)
    if expl.values.ndim == 3:            # sklearn trees return one slice per class
        expl = expl[:, :, 1]
    return expl


def to_raw_units(expl, X_raw):
    """Copy of ``expl`` whose feature values are shown in the passenger's own units.

    SHAP values are unchanged. Only the displayed values and names change:
    scaled Likert ratings go back to the 0–5 rating, ``log1p_Departure Delay``
    back to minutes, ``Class`` gets its tier names in the label. One-hot
    columns and ``_is_na`` flags are already 0/1. All the transforms involved
    are monotone, so beeswarm colours keep their order.
    """
    import shap

    X_raw = X_raw.reset_index(drop=True)
    names, cols = [], []
    for j, name in enumerate(expl.feature_names):
        raw_name = name.removeprefix("log1p_")
        if name.startswith("log1p_") and raw_name in X_raw:
            names.append(f"{raw_name} (min)" if "Delay" in raw_name else raw_name)
            cols.append(X_raw[raw_name].to_numpy(dtype=float))
        elif name in X_raw and pd.api.types.is_numeric_dtype(X_raw[name]):
            names.append(f"{name} (0–5)" if name != "Age" else name)
            cols.append(X_raw[name].to_numpy(dtype=float))
        else:
            label = {"Class": "Class (0 Eco · 1 Eco+ · 2 Bus)"}.get(name, name)
            names.append(label)
            cols.append(np.asarray(expl.data[:, j], dtype=float))
    return shap.Explanation(values=expl.values, base_values=expl.base_values,
                            data=np.column_stack(cols), feature_names=names)


def pick_shap_cases(y_true, y_pred, proba, random_state=RANDOM_STATE):
    """Positions of one correct and one incorrect test prediction.

    The incorrect case is the most confident error: the clearest failure to
    explain. The correct case is a seeded random correct prediction, so it is a
    typical passenger rather than a cherry-picked extreme.
    """
    y_true, y_pred, proba = map(np.asarray, (y_true, y_pred, proba))
    wrong = np.flatnonzero(y_true != y_pred)
    right = np.flatnonzero(y_true == y_pred)
    i_wrong = int(wrong[np.argmax(np.abs(proba[wrong] - 0.5))])
    i_right = int(np.random.RandomState(random_state).choice(right))
    return i_right, i_wrong


__all__ = [
    "CLASS_NAMES", "SCORING", "METRICS", "PRIMARY", "final_selector",
    "make_model_pipeline", "score_predictions", "evaluate_untuned", "tune_model",
    "tuning_report", "cv_table", "best_model_name", "predict_test", "comparison_table",
    "EarlyStoppingXGBClassifier", "plot_early_stopping", "plot_confusion_matrices",
    "shap_explanation", "to_raw_units", "pick_shap_cases",
]
