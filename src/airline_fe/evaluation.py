"""Evaluation protocol: CV harness, ablation log, diagnostics, selection (spec §3.1–3.5).

Everything that is passed into a ``Pipeline`` (``mi_subsampled``, ``one_std_refit``,
``CappedRFECV``) is defined at module level so fitted pipelines stay picklable
by ``joblib`` for the Week 4 app.
"""

from functools import partial

import numpy as np
import pandas as pd
import warnings
from pandas.io.formats.style import Styler
from sklearn.base import clone
from sklearn.feature_selection import RFE, RFECV, SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from airline_fe.config import LIKERT, NOMINAL, ORDINAL, RANDOM_STATE
from airline_fe.pipeline import EXPECTED_WIDTH, STEP_KW, build_pipeline

# ---------------------------------------------------------------------------
# §3.1 CV harness
# ---------------------------------------------------------------------------
N_SPLITS = 5
CV = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"}
METRICS = tuple(SCORING)


def run_cv(pipe, X, y, cv=CV, n_jobs=-1, return_estimator=False):
    """Outer 5-fold CV on training data only, with train scores for the gap check."""
    return cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=n_jobs,
                          return_train_score=True, return_estimator=return_estimator)


# ---------------------------------------------------------------------------
# §3.2 Ablation log
# ---------------------------------------------------------------------------
def _fmt_or_dash(fmt, v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


class _LogStyler(Styler):
    """Styler with a readable text/plain fallback (used by LaTeX-based PDF export)."""
    text_repr = ""

    def __repr__(self):
        return self.text_repr or super().__repr__()


LOG_COLUMNS = [
    "step", "description", "n_features",
    "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std", "f1_mean", "f1_std",
    "train_roc_auc_mean", "train_val_gap",
    "ref", "delta_prev_roc_auc", "delta_base_roc_auc", "paired_delta_std",
    "fit_time_s", "verdict",
]


def verdict(delta, paired_std, min_effect=0.001):
    """Improved / Worsened only when Δ exceeds max(paired_std, min_effect)."""
    if delta is None or np.isnan(delta):
        return "Baseline"
    threshold = max(paired_std, min_effect)
    if delta > threshold:
        return "Improved"
    if delta < -threshold:
        return "Worsened"
    return "Maintained"


class AblationLog:
    """Collects one row per ablation step and keeps per-fold scores for paired Δs.

    Because every step uses the same ``CV`` object, fold *i* is the same rows for
    every step, so ``scores[step] - scores[ref]`` is a paired comparison.
    """

    def __init__(self, base="S0"):
        self.base = base
        self.rows = []
        self.fold_scores = {}   # step -> {metric: ndarray(n_splits)}
        self.cv_results = {}    # step -> raw cross_validate dict

    # -- core -------------------------------------------------------------
    def log_step(self, step, description, X, y, pipe=None, ref=None,
                 expected_width=None, n_features=None, min_effect=0.001,
                 cv=CV, n_jobs=-1, return_estimator=False):
        """Run CV for ``step`` and append its row.

        ``pipe`` defaults to ``build_pipeline(**STEP_KW[step])``. ``ref`` defaults to
        the previously logged step; pass ``ref="S1"`` on S2 so it skips S1b.
        Width is asserted against ``EXPECTED_WIDTH[step]`` when known.
        """
        if step in self.fold_scores:
            raise ValueError(f"step {step!r} already logged")
        if pipe is None:
            pipe = build_pipeline(**STEP_KW[step])
        if n_features is None:
            n_features = self._n_features(pipe, X, y)

        if expected_width is None:
            expected_width = EXPECTED_WIDTH.get(step)
        if expected_width is not None:
            assert n_features == expected_width, (
                f"{step}: preprocessor emits {n_features} cols, expected {expected_width}")

        res = run_cv(pipe, X, y, cv=cv, n_jobs=n_jobs, return_estimator=return_estimator)
        self.cv_results[step] = res
        self.fold_scores[step] = {m: np.asarray(res[f"test_{m}"]) for m in METRICS}
        auc = self.fold_scores[step]["roc_auc"]

        if ref is None and self.rows:
            ref = self.rows[-1]["step"]
        d_prev, paired_std = self._paired(step, ref)
        d_base, _ = self._paired(step, self.base if step != self.base else None)

        row = {"step": step, "description": description, "n_features": n_features}
        for m in METRICS:
            row[f"{m}_mean"] = float(np.mean(res[f"test_{m}"]))
            row[f"{m}_std"] = float(np.std(res[f"test_{m}"]))
        row["train_roc_auc_mean"] = float(np.mean(res["train_roc_auc"]))
        row["train_val_gap"] = row["train_roc_auc_mean"] - row["roc_auc_mean"]
        row["ref"] = ref
        row["delta_prev_roc_auc"] = d_prev
        row["delta_base_roc_auc"] = d_base
        row["paired_delta_std"] = paired_std
        row["fit_time_s"] = float(np.mean(res["fit_time"]))
        row["verdict"] = verdict(d_prev, paired_std, min_effect=min_effect)
        self.rows.append(row)
        assert len(auc) == cv.get_n_splits()
        return res

    def _paired(self, step, ref):
        if ref is None:
            return np.nan, np.nan
        if ref not in self.fold_scores:
            raise KeyError(f"reference step {ref!r} has not been logged")
        diff = self.fold_scores[step]["roc_auc"] - self.fold_scores[ref]["roc_auc"]
        return float(diff.mean()), float(diff.std(ddof=1))

    @staticmethod
    def _n_features(pipe, X, y):
        """Width of the matrix the model sees: preprocess (+ selector if not passthrough)."""
        est = clone(pipe)
        if hasattr(est, "best_estimator_") or hasattr(est, "param_grid"):
            est = est.fit(X, y).best_estimator_
        head = est[:-1]
        return head.fit(X, y).transform(X.iloc[:5] if hasattr(X, "iloc") else X[:5]).shape[1]

    # -- output -----------------------------------------------------------
    def to_frame(self):
        return pd.DataFrame(self.rows, columns=LOG_COLUMNS)

    @staticmethod
    def _formats():
        """Display formats. Deltas and the paired std get 5 d.p. so that values of
        order 1e-5 (and Δ vs the 0.001 floor) are not rounded to 0.0000 / 0.0010."""
        fmt = {c: "{:.4f}" for c in LOG_COLUMNS
               if c.endswith(("_mean", "_std", "_gap", "_auc"))}
        fmt.update({"delta_prev_roc_auc": "{:+.5f}", "delta_base_roc_auc": "{:+.5f}",
                    "paired_delta_std": "{:.5f}", "fit_time_s": "{:.2f}"})
        return fmt

    def styled(self):
        """Styled log for HTML / webpdf. Its text/plain repr is the same formatted
        table, so a LaTeX ``nbconvert --to pdf`` export prints the table, not
        ``<pandas…Styler at 0x…>``."""
        df = self.to_frame()
        fmt = self._formats()
        colors = {"Improved": "color: green", "Worsened": "color: red",
                  "Maintained": "color: gray"}
        sty = _LogStyler(df)
        sty.text_repr = df.to_string(index=False, na_rep="—", formatters={
            c: partial(_fmt_or_dash, f) for c, f in fmt.items()})
        return (sty.format(fmt, na_rep="—")
                .map(lambda v: colors.get(v, ""), subset=["verdict"]))

    def to_csv(self, path):
        df = self.to_frame()
        df.to_csv(path, index=False, float_format="%.6f")
        return df


def log_step(log, step, description, X, y, **kw):
    """Functional alias for ``AblationLog.log_step`` (spec name ``evaluation.log_step``)."""
    return log.log_step(step, description, X, y, **kw)


def coef_std_across_folds(cv_result, features=None):
    """Per-feature mean/std of LR coefficients over outer folds (S0 vs S1 stability).

    Requires ``run_cv(..., return_estimator=True)`` / ``log_step(..., return_estimator=True)``.
    """
    rows = []
    for est in cv_result["estimator"]:
        names = est[:-1].get_feature_names_out()
        rows.append(pd.Series(est[-1].coef_.ravel(), index=names))
    coefs = pd.DataFrame(rows)
    out = pd.DataFrame({"coef_mean": coefs.mean(), "coef_std": coefs.std(ddof=1)})
    return out.loc[features] if features is not None else out


# ---------------------------------------------------------------------------
# §3.4 Variance & correlation diagnostics (pure statistics on X_train, no fitting)
# ---------------------------------------------------------------------------
def flag_variance_table(X_likert, na_value=0, threshold=0.0099):
    """Prevalence & Bernoulli variance of each ``<col>_is_na`` candidate flag."""
    flags = X_likert.eq(na_value).astype(float)
    tbl = pd.DataFrame({
        "flag": [f"{c}_is_na" for c in X_likert.columns],
        "n_zero": flags.sum().astype(int).to_numpy(),
        "prevalence": flags.mean().to_numpy(),
        "variance": flags.var(ddof=0).to_numpy(),
    })
    tbl["decision"] = np.where(tbl["variance"] >= threshold, "keep",
                               "remove (near-zero variance)")
    return tbl.sort_values("variance", ascending=False, ignore_index=True)


def corr_pairs(df, method="pearson", lower=0.60, upper=None):
    """All column pairs with ``lower < |r|`` (and ``|r| <= upper`` if given), sorted."""
    corr = df.corr(method=method)
    iu = np.triu_indices_from(corr, k=1)
    pairs = pd.DataFrame({"feature_a": corr.index[iu[0]], "feature_b": corr.columns[iu[1]],
                          "r": corr.to_numpy()[iu]})
    absr = pairs["r"].abs()
    keep = absr > lower
    if upper is not None:
        keep &= absr <= upper
    return (pairs[keep].assign(abs_r=absr[keep])
            .sort_values("abs_r", ascending=False, ignore_index=True))


# ---------------------------------------------------------------------------
# §3.5 Feature selection
# ---------------------------------------------------------------------------
DISCRETE_SUFFIX = "_is_na"


def discrete_mask(feature_names):
    """Boolean mask of discrete columns (OHE dummies, Class, Likert, flags) by name.

    Name-based rather than positional so it is correct for any preprocessor
    configuration; for the S4 output it marks 22 of 26 columns.
    """
    names = list(feature_names)
    ohe_prefixes = tuple(f"{c}_" for c in NOMINAL)
    return np.array([n.startswith(ohe_prefixes) or n in ORDINAL or n in LIKERT
                     or n.endswith(DISCRETE_SUFFIX) for n in names], dtype=bool)


def mi_subsampled(X, y, discrete_mask=None, n=20_000, random_state=RANDOM_STATE):
    """``mutual_info_classif`` on a stratified ``n``-row subsample of the fit data.

    kNN MI is O(n log n) per feature; 20k rows keeps the nested sweep tractable.
    The mask tells MI which columns are discrete so ties are not treated as kNN
    neighbours. Deterministic for a given ``random_state``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if n is not None and X.shape[0] > n:
        rng = np.random.RandomState(random_state)
        idx = []
        for cls in np.unique(y):
            cls_idx = np.flatnonzero(y == cls)
            k = int(round(n * len(cls_idx) / len(y)))
            idx.append(rng.choice(cls_idx, size=k, replace=False))
        idx = np.sort(np.concatenate(idx))
        X, y = X[idx], y[idx]
    mask = "auto" if discrete_mask is None else np.asarray(discrete_mask, dtype=bool)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Clustering metrics expects discrete")
        return mutual_info_classif(X, y, discrete_features=mask, random_state=random_state)


def one_std_refit(cv_results):
    """GridSearchCV ``refit`` callable: smallest ``k`` within 1 std of the best mean."""
    means = np.asarray(cv_results["mean_test_score"])
    stds = np.asarray(cv_results["std_test_score"])
    ks = np.asarray(cv_results["param_select__k"], dtype=float)
    best = int(np.nanargmax(means))
    threshold = means[best] - stds[best]
    ok = np.flatnonzero(means >= threshold)
    return int(ok[np.argmin(ks[ok])])


K_GRID = [8, 10, 12, 14, 16, 18, 20]


def make_mi_score_func(mask, n=20_000, random_state=RANDOM_STATE):
    """Picklable ``partial`` of ``mi_subsampled`` for ``SelectKBest``."""
    return partial(mi_subsampled, discrete_mask=mask, n=n, random_state=random_state)


def make_selectkbest_search(score_func, k_grid=K_GRID, inner_cv=None, **pre_kw):
    """Method A: ``GridSearchCV`` over ``select__k`` with the one-std refit rule."""
    if inner_cv is None:
        inner_cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pipe = build_pipeline(SelectKBest(score_func), **pre_kw)
    return GridSearchCV(pipe, param_grid={"select__k": list(k_grid)}, cv=inner_cv,
                        refit=one_std_refit, scoring="roc_auc", n_jobs=1)


class CappedRFECV(RFECV):
    """``RFECV`` whose final subset is capped at ``max_features`` (spec §3.5 B).

    After the usual RFECV fit, if the CV-optimal size exceeds the cap, the best
    size ``<= max_features`` by mean CV score is chosen and a plain ``RFE`` is
    refit to it; ``support_``/``ranking_``/``estimator_`` are overwritten so
    ``transform``/``get_support``/``get_feature_names_out`` behave like RFECV.
    """

    def __init__(self, estimator, *, max_features=20, step=1, min_features_to_select=5,
                 cv=None, scoring="roc_auc", verbose=0, n_jobs=1, importance_getter="auto"):
        super().__init__(estimator, step=step, min_features_to_select=min_features_to_select,
                         cv=cv, scoring=scoring, verbose=verbose, n_jobs=n_jobs,
                         importance_getter=importance_getter)
        self.max_features = max_features

    def fit(self, X, y, **params):
        super().fit(X, y, **params)
        self.uncapped_n_features_ = int(self.n_features_)
        self.capped_ = False
        if self.n_features_ > self.max_features:
            sizes = np.asarray(self.cv_results_["n_features"])
            means = np.asarray(self.cv_results_["mean_test_score"])
            allowed = sizes <= self.max_features
            if not allowed.any():
                raise ValueError(f"no evaluated subset has <= {self.max_features} features")
            n_star = int(sizes[allowed][np.argmax(means[allowed])])
            rfe = RFE(clone(self.estimator), n_features_to_select=n_star, step=self.step,
                      importance_getter=self.importance_getter).fit(X, y, **params)
            self.n_features_ = rfe.n_features_
            self.support_ = rfe.support_
            self.ranking_ = rfe.ranking_
            self.estimator_ = rfe.estimator_
            self.capped_ = True
        return self


def selected_features(fitted_pipe):
    """Names kept by the ``select`` step of a fitted pipeline (or GridSearchCV)."""
    pipe = getattr(fitted_pipe, "best_estimator_", fitted_pipe)
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    return list(np.asarray(names)[pipe.named_steps["select"].get_support()])


def selection_frequency(estimators, feature_names=None):
    """Per-feature count (0..n_folds) of how often each outer-fold selector kept it."""
    counts = None
    for est in estimators:
        pipe = getattr(est, "best_estimator_", est)
        names = pipe.named_steps["preprocess"].get_feature_names_out()
        s = pd.Series(pipe.named_steps["select"].get_support().astype(int), index=names)
        counts = s if counts is None else counts.add(s, fill_value=0)
    counts = counts.astype(int)
    if feature_names is not None:
        counts = counts.reindex(feature_names, fill_value=0)
    return counts


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def pick_winner(res_a, res_b, n_a, n_b, min_effect=0.001):
    """Decision rule §3.5: AUC (beyond max(paired std, min_effect)) → fewer features → CappedRFECV."""
    diff = np.asarray(res_a["test_roc_auc"]) - np.asarray(res_b["test_roc_auc"])
    delta, noise = float(diff.mean()), float(diff.std(ddof=1))
    threshold = max(noise, min_effect)
    if abs(delta) > threshold:
        return ("A" if delta > 0 else "B"), f"ROC-AUC delta(A-B)={delta:+.4f} beyond threshold {threshold:.4f}"
    if n_a != n_b:
        return ("A" if n_a < n_b else "B"), f"AUC tied (|delta|={abs(delta):.4f} <= {threshold:.4f}); fewer features"
    return "B", "AUC and feature count tied; default to CappedRFECV"


__all__ = [
    "CV", "SCORING", "METRICS", "run_cv", "AblationLog", "log_step", "verdict",
    "coef_std_across_folds", "flag_variance_table", "corr_pairs", "discrete_mask",
    "mi_subsampled", "one_std_refit", "K_GRID", "make_mi_score_func",
    "make_selectkbest_search", "CappedRFECV", "selected_features",
    "selection_frequency", "jaccard", "pick_winner",
]
