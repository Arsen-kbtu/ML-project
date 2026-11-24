"""Feature-selection helpers that need to be importable (so joblib can reload them)."""

import warnings

import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import RFE, RFECV, mutual_info_classif


def mi_subsampled(X, y, discrete_mask=None, n=20_000, random_state=42):
    """Mutual information on a random subsample of the training fold.

    kNN-based MI on the full ~83k-row fold is slow and the ranking barely
    changes, so we score a fixed-size subsample. Discrete columns (dummies,
    Likert ratings, flags) are marked so MI is not distorted by ties.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    if len(y) > n:
        idx = rng.choice(len(y), size=n, replace=False)
        X, y = X[idx], y[idx]
    discrete = "auto" if discrete_mask is None else np.asarray(discrete_mask, dtype=bool)
    with warnings.catch_warnings():
        # scaled ratings are discrete but not integers; sklearn warns about that, harmless here
        warnings.filterwarnings("ignore", message="Clustering metrics expects discrete values")
        return mutual_info_classif(X, y, discrete_features=discrete, random_state=random_state)


class CappedRFECV(RFECV):
    """RFECV that never keeps more than ``max_features`` columns.

    Plain RFECV picks the best-scoring size, which for logistic regression on
    this data tends to be "almost everything". If that size is above the cap,
    we pick the best size *within* the cap from the same CV curve and refit a
    plain RFE with that size, so transform()/get_support() still work.
    """

    def __init__(
        self,
        estimator,
        *,
        max_features=20,
        step=1,
        min_features_to_select=1,
        cv=None,
        scoring=None,
        verbose=0,
        n_jobs=None,
        importance_getter="auto",
    ):
        super().__init__(
            estimator,
            step=step,
            min_features_to_select=min_features_to_select,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
            n_jobs=n_jobs,
            importance_getter=importance_getter,
        )
        self.max_features = max_features

    def fit(self, X, y, **params):
        super().fit(X, y, **params)
        self.uncapped_n_features_ = int(self.n_features_)
        if self.n_features_ > self.max_features:
            sizes = np.asarray(self.cv_results_["n_features"])
            scores = np.asarray(self.cv_results_["mean_test_score"])
            allowed = sizes <= self.max_features
            n_star = int(sizes[allowed][np.argmax(scores[allowed])])
            rfe = RFE(
                clone(self.estimator),
                n_features_to_select=n_star,
                step=self.step,
                importance_getter=self.importance_getter,
            ).fit(X, y)
            self.support_ = rfe.support_
            self.ranking_ = rfe.ranking_
            self.n_features_ = int(rfe.n_features_)
            self.estimator_ = rfe.estimator_
        return self
