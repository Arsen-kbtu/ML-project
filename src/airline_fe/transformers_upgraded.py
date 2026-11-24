"""Optional, CV-testable feature hypotheses beyond the canonical pipeline."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import SplineTransformer
from sklearn.utils.validation import check_is_fitted

from .config import DELAY_PAIR, LIKERT
from .config_upgraded import TRAVEL_CLASS


class DelayOccurrenceFlagger(BaseEstimator, TransformerMixin):
    """Create one binary feature per delay input indicating delay > 0."""

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float) if not isinstance(X, pd.DataFrame) else X.to_numpy(dtype=float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("DelayOccurrenceFlagger expects departure and arrival columns.")
        self.n_features_in_ = 2
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            if list(self.feature_names_in_) != DELAY_PAIR:
                raise ValueError(f"Expected columns in order {DELAY_PAIR}.")
        else:
            self.feature_names_in_ = np.asarray(DELAY_PAIR, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, ["n_features_in_", "feature_names_in_"])
        values = np.asarray(X, dtype=float) if not isinstance(X, pd.DataFrame) else X.to_numpy(dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("DelayOccurrenceFlagger input width changed after fit.")
        # A missing delay is not silently called a non-delay; handle missingness
        # in the surrounding pipeline before applying this candidate.
        if not np.isfinite(values).all():
            raise ValueError("Impute delay values before applying occurrence flags.")
        return (values > self.threshold).astype(float)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_features_in_")
        names = self.feature_names_in_ if input_features is None else np.asarray(input_features, dtype=object)
        if len(names) != 2:
            raise ValueError("Expected two delay input feature names.")
        return np.asarray([f"{name}_occurred" for name in names], dtype=object)


class TravelClassInteraction(BaseEstimator, TransformerMixin):
    """One-hot encode the observed Travel Type × Class combinations."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            if list(X.columns) != TRAVEL_CLASS:
                raise ValueError(f"Expected columns in order {TRAVEL_CLASS}.")
            values = X.to_numpy(dtype=object)
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            values = np.asarray(X, dtype=object)
            if values.ndim != 2 or values.shape[1] != 2:
                raise ValueError("TravelClassInteraction expects two columns.")
            self.feature_names_in_ = np.asarray(TRAVEL_CLASS, dtype=object)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("TravelClassInteraction expects two columns.")
        self.n_features_in_ = 2
        combos = np.asarray([self._combine(row) for row in values], dtype=object).reshape(-1, 1)
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder_.fit(combos)
        self.categories_ = self.encoder_.categories_[0].copy()
        return self

    @staticmethod
    def _combine(row):
        return f"{row[0]}__{row[1]}"

    def transform(self, X):
        check_is_fitted(self, ["encoder_", "n_features_in_"])
        values = X.to_numpy(dtype=object) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=object)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("TravelClassInteraction input width changed after fit.")
        combos = np.asarray([self._combine(row) for row in values], dtype=object).reshape(-1, 1)
        return self.encoder_.transform(combos)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["encoder_", "n_features_in_"])
        return np.asarray(
            [f"TravelClass_{category}" for category in self.categories_], dtype=object
        )


class ServiceRatingSummary(BaseEstimator, TransformerMixin):
    """Append mean applicable service rating and number of applicable ratings."""

    def __init__(self, na_value=0):
        self.na_value = na_value

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float) if not isinstance(X, pd.DataFrame) else X.to_numpy(dtype=float)
        if values.ndim != 2:
            raise ValueError("ServiceRatingSummary expects a 2D rating matrix.")
        self.n_features_in_ = values.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            if self.n_features_in_ == len(LIKERT) and list(self.feature_names_in_) != LIKERT:
                raise ValueError("Likert columns do not match the configured schema.")
        else:
            self.feature_names_in_ = np.asarray(
                [f"x{i}" for i in range(self.n_features_in_)], dtype=object
            )
        return self

    def transform(self, X):
        check_is_fitted(self, ["n_features_in_", "feature_names_in_"])
        values = np.asarray(X, dtype=float) if not isinstance(X, pd.DataFrame) else X.to_numpy(dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("ServiceRatingSummary input width changed after fit.")
        applicable = np.isfinite(values) & (values != self.na_value)
        counts = applicable.sum(axis=1)
        totals = np.where(applicable, values, 0.0).sum(axis=1)
        means = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
        return np.column_stack([means, counts.astype(float)])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["n_features_in_", "feature_names_in_"])
        return np.asarray(["ServiceRating_mean", "ServiceRating_coverage"], dtype=object)


class LikertLevelEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode integer rating levels 0..5 to test non-linear effects."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float) if not isinstance(X, pd.DataFrame) else X.to_numpy(dtype=float)
        if values.ndim != 2:
            raise ValueError("LikertLevelEncoder expects a 2D rating matrix.")
        if not np.isfinite(values).all() or not np.isin(values, np.arange(6)).all():
            raise ValueError("Ratings must be finite integers in the range 0..5.")
        self.n_features_in_ = values.shape[1]
        self.feature_names_in_ = (
            np.asarray(X.columns, dtype=object)
            if isinstance(X, pd.DataFrame)
            else np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        )
        self.encoders_ = []
        for col in range(self.n_features_in_):
            encoder = OneHotEncoder(
                categories=[np.arange(6)], handle_unknown="ignore", sparse_output=False
            )
            encoder.fit(values[:, col].reshape(-1, 1))
            self.encoders_.append(encoder)
        return self

    def transform(self, X):
        check_is_fitted(self, ["encoders_", "n_features_in_"])
        values = np.asarray(X, dtype=float) if not isinstance(X, pd.DataFrame) else X.to_numpy(dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("LikertLevelEncoder input width changed after fit.")
        return np.hstack(
            [
                encoder.transform(values[:, i].reshape(-1, 1))
                for i, encoder in enumerate(self.encoders_)
            ]
        )

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["feature_names_in_", "n_features_in_"])
        columns = self.feature_names_in_ if input_features is None else np.asarray(input_features, dtype=object)
        if len(columns) != self.n_features_in_:
            raise ValueError("input_features length does not match fitted input width.")
        return np.asarray(
            [f"{column}_rating_{level}" for column in columns for level in range(6)],
            dtype=object,
        )


class FlightDistanceSpline(BaseEstimator, TransformerMixin):
    """Spline basis candidate for nonlinear Flight Distance effects."""

    def __init__(self, n_knots=5, degree=3):
        self.n_knots = n_knots
        self.degree = degree

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float) if not isinstance(X, pd.DataFrame) else X.to_numpy(dtype=float)
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError("FlightDistanceSpline expects one numeric column.")
        if not np.isfinite(values).all():
            raise ValueError("Impute Flight Distance before applying its spline.")
        self.n_features_in_ = 1
        self.feature_names_in_ = (
            np.asarray(X.columns, dtype=object)
            if isinstance(X, pd.DataFrame)
            else np.asarray(["Flight Distance"], dtype=object)
        )
        self.spline_ = SplineTransformer(
            n_knots=self.n_knots, degree=self.degree, include_bias=False
        ).fit(values)
        return self

    def transform(self, X):
        check_is_fitted(self, ["spline_", "n_features_in_"])
        values = np.asarray(X, dtype=float) if not isinstance(X, pd.DataFrame) else X.to_numpy(dtype=float)
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError("FlightDistanceSpline input width changed after fit.")
        return self.spline_.transform(values)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["spline_", "n_features_in_"])
        name = self.feature_names_in_[0] if input_features is None else np.asarray(input_features, dtype=object)[0]
        return np.asarray(
            [f"{name}_spline_{i}" for i in range(self.spline_.n_features_out_)],
            dtype=object,
        )


class AgeBinner(BaseEstimator, TransformerMixin):
    """Apply fixed, interpretable age bands as a one-hot candidate feature."""

    def __init__(self, bins=(0, 25, 40, 60, np.inf)):
        self.bins = bins

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float) if not isinstance(X, pd.DataFrame) else X.to_numpy(dtype=float)
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError("AgeBinner expects one numeric column.")
        if not np.isfinite(values).all():
            raise ValueError("Impute Age before applying age bins.")
        if len(self.bins) < 3 or any(b >= a for b, a in zip(self.bins, self.bins[1:])):
            raise ValueError("bins must be strictly increasing and contain at least two intervals.")
        self.n_features_in_ = 1
        self.feature_names_in_ = (
            np.asarray(X.columns, dtype=object)
            if isinstance(X, pd.DataFrame)
            else np.asarray(["Age"], dtype=object)
        )
        self.bin_edges_ = np.asarray(self.bins, dtype=float)
        self.n_bins_ = len(self.bin_edges_) - 1
        return self

    def transform(self, X):
        check_is_fitted(self, ["bin_edges_", "n_features_in_"])
        values = np.asarray(X, dtype=float) if not isinstance(X, pd.DataFrame) else X.to_numpy(dtype=float)
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError("AgeBinner input width changed after fit.")
        labels = np.digitize(values[:, 0], self.bin_edges_[1:-1], right=False)
        return np.eye(self.n_bins_, dtype=float)[labels]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["bin_edges_", "n_features_in_"])
        name = self.feature_names_in_[0] if input_features is None else np.asarray(input_features, dtype=object)[0]
        return np.asarray(
            [f"{name}_band_{i}" for i in range(self.n_bins_)], dtype=object
        )
