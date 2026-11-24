"""Leakage-safe custom transformers used by the canonical feature pipeline."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .config import DELAY_PAIR, LIKERT


def _as_2d_array(X, expected_width, transformer_name):
    """Validate a DataFrame/array input and return its numeric 2D values."""
    if isinstance(X, pd.DataFrame):
        values = X.to_numpy(dtype=float)
    else:
        values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[1] != expected_width:
        raise ValueError(
            f"{transformer_name} expects a 2D input with {expected_width} "
            f"columns; received shape {values.shape}."
        )
    return values


class DelayFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create net delay recovery while learning imputation on fit data only.

    Input order is departure delay, then arrival delay. For missing arrival
    values, ``net_median`` imputes the net delay directly; ``arrival_median``
    imputes arrival delay before subtraction.
    """

    def __init__(self, impute_strategy="net_median"):
        self.impute_strategy = impute_strategy

    def fit(self, X, y=None):
        if self.impute_strategy not in {"net_median", "arrival_median"}:
            raise ValueError(
                "impute_strategy must be 'net_median' or 'arrival_median'."
            )
        values = _as_2d_array(X, 2, self.__class__.__name__)
        departure, arrival = values[:, 0], values[:, 1]
        valid_net = np.isfinite(departure) & np.isfinite(arrival)
        valid_departure = np.isfinite(departure)
        valid_arrival = np.isfinite(arrival)
        if not valid_net.any() or not valid_departure.any() or not valid_arrival.any():
            raise ValueError("Cannot fit delay imputer: insufficient finite values.")

        net = departure[valid_net] - arrival[valid_net]
        self.median_net_ = float(np.median(net))
        self.median_arrival_ = float(np.median(arrival[valid_arrival]))
        self.median_departure_ = float(np.median(departure[valid_departure]))
        self.n_features_in_ = 2
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            if list(self.feature_names_in_) != DELAY_PAIR:
                raise ValueError(f"Expected columns in order {DELAY_PAIR}.")
        else:
            self.feature_names_in_ = np.asarray(DELAY_PAIR, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(
            self,
            ["median_net_", "median_arrival_", "median_departure_", "n_features_in_"],
        )
        values = _as_2d_array(X, 2, self.__class__.__name__)
        departure, arrival = values[:, 0].copy(), values[:, 1].copy()

        missing_departure = ~np.isfinite(departure)
        missing_arrival = ~np.isfinite(arrival)
        both_missing = missing_departure & missing_arrival
        only_arrival_missing = ~missing_departure & missing_arrival
        only_departure_missing = missing_departure & ~missing_arrival

        net = departure - arrival
        if self.impute_strategy == "net_median":
            net[only_arrival_missing] = self.median_net_
        else:
            net[only_arrival_missing] = departure[only_arrival_missing] - self.median_arrival_
        net[only_departure_missing] = (
            self.median_departure_ - arrival[only_departure_missing]
        )
        net[both_missing] = self.median_net_
        return net.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_features_in_")
        return np.asarray(["Net Delay Recovery"], dtype=object)


class LikertZeroFlagger(BaseEstimator, TransformerMixin):
    """Create float indicators for zero-valued (not applicable) ratings."""

    def __init__(self, na_value=0):
        self.na_value = na_value

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float) if not isinstance(X, pd.DataFrame) else X.to_numpy(dtype=float)
        if values.ndim != 2:
            raise ValueError("LikertZeroFlagger expects a 2D input.")
        self.n_features_in_ = values.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            self.feature_names_in_ = np.asarray(
                [f"x{i}" for i in range(self.n_features_in_)], dtype=object
            )
        if self.n_features_in_ == len(LIKERT) and isinstance(X, pd.DataFrame):
            if list(self.feature_names_in_) != LIKERT:
                raise ValueError("Likert columns do not match the configured schema.")
        return self

    def transform(self, X):
        check_is_fitted(self, ["n_features_in_", "feature_names_in_"])
        values = np.asarray(X, dtype=float) if not isinstance(X, pd.DataFrame) else X.to_numpy(dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} Likert columns; received shape {values.shape}."
            )
        return (values == self.na_value).astype(float)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["n_features_in_", "feature_names_in_"])
        if input_features is None:
            columns = self.feature_names_in_
        else:
            columns = np.asarray(input_features, dtype=object)
            if len(columns) != self.n_features_in_:
                raise ValueError("input_features length does not match fitted input width.")
        return np.asarray([f"{name}_is_na" for name in columns], dtype=object)
