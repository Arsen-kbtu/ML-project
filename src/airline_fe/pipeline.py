"""Leak-free preprocessing and modelling pipeline (spec §2.4).

Every fitted step lives inside one ``Pipeline`` built here, so the notebook,
CV harness and Week 4 app share identical, picklable objects.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
)

from airline_fe.config import (NOMINAL, ORDINAL, CLASS_ORDER, AGE, SKEWED,
                               DELAY_PAIR, LIKERT, RANDOM_STATE)  # LIKERT = 14 cols
from airline_fe.transformers import DelayFeatureEngineer, LikertZeroFlagger


def log1p_names(transformer, input_features):
    """Module-level (picklable) name function; a lambda breaks joblib.dump."""
    return [f"log1p_{c}" for c in input_features]


def _numeric(log=False):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if log:
        steps.append(("log1p", FunctionTransformer(np.log1p, inverse_func=np.expm1,
                                                   feature_names_out=log1p_names)))
    steps.append(("scale", RobustScaler()))
    return Pipeline(steps)


def build_preprocessor(delay="net", log_skewed=True, flags="nzv",
                       impute_strategy="net_median", flag_threshold=0.0099):
    """Defaults = final (S4) configuration; switches build S0–S3 / S1b (§3.3)."""
    if delay not in {"raw", "net"} or flags not in {"none", "all", "nzv"}:
        raise ValueError(f"bad switch: delay={delay!r}, flags={flags!r}")

    branches = [
        ("nominal",    OneHotEncoder(drop="first", handle_unknown="ignore",
                                     sparse_output=False), NOMINAL),
        ("class_ord",  OrdinalEncoder(categories=CLASS_ORDER,
                                      handle_unknown="use_encoded_value",
                                      unknown_value=-1), ORDINAL),
        ("age",        _numeric(), AGE),
        ("log_skewed", _numeric(log=log_skewed), SKEWED),   # log off at S0/S1
    ]
    if delay == "raw":                                       # S0: both collinear delays
        branches.append(("arrival_raw", _numeric(), ["Arrival Delay"]))
    else:                                                    # S1+: combine, drop raw Arrival
        branches.append(("net_delay", Pipeline([
            ("net",   DelayFeatureEngineer(impute_strategy=impute_strategy)),
            ("yj",    PowerTransformer(method="yeo-johnson", standardize=False)),
            ("scale", RobustScaler())]), DELAY_PAIR))
    branches.append(("likert", _numeric(), LIKERT))
    if flags != "none":
        # "all" must use passthrough: VarianceThreshold(0.0) would still drop
        # the 4 flags that are constant on train (Check-in, Gate, Seat, Baggage).
        nzv = VarianceThreshold(flag_threshold) if flags == "nzv" else "passthrough"
        branches.append(("likert_flags", Pipeline([("flag", LikertZeroFlagger()),
                                                   ("nzv",  nzv)]), LIKERT))

    return ColumnTransformer(branches, remainder="drop",
                             verbose_feature_names_out=False).set_output(transform="pandas")


def build_pipeline(selector="passthrough", estimator=None, **pre_kw):
    if estimator is None:
        estimator = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    return Pipeline([("preprocess", build_preprocessor(**pre_kw)),
                     ("select",     selector),
                     ("model",      estimator)])


# Ablation configurations (§3.3) and the output width each must produce on X_train.
STEP_KW = {
    "S0":  dict(delay="raw", log_skewed=False, flags="none"),
    "S1":  dict(delay="net", log_skewed=False, flags="none"),
    "S1b": dict(delay="net", log_skewed=False, flags="none", impute_strategy="arrival_median"),
    "S2":  dict(delay="net", log_skewed=True,  flags="none"),
    "S3":  dict(delay="net", log_skewed=True,  flags="all"),
    "S4":  dict(delay="net", log_skewed=True,  flags="nzv"),   # == defaults
}
EXPECTED_WIDTH = {"S0": 22, "S1": 22, "S1b": 22, "S2": 22, "S3": 36, "S4": 26}
