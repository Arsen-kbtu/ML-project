"""Executable assertions for the custom feature transformers."""

import io

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from airline_fe.config import DELAY_PAIR, LIKERT
from airline_fe.transformers import DelayFeatureEngineer, LikertZeroFlagger
from airline_fe.transformers_upgraded import (
    AgeBinner,
    DelayOccurrenceFlagger,
    FlightDistanceSpline,
    LikertLevelEncoder,
    ServiceRatingSummary,
    TravelClassInteraction,
)


def test_canonical_transformers_clone_and_feature_names():
    delay = DelayFeatureEngineer()
    flagger = LikertZeroFlagger()
    assert clone(delay).get_params() == delay.get_params()
    assert clone(flagger).get_params() == flagger.get_params()

    delay_pipe = Pipeline([("delay", delay)])
    delay_pipe.fit(pd.DataFrame([[2.0, 1.0], [0.0, 0.0]], columns=DELAY_PAIR))
    assert delay_pipe.named_steps["delay"].get_feature_names_out().tolist() == [
        "Net Delay Recovery"
    ]

    likert_pipe = Pipeline([("flags", flagger)])
    likert_frame = pd.DataFrame(
        np.zeros((2, len(LIKERT))), columns=LIKERT
    )
    likert_pipe.fit(likert_frame)
    assert likert_pipe.named_steps["flags"].get_feature_names_out().tolist() == [
        f"{column}_is_na" for column in LIKERT
    ]


def test_delay_fit_state_does_not_change_on_transform():
    frame = pd.DataFrame(
        [[0.0, 0.0], [10.0, 5.0], [4.0, np.nan], [np.nan, 3.0]],
        columns=DELAY_PAIR,
    )
    pipe = Pipeline([("delay", DelayFeatureEngineer())])
    pipe.fit(frame.iloc[1:])
    fitted_median = pipe.named_steps["delay"].median_net_
    pipe.transform(frame.iloc[:1])
    assert pipe.named_steps["delay"].median_net_ == fitted_median
    assert pipe.transform(frame).shape == (4, 1)
    assert np.isfinite(pipe.transform(frame)).all()


def test_delay_edge_cases_and_imputation_strategies():
    frame = pd.DataFrame(
        [[10.0, 5.0], [20.0, 10.0], [4.0, np.nan], [np.nan, 3.0], [np.nan, np.nan]],
        columns=DELAY_PAIR,
    )
    net_pipe = Pipeline([("delay", DelayFeatureEngineer("net_median"))]).fit(frame)
    assert np.allclose(net_pipe.transform(frame).ravel(), [5.0, 10.0, 7.5, 7.0, 7.5])
    arrival_pipe = Pipeline(
        [("delay", DelayFeatureEngineer("arrival_median"))]
    ).fit(frame)
    assert np.allclose(arrival_pipe.transform(frame).ravel(), [5.0, 10.0, -1.0, 7.0, 7.5])

    # ndarray fallback and names
    array_pipe = Pipeline([("delay", DelayFeatureEngineer())]).fit(frame.to_numpy())
    assert array_pipe.named_steps["delay"].get_feature_names_out().tolist() == [
        "Net Delay Recovery"
    ]


def test_likert_flags_dataframe_and_ndarray():
    frame = pd.DataFrame([[0, 2, 0], [5, 0, 1]], columns=["a", "b", "c"])
    pipe = Pipeline([("flags", LikertZeroFlagger())]).fit(frame)
    assert np.array_equal(pipe.transform(frame), [[1, 0, 1], [0, 1, 0]])
    assert pipe.named_steps["flags"].get_feature_names_out().tolist() == [
        "a_is_na",
        "b_is_na",
        "c_is_na",
    ]
    array_pipe = Pipeline([("flags", LikertZeroFlagger())]).fit(frame.to_numpy())
    assert array_pipe.named_steps["flags"].get_feature_names_out().tolist() == [
        "x0_is_na",
        "x1_is_na",
        "x2_is_na",
    ]


def test_canonical_pipeline_joblib_round_trip():
    frame = pd.DataFrame(
        [[0.0, 0.0], [10.0, 5.0], [4.0, np.nan], [np.nan, 3.0]],
        columns=DELAY_PAIR,
    )
    pipe = Pipeline([("delay", DelayFeatureEngineer())]).fit(frame)
    expected = pipe.transform(frame)
    buffer = io.BytesIO()
    joblib.dump(pipe, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)
    assert np.allclose(restored.transform(frame), expected)


def test_upgraded_hypothesis_transformers():
    delay_frame = pd.DataFrame([[0, 0], [10, 0], [0, 4]], columns=DELAY_PAIR)
    occurrence = Pipeline([("flags", DelayOccurrenceFlagger())]).fit(delay_frame)
    assert np.array_equal(occurrence.transform(delay_frame), [[0, 0], [1, 0], [0, 1]])

    travel = pd.DataFrame(
        [["Business", "Business"], ["Personal", "Economy"]],
        columns=["Type of Travel", "Class"],
    )
    interaction = Pipeline([("interaction", TravelClassInteraction())]).fit(travel)
    transformed = interaction.transform(travel)
    assert transformed.shape == (2, 2)
    assert np.array_equal(transformed.sum(axis=1), [1, 1])
    assert len(interaction.named_steps["interaction"].get_feature_names_out()) == 2

    ratings = pd.DataFrame([[5, 0, 3], [0, 0, 0]], columns=["a", "b", "c"])
    summary = Pipeline([("summary", ServiceRatingSummary())]).fit(ratings)
    assert np.allclose(summary.transform(ratings), [[4, 2], [0, 0]])
    assert summary.named_steps["summary"].get_feature_names_out().tolist() == [
        "ServiceRating_mean",
        "ServiceRating_coverage",
    ]


def test_upgraded_non_linearity_candidates():
    ratings = pd.DataFrame([[0, 2], [5, 3], [1, 4]], columns=["rating_a", "rating_b"])
    rating_pipe = Pipeline([("levels", LikertLevelEncoder())]).fit(ratings)
    assert rating_pipe.transform(ratings).shape == (3, 12)
    assert len(rating_pipe.named_steps["levels"].get_feature_names_out()) == 12

    distance = pd.DataFrame({"Flight Distance": [100.0, 500.0, 1000.0, 3000.0]})
    spline_pipe = Pipeline([("spline", FlightDistanceSpline())]).fit(distance)
    assert spline_pipe.transform(distance).shape[0] == len(distance)
    assert spline_pipe.transform(distance).shape[1] == len(
        spline_pipe.named_steps["spline"].get_feature_names_out()
    )

    ages = pd.DataFrame({"Age": [18.0, 30.0, 50.0, 70.0]})
    age_pipe = Pipeline([("bands", AgeBinner())]).fit(ages)
    assert np.array_equal(age_pipe.transform(ages).sum(axis=1), np.ones(4))
    assert age_pipe.transform(ages).shape == (4, 4)
