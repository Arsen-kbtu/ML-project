"""Optional feature-engineering hypotheses layered on the canonical flow.

These are experiment candidates, not claims of measured improvement. Keep the
canonical preprocessing as the baseline and compare each candidate with the
same training-only CV folds before adopting it.
"""

from .config import CLASS_ORDER, LIKERT, RANDOM_STATE

TRAVEL_CLASS = ["Type of Travel", "Class"]

# Prior hypotheses only. Empirical effectiveness must be filled from CV results.
HYPOTHESES = {
    "delay_occurrence": {
        "expected_effectivity": 0.68,
        "reason": "Separate whether a delay occurred from its right-tailed magnitude.",
        "comparison": "canonical delay branch vs canonical branch plus delay flags",
    },
    "travel_class_interaction": {
        "expected_effectivity": 0.57,
        "reason": "The effect of cabin class may differ between business and personal travel.",
        "comparison": "canonical pipeline vs pipeline plus travel/class interaction",
    },
    "service_summary": {
        "expected_effectivity": 0.54,
        "reason": "A compact service-quality summary may capture shared signal across ratings.",
        "comparison": "canonical pipeline vs pipeline plus mean and coverage summaries",
    },
    "likert_nonlinearity": {
        "expected_effectivity": 0.62,
        "reason": "Ordinal rating effects may not be linear on the logistic scale.",
        "comparison": "canonical numeric ratings vs one-hot rating levels, retaining zero flags",
    },
    "flight_distance_nonlinearity": {
        "expected_effectivity": 0.52,
        "reason": "Distance may have a nonlinear effect beyond the canonical log1p transform.",
        "comparison": "canonical log1p distance vs spline-transformed distance",
    },
    "age_bins": {
        "expected_effectivity": 0.38,
        "reason": "Age effects may be nonlinear, though binning can discard useful information.",
        "comparison": "canonical continuous age vs fixed, interpretable age intervals",
    },
}

__all__ = [
    "CLASS_ORDER",
    "HYPOTHESES",
    "LIKERT",
    "RANDOM_STATE",
    "TRAVEL_CLASS",
]
