"""Canonical feature names and reproducibility settings."""

RANDOM_STATE = 42
TARGET = "Satisfaction"
ID_COL = "ID"

NOMINAL = ["Gender", "Customer Type", "Type of Travel"]
ORDINAL = ["Class"]
CLASS_ORDER = [["Economy", "Economy Plus", "Business"]]
AGE = ["Age"]
SKEWED = ["Flight Distance", "Departure Delay"]
DELAY_PAIR = ["Departure Delay", "Arrival Delay"]
LIKERT = [
    "Departure and Arrival Time Convenience",
    "Ease of Online Booking",
    "Check-in Service",
    "Online Boarding",
    "Gate Location",
    "On-board Service",
    "Seat Comfort",
    "Leg Room Service",
    "Cleanliness",
    "Food and Drink",
    "In-flight Service",
    "In-flight Wifi Service",
    "In-flight Entertainment",
    "Baggage Handling",
]
