"""
Central configuration for the Student Performance Prediction pipeline.

All magic numbers, paths, feature groups, and hyperparameters live here.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_DATASET = DATA_DIR / "AI-Data.csv"

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Data split ────────────────────────────────────────────────────────────────
TEST_SIZE = 0.30

# ── Cross-validation ─────────────────────────────────────────────────────────
CV_FOLDS = 5

# ── Target column ────────────────────────────────────────────────────────────
TARGET_COL = "Class"

# ── Feature groups ────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "raisedhands",
    "VisITedResources",
    "AnnouncementsView",
    "Discussion",
]

NOMINAL_FEATURES = [
    "gender",
    "NationalITy",
    "PlaceofBirth",
    "SectionID",
    "Topic",
    "Semester",
    "Relation",
    "ParentAnsweringSurvey",
    "ParentschoolSatisfaction",
    "StudentAbsenceDays",
]

ORDINAL_FEATURES = ["StageID", "GradeID"]

# Ordered categories for ordinal encoding
STAGE_ORDER = ["lowerlevel", "MiddleSchool", "HighSchool"]
GRADE_ORDER = [f"G-{i:02d}" for i in range(1, 13)]  # G-01 through G-12

# ── Input prompts for interactive prediction ──────────────────────────────────
INPUT_PROMPTS = {
    "gender": ("Gender (M/F): ", ["M", "F"]),
    "NationalITy": ("Nationality (e.g., KW, Jordan, USA): ", None),
    "PlaceofBirth": ("Place of Birth (e.g., KuwaIT, Jordan): ", None),
    "StageID": ("Stage (lowerlevel, MiddleSchool, HighSchool): ",
                ["lowerlevel", "MiddleSchool", "HighSchool"]),
    "GradeID": ("Grade (G-02 to G-12): ", None),
    "SectionID": ("Section (A, B, C): ", ["A", "B", "C"]),
    "Topic": ("Topic (e.g., IT, Math, Science, English): ", None),
    "Semester": ("Semester (F or S): ", ["F", "S"]),
    "Relation": ("Relation (Father or Mum): ", ["Father", "Mum"]),
    "raisedhands": ("Raised Hands (0-100): ", "int"),
    "VisITedResources": ("Visited Resources (0-100): ", "int"),
    "AnnouncementsView": ("Announcements Viewed (0-100): ", "int"),
    "Discussion": ("Discussions (0-100): ", "int"),
    "ParentAnsweringSurvey": ("Parent Answered Survey (Yes/No): ", ["Yes", "No"]),
    "ParentschoolSatisfaction": ("Parent School Satisfaction (Good/Bad): ",
                                 ["Good", "Bad"]),
    "StudentAbsenceDays": ("Absence Days (Under-7 or Above-7): ",
                           ["Under-7", "Above-7"]),
}

# All feature columns (for interactive prediction input ordering)
ALL_FEATURE_COLS = NUMERIC_FEATURES + NOMINAL_FEATURES + ORDINAL_FEATURES
