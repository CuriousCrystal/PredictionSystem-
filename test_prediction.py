import logging
import pandas as pd
from unittest.mock import patch

from src.config import TARGET_COL
from src.data_loader import load_data
from src.preprocessing import encode_target, split_data
from src.models import get_pipelines, train_model
from main import predict_single_input

# Suppress debug logs
logging.getLogger().setLevel(logging.ERROR)

def run_prediction_test():
    print("Loading data and training a model to test prediction...")
    data = load_data()
    X_all = data.drop(TARGET_COL, axis=1)
    y_all = data[TARGET_COL]
    y_all_enc, label_encoder = encode_target(y_all)
    X_train, X_test, y_train, y_test = split_data(X_all, y_all_enc)
    
    # We only train a quick Decision Tree for this test
    pipelines = get_pipelines()
    trained_pipelines = {
        "Decision Tree": train_model(pipelines["Decision Tree"], X_train, y_train)
    }
    
    # The interactive prompt asks for inputs in this order:
    # NUMERIC: raisedhands, VisITedResources, AnnouncementsView, Discussion
    # NOMINAL: gender, NationalITy, PlaceofBirth, SectionID, Topic, Semester, Relation, ParentAnsweringSurvey, ParentschoolSatisfaction, StudentAbsenceDays
    # ORDINAL: StageID, GradeID
    simulated_inputs = [
        "85", "92", "40", "60",                                      # Numeric
        "M", "KW", "KuwaIT", "A", "IT", "F", "Father", "Yes", "Good", "Under-7", # Nominal
        "MiddleSchool", "G-07"                                       # Ordinal
    ]
    
    print("\nSimulating interactive inputs for a new student:")
    print("-" * 50)
    with patch('builtins.input', side_effect=simulated_inputs) as mock_input:
        predict_single_input(trained_pipelines, label_encoder)

if __name__ == "__main__":
    run_prediction_test()
