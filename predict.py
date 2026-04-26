"""
Standalone Prediction Script

Loads the saved best model and label encoder to predict performance
for a single student.

Usage:
    python predict.py           # Interactive prediction
    python predict.py --mock    # Run with hardcoded mock/garbage values for testing
"""

import os
import sys
import joblib
import pandas as pd

from src.config import (
    ALL_FEATURE_COLS,
    INPUT_PROMPTS,
    RESULTS_DIR,
)
from src.analyzer import generate_student_analysis


def load_artifacts():
    """Load the saved best model and label encoder."""
    model_path = os.path.join(RESULTS_DIR, "best_model.joblib")
    le_path = os.path.join(RESULTS_DIR, "label_encoder.joblib")

    if not os.path.exists(model_path) or not os.path.exists(le_path):
        print("[!] Error: Saved model or label encoder not found.")
        print("[!] Please run `python main.py` first to train and save the model.")
        sys.exit(1)

    return joblib.load(model_path), joblib.load(le_path)


def predict(model, label_encoder, inputs):
    """Predict performance class from a dict of inputs."""
    input_df = pd.DataFrame([inputs])
    pred = model.predict(input_df)
    label = label_encoder.inverse_transform(pred)[0]
    return label


def main():
    args = set(sys.argv[1:])
    use_mock = "--mock" in args

    print("+" + "=" * 58 + "+")
    print("|          Student Performance Prediction Engine          |")
    print("+" + "=" * 58 + "+")

    model, label_encoder = load_artifacts()
    
    if use_mock:
        print("\n[i] Running in --mock mode with predefined garbage values...")
        # Hardcoded garbage values matching feature columns
        inputs = {
            "raisedhands": 95,
            "VisITedResources": 88,
            "AnnouncementsView": 50,
            "Discussion": 70,
            "gender": "F",
            "NationalITy": "Jordan",
            "PlaceofBirth": "Jordan",
            "SectionID": "B",
            "Topic": "Math",
            "Semester": "S",
            "Relation": "Mum",
            "ParentAnsweringSurvey": "Yes",
            "ParentschoolSatisfaction": "Good",
            "StudentAbsenceDays": "Under-7",
            "StageID": "HighSchool",
            "GradeID": "G-10"
        }
        
        print("\n--- Input Data ---")
        for k, v in inputs.items():
            print(f"  {k:25s}: {v}")
            
    else:
        print("\n[i] Enter student details below:")
        inputs = {}
        for col in ALL_FEATURE_COLS:
            prompt, valid = INPUT_PROMPTS.get(col, (f"{col}: ", None))
            while True:
                val = input(f"  {prompt}")
                if valid == "int":
                    try:
                        inputs[col] = int(val)
                        break
                    except ValueError:
                        print("    [!] Please enter a number.")
                elif valid is not None and val not in valid:
                    print(f"    [!] Choose from: {valid}")
                else:
                    inputs[col] = val
                    break

    # Predict and display
    print("\n" + "=" * 60)
    print("PREDICTION RESULT & ANALYSIS")
    print("=" * 60)
    
    result = predict(model, label_encoder, inputs)
    
    print(f"  Predicted Performance Class: [{result}]\n")
    
    # Generate and print the text-based analysis
    analysis_text = generate_student_analysis(inputs, result)
    print(analysis_text)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
