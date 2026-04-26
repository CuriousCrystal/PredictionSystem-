"""
Simulation and Visual Reporting Script

Automatically simulates a random student (Male or Female), predicts their
performance, and generates a visual analysis report (PNG).
"""

import random
import os
import joblib
from src.config import RESULTS_DIR, ALL_FEATURE_COLS
from src.analyzer import generate_student_analysis
from src.report_generator import save_analysis_report_as_image

def simulate_student():
    """Generates a random student profile (Male or Female)."""
    is_male = random.choice([True, False])
    gender = "M" if is_male else "F"
    
    # Simulate realistic 'garbage' values
    inputs = {
        "gender": gender,
        "raisedhands": random.randint(10, 95),
        "VisITedResources": random.randint(10, 95),
        "AnnouncementsView": random.randint(5, 90),
        "Discussion": random.randint(5, 90),
        "NationalITy": random.choice(["KW", "Jordan", "USA", "Egypt"]),
        "PlaceofBirth": random.choice(["KuwaIT", "Jordan", "USA", "Egypt"]),
        "SectionID": random.choice(["A", "B", "C"]),
        "Topic": random.choice(["IT", "Math", "Science", "English"]),
        "Semester": random.choice(["F", "S"]),
        "Relation": random.choice(["Father", "Mum"]),
        "ParentAnsweringSurvey": random.choice(["Yes", "No"]),
        "ParentschoolSatisfaction": random.choice(["Good", "Bad"]),
        "StudentAbsenceDays": random.choice(["Under-7", "Above-7"]),
        "StageID": random.choice(["lowerlevel", "MiddleSchool", "HighSchool"]),
        "GradeID": random.choice(["G-02", "G-07", "G-10"])
    }
    return inputs

def main():
    print("+" + "=" * 58 + "+")
    print("|      Student Performance - Automated Visual Report       |")
    print("+" + "=" * 58 + "+")

    # Load artifacts
    model_path = os.path.join(RESULTS_DIR, "best_model.joblib")
    le_path = os.path.join(RESULTS_DIR, "label_encoder.joblib")

    if not os.path.exists(model_path) or not os.path.exists(le_path):
        print("[!] Error: Models not found. Please run main.py first.")
        return

    model = joblib.load(model_path)
    label_encoder = joblib.load(le_path)

    # 1. Simulate Student
    inputs = simulate_student()
    print(f"\n[i] Simulated Student: {inputs['gender']} student in {inputs['Topic']} class.")

    # 2. Predict
    import pandas as pd
    input_df = pd.DataFrame([inputs])
    pred_int = model.predict(input_df)
    prediction = label_encoder.inverse_transform(pred_int)[0]
    
    # 3. Generate Text Analysis
    analysis_text = generate_student_analysis(inputs, prediction)
    
    # 4. Generate Visual Image Report
    # Include input summary in the text for the image
    report_text = f"STUDENT PROFILE:\n"
    report_text += f"Gender: {inputs['gender']} | Topic: {inputs['Topic']} | Grade: {inputs['GradeID']}\n"
    report_text += f"Attendance: {inputs['StudentAbsenceDays']} | Parent Survey: {inputs['ParentAnsweringSurvey']}\n"
    report_text += f"Participation (Hands/Resources): {inputs['raisedhands']}/{inputs['VisITedResources']}\n"
    report_text += "-" * 50 + "\n"
    report_text += analysis_text

    image_path = save_analysis_report_as_image(report_text)
    
    print(f"\n[OK] Analysis Complete.")
    print(f"     Predicted Class: {prediction}")
    print(f"     Report saved to: {image_path}")

if __name__ == "__main__":
    main()
