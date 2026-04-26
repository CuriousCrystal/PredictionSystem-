"""
Text-based analysis generation for student predictions.
"""

def generate_student_analysis(inputs: dict, prediction: str) -> str:
    """
    Generates a human-readable analysis explaining why the model likely 
    made its prediction based on the raw input values.
    """
    analysis = []
    
    # 1. Academic Engagement
    hands = int(inputs.get("raisedhands", 0))
    resources = int(inputs.get("VisITedResources", 0))
    
    if hands >= 70 and resources >= 70:
        analysis.append("[Analysis] Extremely High Academic Engagement:")
        analysis.append("   The student shows strong classroom participation and actively reviews course materials.")
        analysis.append("   These are the strongest indicators of success in the dataset.\n")
    elif hands < 30 and resources < 30:
        analysis.append("[Analysis] Low Academic Engagement:")
        analysis.append("   The student rarely raises their hand or visits course resources.")
        analysis.append("   This lack of engagement is a heavy predictor of lower performance.\n")
    else:
        analysis.append("[Analysis] Average Academic Engagement:")
        analysis.append("   The student shows moderate classroom participation and resource usage.\n")
        
    # 2. Attendance
    absences = inputs.get("StudentAbsenceDays", "")
    if absences == "Under-7":
        analysis.append("[Analysis] Strong Attendance:")
        analysis.append("   The student rarely misses class, avoiding the heavy penalties associated with frequent absences.\n")
    elif absences == "Above-7":
        analysis.append("[Analysis] Poor Attendance:")
        analysis.append("   The student has missed more than 7 days of class. This is a significant risk factor for underperformance.\n")
        
    # 3. Parental Support
    survey = inputs.get("ParentAnsweringSurvey", "")
    satisfaction = inputs.get("ParentschoolSatisfaction", "")
    if survey == "Yes" and satisfaction == "Good":
        analysis.append("[Analysis] Strong Parental Support:")
        analysis.append("   Active and satisfied parents correlate strongly with high-performing students.\n")
    elif survey == "No" or satisfaction == "Bad":
        analysis.append("[Analysis] Lower Parental Engagement/Satisfaction:")
        analysis.append("   Lack of parental survey response or low school satisfaction can be an indicator of lower overall support.\n")

    # Conclusion
    analysis.append("[Summary] Summary Analysis:")
    if prediction == "H":
        analysis.append("   The algorithm recognizes that strong behavioral habits and attendance statistically guarantee a High (H) final grade.")
    elif prediction == "M":
        analysis.append("   The student's mixed metrics result in an average Medium (M) predicted performance. Improving engagement or attendance could boost this.")
    else:
        analysis.append("   The model flags this student as high risk for a Low (L) grade due to lower engagement or poor attendance. Early intervention is recommended.")
        
    return "\n".join(analysis)
