# app/health_advice.py

def generate_health_advice(user_input):
    """
    Returns two lists:
      - advice_left : Medical Advice
      - advice_right: Lifestyle Tips
    """
    advice_left = []
    advice_right = []

    age         = user_input.get("age", 0)
    bmi         = user_input.get("bmi", 0)
    cholesterol = int(user_input.get("cholesterol", 1))
    glucose     = int(user_input.get("gluc", 1))
    smoke       = int(user_input.get("smoke", 0))
    alco        = int(user_input.get("alco", 0))
    active      = int(user_input.get("active", 1))

    # ─── Medical Advice ────────────────────────────────────
    # Cholesterol
    if cholesterol == 2:
        advice_left.append("Your cholesterol is mildly elevated; reduce saturated fats & fried foods.")
    elif cholesterol == 3:
        advice_left.append("High cholesterol detected; increase soluble fiber (oats, legumes) & discuss meds.")

    # Glucose
    if glucose == 2:
        advice_left.append("Glucose above normal; cut down on sugary drinks and snacks.")
    elif glucose == 3:
        advice_left.append("High glucose levels; monitor for pre‑diabetes and consult your physician.")

    # BMI
    if bmi < 18.5:
        advice_left.append("Underweight: Focus on nutrient‑dense meals to support healthy weight gain.")
    elif bmi < 25:
        advice_left.append("BMI in healthy range: Maintain diet & exercise habits.")
    elif bmi < 30:
        advice_left.append("Overweight: Aim for 5‑10% body weight loss to improve heart health.")
    else:
        advice_left.append("Obese: Work with a dietitian for a structured weight‑loss plan.")

    # ─── Lifestyle Tips ────────────────────────────────────
    # Smoking
    if smoke == 1:
        advice_right.append("Smoking increases heart risk; seek a quitting program or support group.")
    else:
        advice_right.append("Great job staying smoke‑free!")

    # Alcohol
    if alco == 1:
        advice_right.append("Limit alcohol: no more than 1 drink/day (women) or 2/day (men).")
    else:
        advice_right.append("Abstaining from alcohol benefits your heart.")

    # Physical activity
    if active == 0:
        advice_right.append("Try 30 min brisk walking at least 5 days a week.")
    else:
        advice_right.append("Keep up your regular exercise routine!")

    # Age‑based tip
    if age >= 50:
        advice_right.append("Annual cardiovascular screenings are recommended after age 50.")

    return advice_left, advice_right
