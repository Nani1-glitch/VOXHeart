from flask import Flask, render_template, request, send_file
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import joblib
from health_advice import generate_health_advice
from dotenv import load_dotenv
import openai
from voice_input import collect_user_voice_input
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ─── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
# Holds last processed session for PDF export
last_session = {}

# ─── Helpers ────────────────────────────────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_from_input(X_scaled):
    data = np.load("models/lr_weights.npz")
    W, b = data["W"], data["b"]
    A = sigmoid(np.dot(X_scaled, W) + b)
    return int((A >= 0.5).astype(int)[0][0])

def generate_natural_explanation(user_data):
    prompt = f"""
You are a medical assistant. Based on the following user inputs, write a simple 3-line explanation of their heart disease risk:

- Age: {user_data['age']}
- Gender: {"Male" if user_data['gender']==2 else "Female"}
- Height: {user_data['height']} cm
- Weight: {user_data['weight']} kg
- Systolic BP: {user_data['ap_hi']} mmHg
- Diastolic BP: {user_data['ap_lo']} mmHg
- Cholesterol Level: {user_data['cholesterol']}
- Glucose Level: {user_data['gluc']}
- Smoker: {"Yes" if user_data['smoke']==1 else "No"}
- Alcohol Intake: {"Yes" if user_data['alco']==1 else "No"}
- Physical Activity: {"Yes" if user_data['active']==1 else "No"}

Explain in 3 lines what this means for the person’s heart health.
"""
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful, concise medical assistant."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return resp.choices[0].message["content"].strip()
    except Exception as e:
        return f"⚠️ Unable to generate explanation: {e}"

# ─── Core Processing ────────────────────────────────────────────────────────────
def process_user_input(user_data):
    # 1. Basic validation to avoid zero‑division
    if user_data["height"] <= 0 or user_data["weight"] <= 0:
        raise ValueError("Height and weight must be greater than zero.")

    # 2. Scale & predict
    features = [[
        user_data["age"], user_data["gender"], user_data["height"], user_data["weight"],
        user_data["ap_hi"], user_data["ap_lo"], user_data["cholesterol"], user_data["gluc"],
        user_data["smoke"], user_data["alco"], user_data["active"]
    ]]
    scaler = joblib.load("models/scaler.pkl")
    X_scaled = scaler.transform(features)
    prediction = predict_from_input(X_scaled)

    # 3. BMI calc & category
    bmi = user_data["weight"] / ((user_data["height"]/100)**2)
    if bmi < 18.5:       bmi_cat = "Underweight"
    elif bmi < 25:       bmi_cat = "Normal"
    elif bmi < 30:       bmi_cat = "Overweight"
    else:                bmi_cat = "Obese"

    # 4. Prepare filesystem paths
    timestamp = int(time.time())
    plots_dir = os.path.join(app.static_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    bmi_abs = os.path.join(plots_dir, f"bmi_{timestamp}.png")
    bp_abs  = os.path.join(plots_dir, f"bp_{timestamp}.png")

    # 5. Draw & save BMI plot
    plt.figure(figsize=(6,2))
    plt.axhline(1, xmin=0, xmax=4, color="gray", linewidth=12)
    clr = "green" if bmi<25 else "orange" if bmi<30 else "red"
    plt.plot([bmi],[1],"o",color=clr,markersize=18)
    plt.yticks([]); plt.xticks([15,18.5,25,30,40],["15","18.5","25","30","40"])
    plt.title(f"BMI: {bmi:.1f} ({bmi_cat})")
    plt.tight_layout(); plt.savefig(bmi_abs); plt.close()

    # 6. Draw & save BP plot
    plt.figure(figsize=(5,3))
    plt.bar(["Systolic","Diastolic"], [user_data["ap_hi"],user_data["ap_lo"]], color=["skyblue","lightgreen"])
    plt.axhline(120, color="blue", linestyle="--", label="Normal Systolic")
    plt.axhline(80,  color="green", linestyle="--", label="Normal Diastolic")
    plt.title("Your Blood Pressure"); plt.legend()
    plt.tight_layout(); plt.savefig(bp_abs); plt.close()

    # 7. AI explanation & health advice
    explanation       = generate_natural_explanation(user_data)
    advice_left, advice_right = generate_health_advice({**user_data, "bmi": bmi})

    # 8. Persist row
    os.makedirs("data/retrieved", exist_ok=True)
    row = { **user_data, "prediction": prediction }
    pd.DataFrame([row]).to_csv(
        "data/retrieved/predictions.csv",
        mode="a",
        header=not os.path.exists("data/retrieved/predictions.csv"),
        index=False
    )

    # 9. Store for PDF
    last_session.clear()
    last_session.update({
        "user_data":         row,
        "explanation":       explanation,
        "plot_bmi":          bmi_abs,
        "plot_bp":           bp_abs,
        "advice_left":       advice_left,
        "advice_right":      advice_right
    })

    # 10. Return template‑relative paths
    rel_bmi = f"plots/{os.path.basename(bmi_abs)}"
    rel_bp  = f"plots/{os.path.basename(bp_abs)}"
    return prediction, rel_bmi, rel_bp, advice_left, advice_right, explanation

# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET","POST"])
def index():
    error = None
    if request.method=="POST":
        try:
            form = request.form
            getf = lambda k, d: float(form.get(k) or d)
            user_data = {
                "age":         getf("age",0),
                "gender":      getf("gender",1),
                "height":      getf("height",0),
                "weight":      getf("weight",0),
                "ap_hi":       getf("ap_hi",0),
                "ap_lo":       getf("ap_lo",0),
                "cholesterol": getf("cholesterol",1),
                "gluc":        getf("gluc",1),
                "smoke":       getf("smoke",0),
                "alco":        getf("alco",0),
                "active":      getf("active",1),
            }
            vals = process_user_input(user_data)
            return render_template("index.html",
                                   prediction=vals[0],
                                   plot_bmi=vals[1],
                                   plot_bp=vals[2],
                                   advice_left=vals[3],
                                   advice_right=vals[4],
                                   explanation=vals[5])
        except Exception as e:
            error = str(e)

    return render_template("index.html", error=error)

@app.route("/download_pdf")
def download_pdf():
    if not last_session:
        return "❌ No report to export.", 400

    data        = last_session["user_data"]
    explanation = last_session["explanation"]
    adv_l       = last_session["advice_left"]
    adv_r       = last_session["advice_right"]
    plot_bmi    = last_session["plot_bmi"]
    plot_bp     = last_session["plot_bp"]

    # Prepare PDF
    pdf_dir = os.path.join(app.static_folder, "pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"voxheart_report_{int(time.time())}.pdf")

    c = canvas.Canvas(pdf_path, pagesize=A4)
    w,h = A4

    # Logo
    logo_file = os.path.join(app.static_folder, "logo.png")
    if os.path.exists(logo_file):
        c.drawImage(ImageReader(logo_file), 50, h-100, width=80, height=60, mask="auto")

    c.setFont("Helvetica-Bold",16)
    c.drawString(150, h-70, "VOXHEART - Heart Disease Report")
    c.line(50, h-75, 540, h-75)

    # User info
    y = h-110; c.setFont("Helvetica",11)
    for k,v in data.items():
        c.drawString(50,y,f"{k.capitalize()}: {v}")
        y -= 18
    y -= 10

    # Charts
    if os.path.exists(plot_bmi):
        c.drawImage(ImageReader(plot_bmi), 50, y-120, width=220, height=70)
    if os.path.exists(plot_bp):
        c.drawImage(ImageReader(plot_bp), 300, y-120, width=220, height=90)
    y -= 140

    # Explanation
    c.setFont("Helvetica-Bold",12); c.drawString(50,y,"AI Explanation:")
    y -= 20; c.setFont("Helvetica",10)
    for line in explanation.split("\n"):
        c.drawString(60,y,line); y -= 14
    y -= 10

    # Medical advice
    c.setFont("Helvetica-Bold",12); c.drawString(50,y,"Medical Advice:")
    y -= 16; c.setFont("Helvetica",10)
    for item in adv_l:
        c.drawString(60,y,"- "+item); y -= 14
    y -= 10

    # Lifestyle tips
    c.setFont("Helvetica-Bold",12); c.drawString(300,y+10,"Lifestyle Tips:")
    c.setFont("Helvetica",10)
    for item in adv_r:
        c.drawString(310,y,"- "+item); y -= 14

    c.save()
    return send_file(pdf_path, as_attachment=True)

@app.route("/voice")
def voice_input():
    user_data, transcript = collect_user_voice_input()
    if not user_data:
        return render_template("index.html",
                               prediction=None,
                               explanation=None,
                               advice_left=["⚠️ Voice input failed."],
                               advice_right=[]
                              )

    # fill defaults
    user_data.setdefault("gender",1)
    user_data.setdefault("cholesterol",1)
    user_data.setdefault("gluc",1)
    user_data.setdefault("smoke",0)
    user_data.setdefault("alco",0)
    user_data.setdefault("active",1)

    p, b1, b2, al, ar, ex = process_user_input(user_data)
    return render_template("index.html",
                           prediction=p,
                           plot_bmi=b1,
                           plot_bp=b2,
                           advice_left=transcript["left"]+al,
                           advice_right=transcript["right"]+ar,
                           explanation=ex)

if __name__ == "__main__":
    app.run(debug=True)
