from flask import Flask, render_template, request, send_file, url_for
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

# ReportLab imports
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle

# ─── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
last_session = {}  # holds data for PDF export

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
                {"role":"system","content":"You are a helpful, concise medical assistant."},
                {"role":"user",  "content":prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return resp.choices[0].message["content"].strip()
    except Exception as e:
        return f"⚠️ Unable to generate explanation: {e}"

def process_user_input(user_data):
    if user_data["height"] <= 0 or user_data["weight"] <= 0:
        raise ValueError("Height and weight must be greater than zero.")

    features = [[
        user_data["age"], user_data["gender"], user_data["height"], user_data["weight"],
        user_data["ap_hi"], user_data["ap_lo"], user_data["cholesterol"], user_data["gluc"],
        user_data["smoke"], user_data["alco"], user_data["active"]
    ]]
    scaler = joblib.load("models/scaler.pkl")
    X_scaled = scaler.transform(features)
    prediction = predict_from_input(X_scaled)

    bmi = user_data["weight"] / ((user_data["height"]/100)**2)
    if bmi < 18.5:       bmi_cat = "Underweight"
    elif bmi < 25:       bmi_cat = "Normal"
    elif bmi < 30:       bmi_cat = "Overweight"
    else:                bmi_cat = "Obese"

    ts = int(time.time())
    plot_dir = os.path.join(app.static_folder, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    bmi_png = os.path.join(plot_dir, f"bmi_{ts}.png")
    bp_png  = os.path.join(plot_dir, f"bp_{ts}.png")

    # Draw BMI plot
    plt.figure(figsize=(6,2))
    plt.axhline(1, xmin=0, xmax=4, color="gray", linewidth=12)
    color = "green" if bmi<25 else "orange" if bmi<30 else "red"
    plt.plot([bmi],[1],"o",color=color,markersize=18)
    plt.yticks([]); plt.xticks([15,18.5,25,30,40],["15","18.5","25","30","40"])
    plt.title(f"BMI: {bmi:.1f} ({bmi_cat})")
    plt.tight_layout(); plt.savefig(bmi_png); plt.close()

    # Draw BP plot
    plt.figure(figsize=(5,3))
    plt.bar(["Systolic","Diastolic"], [user_data["ap_hi"],user_data["ap_lo"]],
            color=["skyblue","lightgreen"])
    plt.axhline(120, color="blue", linestyle="--", label="Normal Systolic")
    plt.axhline(80,  color="green", linestyle="--", label="Normal Diastolic")
    plt.title("Your Blood Pressure"); plt.legend()
    plt.tight_layout(); plt.savefig(bp_png); plt.close()

    explanation = generate_natural_explanation(user_data)
    adv_l, adv_r = generate_health_advice({**user_data,"bmi":bmi})

    os.makedirs("data/retrieved", exist_ok=True)
    row = { **user_data, "prediction": prediction }
    pd.DataFrame([row]).to_csv(
        "data/retrieved/predictions.csv",
        mode="a",
        header=not os.path.exists("data/retrieved/predictions.csv"),
        index=False
    )

    last_session.clear()
    last_session.update({
        "user_data":    row,
        "explanation":  explanation,
        "plot_bmi":     bmi_png,
        "plot_bp":      bp_png,
        "advice_left":  adv_l,
        "advice_right": adv_r
    })

    return (
        prediction,
        f"plots/{os.path.basename(bmi_png)}",
        f"plots/{os.path.basename(bp_png)}",
        adv_l, adv_r, explanation
    )

# ─── Flask Routes ──────────────────────────────────────────────────────────────
@app.route("/", methods=["GET","POST"])
def index():
    error = None
    if request.method=="POST":
        try:
            f    = request.form
            getf = lambda k,d: float(f.get(k) or d)
            user_data = {
                "age":getf("age",0), "gender":getf("gender",1),
                "height":getf("height",0), "weight":getf("weight",0),
                "ap_hi":getf("ap_hi",0), "ap_lo":getf("ap_lo",0),
                "cholesterol":getf("cholesterol",1), "gluc":getf("gluc",1),
                "smoke":getf("smoke",0), "alco":getf("alco",0),
                "active":getf("active",1)
            }
            vals = process_user_input(user_data)
            return render_template("index.html",
                                   prediction   = vals[0],
                                   plot_bmi     = vals[1],
                                   plot_bp      = vals[2],
                                   advice_left  = vals[3],
                                   advice_right = vals[4],
                                   explanation  = vals[5])
        except Exception as e:
            error = str(e)
    return render_template("index.html", error=error)

@app.route("/download_pdf")
def download_pdf():
    if not last_session:
        return "❌ No report to export.", 400

    data    = last_session["user_data"]
    adv_l   = last_session["advice_left"]
    adv_r   = last_session["advice_right"]
    bmi_p   = last_session["plot_bmi"]
    bp_p    = last_session["plot_bp"]

    # Prepare PDF
    pdf_dir = os.path.join(app.static_folder, "pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_fp  = os.path.join(pdf_dir, f"voxheart_report_{int(time.time())}.pdf")
    c       = canvas.Canvas(pdf_fp, pagesize=A4)
    w, h    = A4

    # Logo & Title
    logo_fp = os.path.join(app.static_folder, "logo.png")
    if os.path.exists(logo_fp):
        size = 30 * mm
        c.drawImage(ImageReader(logo_fp),
                    (w-size)/2, h-45*mm,
                    size, size, mask="auto")
    c.setFont("Helvetica-Bold",16)
    c.drawCentredString(w/2, h-52*mm, "VOXHEART – Heart Disease Report")
    c.line(20*mm, h-54*mm, w-20*mm, h-54*mm)

    # Two-column grid
    margin_x = 20*mm
    gutter   = 8*mm
    col_w    = (w - 2*margin_x - gutter)/2
    top_y    = h - 60*mm

    # 1) User-details table (left)
    rows = [["Field","Value"]]
    for k,v in data.items():
        pretty = k.replace("_"," ").capitalize()
        if k=="gender": v = "Male" if v==2 else "Female"
        if k in ("smoke","alco","active"):
            v = "Yes" if v==1 else "No"
        rows.append([pretty, str(v)])
    tbl = Table(rows, colWidths=[col_w*0.4, col_w*0.6], rowHeights=6*mm)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#203a43")),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("INNERGRID",   (0,0), (-1,-1), 0.25, colors.white),
        ("BOX",         (0,0), (-1,-1), 0.5, colors.white),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
    ]))
    tbl.wrapOn(c, w, h)
    tbl.drawOn(c, margin_x, top_y - tbl._height)

    # 2) Charts (right)
    chart_x = margin_x + col_w + gutter
    bmi_h   = 35 * mm
    bmi_y   = top_y - bmi_h
    if os.path.exists(bmi_p):
        c.drawImage(ImageReader(os.path.join(app.static_folder, bmi_p)),
                    chart_x, bmi_y, col_w, bmi_h, mask="auto")
    bp_h    = 35 * mm
    bp_y    = bmi_y - bp_h - 5*mm
    if os.path.exists(bp_p):
        c.drawImage(ImageReader(os.path.join(app.static_folder, bp_p)),
                    chart_x, bp_y, col_w, bp_h, mask="auto")

    # 3) Medical Advice (below BP chart)
    advice_y = bp_y - 10*mm
    c.setFont("Helvetica-Bold",11)
    c.drawString(margin_x, advice_y, "Medical Advice:")
    c.setFont("Helvetica",9)
    ty = advice_y - 6*mm
    for line in adv_l:
        c.drawString(margin_x + 2*mm, ty, f"• {line}")
        ty -= 5*mm

    # 4) Lifestyle Tips (beneath Medical Advice)
    ty -= 5*mm  # extra spacing
    c.setFont("Helvetica-Bold",11)
    c.drawString(margin_x, ty, "Lifestyle Tips:")
    c.setFont("Helvetica",9)
    ty -= 6*mm
    for line in adv_r:
        c.drawString(margin_x + 2*mm, ty, f"• {line}")
        ty -= 5*mm

    c.save()
    return send_file(pdf_fp, as_attachment=True)

@app.route("/voice")
def voice():
    user_data, transcript = collect_user_voice_input()
    if not user_data:
        return render_template("index.html",
                               prediction=None,
                               explanation=None,
                               advice_left=["⚠️ Voice input failed."],
                               advice_right=[])
    user_data.setdefault("gender",1)
    user_data.setdefault("cholesterol",1)
    user_data.setdefault("gluc",1)
    user_data.setdefault("smoke",0)
    user_data.setdefault("alco",0)
    user_data.setdefault("active",1)

    p,b1,b2,al,ar,ex = process_user_input(user_data)
    return render_template("index.html",
                           prediction   = p,
                           plot_bmi     = b1,
                           plot_bp      = b2,
                           advice_left  = transcript["left"] + al,
                           advice_right = transcript["right"] + ar,
                           explanation  = ex)

if __name__ == "__main__":
    app.run(debug=True)
