# 🫀 VOXHEART - AI-Powered Heart Disease Predictor

**VOXHEART** is an intelligent, voice-enhanced, and visually rich heart disease prediction system. It uses machine learning, personalized medical advice, and OpenAI-generated explanations to empower users with real-time health insights.

> 🔬 Developed by [@Nithin & @Lalitha](https://github.com/Nani1-glitch) — with ❤️ for precision & professionalism.

---

## 🚀 Features

- 🧠 **ML-Powered Prediction** using Logistic Regression  
- 🗣️ **Voice Input** for hands-free health assessment *(temporarily disabled for refinement)*  
- 📊 **BMI & Blood Pressure Visualizations**  
- 💬 **AI-Generated Explanation** via OpenAI (GPT-3.5)  
- 📝 **Personalized Medical & Lifestyle Suggestions**  
- 📄 **Professional Hospital-style PDF Export** *(with charts, logo & results)*  
- 📦 **Data Logging** for future prediction tracking  
- 📞 **Coming Soon**: Phone Notification Reminders  

---

## 🏗️ Tech Stack

- **Frontend**: HTML, CSS, JavaScript (vanilla + Web Speech API)  
- **Backend**: Flask (Python), NumPy, Matplotlib, Pandas, Joblib  
- **AI/ML**: Logistic Regression, Scikit-learn, OpenAI API  
- **PDF Export**: ReportLab  
- **Voice Module**: Web Speech API + JavaScript *(in progress)*  

---

## 📂 Project Structure

heart_disease_predictor/  
├── app/  
│   ├── static/ # Plots, Assets  
│   ├── templates/  
│   │   └── index.html # Main UI Template  
│   ├── app.py # Flask App  
│   ├── generate_pdf.py # PDF Report Generator  
│   ├── health_advice.py # Custom Advice Logic  
│   ├── voice_input.py # Voice Form Handling (WIP)  
│   └── models/  
│       ├── lr_weights.npz # Logistic Regression Weights  
│       └── scaler.pkl # Feature Scaler  
├── data/  
│   └── retrieved/  
│       └── predictions.csv # Saved Results  
├── README.md  
└── requirements.txt

---

## 🖥️ Local Setup

### 🔧 Requirements

- Python 3.10+  
- Flask  
- Matplotlib  
- Pandas  
- Scikit-learn  
- OpenAI  
- ReportLab  

### 📦 Installation

```bash
git clone https://github.com/your-username/voxheart.git
cd voxheart
pip install -r requirements.txt
🔐 Add your OpenAI key to .env as:
OPENAI_API_KEY=your-api-key ```

▶️ Run the App
```bash
cd app
python app.py
Now open your browser and go to:
📍 http://127.0.0.1:5000 ```

**### 📄 Sample Output PDF**
The PDF report contains:

- VOXHEART Logo 🩺  
- Prediction Summary ✅/❌  
- User Inputs & BMI Stats  
- Explanation by GPT 🧠  
- BMI & BP Charts  
- Medical & Lifestyle Advice (dual column layout)  

🧾 Exported via ReportLab, styled like a real hospital report.

**### ⚙️ Upcoming Features ** 
✅ Mobile/Email Reminder Notification  
✅ Chart.js Animated Graphs  
✅ Full Voice-Controlled Form Submission (Enhanced)  
✅ Auto Email Health Summary Report  
