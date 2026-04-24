<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<h1 align="center">🔬 VitaminVision</h1>
<p align="center">
  <strong>AI-Powered Nutrient Detection from Food Images</strong><br/>
  Upload a photo of any food → get instant vitamin predictions with confidence scores
</p>

---

## 🚀 Live Demo

👉 **[vitaminvision.streamlit.app](https://vitaminvision.streamlit.app)**

> The app runs in **demo mode** when the model file is not present. Place a trained model at `models/my_model.h5` for real predictions.

---

## 📋 Problem Statement

Tracking vitamin intake is difficult for most people. Existing diet-tracking tools focus on calories rather than micronutrients. VitaminVision explores whether **computer vision** can be used to analyze food images and provide vitamin-related insights.

> ⚠️ **Disclaimer:** Vitamins are chemical properties and cannot be directly inferred from images alone. This project is a **learning-oriented prototype**, not a medical-grade system.

---

## 🧠 How It Works

1. **Upload** a food image (JPG, PNG, WEBP)
2. **AI Analysis** — VGG19-based CNN processes the image
3. **Results** — Dominant vitamin prediction with confidence, benefits, and food sources

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit (deployed) / HTML + CSS (Flask local) |
| **Backend** | Python, Flask |
| **ML Model** | TensorFlow, Keras, VGG19 (Transfer Learning) |
| **Deployment** | Streamlit Cloud |

---

## 📁 Project Structure

```
VitaminVision/
├── streamlit_app.py          # Streamlit app (for deployment)
├── app.py                    # Flask app (for local development)
├── requirements.txt          # Python dependencies
├── .gitignore
├── README.md
│
├── models/                   # Place my_model.h5 here
├── templates/                # Flask HTML templates
│   ├── index.html
│   └── predict.html
├── static/css/               # Flask CSS
│   └── style.css
├── notebooks/                # Training notebook
│   └── Projectvv.ipynb
├── docs/                     # Documentation (consolidated)
│   ├── planning/
│   ├── data/
│   ├── model/
│   ├── optimization/
│   └── reports/
├── assets/demo/              # Screenshots and demo video
│
│  ── Academic Project Phases (certification) ──
├── 1. Project Initialization and Planning Phase/
├── 2. Data Collection and Preprocessing Phase/
├── 3. Model Development Phase/
├── 4. Model Optimization and Tuning Phase/
├── 5. Project Executable Files/
└── 6. Documentation and demonstration/
```

---

## ▶️ Quick Start

### Streamlit (Recommended)
```bash
git clone https://github.com/adar-shh04/VitaminVision.git
cd VitaminVision
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Flask (Local Development)
```bash
pip install flask tensorflow numpy Pillow
python app.py
# Visit http://127.0.0.1:5000
```

---

## 🧪 Model Details

| Property | Value |
|----------|-------|
| Architecture | VGG19 (Transfer Learning) |
| Pre-trained Weights | ImageNet |
| Input Size | 224 × 224 × 3 |
| Output Classes | Vitamin A, B, C, D, E |
| Preprocessing | Resize, Normalize (0–1), Augmentation |
| Regularization | Dropout, Early Stopping |

---

## 📊 Limitations & Learnings

- Vitamin content is **not visually deterministic** — this is a classification proxy
- Dataset size and label quality limited model performance
- Similar-looking foods caused misclassification
- The project demonstrates: ML problem framing, dataset bias, model evaluation beyond accuracy

---

## 🙏 Acknowledgments

- Built as part of the **SmartInternz Summer Project Internship**
- VIT University, Vellore, India

---

<p align="center">
  Built with ❤️ and Deep Learning
</p>
