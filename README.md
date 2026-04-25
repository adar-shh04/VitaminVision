<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/MongoDB-Atlas-47A248?style=for-the-badge&logo=mongodb&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
</p>

<h1 align="center">🔬 VitaminVision</h1>
<p align="center">
  <strong>Production-Grade AI Nutrient Detection System</strong><br/>
  Decoupled FastAPI backend + MongoDB storage + Streamlit frontend
</p>

---

## 🌟 Project Evolution
Originally built as a monolithic Streamlit app, **VitaminVision** has been refactored into a modern, decoupled architecture using **Clean Architecture** principles. It now features a high-performance FastAPI backend, persistent MongoDB storage, and a lightweight Streamlit client.

---

## 🛠️ Modern Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend API** | FastAPI (Python), Uvicorn |
| **Database** | MongoDB Atlas (NoSQL) |
| **Database Driver** | Motor (Asynchronous PyMongo) |
| **Frontend** | Streamlit |
| **ML Model** | TensorFlow/Keras (VGG19 Transfer Learning) |
| **Validation** | Pydantic v2 |
| **Environment** | Python Dotenv, Pydantic Settings |

---

## 🧠 Key Features
- **Decoupled Architecture**: Separated frontend (Streamlit) from backend (FastAPI) for better scalability.
- **Async Database Integration**: Real-time logging of predictions to MongoDB Atlas using asynchronous drivers.
- **Singleton ML Inference**: Optimized performance by loading the VGG19 model once on startup.
- **Historical Tracking**: A dedicated dashboard to view past predictions and nutritional insights.
- **Automated Validation**: Strict request/response validation using Pydantic models.

---

## 📁 Project Structure

```
VitaminVision/
├── backend/                  # FastAPI Backend Service
│   ├── app/
│   │   ├── main.py           # API Entry point & Middleware
│   │   ├── routes/           # REST Endpoints (/predict, /history)
│   │   ├── services/         # ML Inference & Business Logic
│   │   ├── db/               # MongoDB Connection Logic
│   │   ├── models/           # DB Domain Models
│   │   └── schemas/          # Pydantic Request/Response Models
│   ├── .env                  # Configuration (DB URIs)
│   └── requirements.txt      # Backend Dependencies
│
├── streamlit_app.py          # Refactored Frontend Client
├── models/                   # AI Model storage (my_model.h5)
├── notebooks/                # Model Training (VGG19)
└── docs/                     # Research & Planning Documentation
```

---

## ▶️ Running Locally

### 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
# Add your MONGODB_URI to .env
uvicorn app.main:app --reload
```
*API docs available at: http://localhost:8000/docs*

### 2. Frontend Setup
```bash
# In a new terminal at root
streamlit run streamlit_app.py
```

---

## 🧪 Model Architecture
- **Base Model**: VGG19 (Pre-trained on ImageNet)
- **Classification**: 5 Classes (Vitamin A, B, C, D, E)
- **Input**: 224x224 RGB Images
- **Optimization**: Dropout, Early Stopping, and Learning Rate Scheduling.

---

## 📊 Limitations & Learnings
- **Visual Proxy**: Vitamin content is inferred via food classification; it is not a direct chemical analysis.
- **Asynchronous Design**: Learned the importance of non-blocking I/O when handling both ML inference and database writes.
- **Separation of Concerns**: Migrating from monolith to microservices-style architecture significantly improved code readability and deployment flexibility.

---

<p align="center">
  Built with ❤️ and Modern Backend Best Practices
</p>
