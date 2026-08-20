# ImmuCore

**ML-Powered Health Risk Prediction**

ImmuCore is a focused, single-pipeline application: a user enters their health-check
numbers, a trained machine learning model predicts their risk for diabetes, and an
AI layer translates that raw prediction into plain-language severity and actionable
precautions.

> **⚠️ Medical Disclaimer:** ImmuCore is a student/demo project. It does **not**
> provide medical advice, diagnosis, or treatment. Always consult a qualified
> healthcare provider for medical decisions.

---

## How It Works

```
┌─────────────┐      ┌─────────────┐      ┌─────────────────┐
│  User fills  │ ---> │  ML model   │ ---> │  AI explanation  │
│  health form │      │  predicts   │      │  + precautions   │
└─────────────┘      │  risk class │      └─────────────────┘
                     │  + score    │
                     └─────────────┘
```

1. **Input** — User fills a short form with health-check values (glucose, BMI, blood
   pressure, age, etc.)
2. **Predict** — The backend runs a trained ML model and returns a risk class and
   confidence score
3. **Explain** — An LLM call generates a plain-language severity read and concrete
   precautions

---

## Tech Stack

| Layer     | Technology                        |
|-----------|-----------------------------------|
| ML Model  | scikit-learn (Python)             |
| API       | FastAPI                           |
| AI Layer  | LLM API (prompt-based)           |
| Frontend  | Plain HTML, CSS, vanilla JS      |
| Dataset   | Pima Indians Diabetes (Kaggle)   |

---

## Project Structure

```
ImmuCore/
├── model/                  # ML pipeline
│   ├── data/               # Raw dataset (CSV)
│   ├── notebooks/          # EDA notebooks and charts
│   ├── src/                # Python scripts — data loading, preprocessing, EDA
│   └── exports/            # Trained model artifacts (after Week 2)
├── api/                    # FastAPI backend (Week 3)
├── web/                    # Frontend — HTML, CSS, vanilla JS
│   ├── index.html
│   ├── style.css
│   └── script.js
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Dataset — Pima Indians Diabetes

- **Source:** [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Samples:** 768
- **Task:** Binary classification (diabetic vs. not diabetic)
- **Features (8):**

| Feature                    | Description                          | Unit / Range         |
|----------------------------|--------------------------------------|----------------------|
| Pregnancies                | Number of pregnancies                | 0–17                 |
| Glucose                    | Plasma glucose (2-hr oral GTT)       | mg/dL                |
| BloodPressure              | Diastolic blood pressure             | mm Hg                |
| SkinThickness              | Triceps skinfold thickness           | mm                   |
| Insulin                    | 2-hour serum insulin                 | μU/mL                |
| BMI                        | Body mass index                      | kg/m²                |
| DiabetesPedigreeFunction   | Diabetes pedigree (family history)   | 0.0–2.5              |
| Age                        | Age in years                         | 21–81                |

- **Target:** `Outcome` — 1 = diabetic, 0 = not diabetic

All features are numeric and represent values a person can realistically self-report
or obtain from a standard health check-up.

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/ImmuForge/ImmuCore.git
cd ImmuCore

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify the dataset loads
python model/src/load_data.py

# 5. Run the EDA script
python model/src/eda.py
```

### View the Frontend

Open `web/index.html` directly in your browser — no build step or server needed.

---

## Build Roadmap

| Week | Owner  | Goal                                                    | Status      |
|------|--------|---------------------------------------------------------|-------------|
| 1    | Dhruv  | Scaffolding, dataset, EDA, frontend shell               | ✅ Done     |
| 2    | Tushar | Train & evaluate model, export artifact, prompt template | ⬜ Upcoming |
| 3    | Dhruv  | FastAPI, AI explanation layer, wire frontend, deploy     | ⬜ Upcoming |
| 4    | Tushar | Testing, bias checks, model card, final docs            | ⬜ Upcoming |

---

## Team

- **Dhruv** — Web development + ML integration
- **Tushar** — ML modeling + evaluation

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.