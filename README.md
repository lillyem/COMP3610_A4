# COMP 3610 – Assignment 4: MLOps & Model Deployment

**Name:** Sonali Maharaj  
**Student ID:** 816034459  
**Course:** COMP 3610 – Big Data Analytics  

---

## Overview

This project deploys a trained Random Forest regression model as a production-ready prediction service

**Stack:** MLflow · FastAPI · Docker · scikit-learn

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Docker Desktop | Latest |
| pip | Latest |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/lillyem/COMP3610_A4.git
cd COMP3610_A4
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open `assignment4.ipynb` in Jupyter and run all cells sequentially. This will:

- Download the NYC taxi dataset
- Preprocess data using the Assignment 2 pipeline
- Train and log models to MLflow
- Register the best model
- Save model artifacts to `models/`

Note: The `models/` directory must contain the trained model artifacts (`model.pkl`, `feature_columns.pkl`, `zone_lookup.pkl`). These are generated when running the notebook.

### 4. Start the MLflow UI (optional)

```bash
mlflow ui --port 5000
```

Visit [http://localhost:5000](http://localhost:5000) to view tracked experiments.

### 5. Run the API locally

```bash
uvicorn app:app --reload --port 8000
```

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for the Swagger UI.

### 6. Run tests

```bash
pytest test_app.py -v
```

### 7. Run with Docker Compose

Docker Compose orchestrates both the API and MLflow services in a shared network environment.


```bash
docker compose up --build
```

This starts:
- **API service** on [http://localhost:8000](http://localhost:8000)
- **MLflow UI** on [http://localhost:5000](http://localhost:5000)

### 8. Test the containerized API

```bash
curl http://localhost:8000/health
```

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"trip_distance\": 3.5, \"fare_amount\": 18.0, \"pickup_hour\": 14, \"pickup_day_of_week\": 2, \"passenger_count\": 1, \"VendorID\": 1, \"RatecodeID\": 1, \"PULocationID\": 140, \"DOLocationID\": 236, \"trip_duration_minutes\": 15.0}"
```

### 9. Stop services

```bash
docker compose down
```

---

## Project Structure

```
├── assignment4.ipynb       ← Jupyter notebook documenting all work
├── app.py                  ← FastAPI application
├── test_app.py             ← pytest test suite
├── Dockerfile              ← Container recipe
├── docker-compose.yml      ← Multi-service orchestration
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── .gitignore              ← Git exclusions
├── .dockerignore           ← Docker build exclusions
└── models/                 ← Saved model artifacts (gitignored)
    ├── model.pkl
    ├── feature_columns.pkl
    └── zone_lookup.pkl
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Welcome message |
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions (max 100) |
| GET | `/docs` | Swagger UI |

---

## Model Details

- **Algorithm:** Tuned RandomForestRegressor
- **Target:** `tip_amount`
- **Dataset:** NYC Yellow Taxi Trip Records (January 2024)
- **Test metrics:** MAE = 1.164, RMSE = 2.285, R² = 0.629