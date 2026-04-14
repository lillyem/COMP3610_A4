"""
FastAPI application for NYC Taxi Tip Amount Prediction.
Serves predictions from a trained Random Forest model using the same
preprocessing pipeline as Assignment 2.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import List
import joblib
import numpy as np
import pandas as pd
import uuid
import time
import os


# Global variables

ml_model = None
feature_columns = None
zone_lookup = None
start_time = None

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
FEATURE_COLUMNS_PATH = os.getenv("FEATURE_COLUMNS_PATH", "models/feature_columns.pkl")
ZONE_LOOKUP_PATH = os.getenv("ZONE_LOOKUP_PATH", "models/zone_lookup.pkl")
MODEL_VERSION = "1.0.0"



# Pydantic input / output schemas

class TripInput(BaseModel):
    """Input schema for a single taxi trip prediction."""
    trip_distance: float = Field(..., gt=0, le=100, description="Trip distance in miles")
    fare_amount: float = Field(..., ge=0, le=500, description="Fare amount in dollars")
    pickup_hour: int = Field(..., ge=0, le=23, description="Hour of pickup (0-23)")
    pickup_day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    passenger_count: int = Field(..., ge=1, le=9, description="Number of passengers")
    VendorID: int = Field(..., ge=1, le=6, description="Vendor ID")
    RatecodeID: int = Field(..., ge=1, le=99, description="Rate code ID")
    store_and_fwd_flag: int = Field(default=0, ge=0, le=1, description="Store and forward flag")
    PULocationID: int = Field(..., ge=1, le=265, description="Pickup location ID")
    DOLocationID: int = Field(..., ge=1, le=265, description="Dropoff location ID")
    payment_type: int = Field(default=1, eq=1, description="Payment type (1 = credit card only)")
    extra: float = Field(default=0.0, ge=0, le=100, description="Extra charges")
    mta_tax: float = Field(default=0.5, ge=0, le=100, description="MTA tax")
    tolls_amount: float = Field(default=0.0, ge=0, le=200, description="Tolls amount")
    improvement_surcharge: float = Field(default=1.0, ge=0, le=10, description="Improvement surcharge")
    congestion_surcharge: float = Field(default=2.5, ge=0, le=100, description="Congestion surcharge")
    Airport_fee: float = Field(default=0.0, ge=0, le=100, description="Airport fee")
    trip_duration_minutes: float = Field(..., gt=0, le=1440, description="Trip duration in minutes")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "trip_distance": 3.5,
                    "fare_amount": 18.0,
                    "pickup_hour": 14,
                    "pickup_day_of_week": 2,
                    "passenger_count": 1,
                    "VendorID": 1,
                    "RatecodeID": 1,
                    "store_and_fwd_flag": 0,
                    "PULocationID": 140,
                    "DOLocationID": 236,
                    "payment_type": 1,
                    "extra": 1.0,
                    "mta_tax": 0.5,
                    "tolls_amount": 0.0,
                    "improvement_surcharge": 1.0,
                    "congestion_surcharge": 2.5,
                    "Airport_fee": 0.0,
                    "trip_duration_minutes": 15.0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for a single prediction."""
    tip_amount: float
    prediction_id: str
    model_version: str


class BatchInput(BaseModel):
    """Input schema for batch predictions (max 100 records)."""
    records: List[TripInput] = Field(..., max_length=100)


class BatchResponse(BaseModel):
    """Output schema for batch predictions."""
    predictions: List[PredictionResponse]
    count: int
    processing_time_ms: float



# Preprocessing 

def preprocess_input(trip: TripInput) -> pd.DataFrame:
    """
    Convert a single TripInput into a feature DataFrame.
    """
    data = {
        "VendorID": trip.VendorID,
        "passenger_count": trip.passenger_count,
        "trip_distance": trip.trip_distance,
        "RatecodeID": trip.RatecodeID,
        "store_and_fwd_flag": trip.store_and_fwd_flag,
        "PULocationID": trip.PULocationID,
        "DOLocationID": trip.DOLocationID,
        "payment_type": trip.payment_type,
        "fare_amount": trip.fare_amount,
        "extra": trip.extra,
        "mta_tax": trip.mta_tax,
        "tolls_amount": trip.tolls_amount,
        "improvement_surcharge": trip.improvement_surcharge,
        "congestion_surcharge": trip.congestion_surcharge,
        "Airport_fee": trip.Airport_fee,
        "trip_duration_minutes": trip.trip_duration_minutes,
        "pickup_hour": trip.pickup_hour,
        "pickup_day_of_week": trip.pickup_day_of_week,
    }

    # Engineered features 
    data["is_weekend"] = int(trip.pickup_day_of_week in (5, 6))
    data["log_trip_distance"] = np.log1p(trip.trip_distance)

    # trip_speed_mph: avoid division by zero
    if trip.trip_duration_minutes > 0:
        data["trip_speed_mph"] = trip.trip_distance / (trip.trip_duration_minutes / 60)
    else:
        data["trip_speed_mph"] = 0.0

    # fare_per_mile: avoid division by zero
    if trip.trip_distance > 0:
        data["fare_per_mile"] = trip.fare_amount / trip.trip_distance
    else:
        data["fare_per_mile"] = 0.0

    # fare_per_minute: avoid division by zero
    if trip.trip_duration_minutes > 0:
        data["fare_per_minute"] = trip.fare_amount / trip.trip_duration_minutes
    else:
        data["fare_per_minute"] = 0.0

    # Borough lookup using zone_lookup dict
    pickup_borough = zone_lookup.get(trip.PULocationID, "Unknown")
    dropoff_borough = zone_lookup.get(trip.DOLocationID, "Unknown")

    # One-hot encode boroughs 
    boroughs = ["Bronx", "Brooklyn", "EWR", "Manhattan", "Queens", "Staten Island", "Unknown"]
    for b in boroughs:
        data[f"pickup_{b}"] = (pickup_borough == b)
        data[f"dropoff_{b}"] = (dropoff_borough == b)

    df = pd.DataFrame([data])

    # Reindex to match training column order exactly, fill missing with 0
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df



# Lifespan handler – load model once at startup

async def lifespan(app: FastAPI):
    """Load model artifacts once at startup."""
    global ml_model, feature_columns, zone_lookup, start_time

    ml_model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    zone_lookup = joblib.load(ZONE_LOOKUP_PATH)
    start_time = time.time()
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Feature columns: {len(feature_columns)}")

    yield

    print("Shutting down...")



# FastAPI app

app = FastAPI(
    title="NYC Taxi Tip Predictor",
    description="Predicts tip_amount for NYC yellow taxi trips using a tuned Random Forest model.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)



# Endpoints

@app.get("/")
def root():
    return {"message": "NYC Taxi Tip Predictor API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: TripInput):
    """Return a single tip_amount prediction."""
    features = preprocess_input(input_data)
    prediction = ml_model.predict(features)[0]

    return PredictionResponse(
        tip_amount=round(float(prediction), 2),
        prediction_id=str(uuid.uuid4()),
        model_version=MODEL_VERSION,
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchInput):
    """Return predictions for a batch of trip records (max 100)."""
    start = time.time()
    predictions = []

    for record in batch.records:
        features = preprocess_input(record)
        pred = ml_model.predict(features)[0]
        predictions.append(
            PredictionResponse(
                tip_amount=round(float(pred), 2),
                prediction_id=str(uuid.uuid4()),
                model_version=MODEL_VERSION,
            )
        )

    elapsed_ms = (time.time() - start) * 1000

    return BatchResponse(
        predictions=predictions,
        count=len(predictions),
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.get("/health")
def health_check():
    """Return API health status."""
    return {
        "status": "healthy",
        "model_loaded": ml_model is not None,
        "model_version": MODEL_VERSION,
        "uptime_seconds": round(time.time() - start_time, 1) if start_time else 0,
    }


@app.get("/model/info")
def model_info():
    """Return metadata about the loaded model."""
    return {
        "model_name": "taxi-tip-regressor",
        "version": MODEL_VERSION,
        "algorithm": "RandomForestRegressor (tuned)",
        "hyperparameters": {
            "n_estimators": 40,
            "max_depth": 12,
            "min_samples_split": 40,
            "min_samples_leaf": 8,
            "max_features": "sqrt",
            "random_state": 42,
        },
        "features": list(feature_columns) if feature_columns is not None else [],
        "metrics": {
            "MAE": 1.1636,
            "RMSE": 2.2842,
            "R2": 0.6292,
        },
        "trained_date": "2026-04-13",
        "dataset": "NYC Yellow Taxi Trip Records (Jan 2024)",
    }



# Global exception handler

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unexpected errors and return structured JSON (no internal details)."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again.",
        },
    )
