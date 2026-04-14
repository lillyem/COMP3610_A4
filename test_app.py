"""
Test suite for the NYC Taxi Tip Predictor API.
Covers: valid predictions, batch predictions, invalid input,
health check, model info, and edge cases.
"""

import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    """Create a TestClient that runs FastAPI startup/shutdown events."""
    with TestClient(app) as test_client:
        yield test_client


# A valid sample trip matching the Assignment 2 feature schema
VALID_TRIP = {
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


def test_root(client):
    """Root endpoint returns a welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health(client):
    """Health check returns healthy status with model loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_version" in data
    assert "uptime_seconds" in data


def test_predict_valid(client):
    """A valid trip returns a prediction with all required fields."""
    response = client.post("/predict", json=VALID_TRIP)
    assert response.status_code == 200
    data = response.json()
    assert "tip_amount" in data
    assert "prediction_id" in data
    assert "model_version" in data
    assert isinstance(data["tip_amount"], float)


def test_batch_prediction(client):
    """Batch endpoint returns correct count of predictions."""
    records = [VALID_TRIP] * 3
    response = client.post("/predict/batch", json={"records": records})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert len(data["predictions"]) == 3
    assert "processing_time_ms" in data


def test_model_info(client):
    """Model info endpoint returns metadata."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "features" in data
    assert "metrics" in data
    assert "hyperparameters" in data


# --- Validation tests ---

def test_predict_missing_field(client):
    """Missing required field should return 422."""
    incomplete = {"trip_distance": 3.5}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_invalid_type(client):
    """String where float expected should return 422."""
    bad_trip = VALID_TRIP.copy()
    bad_trip["trip_distance"] = "not a number"
    response = client.post("/predict", json=bad_trip)
    assert response.status_code == 422


def test_predict_out_of_range_negative_distance(client):
    """Negative trip_distance (violates gt=0) should return 422."""
    bad_trip = VALID_TRIP.copy()
    bad_trip["trip_distance"] = -1.0
    response = client.post("/predict", json=bad_trip)
    assert response.status_code == 422


def test_predict_pickup_hour_out_of_range(client):
    """pickup_hour > 23 should return 422."""
    bad_trip = VALID_TRIP.copy()
    bad_trip["pickup_hour"] = 25
    response = client.post("/predict", json=bad_trip)
    assert response.status_code == 422


def test_batch_prediction_too_large(client):
    """More than 100 records should return 422."""
    records = [VALID_TRIP] * 101
    response = client.post("/predict/batch", json={"records": records})
    assert response.status_code == 422


# --- Edge-case tests ---

def test_predict_short_trip(client):
    """Very short trip with minimal fare should still return a valid prediction."""
    edge_trip = VALID_TRIP.copy()
    edge_trip["trip_distance"] = 0.1
    edge_trip["fare_amount"] = 3.0
    edge_trip["trip_duration_minutes"] = 2.0
    response = client.post("/predict", json=edge_trip)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["tip_amount"], float)


def test_predict_high_fare(client):
    """Extreme fare values should still produce a prediction."""
    edge_trip = VALID_TRIP.copy()
    edge_trip["fare_amount"] = 450.0
    edge_trip["trip_distance"] = 50.0
    edge_trip["trip_duration_minutes"] = 120.0
    response = client.post("/predict", json=edge_trip)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["tip_amount"], float)