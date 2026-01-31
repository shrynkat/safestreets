"""
SafeStreets Risk Prediction API

FastAPI application for serving accident risk predictions using the trained XGBoost model.

Endpoints:
- POST /predict: Single prediction
- POST /predict/batch: Batch predictions
- GET /health: Health check
- GET /model/info: Model metadata
"""

import json
import logging
import pickle
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models import (
    AccidentFeatures,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ContributingFactor,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RiskLevel,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "risk_model.pkl"
METRICS_PATH = PROJECT_ROOT / "ml" / "models" / "metrics.json"

# Global state
app_state = {
    "model": None,
    "feature_names": None,
    "feature_importances": None,
    "metrics": None,
    "model_loaded": False,
    "start_time": None,
    "model_version": "1.0.0",
    "training_date": None
}

# Feature encoding mappings (must match training pipeline)
TIME_PERIOD_MAP = {
    'Night': 0,
    'Morning': 1,
    'Afternoon': 2,
    'Evening': 3
}

# Common weather conditions for encoding
WEATHER_CONDITIONS = [
    'Clear', 'Cloudy', 'Overcast', 'Rain', 'Light Rain', 'Heavy Rain',
    'Snow', 'Light Snow', 'Heavy Snow', 'Fog', 'Haze', 'Thunderstorm',
    'Drizzle', 'Sleet', 'Smoke', 'Windy', 'Fair', 'Partly Cloudy'
]


def get_hour_to_time_period(hour: int) -> int:
    """Convert hour to time period encoding."""
    if 6 <= hour < 12:
        return 1  # Morning
    elif 12 <= hour < 18:
        return 2  # Afternoon
    elif 18 <= hour < 22:
        return 3  # Evening
    else:
        return 0  # Night


def encode_weather_condition(condition: Optional[str]) -> int:
    """Encode weather condition to integer."""
    if condition is None:
        return 0
    condition_lower = condition.lower()
    for i, wc in enumerate(WEATHER_CONDITIONS):
        if wc.lower() in condition_lower or condition_lower in wc.lower():
            return i
    return 0  # Default to 'Clear'


def load_model():
    """Load the trained model and metadata."""
    logger.info(f"Loading model from {MODEL_PATH}...")

    if not MODEL_PATH.exists():
        logger.error(f"Model file not found: {MODEL_PATH}")
        return False

    try:
        # Load model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        app_state["model"] = model
        app_state["feature_names"] = model.get_booster().feature_names
        app_state["feature_importances"] = dict(
            zip(app_state["feature_names"], model.feature_importances_)
        )

        logger.info(f"Model loaded successfully with {len(app_state['feature_names'])} features")

        # Load metrics if available
        if METRICS_PATH.exists():
            with open(METRICS_PATH, 'r') as f:
                metrics_data = json.load(f)
            app_state["metrics"] = metrics_data.get("metrics", {})
            app_state["training_date"] = metrics_data.get("timestamp")
            logger.info("Model metrics loaded")

        app_state["model_loaded"] = True
        return True

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def warm_up_model():
    """Perform model warm-up with a dummy prediction."""
    if not app_state["model_loaded"]:
        return

    logger.info("Warming up model...")
    try:
        # Create dummy input
        dummy_features = pd.DataFrame([{
            name: 0.0 for name in app_state["feature_names"]
        }])
        app_state["model"].predict_proba(dummy_features)
        logger.info("Model warm-up completed")
    except Exception as e:
        logger.warning(f"Model warm-up failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting SafeStreets Risk Prediction API...")
    app_state["start_time"] = time.time()

    if load_model():
        warm_up_model()
    else:
        logger.warning("API starting without model - predictions will fail")

    yield

    # Shutdown
    logger.info("Shutting down API...")


# Initialize FastAPI app
app = FastAPI(
    title="SafeStreets Risk Prediction API",
    description="""
    API for predicting accident risk based on location, weather, and road conditions.

    ## Features
    - Real-time risk predictions using trained XGBoost model
    - Batch prediction support for multiple locations
    - Risk level classification (LOW/MEDIUM/HIGH)
    - Contributing factor analysis

    ## Usage
    Send location and environmental features to get risk predictions.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            details={"path": str(request.url)}
        ).model_dump(mode='json')
    )


def preprocess_features(features: AccidentFeatures) -> pd.DataFrame:
    """
    Preprocess input features to match training pipeline.

    Transforms raw input features into the format expected by the model.
    """
    # Determine time period
    if features.time_period:
        time_period_encoded = TIME_PERIOD_MAP.get(features.time_period.value, 0)
    else:
        time_period_encoded = get_hour_to_time_period(features.hour)

    # Calculate derived features
    is_rush_hour = 1 if (7 <= features.hour <= 9) or (16 <= features.hour <= 19) else 0
    poor_weather = 1 if (features.visibility_mi < 5) or (features.precipitation_in > 0) else 0
    extreme_temp = 1 if (features.temperature_f < 32) or (features.temperature_f > 95) else 0

    # Road feature count
    road_features = [
        features.has_amenity, features.has_bump, features.has_crossing,
        features.has_give_way, features.has_junction, features.has_no_exit,
        features.has_railway, features.has_roundabout, features.has_station,
        features.has_stop, features.has_traffic_calming, features.has_traffic_signal
    ]
    road_feature_count = sum(1 for f in road_features if f)

    # Encode categorical features
    # State encoding (simple hash-based for inference)
    state_encoded = hash(features.state or '') % 100 if features.state else 0

    # City encoding (simple hash-based for inference)
    city_encoded = hash(features.city or '') % 10000 if features.city else 0

    # Weather encoding
    weather_encoded = encode_weather_condition(features.weather_condition)

    # Build feature dictionary matching training order
    feature_dict = {
        'hour': features.hour,
        'day_of_week': features.day_of_week,
        'is_weekend': int(features.is_weekend),
        'month': features.month,
        'time_period_encoded': time_period_encoded,
        'is_rush_hour': is_rush_hour,
        'state_encoded': state_encoded,
        'city_encoded': city_encoded,
        'latitude': features.latitude,
        'longitude': features.longitude,
        'temperature_f': features.temperature_f,
        'humidity_pct': features.humidity_pct,
        'pressure_in': features.pressure_in,
        'visibility_mi': features.visibility_mi,
        'wind_speed_mph': features.wind_speed_mph,
        'precipitation_in': features.precipitation_in,
        'weather_encoded': weather_encoded,
        'poor_weather': poor_weather,
        'extreme_temp': extreme_temp,
        'has_amenity': int(features.has_amenity),
        'has_bump': int(features.has_bump),
        'has_crossing': int(features.has_crossing),
        'has_give_way': int(features.has_give_way),
        'has_junction': int(features.has_junction),
        'has_no_exit': int(features.has_no_exit),
        'has_railway': int(features.has_railway),
        'has_roundabout': int(features.has_roundabout),
        'has_station': int(features.has_station),
        'has_stop': int(features.has_stop),
        'has_traffic_calming': int(features.has_traffic_calming),
        'has_traffic_signal': int(features.has_traffic_signal),
        'road_feature_count': road_feature_count,
        'distance_mi': features.distance_mi
    }

    # Create DataFrame with correct column order
    if app_state["feature_names"]:
        # Use model's feature order
        ordered_features = {}
        for name in app_state["feature_names"]:
            ordered_features[name] = feature_dict.get(name, 0.0)
        return pd.DataFrame([ordered_features])
    else:
        return pd.DataFrame([feature_dict])


def get_risk_level(probability: float) -> RiskLevel:
    """Classify risk level based on probability."""
    if probability < 0.3:
        return RiskLevel.LOW
    elif probability < 0.6:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.HIGH


def get_contributing_factors(
    features_df: pd.DataFrame,
    top_n: int = 5
) -> list[ContributingFactor]:
    """Get top contributing factors for the prediction."""
    if not app_state["feature_importances"]:
        return []

    factors = []
    importances = app_state["feature_importances"]

    # Sort features by importance
    sorted_features = sorted(
        importances.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    for feature_name, importance in sorted_features:
        value = None
        if feature_name in features_df.columns:
            value = float(features_df[feature_name].iloc[0])

        factors.append(ContributingFactor(
            feature=feature_name,
            importance=round(importance, 4),
            value=value
        ))

    return factors


def make_prediction(features: AccidentFeatures) -> PredictionResponse:
    """Generate prediction for a single instance."""
    if not app_state["model_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )

    # Preprocess features
    features_df = preprocess_features(features)

    # Get prediction
    model = app_state["model"]
    probabilities = model.predict_proba(features_df)

    # Risk score is probability of high severity (class 1)
    risk_score = float(probabilities[0][1])

    # Calculate confidence (distance from 0.5)
    confidence = abs(risk_score - 0.5) * 2

    # Get risk level
    risk_level = get_risk_level(risk_score)

    # Get contributing factors
    contributing_factors = get_contributing_factors(features_df)

    return PredictionResponse(
        risk_score=round(risk_score, 4),
        risk_level=risk_level,
        confidence=round(confidence, 4),
        contributing_factors=contributing_factors,
        prediction_timestamp=datetime.utcnow()
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        422: {"model": ErrorResponse, "description": "Validation error"}
    },
    summary="Single Risk Prediction",
    description="Predict accident risk for a single location with given features."
)
async def predict(features: AccidentFeatures) -> PredictionResponse:
    """
    Generate a risk prediction for a single location.

    The prediction includes:
    - **risk_score**: Probability of high-severity accident (0-1)
    - **risk_level**: Classification (LOW/MEDIUM/HIGH)
    - **confidence**: How confident the model is in the prediction
    - **contributing_factors**: Top features influencing the prediction
    """
    logger.info(f"Prediction request received for lat={features.latitude}, lon={features.longitude}")

    try:
        prediction = make_prediction(features)
        logger.info(f"Prediction: risk_score={prediction.risk_score}, level={prediction.risk_level}")
        return prediction

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        422: {"model": ErrorResponse, "description": "Validation error"}
    },
    summary="Batch Risk Predictions",
    description="Predict accident risk for multiple locations."
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Generate risk predictions for multiple locations.

    Accepts up to 1000 instances in a single request.
    Returns predictions in the same order as input.
    """
    start_time = time.time()
    logger.info(f"Batch prediction request received for {len(request.instances)} instances")

    if not app_state["model_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )

    try:
        predictions = []
        for features in request.instances:
            prediction = make_prediction(features)
            predictions.append(prediction)

        processing_time = (time.time() - start_time) * 1000

        logger.info(f"Batch prediction completed: {len(predictions)} predictions in {processing_time:.2f}ms")

        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            processing_time_ms=round(processing_time, 2)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the API."
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint for monitoring and load balancers.

    Returns the service status, whether the model is loaded, and uptime.
    """
    uptime = time.time() - app_state["start_time"] if app_state["start_time"] else 0

    status = "healthy" if app_state["model_loaded"] else "degraded"

    return HealthResponse(
        status=status,
        model_loaded=app_state["model_loaded"],
        uptime_seconds=round(uptime, 2),
        timestamp=datetime.utcnow()
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    },
    summary="Model Information",
    description="Get metadata about the loaded model."
)
async def model_info() -> ModelInfoResponse:
    """
    Get information about the currently loaded model.

    Includes model version, training date, features, and performance metrics.
    """
    if not app_state["model_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )

    return ModelInfoResponse(
        model_version=app_state["model_version"],
        model_type="XGBClassifier",
        training_date=app_state["training_date"],
        feature_count=len(app_state["feature_names"]) if app_state["feature_names"] else 0,
        feature_names=app_state["feature_names"] or [],
        performance_metrics=app_state["metrics"] or {}
    )


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirects to docs."""
    return {
        "message": "SafeStreets Risk Prediction API",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
