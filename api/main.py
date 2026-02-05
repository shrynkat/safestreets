"""
SafeStreets Risk Prediction API

FastAPI application for serving accident risk predictions using the trained XGBoost model.

Endpoints:
- POST /predict: Single prediction
- POST /predict/batch: Batch predictions
- POST /analyze-route: Analyze route alternatives for safety
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
    DangerZone,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RiskLevel,
    RouteAnalysisRequest,
    RouteAnalysisResponse,
    RouteDetails,
)

# Import routing modules
from routing.directions_api import get_routes
from routing.risk_scorer import score_entire_route

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


def get_route_risk_level(risk_score: float) -> str:
    """Convert risk score to risk level string."""
    if risk_score < 0.3:
        return "LOW"
    elif risk_score < 0.5:
        return "MEDIUM"
    elif risk_score < 0.7:
        return "HIGH"
    else:
        return "CRITICAL"


@app.post(
    "/analyze-route",
    response_model=RouteAnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal error"}
    },
    summary="Analyze Route Safety",
    description="""
    Analyze multiple route alternatives between origin and destination for accident risk.

    This endpoint:
    1. Fetches route alternatives from Google Maps Directions API
    2. Scores each route segment using the trained ML model
    3. Calculates overall safety metrics for each route
    4. Identifies danger zones (high-risk segments)
    5. Returns routes sorted by safety score (safest first)

    ## Example Request

    ```bash
    curl -X POST "http://localhost:8000/analyze-route" \\
      -H "Content-Type: application/json" \\
      -d '{
        "origin": "Tempe, AZ",
        "destination": "Sedona, AZ",
        "departure_time": "2024-02-01T17:00:00",
        "weather_condition": "Clear",
        "temperature_f": 70.0,
        "visibility_mi": 10.0
      }'
    ```
    """
)
async def analyze_route(request: RouteAnalysisRequest) -> RouteAnalysisResponse:
    """
    Analyze route alternatives for accident risk.

    Fetches multiple route options and scores each for safety based on:
    - Time of day and day of week
    - Weather conditions
    - Road features along the route
    - Historical accident patterns

    Returns routes sorted by safety score with the safest option first.
    """
    logger.info(f"Route analysis request: {request.origin} -> {request.destination}")

    try:
        # Step 1: Fetch routes from Google Maps API
        # Google rejects past departure_time values, so pass None for past times.
        # Route geometry doesn't change — only traffic does — so we still use
        # the original departure_time for risk scoring (hour/day_of_week patterns).
        routing_departure = request.departure_time
        if request.departure_time and request.departure_time < datetime.utcnow():
            logger.info(f"Departure time {request.departure_time} is in the past, fetching routes without traffic timing")
            routing_departure = None

        try:
            routes = get_routes(
                origin=request.origin,
                destination=request.destination,
                departure_time=routing_departure
            )
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            raise HTTPException(
                status_code=503,
                detail="Google Maps API not configured. Please check API key."
            )
        except Exception as e:
            error_msg = str(e)
            if "REQUEST_DENIED" in error_msg:
                raise HTTPException(
                    status_code=503,
                    detail="Google Maps API access denied. Please check API key and billing."
                )
            elif "ZERO_RESULTS" in error_msg or "NOT_FOUND" in error_msg:
                raise HTTPException(
                    status_code=400,
                    detail=f"No routes found between '{request.origin}' and '{request.destination}'. Please check the addresses."
                )
            else:
                logger.error(f"Directions API error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to fetch routes: {error_msg}"
                )

        if not routes:
            raise HTTPException(
                status_code=400,
                detail=f"No routes found between '{request.origin}' and '{request.destination}'."
            )

        logger.info(f"Fetched {len(routes)} route alternatives")

        # Step 2: Score each route
        analyzed_routes = []

        for i, route in enumerate(routes, 1):
            try:
                # Score the route
                risk_result = score_entire_route(
                    waypoints=route['waypoints'],
                    departure_time=request.departure_time,
                    weather_condition=request.weather_condition,
                    temperature_f=request.temperature_f,
                    visibility_mi=request.visibility_mi
                )

                # Build danger zones list
                danger_zones = []
                for dz in risk_result.get('danger_zones', []):
                    danger_zones.append(DangerZone(
                        segment=dz['segment_index'],
                        risk=round(dz['risk_score'], 3),
                        location=f"({dz['lat']:.5f}, {dz['lon']:.5f})",
                        risk_level=dz['risk_level']
                    ))

                # Create route details
                route_details = RouteDetails(
                    route_id=i,
                    summary=route['summary'],
                    distance_text=route['distance_text'],
                    distance_meters=route['distance_meters'],
                    duration_text=route['duration_text'],
                    duration_seconds=route['duration_seconds'],
                    risk_score=round(risk_result['overall_risk'], 3),
                    risk_level=get_route_risk_level(risk_result['overall_risk']),
                    safety_score=round(risk_result['safety_score'], 1),
                    max_risk=round(risk_result['max_risk'], 3),
                    min_risk=round(risk_result['min_risk'], 3),
                    segments_analyzed=risk_result['segments_scored'],
                    danger_zones=danger_zones,
                    recommendation=None
                )

                analyzed_routes.append(route_details)

            except Exception as e:
                logger.error(f"Error scoring route {i}: {e}")
                # Include route with default risk values if scoring fails
                analyzed_routes.append(RouteDetails(
                    route_id=i,
                    summary=route['summary'],
                    distance_text=route['distance_text'],
                    distance_meters=route['distance_meters'],
                    duration_text=route['duration_text'],
                    duration_seconds=route['duration_seconds'],
                    risk_score=0.5,
                    risk_level="UNKNOWN",
                    safety_score=50.0,
                    max_risk=0.5,
                    min_risk=0.5,
                    segments_analyzed=0,
                    danger_zones=[],
                    recommendation="Scoring failed - use with caution"
                ))

        # Step 3: Sort routes by safety score (highest first = safest)
        analyzed_routes.sort(key=lambda r: r.safety_score, reverse=True)

        # Step 4: Mark best route as recommended
        if analyzed_routes:
            best_route = analyzed_routes[0]
            if best_route.risk_level != "UNKNOWN":
                best_route.recommendation = "Recommended - Safest option"

                # Add context for other routes
                for route in analyzed_routes[1:]:
                    safety_diff = best_route.safety_score - route.safety_score
                    if safety_diff > 10:
                        route.recommendation = f"Higher risk - {safety_diff:.1f} points less safe"
                    elif safety_diff > 5:
                        route.recommendation = "Slightly higher risk"
                    else:
                        route.recommendation = "Similar safety level"

        logger.info(f"Route analysis complete: {len(analyzed_routes)} routes scored")

        return RouteAnalysisResponse(
            routes=analyzed_routes,
            total_routes=len(analyzed_routes),
            origin=request.origin,
            destination=request.destination,
            weather_condition=request.weather_condition,
            temperature_f=request.temperature_f,
            visibility_mi=request.visibility_mi,
            analysis_timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Route analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Route analysis failed: {str(e)}"
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
