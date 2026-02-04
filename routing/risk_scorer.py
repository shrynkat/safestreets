"""
SafeStreets Route Risk Scorer

Provides functions to calculate accident risk scores for route segments
using the trained XGBoost model. Enables comparison of route alternatives
based on safety metrics.
"""

import logging
import math
import pickle
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "risk_model.pkl"

# Feature order must match training exactly (33 features)
FEATURE_NAMES = [
    "hour", "day_of_week", "is_weekend", "month", "time_period_encoded",
    "is_rush_hour", "state_encoded", "city_encoded", "latitude", "longitude",
    "temperature_f", "humidity_pct", "pressure_in", "visibility_mi",
    "wind_speed_mph", "precipitation_in", "weather_encoded", "poor_weather",
    "extreme_temp", "has_amenity", "has_bump", "has_crossing", "has_give_way",
    "has_junction", "has_no_exit", "has_railway", "has_roundabout", "has_station",
    "has_stop", "has_traffic_calming", "has_traffic_signal", "road_feature_count",
    "distance_mi"
]

# Weather condition encoding (matching training data)
WEATHER_ENCODING = {
    "clear": 0,
    "fair": 1,
    "cloudy": 2,
    "overcast": 3,
    "fog": 4,
    "haze": 5,
    "smoke": 6,
    "rain": 7,
    "drizzle": 8,
    "thunderstorm": 9,
    "snow": 10,
    "sleet": 11,
    "ice": 12,
    "freezing": 13,
    "windy": 14,
    "dust": 15,
    "sand": 16,
}

# Poor weather conditions
POOR_WEATHER_CONDITIONS = {
    "rain", "drizzle", "thunderstorm", "snow", "sleet",
    "ice", "freezing", "fog", "haze", "smoke"
}

# Time period encoding
TIME_PERIOD_ENCODING = {
    "night": 0,       # 0-5
    "morning": 1,     # 6-9
    "midday": 2,      # 10-15
    "evening": 3,     # 16-19
    "late_night": 4   # 20-23
}


@lru_cache(maxsize=1)
def load_model():
    """
    Load the trained risk prediction model from disk.

    Uses LRU cache to ensure model is only loaded once per session.

    Returns:
        Trained XGBoost model object.

    Raises:
        FileNotFoundError: If model file doesn't exist.
        Exception: If model loading fails.
    """
    logger.info(f"Loading model from {MODEL_PATH}...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Please train the model first using: python -m ml.train_model"
        )

    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points using Haversine formula.

    Args:
        lat1: Latitude of first point in degrees.
        lon1: Longitude of first point in degrees.
        lat2: Latitude of second point in degrees.
        lon2: Longitude of second point in degrees.

    Returns:
        Distance between the points in miles.

    Example:
        >>> dist = calculate_distance(33.4255, -111.9400, 34.8697, -111.7610)
        >>> print(f"{dist:.1f} miles")
        100.5 miles
    """
    # Earth's radius in miles
    R = 3959.0

    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    # Haversine formula
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def get_time_period(hour: int) -> str:
    """
    Determine time period based on hour of day.

    Args:
        hour: Hour of day (0-23).

    Returns:
        Time period string: 'night', 'morning', 'midday', 'evening', or 'late_night'.
    """
    if 0 <= hour < 6:
        return "night"
    elif 6 <= hour < 10:
        return "morning"
    elif 10 <= hour < 16:
        return "midday"
    elif 16 <= hour < 20:
        return "evening"
    else:
        return "late_night"


def prepare_features(
    lat: float,
    lon: float,
    hour: int,
    day_of_week: int,
    weather: str,
    temp: float,
    visibility: float,
    month: int = 6,
    has_traffic_signal: bool = False,
    has_junction: bool = False,
    has_crossing: bool = False,
    has_stop: bool = False,
    humidity_pct: float = 50.0,
    pressure_in: float = 29.9,
    wind_speed_mph: float = 5.0,
    precipitation_in: float = 0.0,
    state_encoded: int = 0,
    city_encoded: int = 0,
    distance_mi: float = 0.1
) -> np.ndarray:
    """
    Create feature vector matching the model's training format.

    Converts raw inputs into the 33-feature array expected by the model,
    including derived features like is_weekend, is_rush_hour, etc.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
        hour: Hour of day (0-23).
        day_of_week: Day of week (0=Sunday, 6=Saturday).
        weather: Weather condition string (e.g., "Clear", "Rain").
        temp: Temperature in Fahrenheit.
        visibility: Visibility in miles.
        month: Month of year (1-12), default 6.
        has_traffic_signal: Whether location has traffic signal.
        has_junction: Whether location is at a junction.
        has_crossing: Whether location has pedestrian crossing.
        has_stop: Whether location has stop sign.
        humidity_pct: Humidity percentage (0-100), default 50.
        pressure_in: Barometric pressure in inches, default 29.9.
        wind_speed_mph: Wind speed in mph, default 5.
        precipitation_in: Precipitation in inches, default 0.
        state_encoded: Encoded state ID, default 0.
        city_encoded: Encoded city ID, default 0.
        distance_mi: Road segment distance in miles, default 0.1.

    Returns:
        numpy array of shape (33,) with features in correct order.
    """
    # Derived features
    is_weekend = 1 if day_of_week in [0, 6] else 0  # Sunday=0, Saturday=6
    time_period = get_time_period(hour)
    time_period_encoded = TIME_PERIOD_ENCODING.get(time_period, 2)
    is_rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18] else 0

    # Weather encoding
    weather_lower = weather.lower()
    weather_encoded = 0  # Default to clear
    for key, value in WEATHER_ENCODING.items():
        if key in weather_lower:
            weather_encoded = value
            break

    # Weather-derived features
    poor_weather = 1 if any(cond in weather_lower for cond in POOR_WEATHER_CONDITIONS) else 0
    extreme_temp = 1 if temp < 32 or temp > 95 else 0

    # Road features (default to False for unknown locations)
    has_amenity = 0
    has_bump = 0
    has_give_way = 0
    has_no_exit = 0
    has_railway = 0
    has_roundabout = 0
    has_station = 0
    has_traffic_calming = 0

    # Count road features
    road_feature_count = sum([
        has_traffic_signal, has_junction, has_crossing, has_stop,
        has_amenity, has_bump, has_give_way, has_no_exit,
        has_railway, has_roundabout, has_station, has_traffic_calming
    ])

    # Build feature array in exact order
    features = np.array([
        hour,                    # 0
        day_of_week,            # 1
        is_weekend,             # 2
        month,                  # 3
        time_period_encoded,    # 4
        is_rush_hour,           # 5
        state_encoded,          # 6
        city_encoded,           # 7
        lat,                    # 8 - latitude
        lon,                    # 9 - longitude
        temp,                   # 10 - temperature_f
        humidity_pct,           # 11
        pressure_in,            # 12
        visibility,             # 13 - visibility_mi
        wind_speed_mph,         # 14
        precipitation_in,       # 15
        weather_encoded,        # 16
        poor_weather,           # 17
        extreme_temp,           # 18
        has_amenity,            # 19
        has_bump,               # 20
        int(has_crossing),      # 21
        has_give_way,           # 22
        int(has_junction),      # 23
        has_no_exit,            # 24
        has_railway,            # 25
        has_roundabout,         # 26
        has_station,            # 27
        int(has_stop),          # 28
        has_traffic_calming,    # 29
        int(has_traffic_signal),# 30
        road_feature_count,     # 31
        distance_mi             # 32
    ], dtype=np.float64)

    return features


def get_risk_level(risk_score: float) -> str:
    """
    Convert numeric risk score to categorical level.

    Args:
        risk_score: Risk probability (0-1).

    Returns:
        Risk level string: 'LOW', 'MODERATE', 'HIGH', or 'CRITICAL'.
    """
    if risk_score < 0.3:
        return "LOW"
    elif risk_score < 0.5:
        return "MODERATE"
    elif risk_score < 0.7:
        return "HIGH"
    else:
        return "CRITICAL"


def score_route_segment(
    lat: float,
    lon: float,
    hour: int,
    day_of_week: int,
    weather_condition: str,
    temperature_f: float,
    visibility_mi: float,
    has_traffic_signal: bool = False,
    has_junction: bool = False
) -> Dict:
    """
    Calculate risk score for a single route segment.

    Uses the trained model to predict accident probability for the given
    location and conditions.

    Args:
        lat: Latitude of segment location.
        lon: Longitude of segment location.
        hour: Hour of day (0-23).
        day_of_week: Day of week (0=Sunday, 6=Saturday).
        weather_condition: Current weather (e.g., "Clear", "Rain", "Snow").
        temperature_f: Temperature in Fahrenheit.
        visibility_mi: Visibility in miles.
        has_traffic_signal: Whether segment has traffic signal.
        has_junction: Whether segment is at a junction.

    Returns:
        Dictionary containing:
            - risk_score: Probability of high-severity accident (0-1)
            - risk_level: Categorical level (LOW/MODERATE/HIGH/CRITICAL)
            - confidence: Model confidence in prediction (0-1)

    Example:
        >>> result = score_route_segment(
        ...     lat=33.4255, lon=-111.9400,
        ...     hour=17, day_of_week=5,
        ...     weather_condition="Clear",
        ...     temperature_f=85, visibility_mi=10
        ... )
        >>> print(f"Risk: {result['risk_score']:.2f} ({result['risk_level']})")
    """
    try:
        model = load_model()

        # Prepare features
        features = prepare_features(
            lat=lat,
            lon=lon,
            hour=hour,
            day_of_week=day_of_week,
            weather=weather_condition,
            temp=temperature_f,
            visibility=visibility_mi,
            has_traffic_signal=has_traffic_signal,
            has_junction=has_junction
        )

        # Reshape for prediction
        X = features.reshape(1, -1)

        # Get probability prediction
        proba = model.predict_proba(X)[0]
        risk_score = proba[1]  # Probability of high-severity (class 1)
        confidence = max(proba)  # Higher of the two probabilities

        return {
            "risk_score": float(risk_score),
            "risk_level": get_risk_level(risk_score),
            "confidence": float(confidence)
        }

    except Exception as e:
        logger.error(f"Error scoring segment at ({lat}, {lon}): {e}")
        return {
            "risk_score": 0.5,  # Default to moderate risk on error
            "risk_level": "UNKNOWN",
            "confidence": 0.0
        }


def score_entire_route(
    waypoints: List[Tuple[float, float]],
    departure_time: datetime,
    weather_condition: str = "Clear",
    temperature_f: float = 70.0,
    visibility_mi: float = 10.0
) -> Dict:
    """
    Calculate comprehensive risk scores for an entire route.

    Samples waypoints approximately every 1 mile and aggregates risk scores
    to provide overall route safety metrics.

    Args:
        waypoints: List of (latitude, longitude) tuples from directions API.
        departure_time: Departure datetime for time-based risk factors.
        weather_condition: Weather condition string, default "Clear".
        temperature_f: Temperature in Fahrenheit, default 70.
        visibility_mi: Visibility in miles, default 10.

    Returns:
        Dictionary containing:
            - overall_risk: Average risk across all segments (0-1)
            - max_risk: Highest individual segment risk (0-1)
            - min_risk: Lowest individual segment risk (0-1)
            - safety_score: Inverted overall risk (0-100, higher is safer)
            - segments_scored: Number of segments evaluated
            - danger_zones: List of segments with risk > 0.6
            - segment_scores: List of all segment risk data

    Example:
        >>> routes = get_routes("Tempe, AZ", "Sedona, AZ")
        >>> departure = datetime(2024, 1, 5, 17, 0)  # Friday 5 PM
        >>> risk = score_entire_route(routes[0]['waypoints'], departure)
        >>> print(f"Safety Score: {risk['safety_score']:.1f}/100")
    """
    if not waypoints or len(waypoints) < 2:
        logger.warning("Insufficient waypoints for route scoring")
        return {
            "overall_risk": 0.0,
            "max_risk": 0.0,
            "min_risk": 0.0,
            "safety_score": 100.0,
            "segments_scored": 0,
            "danger_zones": [],
            "segment_scores": []
        }

    # Extract time components
    hour = departure_time.hour
    day_of_week = departure_time.weekday()  # Monday=0, Sunday=6
    # Convert to Sunday=0 format used in training
    day_of_week = (day_of_week + 1) % 7

    logger.info(f"Scoring route with {len(waypoints)} waypoints...")
    logger.info(f"Departure: {departure_time.strftime('%A %I:%M %p')}, Weather: {weather_condition}")

    # Sample waypoints approximately every 1 mile
    sampled_points = []
    cumulative_distance = 0.0
    sample_interval = 1.0  # miles

    sampled_points.append(waypoints[0])

    for i in range(1, len(waypoints)):
        dist = calculate_distance(
            waypoints[i-1][0], waypoints[i-1][1],
            waypoints[i][0], waypoints[i][1]
        )
        cumulative_distance += dist

        if cumulative_distance >= sample_interval:
            sampled_points.append(waypoints[i])
            cumulative_distance = 0.0

    # Always include final waypoint
    if sampled_points[-1] != waypoints[-1]:
        sampled_points.append(waypoints[-1])

    logger.info(f"Sampled {len(sampled_points)} points for scoring")

    # Score each sampled segment
    segment_scores = []
    risk_scores = []

    for i, (lat, lon) in enumerate(sampled_points):
        # Estimate if this might be a junction (simplified heuristic)
        # Real implementation would use road network data
        has_junction = False
        has_signal = False

        result = score_route_segment(
            lat=lat,
            lon=lon,
            hour=hour,
            day_of_week=day_of_week,
            weather_condition=weather_condition,
            temperature_f=temperature_f,
            visibility_mi=visibility_mi,
            has_traffic_signal=has_signal,
            has_junction=has_junction
        )

        segment_data = {
            "segment_index": i,
            "lat": lat,
            "lon": lon,
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "confidence": result["confidence"]
        }

        segment_scores.append(segment_data)
        risk_scores.append(result["risk_score"])

    # Calculate aggregate metrics
    overall_risk = np.mean(risk_scores)
    max_risk = np.max(risk_scores)
    min_risk = np.min(risk_scores)
    safety_score = 100 - (overall_risk * 100)

    # Identify danger zones (risk > 0.6)
    danger_zones = [
        seg for seg in segment_scores
        if seg["risk_score"] > 0.6
    ]

    logger.info(f"Route scored: Overall risk={overall_risk:.3f}, "
                f"Safety score={safety_score:.1f}, "
                f"Danger zones={len(danger_zones)}")

    return {
        "overall_risk": float(overall_risk),
        "max_risk": float(max_risk),
        "min_risk": float(min_risk),
        "safety_score": float(safety_score),
        "segments_scored": len(segment_scores),
        "danger_zones": danger_zones,
        "segment_scores": segment_scores,
        "weather_condition": weather_condition,
        "temperature_f": temperature_f,
        "visibility_mi": visibility_mi,
        "departure_time": departure_time.isoformat()
    }


if __name__ == "__main__":
    # Quick test with mock data
    print("Testing risk scorer...")

    # Test single segment
    result = score_route_segment(
        lat=33.4255,
        lon=-111.9400,
        hour=17,
        day_of_week=5,  # Friday
        weather_condition="Clear",
        temperature_f=85,
        visibility_mi=10
    )
    print(f"\nSingle segment test:")
    print(f"  Risk Score: {result['risk_score']:.3f}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Confidence: {result['confidence']:.3f}")

    # Test route scoring with mock waypoints
    mock_waypoints = [
        (33.4255, -111.9400),  # Tempe
        (33.5000, -111.8500),
        (33.6000, -111.7500),
        (33.7000, -111.7000),
        (34.8697, -111.7610),  # Sedona
    ]

    departure = datetime(2024, 1, 5, 17, 0)  # Friday 5 PM

    route_result = score_entire_route(
        waypoints=mock_waypoints,
        departure_time=departure,
        weather_condition="Clear",
        temperature_f=70,
        visibility_mi=10
    )

    print(f"\nRoute scoring test:")
    print(f"  Overall Risk: {route_result['overall_risk']:.3f}")
    print(f"  Safety Score: {route_result['safety_score']:.1f}/100")
    print(f"  Max Risk: {route_result['max_risk']:.3f}")
    print(f"  Segments Scored: {route_result['segments_scored']}")
    print(f"  Danger Zones: {len(route_result['danger_zones'])}")
