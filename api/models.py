"""
Pydantic Models for SafeStreets Risk Prediction API

This module defines the request and response schemas for the API endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TimePeriod(str, Enum):
    """Time period of day."""
    NIGHT = "Night"
    MORNING = "Morning"
    AFTERNOON = "Afternoon"
    EVENING = "Evening"


class AccidentFeatures(BaseModel):
    """
    Input features for accident risk prediction.

    All features should match the format expected by the ML model.
    Optional fields will use default values if not provided.
    """

    # Location features
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    state: Optional[str] = Field(None, description="US state abbreviation (e.g., 'CA', 'TX')")
    city: Optional[str] = Field(None, description="City name")

    # Temporal features
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    month: int = Field(default=1, ge=1, le=12, description="Month (1-12)")
    is_weekend: bool = Field(default=False, description="Whether it's a weekend")
    time_period: Optional[TimePeriod] = Field(None, description="Time period of day")

    # Weather features
    temperature_f: float = Field(default=70.0, description="Temperature in Fahrenheit")
    humidity_pct: float = Field(default=50.0, ge=0, le=100, description="Humidity percentage")
    pressure_in: float = Field(default=29.92, description="Barometric pressure in inches")
    visibility_mi: float = Field(default=10.0, ge=0, description="Visibility in miles")
    wind_speed_mph: float = Field(default=5.0, ge=0, description="Wind speed in mph")
    precipitation_in: float = Field(default=0.0, ge=0, description="Precipitation in inches")
    weather_condition: Optional[str] = Field(None, description="Weather condition (e.g., 'Clear', 'Rain', 'Fog')")

    # Road features
    has_amenity: bool = Field(default=False, description="Nearby amenity present")
    has_bump: bool = Field(default=False, description="Speed bump present")
    has_crossing: bool = Field(default=False, description="Crossing present")
    has_give_way: bool = Field(default=False, description="Give way sign present")
    has_junction: bool = Field(default=False, description="Junction present")
    has_no_exit: bool = Field(default=False, description="No exit present")
    has_railway: bool = Field(default=False, description="Railway crossing present")
    has_roundabout: bool = Field(default=False, description="Roundabout present")
    has_station: bool = Field(default=False, description="Station present")
    has_stop: bool = Field(default=False, description="Stop sign present")
    has_traffic_calming: bool = Field(default=False, description="Traffic calming present")
    has_traffic_signal: bool = Field(default=False, description="Traffic signal present")
    distance_mi: float = Field(default=0.0, ge=0, description="Distance affected in miles")

    @field_validator('day_of_week', mode='before')
    @classmethod
    def validate_day_of_week(cls, v):
        if isinstance(v, str):
            days = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                    'friday': 4, 'saturday': 5, 'sunday': 6}
            return days.get(v.lower(), v)
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "latitude": 34.0522,
                    "longitude": -118.2437,
                    "state": "CA",
                    "city": "Los Angeles",
                    "hour": 8,
                    "day_of_week": 0,
                    "month": 3,
                    "is_weekend": False,
                    "temperature_f": 65.0,
                    "humidity_pct": 45.0,
                    "visibility_mi": 10.0,
                    "has_traffic_signal": True,
                    "has_junction": True
                }
            ]
        }
    }


class ContributingFactor(BaseModel):
    """A factor contributing to the risk prediction."""
    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., ge=0, le=1, description="Importance score (0-1)")
    value: Optional[float] = Field(None, description="Feature value for this prediction")


class PredictionResponse(BaseModel):
    """Response for a single risk prediction."""
    risk_score: float = Field(..., ge=0, le=1, description="Risk probability (0-1)")
    risk_level: RiskLevel = Field(..., description="Risk classification")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    contributing_factors: list[ContributingFactor] = Field(
        default_factory=list,
        description="Top contributing factors"
    )
    prediction_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the prediction was made"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "risk_score": 0.73,
                    "risk_level": "HIGH",
                    "confidence": 0.85,
                    "contributing_factors": [
                        {"feature": "visibility_mi", "importance": 0.23, "value": 2.0},
                        {"feature": "hour", "importance": 0.18, "value": 23.0}
                    ],
                    "prediction_timestamp": "2024-01-15T10:30:00"
                }
            ]
        }
    }


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    instances: list[AccidentFeatures] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of feature instances (max 1000)"
    )


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: list[PredictionResponse] = Field(..., description="List of predictions")
    total_count: int = Field(..., description="Total number of predictions")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelInfoResponse(BaseModel):
    """Model metadata response."""
    model_version: str = Field(..., description="Model version identifier")
    model_type: str = Field(..., description="Type of model")
    training_date: Optional[str] = Field(None, description="When model was trained")
    feature_count: int = Field(..., description="Number of input features")
    feature_names: list[str] = Field(..., description="List of feature names")
    performance_metrics: dict = Field(..., description="Model performance metrics")


class ErrorResponse(BaseModel):
    """Error response format."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Route Analysis Models
# =============================================================================

class RouteAnalysisRequest(BaseModel):
    """
    Request body for route analysis endpoint.

    Analyzes multiple route alternatives between origin and destination,
    scoring each for accident risk based on conditions.
    """
    origin: str = Field(
        ...,
        min_length=1,
        description="Starting location (address, place name, or 'lat,lng')"
    )
    destination: str = Field(
        ...,
        min_length=1,
        description="Ending location (address, place name, or 'lat,lng')"
    )
    departure_time: datetime = Field(
        default_factory=datetime.utcnow,
        description="Departure time in ISO format (e.g., '2024-02-01T17:00:00')"
    )
    weather_condition: Optional[str] = Field(
        default=None,
        description="Weather condition override (e.g., 'Clear', 'Rain', 'Snow', 'Fog'). "
                    "Ignored when auto_fetch_weather is True."
    )
    temperature_f: Optional[float] = Field(
        default=None,
        description="Temperature override in Fahrenheit. "
                    "Ignored when auto_fetch_weather is True."
    )
    visibility_mi: Optional[float] = Field(
        default=None,
        ge=0,
        description="Visibility override in miles. "
                    "Ignored when auto_fetch_weather is True."
    )
    auto_fetch_weather: bool = Field(
        default=True,
        description="If True, fetch real-time weather from OpenWeather API along the route. "
                    "If False, use the manual weather_condition/temperature_f/visibility_mi values "
                    "(useful for historical 'what-if' analysis)."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "origin": "Tempe, AZ",
                    "destination": "Sedona, AZ",
                    "departure_time": "2024-02-01T17:00:00",
                    "auto_fetch_weather": True
                },
                {
                    "origin": "Tempe, AZ",
                    "destination": "Sedona, AZ",
                    "departure_time": "2024-02-01T17:00:00",
                    "auto_fetch_weather": False,
                    "weather_condition": "Rain",
                    "temperature_f": 55.0,
                    "visibility_mi": 4.0
                }
            ]
        }
    }


class DangerZone(BaseModel):
    """A high-risk segment along a route."""
    segment: int = Field(..., description="Segment index along the route")
    risk: float = Field(..., ge=0, le=1, description="Risk score for this segment")
    location: str = Field(..., description="Location as '(lat, lon)' string")
    risk_level: str = Field(..., description="Risk level classification")


class RouteDetails(BaseModel):
    """Detailed information about a single route alternative."""
    route_id: int = Field(..., description="Route identifier (1-indexed)")
    summary: str = Field(..., description="Route name/description (e.g., 'I-17 N')")
    distance_text: str = Field(..., description="Human-readable distance (e.g., '126 mi')")
    distance_meters: int = Field(..., description="Distance in meters")
    duration_text: str = Field(..., description="Human-readable duration (e.g., '2 hours 4 mins')")
    duration_seconds: int = Field(..., description="Duration in seconds")
    risk_score: float = Field(..., ge=0, le=1, description="Overall risk score (0-1)")
    risk_level: str = Field(..., description="Risk classification (LOW/MEDIUM/HIGH)")
    safety_score: float = Field(..., ge=0, le=100, description="Safety score (0-100, higher is safer)")
    max_risk: float = Field(..., ge=0, le=1, description="Highest risk segment score")
    min_risk: float = Field(..., ge=0, le=1, description="Lowest risk segment score")
    segments_analyzed: int = Field(..., description="Number of segments scored")
    danger_zones: list[DangerZone] = Field(
        default_factory=list,
        description="List of high-risk segments (risk > 0.6)"
    )
    recommendation: Optional[str] = Field(
        None,
        description="Recommendation text if this is the safest route"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "route_id": 1,
                    "summary": "I-17 N",
                    "distance_text": "126 mi",
                    "distance_meters": 202777,
                    "duration_text": "2 hours 4 mins",
                    "duration_seconds": 7440,
                    "risk_score": 0.489,
                    "risk_level": "MEDIUM",
                    "safety_score": 51.1,
                    "max_risk": 0.577,
                    "min_risk": 0.410,
                    "segments_analyzed": 98,
                    "danger_zones": [],
                    "recommendation": "Recommended - Safest option"
                }
            ]
        }
    }


class RouteAnalysisResponse(BaseModel):
    """Response containing analyzed routes sorted by safety."""
    routes: list[RouteDetails] = Field(
        ...,
        description="List of route alternatives sorted by safety score (safest first)"
    )
    total_routes: int = Field(..., description="Total number of routes analyzed")
    origin: str = Field(..., description="Origin location")
    destination: str = Field(..., description="Destination location")
    actual_weather_used: dict = Field(
        ...,
        description="Weather data actually used for risk scoring "
                    "(auto-fetched or manually provided)"
    )
    analysis_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the analysis was performed"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "routes": [
                        {
                            "route_id": 1,
                            "summary": "I-17 N",
                            "distance_text": "126 mi",
                            "distance_meters": 202777,
                            "duration_text": "2 hours 4 mins",
                            "duration_seconds": 7440,
                            "risk_score": 0.489,
                            "risk_level": "MEDIUM",
                            "safety_score": 51.1,
                            "max_risk": 0.577,
                            "min_risk": 0.410,
                            "segments_analyzed": 98,
                            "danger_zones": [],
                            "recommendation": "Recommended - Safest option"
                        }
                    ],
                    "total_routes": 1,
                    "origin": "Tempe, AZ",
                    "destination": "Sedona, AZ",
                    "actual_weather_used": {
                        "source": "auto",
                        "weather_condition": "Clear",
                        "temperature_f": 72.3,
                        "visibility_mi": 10.0,
                        "humidity_pct": 28.0,
                        "wind_speed_mph": 6.2,
                        "description": "clear sky"
                    },
                    "analysis_timestamp": "2024-02-01T17:00:00Z"
                }
            ]
        }
    }
