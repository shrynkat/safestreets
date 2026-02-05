"""
SafeStreets Time Optimizer Module

Analyzes how departure time affects route safety and recommends
the safest windows for travel.  Route geometry is fetched once and
reused across all time slots since it doesn't change — only the
time-based risk factors (hour, rush hour, time period) vary.

Requires:
    - Google Maps API key (for route fetching)
    - Trained risk model in ml/models/risk_model.pkl
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .directions_api import get_routes
from .risk_scorer import score_entire_route

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard departure hours to evaluate (24-h clock)
_DEFAULT_HOURS = [6, 9, 12, 15, 18, 21]

# ---------------------------------------------------------------------------
# Route cache — avoids repeat Google Maps calls within a session.
# Key: (origin, destination)   Value: list of route dicts
# ---------------------------------------------------------------------------
_route_cache: Dict[Tuple[str, str], List[Dict]] = {}


def _cache_key(origin: str, destination: str) -> Tuple[str, str]:
    return (origin.strip().lower(), destination.strip().lower())


def _get_cached_routes(origin: str, destination: str) -> Optional[List[Dict]]:
    return _route_cache.get(_cache_key(origin, destination))


def _set_cached_routes(
    origin: str, destination: str, routes: List[Dict]
) -> None:
    _route_cache[_cache_key(origin, destination)] = routes


# ---------------------------------------------------------------------------
# Score cache — avoids re-scoring the same (route, hour, weather) combo.
# Key: (origin, destination, hour, weather_condition, temperature_f, visibility_mi)
# Value: score_entire_route result dict
# ---------------------------------------------------------------------------
_score_cache: Dict[tuple, Dict] = {}


def _score_key(
    origin: str,
    destination: str,
    hour: int,
    weather: Dict,
) -> tuple:
    return (
        origin.strip().lower(),
        destination.strip().lower(),
        hour,
        weather.get("weather_condition", "Clear"),
        weather.get("temperature_f", 70.0),
        weather.get("visibility_mi", 10.0),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_routes(origin: str, destination: str) -> List[Dict]:
    """
    Fetch routes for origin/destination, using cache when available.

    Routes are requested without a departure_time so that geometry is
    time-independent and can be reused across multiple time slots.

    Raises:
        ValueError: If no routes are found or API key is missing.
    """
    cached = _get_cached_routes(origin, destination)
    if cached is not None:
        logger.debug(f"Route cache hit for {origin} -> {destination}")
        return cached

    logger.info(f"Fetching routes: {origin} -> {destination}")
    routes = get_routes(origin=origin, destination=destination)

    if not routes:
        raise ValueError(
            f"No routes found between '{origin}' and '{destination}'"
        )

    _set_cached_routes(origin, destination, routes)
    return routes


def _score_at_time(
    origin: str,
    destination: str,
    waypoints: List[Tuple[float, float]],
    departure_dt: datetime,
    weather: Dict,
) -> Dict:
    """
    Score a route's waypoints at a specific departure time.

    Uses the score cache to avoid redundant ML predictions.

    Returns:
        The full dict from ``score_entire_route``.
    """
    key = _score_key(origin, destination, departure_dt.hour, weather)
    cached = _score_cache.get(key)
    if cached is not None:
        logger.debug(f"Score cache hit for hour={departure_dt.hour}")
        return cached

    result = score_entire_route(
        waypoints=waypoints,
        departure_time=departure_dt,
        weather_condition=weather.get("weather_condition", "Clear"),
        temperature_f=weather.get("temperature_f", 70.0),
        visibility_mi=weather.get("visibility_mi", 10.0),
    )

    _score_cache[key] = result
    return result


def _format_hour(hour: int) -> str:
    """Format a 24-h hour as a human-readable string like '6:00 AM'."""
    if hour == 0:
        return "12:00 AM"
    if hour < 12:
        return f"{hour}:00 AM"
    if hour == 12:
        return "12:00 PM"
    return f"{hour - 12}:00 PM"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_departure_times(
    origin: str,
    destination: str,
    date: str,
    weather: Dict,
    hours: Optional[List[int]] = None,
) -> Dict:
    """
    Evaluate safety across multiple departure times on a given date.

    Fetches route geometry once, then scores it at each candidate hour
    to find the safest and most dangerous departure windows.

    Args:
        origin: Starting location (address or "lat,lng").
        destination: Ending location (address or "lat,lng").
        date: Date string in ``YYYY-MM-DD`` format.
        weather: Dict with at least ``weather_condition``,
                 ``temperature_f``, and ``visibility_mi``.
        hours: Optional list of hours (0–23) to test.
               Defaults to [6, 9, 12, 15, 18, 21].

    Returns:
        Dict with:
            - best_time: str – the safest departure time (e.g. "6:00 AM")
            - worst_time: str – the riskiest departure time
            - best_safety_score: float – safety score of the best time
            - worst_safety_score: float – safety score of the worst time
            - improvement_pct: float – percent improvement best vs worst
            - recommendations: list of dicts sorted safest-first, each with
              ``hour``, ``time_label``, ``safety_score``, ``risk_score``,
              ``risk_level``, ``segments_analyzed``, ``danger_zones_count``
            - route_summary: str – name of the scored route

    Example:
        >>> result = analyze_departure_times(
        ...     "Tempe, AZ", "Sedona, AZ", "2024-02-02",
        ...     {"weather_condition": "Clear", "temperature_f": 70, "visibility_mi": 10}
        ... )
        >>> print(result["best_time"], result["best_safety_score"])
        '6:00 AM' 62.3
    """
    test_hours = hours or list(_DEFAULT_HOURS)
    base_date = datetime.strptime(date, "%Y-%m-%d")

    logger.info(
        f"Analyzing departure times for {origin} -> {destination} "
        f"on {date} ({len(test_hours)} slots)"
    )

    # Fetch routes once
    routes = _fetch_routes(origin, destination)
    waypoints = routes[0]["waypoints"]
    route_summary = routes[0]["summary"]

    # Score each time slot
    recommendations: List[Dict] = []
    for hour in test_hours:
        departure_dt = base_date.replace(hour=hour, minute=0, second=0)
        logger.info(f"Scoring departure at {_format_hour(hour)}...")

        result = _score_at_time(
            origin, destination, waypoints, departure_dt, weather
        )

        recommendations.append({
            "hour": hour,
            "time_label": _format_hour(hour),
            "safety_score": round(result["safety_score"], 1),
            "risk_score": round(result["overall_risk"], 3),
            "risk_level": result.get("danger_zones", [{}])[0].get("risk_level", "")
                          if result.get("danger_zones") else
                          ("LOW" if result["overall_risk"] < 0.3 else
                           "MODERATE" if result["overall_risk"] < 0.5 else
                           "HIGH" if result["overall_risk"] < 0.7 else "CRITICAL"),
            "segments_analyzed": result["segments_scored"],
            "danger_zones_count": len(result.get("danger_zones", [])),
        })

    # Sort safest first
    recommendations.sort(key=lambda r: r["safety_score"], reverse=True)

    best = recommendations[0]
    worst = recommendations[-1]

    if worst["safety_score"] > 0:
        improvement_pct = round(
            (best["safety_score"] - worst["safety_score"])
            / worst["safety_score"] * 100,
            1,
        )
    else:
        improvement_pct = 0.0

    logger.info(
        f"Best: {best['time_label']} (safety {best['safety_score']}), "
        f"Worst: {worst['time_label']} (safety {worst['safety_score']}), "
        f"Improvement: {improvement_pct}%"
    )

    return {
        "best_time": best["time_label"],
        "worst_time": worst["time_label"],
        "best_safety_score": best["safety_score"],
        "worst_safety_score": worst["safety_score"],
        "improvement_pct": improvement_pct,
        "recommendations": recommendations,
        "route_summary": route_summary,
    }


def compare_times(
    origin: str,
    destination: str,
    time1: str,
    time2: str,
    weather: Dict,
) -> Dict:
    """
    Compare two specific departure times for safety.

    Args:
        origin: Starting location.
        destination: Ending location.
        time1: First departure time in ISO format (``YYYY-MM-DDTHH:MM:SS``).
        time2: Second departure time in ISO format.
        weather: Dict with ``weather_condition``, ``temperature_f``,
                 ``visibility_mi``.

    Returns:
        Dict with:
            - time1_label / time2_label: human-readable labels
            - time1_safety / time2_safety: safety scores
            - time1_risk / time2_risk: risk scores
            - safer: which time is safer ("time1" or "time2")
            - safety_difference: absolute difference in safety scores
            - recommendation: human-readable verdict

    Example:
        >>> compare_times(
        ...     "Tempe, AZ", "Sedona, AZ",
        ...     "2024-02-02T08:00:00", "2024-02-02T17:00:00",
        ...     {"weather_condition": "Clear", "temperature_f": 70, "visibility_mi": 10}
        ... )
    """
    dt1 = datetime.fromisoformat(time1)
    dt2 = datetime.fromisoformat(time2)

    logger.info(
        f"Comparing {dt1.strftime('%I:%M %p')} vs {dt2.strftime('%I:%M %p')} "
        f"for {origin} -> {destination}"
    )

    routes = _fetch_routes(origin, destination)
    waypoints = routes[0]["waypoints"]

    r1 = _score_at_time(origin, destination, waypoints, dt1, weather)
    r2 = _score_at_time(origin, destination, waypoints, dt2, weather)

    s1 = round(r1["safety_score"], 1)
    s2 = round(r2["safety_score"], 1)
    diff = round(abs(s1 - s2), 1)

    if s1 > s2:
        safer = "time1"
        recommendation = (
            f"Departing at {dt1.strftime('%I:%M %p')} is safer by "
            f"{diff} safety points compared to {dt2.strftime('%I:%M %p')}."
        )
    elif s2 > s1:
        safer = "time2"
        recommendation = (
            f"Departing at {dt2.strftime('%I:%M %p')} is safer by "
            f"{diff} safety points compared to {dt1.strftime('%I:%M %p')}."
        )
    else:
        safer = "equal"
        recommendation = (
            f"Both departure times have the same safety score ({s1})."
        )

    return {
        "time1_label": dt1.strftime("%I:%M %p"),
        "time2_label": dt2.strftime("%I:%M %p"),
        "time1_safety": s1,
        "time2_safety": s2,
        "time1_risk": round(r1["overall_risk"], 3),
        "time2_risk": round(r2["overall_risk"], 3),
        "safer": safer,
        "safety_difference": diff,
        "recommendation": recommendation,
    }


def get_optimal_window(
    origin: str,
    destination: str,
    date: str,
    weather: Dict,
    min_hour: int = 6,
    max_hour: int = 22,
) -> Dict:
    """
    Find the safest continuous 2-hour departure window on a given date.

    Scores every hour from ``min_hour`` to ``max_hour``, then slides a
    2-hour window across the results to find the pair of consecutive
    hours with the highest average safety score.

    Args:
        origin: Starting location.
        destination: Ending location.
        date: Date string in ``YYYY-MM-DD`` format.
        weather: Dict with ``weather_condition``, ``temperature_f``,
                 ``visibility_mi``.
        min_hour: Earliest hour to consider (inclusive, default 6).
        max_hour: Latest hour to consider (inclusive, default 22).

    Returns:
        Dict with:
            - window_start: str – start of optimal window (e.g. "9:00 AM")
            - window_end: str – end of optimal window (e.g. "11:00 AM")
            - window_start_hour: int
            - window_end_hour: int
            - avg_safety_score: float – average safety in the window
            - hourly_scores: list of (hour, safety_score) for all tested hours
            - recommendation: human-readable summary

    Example:
        >>> result = get_optimal_window(
        ...     "Tempe, AZ", "Sedona, AZ", "2024-02-02",
        ...     {"weather_condition": "Clear", "temperature_f": 70, "visibility_mi": 10}
        ... )
        >>> print(result["recommendation"])
    """
    base_date = datetime.strptime(date, "%Y-%m-%d")
    test_hours = list(range(min_hour, max_hour + 1))

    if len(test_hours) < 2:
        raise ValueError(
            f"Need at least 2 hours between min_hour ({min_hour}) "
            f"and max_hour ({max_hour})"
        )

    logger.info(
        f"Finding optimal 2-hour window ({_format_hour(min_hour)}–"
        f"{_format_hour(max_hour)}) for {origin} -> {destination} on {date}"
    )

    routes = _fetch_routes(origin, destination)
    waypoints = routes[0]["waypoints"]

    # Score every hour
    hourly_scores: List[Tuple[int, float]] = []
    for hour in test_hours:
        departure_dt = base_date.replace(hour=hour, minute=0, second=0)
        result = _score_at_time(
            origin, destination, waypoints, departure_dt, weather
        )
        safety = round(result["safety_score"], 1)
        hourly_scores.append((hour, safety))
        logger.debug(f"  {_format_hour(hour)}: safety {safety}")

    # Slide 2-hour window and find best average
    best_avg = -1.0
    best_start_idx = 0
    for i in range(len(hourly_scores) - 1):
        avg = (hourly_scores[i][1] + hourly_scores[i + 1][1]) / 2
        if avg > best_avg:
            best_avg = avg
            best_start_idx = i

    start_hour = hourly_scores[best_start_idx][0]
    end_hour = hourly_scores[best_start_idx + 1][0] + 1  # window covers 2 hours

    recommendation = (
        f"The safest 2-hour departure window is "
        f"{_format_hour(start_hour)} – {_format_hour(end_hour)} "
        f"(avg safety score {round(best_avg, 1)}/100)."
    )

    logger.info(recommendation)

    return {
        "window_start": _format_hour(start_hour),
        "window_end": _format_hour(end_hour),
        "window_start_hour": start_hour,
        "window_end_hour": end_hour,
        "avg_safety_score": round(best_avg, 1),
        "hourly_scores": [
            {"hour": h, "time_label": _format_hour(h), "safety_score": s}
            for h, s in hourly_scores
        ],
        "recommendation": recommendation,
    }
