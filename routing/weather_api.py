"""
SafeStreets Weather API Module

Fetches current weather conditions from the OpenWeather API and maps them
to the categories used by the SafeStreets ML model.  Supports single-point
lookups and averaged weather along an entire route.

Requires OPENWEATHER_API_KEY in the environment (or .env file).
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# Cache: {(lat_rounded, lon_rounded): (timestamp, result)}
_weather_cache: Dict[Tuple[float, float], Tuple[float, Dict]] = {}
_CACHE_TTL_SECONDS = 30 * 60  # 30 minutes

# Maximum sample points when averaging weather along a route
_MAX_SAMPLE_POINTS = 5

# ---------------------------------------------------------------------------
# OpenWeather condition-code → SafeStreets category mapping
#
# Reference: https://openweathermap.org/weather-conditions
#   2xx  Thunderstorm
#   3xx  Drizzle
#   5xx  Rain
#   6xx  Snow
#   7xx  Atmosphere (fog, haze, smoke, …)
#   800  Clear
#   80x  Clouds
# ---------------------------------------------------------------------------

_CODE_CATEGORY_MAP: Dict[int, str] = {
    # Thunderstorm group (200–232)
    200: "Thunderstorm", 201: "Thunderstorm", 202: "Thunderstorm",
    210: "Thunderstorm", 211: "Thunderstorm", 212: "Thunderstorm",
    221: "Thunderstorm", 230: "Thunderstorm", 231: "Thunderstorm",
    232: "Thunderstorm",
    # Drizzle group (300–321)
    300: "Drizzle", 301: "Drizzle", 302: "Drizzle",
    310: "Drizzle", 311: "Drizzle", 312: "Drizzle",
    313: "Drizzle", 314: "Drizzle", 321: "Drizzle",
    # Rain group (500–531)
    500: "Light Rain", 501: "Rain", 502: "Heavy Rain",
    503: "Heavy Rain", 504: "Heavy Rain",
    511: "Sleet",
    520: "Rain", 521: "Rain", 522: "Heavy Rain", 531: "Rain",
    # Snow group (600–622)
    600: "Light Snow", 601: "Snow", 602: "Heavy Snow",
    611: "Sleet", 612: "Sleet", 613: "Sleet",
    615: "Snow", 616: "Snow",
    620: "Snow", 621: "Snow", 622: "Heavy Snow",
    # Atmosphere group (700–781)
    701: "Fog", 711: "Smoke", 721: "Haze",
    731: "Haze", 741: "Fog", 751: "Haze",
    761: "Haze", 762: "Haze",
    771: "Windy", 781: "Windy",
    # Clear (800)
    800: "Clear",
    # Clouds (801–804)
    801: "Partly Cloudy", 802: "Cloudy",
    803: "Cloudy", 804: "Overcast",
}


def _get_api_key() -> str:
    """Return the OpenWeather API key or raise ``ValueError``."""
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENWEATHER_API_KEY not found in environment variables. "
            "Please set it in your .env file or environment."
        )
    return api_key


def _round_coord(value: float, precision: int = 2) -> float:
    """Round a coordinate for cache-key purposes (~1 km granularity)."""
    return round(value, precision)


def _cache_key(lat: float, lon: float) -> Tuple[float, float]:
    return (_round_coord(lat), _round_coord(lon))


def _get_cached(lat: float, lon: float) -> Optional[Dict]:
    """Return cached weather dict if still fresh, else ``None``."""
    key = _cache_key(lat, lon)
    entry = _weather_cache.get(key)
    if entry is None:
        return None
    ts, result = entry
    if time.time() - ts > _CACHE_TTL_SECONDS:
        del _weather_cache[key]
        return None
    return result


def _set_cache(lat: float, lon: float, result: Dict) -> None:
    _weather_cache[_cache_key(lat, lon)] = (time.time(), result)


# ---------------------------------------------------------------------------
# Default weather values (returned on API failure)
# ---------------------------------------------------------------------------

_DEFAULT_WEATHER: Dict = {
    "temperature_f": 70.0,
    "weather_condition": "Clear",
    "visibility_mi": 10.0,
    "humidity_pct": 50.0,
    "wind_speed_mph": 5.0,
    "description": "default (API unavailable)",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_weather_to_category(weather_code: int) -> str:
    """
    Map an OpenWeather condition code to a SafeStreets weather category.

    The SafeStreets ML model expects one of the categories defined in
    ``api.main.WEATHER_CONDITIONS``.  This function translates the numeric
    codes returned by the OpenWeather API to those categories.

    Args:
        weather_code: Integer condition code from the OpenWeather API
                      (e.g. 200 = Thunderstorm, 500 = Light Rain, 800 = Clear).

    Returns:
        A category string such as ``"Clear"``, ``"Rain"``, ``"Snow"``, or
        ``"Fog"``.  Falls back to ``"Clear"`` for unknown codes.

    Example:
        >>> map_weather_to_category(501)
        'Rain'
        >>> map_weather_to_category(800)
        'Clear'
    """
    return _CODE_CATEGORY_MAP.get(weather_code, "Clear")


def get_current_weather(lat: float, lon: float) -> Dict:
    """
    Fetch current weather conditions for a coordinate from OpenWeather.

    Results are cached for 30 minutes (keyed by coordinates rounded to
    2 decimal places) to avoid excessive API calls.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.

    Returns:
        Dict with keys:
            - temperature_f (float): Temperature in Fahrenheit.
            - weather_condition (str): SafeStreets category
              (Clear, Rain, Snow, Fog, Clouds, …).
            - visibility_mi (float): Visibility in miles.
            - humidity_pct (float): Relative humidity percentage.
            - wind_speed_mph (float): Wind speed in mph.
            - description (str): Human-readable description
              (e.g. "light rain").

        On API failure, returns sensible defaults so callers can proceed
        with degraded accuracy rather than crashing.

    Example:
        >>> weather = get_current_weather(33.4255, -111.9400)
        >>> print(weather["weather_condition"])
        'Clear'
    """
    # Check cache first
    cached = _get_cached(lat, lon)
    if cached is not None:
        logger.debug(f"Cache hit for ({lat:.2f}, {lon:.2f})")
        return cached

    try:
        api_key = _get_api_key()
    except ValueError as e:
        logger.error(str(e))
        return dict(_DEFAULT_WEATHER)

    try:
        resp = requests.get(
            OPENWEATHER_BASE_URL,
            params={"lat": lat, "lon": lon, "appid": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.error(f"OpenWeather API request failed: {e}")
        return dict(_DEFAULT_WEATHER)

    try:
        # --- Temperature: Kelvin → Fahrenheit ---
        temp_k = data["main"]["temp"]
        temp_f = (temp_k - 273.15) * 9 / 5 + 32

        # --- Weather condition ---
        weather_entry = data["weather"][0]
        weather_code = weather_entry["id"]
        weather_condition = map_weather_to_category(weather_code)
        description = weather_entry.get("description", weather_condition)

        # --- Visibility: meters → miles (default 10 000 m if absent) ---
        visibility_m = data.get("visibility", 10000)
        visibility_mi = visibility_m / 1609.34

        # --- Humidity ---
        humidity_pct = data["main"].get("humidity", 50.0)

        # --- Wind speed: m/s → mph ---
        wind_speed_ms = data.get("wind", {}).get("speed", 0.0)
        wind_speed_mph = wind_speed_ms * 2.237

        result: Dict = {
            "temperature_f": round(temp_f, 1),
            "weather_condition": weather_condition,
            "visibility_mi": round(visibility_mi, 1),
            "humidity_pct": float(humidity_pct),
            "wind_speed_mph": round(wind_speed_mph, 1),
            "description": description,
        }

        _set_cache(lat, lon, result)
        logger.info(
            f"Weather at ({lat:.2f}, {lon:.2f}): "
            f"{weather_condition}, {temp_f:.0f}°F, vis {visibility_mi:.1f} mi"
        )
        return result

    except (KeyError, TypeError, IndexError) as e:
        logger.error(f"Failed to parse OpenWeather response: {e}")
        return dict(_DEFAULT_WEATHER)


def get_weather_along_route(
    waypoints: List[Tuple[float, float]],
) -> Dict:
    """
    Sample weather at several points along a route and return averaged conditions.

    Picks up to ``_MAX_SAMPLE_POINTS`` evenly-spaced waypoints (always
    including the first and last), queries each, and computes the mean
    values.  If conditions vary significantly along the route a
    ``weather_warning`` key is added to the result.

    Args:
        waypoints: Ordered list of (lat, lon) tuples describing the route.

    Returns:
        Dict with the same keys as :func:`get_current_weather` plus:
            - samples (int): Number of points sampled.
            - weather_warning (str | None): Warning when conditions differ
              significantly along the route (e.g. clear at start, rain at end).

    Example:
        >>> route_wp = [(33.42, -111.94), (34.10, -111.80), (34.87, -111.76)]
        >>> weather = get_weather_along_route(route_wp)
        >>> print(weather["weather_condition"], weather["samples"])
        'Clear' 3
    """
    if not waypoints:
        logger.warning("Empty waypoints list — returning defaults")
        return {**_DEFAULT_WEATHER, "samples": 0, "weather_warning": None}

    # Pick sample indices (always include first & last)
    n = len(waypoints)
    if n <= _MAX_SAMPLE_POINTS:
        indices = list(range(n))
    else:
        step = (n - 1) / (_MAX_SAMPLE_POINTS - 1)
        indices = [round(i * step) for i in range(_MAX_SAMPLE_POINTS)]

    # Fetch weather for each sample point
    samples: List[Dict] = []
    for idx in indices:
        lat, lon = waypoints[idx]
        weather = get_current_weather(lat, lon)
        samples.append(weather)

    # Average numeric fields
    avg_temp = sum(s["temperature_f"] for s in samples) / len(samples)
    avg_vis = sum(s["visibility_mi"] for s in samples) / len(samples)
    avg_hum = sum(s["humidity_pct"] for s in samples) / len(samples)
    avg_wind = sum(s["wind_speed_mph"] for s in samples) / len(samples)

    # Pick the most common (worst-case tiebreak) weather condition.
    # If conditions differ we want to surface the more dangerous one.
    condition_counts: Dict[str, int] = {}
    for s in samples:
        c = s["weather_condition"]
        condition_counts[c] = condition_counts.get(c, 0) + 1
    dominant_condition = max(condition_counts, key=lambda c: condition_counts[c])

    # Build description from the sample matching the dominant condition
    description = dominant_condition
    for s in samples:
        if s["weather_condition"] == dominant_condition:
            description = s["description"]
            break

    # Detect significant variation
    unique_conditions = set(condition_counts.keys())
    weather_warning: Optional[str] = None
    if len(unique_conditions) > 1:
        parts = ", ".join(
            f"{cond} ({cnt}/{len(samples)} points)"
            for cond, cnt in sorted(
                condition_counts.items(), key=lambda x: -x[1]
            )
        )
        weather_warning = f"Weather varies along route: {parts}"
        logger.warning(weather_warning)

    result: Dict = {
        "temperature_f": round(avg_temp, 1),
        "weather_condition": dominant_condition,
        "visibility_mi": round(avg_vis, 1),
        "humidity_pct": round(avg_hum, 1),
        "wind_speed_mph": round(avg_wind, 1),
        "description": description,
        "samples": len(samples),
        "weather_warning": weather_warning,
    }

    logger.info(
        f"Route weather ({len(samples)} samples): "
        f"{dominant_condition}, {avg_temp:.0f}°F avg"
    )
    return result
