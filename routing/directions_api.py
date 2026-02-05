"""
SafeStreets Directions API Module

Provides functions to fetch and parse route alternatives from Google Maps
Directions API for risk comparison analysis.
"""

import math
import os
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import googlemaps
import polyline
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


def _get_api_key() -> str:
    """
    Retrieve Google Maps API key from environment variables.

    Returns:
        str: The Google Maps API key.

    Raises:
        ValueError: If GOOGLE_MAPS_API_KEY is not set in environment.
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_MAPS_API_KEY not found in environment variables. "
            "Please set it in your .env file or environment."
        )
    return api_key


def decode_polyline(polyline_string: str) -> List[Tuple[float, float]]:
    """
    Decode an encoded polyline string into a list of coordinates.

    Google Maps API returns routes as encoded polyline strings for efficiency.
    This function decodes them into usable lat/lng coordinate pairs.

    Args:
        polyline_string: Encoded polyline string from Google Maps API.

    Returns:
        List of (latitude, longitude) tuples representing the route path.

    Example:
        >>> coords = decode_polyline("_p~iF~ps|U_ulLnnqC_mqNvxq`@")
        >>> print(coords[0])
        (38.5, -120.2)
    """
    try:
        coordinates = polyline.decode(polyline_string)
        return coordinates
    except Exception as e:
        logger.error(f"Error decoding polyline: {e}")
        return []


MAX_ROUTES = 3

# Fallback avoid strategies when Google returns fewer than MAX_ROUTES
_AVOID_STRATEGIES = [
    ("highways", "avoid_highways"),
    ("tolls", "avoid_tolls"),
    ("ferries", "avoid_ferries"),
]


def _parse_route(
    route_raw: Dict,
    index: int,
    route_type: str = "fastest",
) -> Optional[Dict]:
    """
    Parse a single route from Google Maps Directions API response.

    Args:
        route_raw: Raw route dict from the API.
        index: 1-based route index (used as fallback summary).
        route_type: Label for how this route was obtained.

    Returns:
        Parsed route dict, or None if parsing fails.
    """
    try:
        leg = route_raw["legs"][0]
        encoded_polyline = route_raw["overview_polyline"]["points"]
        waypoints = decode_polyline(encoded_polyline)

        route_data = {
            "distance_text": leg["distance"]["text"],
            "distance_meters": leg["distance"]["value"],
            "duration_text": leg["duration"]["text"],
            "duration_seconds": leg["duration"]["value"],
            "polyline": encoded_polyline,
            "summary": route_raw.get("summary", f"Route {index}"),
            "waypoints": waypoints,
            "route_type": route_type,
        }

        if "duration_in_traffic" in leg:
            route_data["duration_in_traffic_text"] = leg["duration_in_traffic"]["text"]
            route_data["duration_in_traffic_seconds"] = leg["duration_in_traffic"]["value"]

        return route_data

    except KeyError as e:
        logger.error(f"Error parsing route {index}: Missing key {e}")
        return None


def _routes_are_similar(
    route_a: Dict,
    route_b: Dict,
    threshold_km: float = 1.0,
) -> bool:
    """
    Check if two routes are essentially the same by comparing key waypoints.

    Compares overview polylines first (exact match), then samples waypoints
    at 25%, 50%, and 75% of each route to check geographic similarity.

    Args:
        route_a: First parsed route dict.
        route_b: Second parsed route dict.
        threshold_km: Maximum distance (km) between sample points to
                      consider routes similar.

    Returns:
        True if routes appear to follow the same path.
    """
    if route_a.get("polyline") == route_b.get("polyline"):
        return True

    wp_a = route_a.get("waypoints", [])
    wp_b = route_b.get("waypoints", [])

    if not wp_a or not wp_b:
        return False

    for frac in (0.25, 0.5, 0.75):
        idx_a = int(len(wp_a) * frac)
        idx_b = int(len(wp_b) * frac)

        lat_a, lon_a = wp_a[idx_a]
        lat_b, lon_b = wp_b[idx_b]

        # Approximate: 1 degree ≈ 111 km
        dist_km = math.sqrt((lat_a - lat_b) ** 2 + (lon_a - lon_b) ** 2) * 111
        if dist_km > threshold_km:
            return False

    return True


def get_routes(
    origin: str,
    destination: str,
    departure_time: Optional[datetime] = None
) -> List[Dict]:
    """
    Fetch alternative routes between origin and destination using Google Maps API.

    Retrieves multiple route alternatives and parses them into a structured
    format suitable for risk analysis and comparison.  When Google returns
    fewer than MAX_ROUTES alternatives, additional API calls are made with
    ``avoid`` options (highways, tolls, ferries) to force different paths.
    Duplicate routes are detected via waypoint sampling and filtered out.

    Args:
        origin: Starting location (address, place name, or "lat,lng").
        destination: Ending location (address, place name, or "lat,lng").
        departure_time: Optional departure time for traffic-aware routing.
                       If None, current time is used.

    Returns:
        Up to MAX_ROUTES route dictionaries, each containing:
            - distance_text: Human-readable distance (e.g., "120 mi")
            - distance_meters: Distance in meters (int)
            - duration_text: Human-readable duration (e.g., "2 hours 15 mins")
            - duration_seconds: Duration in seconds (int)
            - polyline: Encoded polyline string
            - summary: Route name/description (e.g., "I-17 N")
            - waypoints: List of (lat, lng) tuples along the route
            - route_type: How this route was obtained
              ("fastest", "avoid_highways", "avoid_tolls", "avoid_ferries")

    Raises:
        ValueError: If API key is not configured.
        googlemaps.exceptions.ApiError: If API request fails.

    Example:
        >>> routes = get_routes("Tempe, AZ", "Sedona, AZ")
        >>> print(f"Found {len(routes)} routes")
        >>> print(routes[0]['summary'], routes[0]['route_type'])
        'I-17 N' 'fastest'
    """
    try:
        # Get API key and initialize client
        api_key = _get_api_key()
        client = googlemaps.Client(key=api_key)

        # Set departure time for traffic-aware routing
        if departure_time is None:
            departure_time = datetime.now()

        logger.info(f"Fetching routes from '{origin}' to '{destination}'")

        # Request directions with alternatives
        directions_result = client.directions(
            origin=origin,
            destination=destination,
            mode="driving",
            alternatives=True,
            departure_time=departure_time
        )

        if not directions_result:
            logger.warning("No routes found for the given origin and destination")
            return []

        # Parse initial alternatives
        routes = []
        for i, route_raw in enumerate(directions_result):
            parsed = _parse_route(route_raw, i + 1, route_type="fastest")
            if parsed:
                routes.append(parsed)
                logger.debug(f"Parsed route {i + 1}: {parsed['summary']}")

        # If Google returned fewer than MAX_ROUTES, request additional
        # alternatives using avoid options to force different paths.
        if len(routes) < MAX_ROUTES:
            base_kwargs = {
                "origin": origin,
                "destination": destination,
                "mode": "driving",
                "departure_time": departure_time,
            }

            for avoid_value, route_type in _AVOID_STRATEGIES:
                if len(routes) >= MAX_ROUTES:
                    break

                try:
                    logger.info(f"Fetching alternative route (avoid={avoid_value})")
                    alt_result = client.directions(
                        **base_kwargs,
                        alternatives=False,
                        avoid=avoid_value,
                    )

                    if alt_result:
                        candidate = _parse_route(
                            alt_result[0], len(routes) + 1, route_type=route_type
                        )
                        if candidate and not any(
                            _routes_are_similar(candidate, r) for r in routes
                        ):
                            routes.append(candidate)
                            logger.info(
                                f"Added unique alternative: {candidate['summary']} ({route_type})"
                            )
                        else:
                            logger.debug(
                                f"Avoid {avoid_value} returned duplicate route, skipping"
                            )

                except Exception as e:
                    logger.warning(f"Failed to fetch avoid={avoid_value} alternative: {e}")

        logger.info(f"Successfully fetched {len(routes)} route alternatives")
        return routes[:MAX_ROUTES]

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise
    except googlemaps.exceptions.ApiError as e:
        logger.error(f"Google Maps API error: {e}")
        raise
    except googlemaps.exceptions.TransportError as e:
        logger.error(f"Network error connecting to Google Maps API: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching routes: {e}")
        raise


if __name__ == "__main__":
    # Quick test
    try:
        routes = get_routes("Tempe, AZ", "Sedona, AZ")
        print(f"Found {len(routes)} routes")
        for route in routes:
            print(f"  - {route['summary']}: {route['distance_text']}, {route['duration_text']}")
    except Exception as e:
        print(f"Error: {e}")
