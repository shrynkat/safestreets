"""
SafeStreets Directions API Module

Provides functions to fetch and parse route alternatives from Google Maps
Directions API for risk comparison analysis.
"""

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


def get_routes(
    origin: str,
    destination: str,
    departure_time: Optional[datetime] = None
) -> List[Dict]:
    """
    Fetch alternative routes between origin and destination using Google Maps API.

    Retrieves multiple route alternatives and parses them into a structured
    format suitable for risk analysis and comparison.

    Args:
        origin: Starting location (address, place name, or "lat,lng").
        destination: Ending location (address, place name, or "lat,lng").
        departure_time: Optional departure time for traffic-aware routing.
                       If None, current time is used.

    Returns:
        List of route dictionaries, each containing:
            - distance_text: Human-readable distance (e.g., "120 mi")
            - distance_meters: Distance in meters (int)
            - duration_text: Human-readable duration (e.g., "2 hours 15 mins")
            - duration_seconds: Duration in seconds (int)
            - polyline: Encoded polyline string
            - summary: Route name/description (e.g., "I-17 N")
            - waypoints: List of (lat, lng) tuples along the route

    Raises:
        ValueError: If API key is not configured.
        googlemaps.exceptions.ApiError: If API request fails.

    Example:
        >>> routes = get_routes("Tempe, AZ", "Sedona, AZ")
        >>> print(f"Found {len(routes)} routes")
        >>> print(routes[0]['summary'])
        'I-17 N'
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

        # Parse each route alternative
        routes = []
        for i, route in enumerate(directions_result):
            try:
                # Extract leg information (assuming single leg for direct routes)
                leg = route["legs"][0]

                # Get the overview polyline for the entire route
                encoded_polyline = route["overview_polyline"]["points"]

                # Decode polyline to waypoints
                waypoints = decode_polyline(encoded_polyline)

                route_data = {
                    "distance_text": leg["distance"]["text"],
                    "distance_meters": leg["distance"]["value"],
                    "duration_text": leg["duration"]["text"],
                    "duration_seconds": leg["duration"]["value"],
                    "polyline": encoded_polyline,
                    "summary": route.get("summary", f"Route {i + 1}"),
                    "waypoints": waypoints
                }

                # Include duration_in_traffic if available
                if "duration_in_traffic" in leg:
                    route_data["duration_in_traffic_text"] = leg["duration_in_traffic"]["text"]
                    route_data["duration_in_traffic_seconds"] = leg["duration_in_traffic"]["value"]

                routes.append(route_data)
                logger.debug(f"Parsed route {i + 1}: {route_data['summary']}")

            except KeyError as e:
                logger.error(f"Error parsing route {i + 1}: Missing key {e}")
                continue

        logger.info(f"Successfully fetched {len(routes)} route alternatives")
        return routes

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
