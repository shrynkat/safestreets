"""
Test script for SafeStreets routing functionality.

Tests the Google Maps Directions API integration by fetching routes
between Tempe, AZ and Sedona, AZ.  Verifies that multiple alternatives
are returned and that each route carries a ``route_type`` label.
"""

from directions_api import get_routes, MAX_ROUTES, _routes_are_similar


def main():
    """Run routing tests with sample origin and destination."""
    origin = "Tempe, AZ"
    destination = "Sedona, AZ"

    print("=" * 60)
    print("SafeStreets Routing Test")
    print("=" * 60)
    print(f"\nOrigin: {origin}")
    print(f"Destination: {destination}")
    print("-" * 60)

    try:
        # Fetch routes
        routes = get_routes(origin=origin, destination=destination)

        # Display results
        print(f"\nNumber of routes found: {len(routes)}")
        print("-" * 60)

        if not routes:
            print("No routes were returned. Check your API key and network connection.")
            return

        # ---- Verify multiple routes ----
        if len(routes) >= 2:
            print(f"\n[PASS] Got {len(routes)} route alternatives (>= 2)")
        else:
            print(f"\n[WARN] Only {len(routes)} route returned — expected >= 2")

        if len(routes) <= MAX_ROUTES:
            print(f"[PASS] Route count ({len(routes)}) within MAX_ROUTES ({MAX_ROUTES})")
        else:
            print(f"[FAIL] Route count ({len(routes)}) exceeds MAX_ROUTES ({MAX_ROUTES})")

        # ---- Verify route_type field ----
        valid_types = {"fastest", "avoid_highways", "avoid_tolls", "avoid_ferries"}
        all_have_type = True
        for route in routes:
            if "route_type" not in route:
                all_have_type = False
                break
            if route["route_type"] not in valid_types:
                all_have_type = False
                break

        if all_have_type:
            print("[PASS] All routes have a valid route_type field")
        else:
            print("[FAIL] Some routes are missing route_type or have invalid values")

        # ---- Verify deduplication ----
        duplicates_found = False
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                if _routes_are_similar(routes[i], routes[j]):
                    duplicates_found = True
                    print(
                        f"[FAIL] Routes {i + 1} and {j + 1} are similar "
                        f"('{routes[i]['summary']}' vs '{routes[j]['summary']}')"
                    )

        if not duplicates_found:
            print("[PASS] No duplicate routes detected")

        # ---- Print details for each route ----
        print("-" * 60)
        for i, route in enumerate(routes, 1):
            print(f"\nRoute {i}: {route['summary']} [{route['route_type']}]")
            print(f"  Distance: {route['distance_text']} ({route['distance_meters']:,} meters)")
            print(f"  Duration: {route['duration_text']} ({route['duration_seconds']:,} seconds)")
            print(f"  Number of waypoints: {len(route['waypoints'])}")

            # Show traffic duration if available
            if "duration_in_traffic_text" in route:
                print(f"  Duration (with traffic): {route['duration_in_traffic_text']}")

        # Print first 3 waypoints of first route
        print("-" * 60)
        print("\nFirst 3 waypoints of first route:")
        first_route_waypoints = routes[0]["waypoints"]
        for j, waypoint in enumerate(first_route_waypoints[:3], 1):
            lat, lng = waypoint
            print(f"  Waypoint {j}: ({lat:.6f}, {lng:.6f})")

        print(f"\n  ... and {len(first_route_waypoints) - 3} more waypoints")
        print("-" * 60)
        print("\nTest completed successfully!")

    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        print("Make sure GOOGLE_MAPS_API_KEY is set in your .env file.")
    except Exception as e:
        print(f"\nError occurred: {type(e).__name__}: {e}")
        print("\nPossible causes:")
        print("  - Invalid API key")
        print("  - API key not enabled for Directions API")
        print("  - Network connectivity issues")
        print("  - Rate limiting")


if __name__ == "__main__":
    main()
