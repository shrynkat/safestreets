"""
Test script for SafeStreets risk scoring functionality.

Tests the ML-based route risk scoring by:
1. Fetching routes from Tempe to Sedona via Google Maps API
2. Scoring each route for safety using the trained model
3. Comparing routes and identifying danger zones
"""

from datetime import datetime

from directions_api import get_routes
from risk_scorer import score_entire_route


def main():
    """Run risk scoring tests with real route data."""
    origin = "Tempe, AZ"
    destination = "Sedona, AZ"

    # Friday at 5 PM
    departure_time = datetime(2024, 1, 5, 17, 0)

    # Weather conditions
    weather_condition = "Clear"
    temperature_f = 70.0
    visibility_mi = 10.0

    print("=" * 70)
    print("SafeStreets Route Risk Scoring Test")
    print("=" * 70)
    print(f"\nOrigin: {origin}")
    print(f"Destination: {destination}")
    print(f"Departure: {departure_time.strftime('%A, %B %d, %Y at %I:%M %p')}")
    print(f"Weather: {weather_condition}, {temperature_f}F, {visibility_mi} mi visibility")
    print("-" * 70)

    try:
        # Step 1: Fetch routes
        print("\nFetching routes from Google Maps API...")
        routes = get_routes(origin=origin, destination=destination)

        if not routes:
            print("No routes were returned. Check your API key configuration.")
            return

        print(f"Found {len(routes)} route alternatives\n")

        # Step 2: Score each route
        route_scores = []

        for i, route in enumerate(routes, 1):
            print(f"\n{'='*70}")
            print(f"ROUTE {i}: {route['summary']}")
            print(f"{'='*70}")
            print(f"Distance: {route['distance_text']} ({route['distance_meters']:,} meters)")
            print(f"Duration: {route['duration_text']}")
            print(f"Waypoints: {len(route['waypoints'])}")

            # Score the route
            print("\nScoring route safety...")
            risk_result = score_entire_route(
                waypoints=route['waypoints'],
                departure_time=departure_time,
                weather_condition=weather_condition,
                temperature_f=temperature_f,
                visibility_mi=visibility_mi
            )

            # Store for comparison
            route_scores.append({
                "route_num": i,
                "summary": route['summary'],
                "distance": route['distance_text'],
                "duration": route['duration_text'],
                "risk_data": risk_result
            })

            # Print results
            print(f"\n--- Risk Assessment ---")
            print(f"Overall Risk: {risk_result['overall_risk']:.3f}")
            print(f"Safety Score: {risk_result['safety_score']:.1f} / 100")
            print(f"Max Risk Segment: {risk_result['max_risk']:.3f}")
            print(f"Min Risk Segment: {risk_result['min_risk']:.3f}")
            print(f"Segments Scored: {risk_result['segments_scored']}")
            print(f"Danger Zones: {len(risk_result['danger_zones'])}")

        # Step 3: Show detailed segment analysis for the first route
        print("\n" + "=" * 70)
        print("DETAILED SEGMENT ANALYSIS - ROUTE 1")
        print("=" * 70)

        if route_scores:
            first_route = route_scores[0]
            segments = first_route['risk_data']['segment_scores']

            # Sort segments by risk
            sorted_by_risk = sorted(segments, key=lambda x: x['risk_score'], reverse=True)

            # Top 3 highest risk segments
            print("\nTop 3 HIGHEST RISK Segments:")
            print("-" * 50)
            for i, seg in enumerate(sorted_by_risk[:3], 1):
                print(f"  {i}. Risk: {seg['risk_score']:.3f} ({seg['risk_level']})")
                print(f"     Location: ({seg['lat']:.5f}, {seg['lon']:.5f})")
                print(f"     Confidence: {seg['confidence']:.3f}")

            # Top 3 safest segments
            print("\nTop 3 SAFEST Segments:")
            print("-" * 50)
            safest = sorted_by_risk[-3:][::-1]  # Last 3, reversed
            for i, seg in enumerate(safest, 1):
                print(f"  {i}. Risk: {seg['risk_score']:.3f} ({seg['risk_level']})")
                print(f"     Location: ({seg['lat']:.5f}, {seg['lon']:.5f})")
                print(f"     Confidence: {seg['confidence']:.3f}")

        # Step 4: Route comparison summary
        if len(route_scores) > 1:
            print("\n" + "=" * 70)
            print("ROUTE COMPARISON SUMMARY")
            print("=" * 70)

            # Sort routes by safety score (higher is better)
            sorted_routes = sorted(
                route_scores,
                key=lambda x: x['risk_data']['safety_score'],
                reverse=True
            )

            print(f"\n{'Rank':<6}{'Route':<25}{'Safety':<12}{'Risk':<10}{'Danger Zones':<15}")
            print("-" * 70)

            for rank, route in enumerate(sorted_routes, 1):
                risk = route['risk_data']
                print(f"{rank:<6}{route['summary']:<25}"
                      f"{risk['safety_score']:.1f}/100{'':<4}"
                      f"{risk['overall_risk']:.3f}{'':<5}"
                      f"{len(risk['danger_zones'])}")

            # Recommendation
            best = sorted_routes[0]
            print(f"\nRECOMMENDATION: Take {best['summary']}")
            print(f"  - Highest safety score: {best['risk_data']['safety_score']:.1f}/100")
            print(f"  - Distance: {best['distance']}")
            print(f"  - Duration: {best['duration']}")

        print("\n" + "=" * 70)
        print("Test completed successfully!")
        print("=" * 70)

    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
    except FileNotFoundError as e:
        print(f"\nModel Error: {e}")
        print("Make sure the ML model has been trained.")
    except Exception as e:
        print(f"\nError occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def test_with_mock_data():
    """
    Test risk scoring with mock waypoints (no API key required).

    Useful for testing the risk scoring logic without Google Maps API access.
    """
    print("=" * 70)
    print("SafeStreets Risk Scoring Test (Mock Data)")
    print("=" * 70)

    # Mock waypoints from Tempe to Sedona (approximate route)
    mock_waypoints = [
        (33.4255, -111.9400),  # Tempe
        (33.4500, -111.9000),
        (33.5000, -111.8500),
        (33.5500, -111.8000),
        (33.6000, -111.7500),
        (33.7000, -111.7200),
        (33.8000, -111.7000),
        (33.9000, -111.6800),
        (34.0000, -111.6600),
        (34.1500, -111.6500),
        (34.3000, -111.6800),
        (34.5000, -111.7000),
        (34.7000, -111.7300),
        (34.8697, -111.7610),  # Sedona
    ]

    departure_time = datetime(2024, 1, 5, 17, 0)  # Friday 5 PM

    print(f"\nMock route: Tempe, AZ to Sedona, AZ")
    print(f"Waypoints: {len(mock_waypoints)}")
    print(f"Departure: {departure_time.strftime('%A %I:%M %p')}")
    print("-" * 70)

    try:
        result = score_entire_route(
            waypoints=mock_waypoints,
            departure_time=departure_time,
            weather_condition="Clear",
            temperature_f=70.0,
            visibility_mi=10.0
        )

        print(f"\n--- Risk Assessment ---")
        print(f"Overall Risk: {result['overall_risk']:.3f}")
        print(f"Safety Score: {result['safety_score']:.1f} / 100")
        print(f"Max Risk: {result['max_risk']:.3f}")
        print(f"Min Risk: {result['min_risk']:.3f}")
        print(f"Segments Scored: {result['segments_scored']}")
        print(f"Danger Zones: {len(result['danger_zones'])}")

        # Show segment breakdown
        print("\nSegment Risk Distribution:")
        segments = result['segment_scores']
        sorted_segs = sorted(segments, key=lambda x: x['risk_score'], reverse=True)

        print("\nTop 3 Highest Risk:")
        for seg in sorted_segs[:3]:
            print(f"  ({seg['lat']:.4f}, {seg['lon']:.4f}): "
                  f"{seg['risk_score']:.3f} ({seg['risk_level']})")

        print("\nTop 3 Safest:")
        for seg in sorted_segs[-3:]:
            print(f"  ({seg['lat']:.4f}, {seg['lon']:.4f}): "
                  f"{seg['risk_score']:.3f} ({seg['risk_level']})")

        print("\n" + "=" * 70)
        print("Mock data test completed!")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\nModel Error: {e}")
    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--mock":
        # Run with mock data (no API key needed)
        test_with_mock_data()
    else:
        # Run with real API data
        main()
