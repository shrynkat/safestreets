"""
Test script for SafeStreets weather integration.

Tests the OpenWeather API integration by fetching weather for Tempe, AZ
and for a sample route from Tempe to Sedona.
"""

from weather_api import get_current_weather, get_weather_along_route, map_weather_to_category


def test_map_weather_to_category():
    """Verify condition-code mapping for representative codes."""
    print("=" * 60)
    print("Test: map_weather_to_category")
    print("=" * 60)

    cases = [
        (200, "Thunderstorm"),
        (300, "Drizzle"),
        (500, "Light Rain"),
        (501, "Rain"),
        (502, "Heavy Rain"),
        (600, "Light Snow"),
        (601, "Snow"),
        (602, "Heavy Snow"),
        (701, "Fog"),
        (741, "Fog"),
        (800, "Clear"),
        (802, "Cloudy"),
        (804, "Overcast"),
        (9999, "Clear"),   # unknown code falls back to Clear
    ]

    all_pass = True
    for code, expected in cases:
        result = map_weather_to_category(code)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] Code {code:>4d} -> {result!r} (expected {expected!r})")

    print()
    if all_pass:
        print("[PASS] All category mappings correct")
    else:
        print("[FAIL] Some category mappings incorrect")
    print()


def test_get_current_weather():
    """Fetch current weather for Tempe, AZ and print a formatted report."""
    print("=" * 60)
    print("Test: get_current_weather (Tempe, AZ)")
    print("=" * 60)

    # Tempe, AZ coordinates
    lat, lon = 33.4255, -111.9400
    weather = get_current_weather(lat, lon)

    print(f"\n  Location:    ({lat}, {lon})")
    print(f"  Condition:   {weather['weather_condition']}")
    print(f"  Description: {weather['description']}")
    print(f"  Temperature: {weather['temperature_f']}°F")
    print(f"  Visibility:  {weather['visibility_mi']} mi")
    print(f"  Humidity:    {weather['humidity_pct']}%")
    print(f"  Wind Speed:  {weather['wind_speed_mph']} mph")

    # Basic sanity checks
    checks = [
        ("temperature_f is numeric", isinstance(weather["temperature_f"], (int, float))),
        ("visibility_mi >= 0", weather["visibility_mi"] >= 0),
        ("humidity_pct 0-100", 0 <= weather["humidity_pct"] <= 100),
        ("wind_speed_mph >= 0", weather["wind_speed_mph"] >= 0),
        ("weather_condition non-empty", bool(weather["weather_condition"])),
        ("description non-empty", bool(weather["description"])),
    ]

    print()
    all_pass = True
    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {label}")

    print()
    if all_pass:
        print("[PASS] Current weather fetch OK")
    else:
        print("[FAIL] Some checks failed")
    print()


def test_get_weather_along_route():
    """Fetch weather along a Tempe → Sedona sample route."""
    print("=" * 60)
    print("Test: get_weather_along_route (Tempe -> Sedona)")
    print("=" * 60)

    # Simplified waypoints: Tempe → Black Canyon City → Camp Verde → Sedona
    waypoints = [
        (33.4255, -111.9400),  # Tempe
        (33.8303, -112.0519),  # Black Canyon City
        (34.2397, -111.8568),  # Camp Verde area
        (34.5601, -111.8215),  # Rimrock area
        (34.8697, -111.7610),  # Sedona
    ]

    weather = get_weather_along_route(waypoints)

    print(f"\n  Samples:     {weather['samples']}")
    print(f"  Condition:   {weather['weather_condition']}")
    print(f"  Description: {weather['description']}")
    print(f"  Avg Temp:    {weather['temperature_f']}°F")
    print(f"  Avg Vis:     {weather['visibility_mi']} mi")
    print(f"  Avg Humidity:{weather['humidity_pct']}%")
    print(f"  Avg Wind:    {weather['wind_speed_mph']} mph")

    if weather["weather_warning"]:
        print(f"\n  WARNING: {weather['weather_warning']}")
    else:
        print("\n  No weather variation warning (consistent conditions)")

    # Sanity checks
    checks = [
        ("samples >= 1", weather["samples"] >= 1),
        ("samples <= len(waypoints)", weather["samples"] <= len(waypoints)),
        ("weather_condition present", bool(weather["weather_condition"])),
        ("weather_warning is str or None",
         weather["weather_warning"] is None or isinstance(weather["weather_warning"], str)),
    ]

    print()
    all_pass = True
    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {label}")

    print()
    if all_pass:
        print("[PASS] Route weather fetch OK")
    else:
        print("[FAIL] Some checks failed")
    print()


def main():
    """Run all weather tests."""
    test_map_weather_to_category()
    test_get_current_weather()
    test_get_weather_along_route()
    print("=" * 60)
    print("All weather tests completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
