"""
Test script for SafeStreets time optimizer.

Tests departure-time analysis for a Tempe → Sedona trip on a Friday,
identifies the safest and most dangerous departure windows, and prints
a user-friendly recommendation report.
"""

from .time_optimizer import (
    analyze_departure_times,
    compare_times,
    get_optimal_window,
)

# Shared weather for all tests
WEATHER = {
    "weather_condition": "Clear",
    "temperature_f": 70.0,
    "visibility_mi": 10.0,
}

ORIGIN = "Tempe, AZ"
DESTINATION = "Sedona, AZ"
# A Friday
DATE = "2024-02-02"


def test_analyze_departure_times():
    """Test full 6-slot departure analysis and print report."""
    print("=" * 60)
    print("Test: analyze_departure_times")
    print(f"  {ORIGIN} -> {DESTINATION}  ({DATE}, Friday)")
    print("=" * 60)

    result = analyze_departure_times(ORIGIN, DESTINATION, DATE, WEATHER)

    print(f"\nRoute analyzed: {result['route_summary']}")
    print(f"\nBest departure:  {result['best_time']}  "
          f"(safety score {result['best_safety_score']})")
    print(f"Worst departure: {result['worst_time']}  "
          f"(safety score {result['worst_safety_score']})")
    print(f"Improvement:     {result['improvement_pct']}% safer at best vs worst")

    # Detailed breakdown
    print(f"\n{'Time':<12} {'Safety':>8} {'Risk':>8} {'Level':<10} {'Danger Zones':>12}")
    print("-" * 54)
    for rec in result["recommendations"]:
        print(
            f"{rec['time_label']:<12} "
            f"{rec['safety_score']:>8.1f} "
            f"{rec['risk_score']:>8.3f} "
            f"{rec['risk_level']:<10} "
            f"{rec['danger_zones_count']:>12}"
        )

    # Sanity checks
    print()
    checks = [
        ("has recommendations", len(result["recommendations"]) > 0),
        ("best_time is a string", isinstance(result["best_time"], str)),
        ("worst_time is a string", isinstance(result["worst_time"], str)),
        ("best >= worst safety",
         result["best_safety_score"] >= result["worst_safety_score"]),
        ("improvement_pct >= 0", result["improvement_pct"] >= 0),
        ("sorted safest-first",
         all(
             result["recommendations"][i]["safety_score"]
             >= result["recommendations"][i + 1]["safety_score"]
             for i in range(len(result["recommendations"]) - 1)
         )),
    ]

    all_pass = True
    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {label}")

    print()
    if all_pass:
        print("[PASS] analyze_departure_times OK")
    else:
        print("[FAIL] Some checks failed")
    print()

    return result


def test_compare_times():
    """Compare morning vs evening departure."""
    print("=" * 60)
    print("Test: compare_times (8 AM vs 5 PM)")
    print("=" * 60)

    result = compare_times(
        ORIGIN,
        DESTINATION,
        f"{DATE}T08:00:00",
        f"{DATE}T17:00:00",
        WEATHER,
    )

    print(f"\n  {result['time1_label']:>10}  safety={result['time1_safety']:<6}  "
          f"risk={result['time1_risk']}")
    print(f"  {result['time2_label']:>10}  safety={result['time2_safety']:<6}  "
          f"risk={result['time2_risk']}")
    print(f"\n  Safer: {result['safer']}")
    print(f"  Difference: {result['safety_difference']} points")
    print(f"  {result['recommendation']}")

    checks = [
        ("safer in valid values", result["safer"] in ("time1", "time2", "equal")),
        ("safety_difference >= 0", result["safety_difference"] >= 0),
        ("recommendation non-empty", bool(result["recommendation"])),
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
        print("[PASS] compare_times OK")
    else:
        print("[FAIL] Some checks failed")
    print()


def test_get_optimal_window():
    """Find best 2-hour window between 6 AM and 10 PM."""
    print("=" * 60)
    print("Test: get_optimal_window (6 AM – 10 PM)")
    print("=" * 60)

    result = get_optimal_window(ORIGIN, DESTINATION, DATE, WEATHER)

    print(f"\n  Optimal window: {result['window_start']} – {result['window_end']}")
    print(f"  Avg safety:     {result['avg_safety_score']}")

    # Hourly breakdown
    print(f"\n  {'Hour':<12} {'Safety':>8}")
    print("  " + "-" * 22)
    for entry in result["hourly_scores"]:
        marker = " <-- best window" if (
            entry["hour"] >= result["window_start_hour"]
            and entry["hour"] <= result["window_end_hour"] - 1
        ) else ""
        print(f"  {entry['time_label']:<12} {entry['safety_score']:>8.1f}{marker}")

    print(f"\n  {result['recommendation']}")

    checks = [
        ("window_start is str", isinstance(result["window_start"], str)),
        ("window_end is str", isinstance(result["window_end"], str)),
        ("avg_safety_score > 0", result["avg_safety_score"] > 0),
        ("hourly_scores populated", len(result["hourly_scores"]) >= 2),
        ("window covers 2 hours",
         result["window_end_hour"] - result["window_start_hour"] == 2),
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
        print("[PASS] get_optimal_window OK")
    else:
        print("[FAIL] Some checks failed")
    print()


def print_user_recommendation(analysis_result):
    """Print a user-friendly travel recommendation."""
    r = analysis_result
    print("=" * 60)
    print("  SAFESTREETS TRAVEL RECOMMENDATION")
    print("=" * 60)
    print(f"\n  Route:  {ORIGIN} -> {DESTINATION}")
    print(f"  Date:   {DATE} (Friday)")
    print(f"  Weather: {WEATHER['weather_condition']}, "
          f"{WEATHER['temperature_f']}°F, "
          f"visibility {WEATHER['visibility_mi']} mi")

    print(f"\n  >>> Depart at {r['best_time']} for the safest trip <<<")
    print(f"      Safety score: {r['best_safety_score']}/100")

    print(f"\n  Avoid departing at {r['worst_time']} if possible")
    print(f"      Safety score: {r['worst_safety_score']}/100")

    print(f"\n  Choosing {r['best_time']} over {r['worst_time']} is "
          f"{r['improvement_pct']}% safer.")
    print("=" * 60)


def main():
    """Run all time optimizer tests."""
    result = test_analyze_departure_times()
    test_compare_times()
    test_get_optimal_window()
    print_user_recommendation(result)
    print("\nAll time optimizer tests completed.")


if __name__ == "__main__":
    main()
