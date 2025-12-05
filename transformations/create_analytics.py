"""
Create Analytics Views
High-level business intelligence queries for risk analysis
"""

import duckdb
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "safestreets.duckdb"

def create_analytics_views():
    """Create analytics layer views"""
    
    print("=" * 70)
    print("CREATING ANALYTICS VIEWS")
    print("=" * 70)
    print(f"Start time: {datetime.now()}\n")
    
    try:
        con = duckdb.connect(str(DB_PATH))
        
        # 1. TOP RISK LOCATIONS
        print("🎯 Creating analytics.top_risk_locations...")
        con.execute("""
            CREATE OR REPLACE VIEW analytics.top_risk_locations AS
            SELECT 
                State,
                City,
                total_accidents,
                high_severity_count,
                ROUND(avg_severity, 2) as avg_severity,
                ROUND((high_severity_count * 100.0 / total_accidents), 2) as high_severity_rate,
                ROUND(avg_duration_minutes, 1) as avg_duration_minutes,
                accidents_at_signals,
                accidents_at_junctions,
                morning_rush_accidents,
                evening_rush_accidents
            FROM staging.location_summary
            WHERE total_accidents >= 100  -- Focus on locations with substantial data
            ORDER BY high_severity_count DESC
        """)
        print("✅ Created analytics.top_risk_locations\n")
        
        # 2. HOURLY RISK PROFILE
        print("⏰ Creating analytics.hourly_risk_profile...")
        con.execute("""
            CREATE OR REPLACE VIEW analytics.hourly_risk_profile AS
            SELECT 
                hour,
                SUM(accident_count) as total_accidents,
                ROUND(AVG(avg_severity), 2) as avg_severity,
                SUM(high_severity_count) as high_severity_count,
                ROUND(AVG(avg_temperature), 1) as avg_temperature,
                ROUND(AVG(avg_visibility), 1) as avg_visibility
            FROM staging.temporal_patterns
            GROUP BY hour
            ORDER BY hour
        """)
        print("✅ Created analytics.hourly_risk_profile\n")
        
        # 3. DAY OF WEEK ANALYSIS
        print("📅 Creating analytics.day_of_week_analysis...")
        con.execute("""
            CREATE OR REPLACE VIEW analytics.day_of_week_analysis AS
            SELECT 
                day_of_week,
                CASE day_of_week
                    WHEN 0 THEN 'Sunday'
                    WHEN 1 THEN 'Monday'
                    WHEN 2 THEN 'Tuesday'
                    WHEN 3 THEN 'Wednesday'
                    WHEN 4 THEN 'Thursday'
                    WHEN 5 THEN 'Friday'
                    WHEN 6 THEN 'Saturday'
                END as day_name,
                is_weekend,
                SUM(accident_count) as total_accidents,
                ROUND(AVG(avg_severity), 2) as avg_severity,
                SUM(high_severity_count) as high_severity_count
            FROM staging.temporal_patterns
            GROUP BY day_of_week, is_weekend
            ORDER BY day_of_week
        """)
        print("✅ Created analytics.day_of_week_analysis\n")
        
        # 4. WEATHER RISK RANKING
        print("🌦️  Creating analytics.weather_risk_ranking...")
        con.execute("""
            CREATE OR REPLACE VIEW analytics.weather_risk_ranking AS
            SELECT 
                weather_condition,
                accident_count,
                high_severity_count,
                ROUND(avg_severity, 2) as avg_severity,
                ROUND((high_severity_count * 100.0 / accident_count), 2) as high_severity_rate,
                ROUND(avg_visibility, 1) as avg_visibility,
                ROUND(avg_temperature, 1) as avg_temperature,
                ROUND(avg_wind_speed, 1) as avg_wind_speed,
                ROUND(avg_precipitation, 2) as avg_precipitation
            FROM staging.weather_impact
            ORDER BY high_severity_rate DESC
        """)
        print("✅ Created analytics.weather_risk_ranking\n")
        
        # 5. STATE SUMMARY
        print("🗺️  Creating analytics.state_summary...")
        con.execute("""
            CREATE OR REPLACE VIEW analytics.state_summary AS
            SELECT 
                State,
                SUM(total_accidents) as total_accidents,
                COUNT(DISTINCT City) as cities_affected,
                ROUND(AVG(avg_severity), 2) as avg_severity,
                SUM(high_severity_count) as high_severity_count,
                SUM(accidents_at_signals) as total_signal_accidents,
                SUM(accidents_at_junctions) as total_junction_accidents
            FROM staging.location_summary
            GROUP BY State
            ORDER BY total_accidents DESC
        """)
        print("✅ Created analytics.state_summary\n")
        
        # 6. RUSH HOUR IMPACT
        print("🚗 Creating analytics.rush_hour_impact...")
        con.execute("""
            CREATE OR REPLACE VIEW analytics.rush_hour_impact AS
            SELECT 
                time_period,
                SUM(accident_count) as total_accidents,
                ROUND(AVG(avg_severity), 2) as avg_severity,
                SUM(high_severity_count) as high_severity_count,
                ROUND((SUM(high_severity_count) * 100.0 / SUM(accident_count)), 2) as high_severity_rate
            FROM staging.temporal_patterns
            GROUP BY time_period
            ORDER BY total_accidents DESC
        """)
        print("✅ Created analytics.rush_hour_impact\n")
        
        # 7. HIGH RISK SEGMENTS (Most dangerous combinations)
        print("⚠️  Creating analytics.high_risk_segments...")
        con.execute("""
            CREATE OR REPLACE VIEW analytics.high_risk_segments AS
            SELECT 
                State,
                City,
                time_period,
                COUNT(*) as accident_count,
                ROUND(AVG(Severity), 2) as avg_severity,
                SUM(CASE WHEN is_high_severity THEN 1 ELSE 0 END) as high_severity_count,
                SUM(CASE WHEN has_traffic_signal THEN 1 ELSE 0 END) as signal_accidents,
                SUM(CASE WHEN has_junction THEN 1 ELSE 0 END) as junction_accidents,
                ROUND(AVG(visibility_mi), 1) as avg_visibility
            FROM staging.accidents
            WHERE Severity >= 2  -- Focus on moderate to severe
            GROUP BY State, City, time_period
            HAVING COUNT(*) >= 50  -- Statistically significant
            ORDER BY high_severity_count DESC
        """)
        print("✅ Created analytics.high_risk_segments\n")
        
        print("=" * 70)
        print("SAMPLE ANALYTICS QUERIES")
        print("=" * 70)
        
        # Sample Query 1: Top 10 most dangerous cities
        print("\n📍 Top 10 Most Dangerous Cities (by high-severity accidents):")
        results = con.execute("""
            SELECT City, State, total_accidents, high_severity_count, avg_severity
            FROM analytics.top_risk_locations
            LIMIT 10
        """).fetchall()
        
        for city, state, total, high_sev, avg_sev in results:
            print(f"  {city:20} {state:3} | Total: {total:>6,} | High Severity: {high_sev:>5,} | Avg: {avg_sev:.2f}")
        
        # Sample Query 2: Peak accident hours
        print("\n⏰ Peak Accident Hours:")
        results = con.execute("""
            SELECT hour, total_accidents, avg_severity
            FROM analytics.hourly_risk_profile
            ORDER BY total_accidents DESC
            LIMIT 5
        """).fetchall()
        
        for hour, total, avg_sev in results:
            print(f"  {hour:02d}:00 | Accidents: {total:>8,} | Avg Severity: {avg_sev:.2f}")
        
        # Sample Query 3: Weather impact
        print("\n🌦️  Most Dangerous Weather Conditions:")
        results = con.execute("""
            SELECT weather_condition, accident_count, high_severity_rate
            FROM analytics.weather_risk_ranking
            WHERE accident_count >= 1000
            ORDER BY high_severity_rate DESC
            LIMIT 5
        """).fetchall()
        
        for weather, count, rate in results:
            print(f"  {weather:30} | Count: {count:>7,} | High Severity Rate: {rate:.1f}%")
        
        # Sample Query 4: Rush hour comparison
        print("\n🚗 Rush Hour vs Other Times:")
        results = con.execute("""
            SELECT time_period, total_accidents, high_severity_rate
            FROM analytics.rush_hour_impact
            ORDER BY total_accidents DESC
        """).fetchall()
        
        for period, total, rate in results:
            print(f"  {period:15} | Accidents: {total:>8,} | High Severity: {rate:.1f}%")
        
        print("\n" + "=" * 70)
        print("✅ ANALYTICS VIEWS COMPLETE!")
        print("=" * 70)
        print("\nAvailable analytics views:")
        print("  - analytics.top_risk_locations")
        print("  - analytics.hourly_risk_profile")
        print("  - analytics.day_of_week_analysis")
        print("  - analytics.weather_risk_ranking")
        print("  - analytics.state_summary")
        print("  - analytics.rush_hour_impact")
        print("  - analytics.high_risk_segments")
        
        print(f"\nEnd time: {datetime.now()}")
        print("=" * 70)
        
        con.close()
        return True
        
    except Exception as e:
        print(f"❌ Error creating analytics views: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_analytics_views()
    sys.exit(0 if success else 1)