"""
Create Staging Tables
Transform raw data into cleaned, standardized format
"""

import duckdb
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "safestreets.duckdb"

def create_staging_tables():
    """Create cleaned staging tables"""
    
    print("=" * 70)
    print("CREATING STAGING TABLES")
    print("=" * 70)
    print(f"Start time: {datetime.now()}\n")
    
    try:
        con = duckdb.connect(str(DB_PATH))
        
        # 1. STAGING.ACCIDENTS - Cleaned and enriched accidents
        print("📊 Creating staging.accidents...")
        con.execute("""
            CREATE OR REPLACE TABLE staging.accidents AS
            SELECT 
                ID,
                Source,
                Severity,
                
                -- Temporal fields (convert VARCHAR to TIMESTAMP first)
                TRY_CAST(Start_Time AS TIMESTAMP) as start_time,
                TRY_CAST(End_Time AS TIMESTAMP) as end_time,
                EXTRACT(YEAR FROM TRY_CAST(Start_Time AS TIMESTAMP)) as year,
                EXTRACT(MONTH FROM TRY_CAST(Start_Time AS TIMESTAMP)) as month,
                EXTRACT(DAY FROM TRY_CAST(Start_Time AS TIMESTAMP)) as day,
                EXTRACT(HOUR FROM TRY_CAST(Start_Time AS TIMESTAMP)) as hour,
                EXTRACT(DOW FROM TRY_CAST(Start_Time AS TIMESTAMP)) as day_of_week,  -- 0=Sunday
                CASE 
                    WHEN EXTRACT(DOW FROM TRY_CAST(Start_Time AS TIMESTAMP)) IN (0, 6) THEN true
                    ELSE false
                END as is_weekend,
                CASE
                    WHEN EXTRACT(HOUR FROM TRY_CAST(Start_Time AS TIMESTAMP)) BETWEEN 6 AND 9 THEN 'Morning Rush'
                    WHEN EXTRACT(HOUR FROM TRY_CAST(Start_Time AS TIMESTAMP)) BETWEEN 10 AND 15 THEN 'Midday'
                    WHEN EXTRACT(HOUR FROM TRY_CAST(Start_Time AS TIMESTAMP)) BETWEEN 16 AND 19 THEN 'Evening Rush'
                    WHEN EXTRACT(HOUR FROM TRY_CAST(Start_Time AS TIMESTAMP)) BETWEEN 20 AND 23 THEN 'Evening'
                    ELSE 'Night'
                END as time_period,
                
                -- Location fields
                Start_Lat as latitude,
                Start_Lng as longitude,
                "Distance(mi)" as distance_mi,
                Street,
                City,
                County,
                State,
                Zipcode,
                Country,
                
                -- Weather fields
                COALESCE("Temperature(F)", 0) as temperature_f,
                COALESCE("Humidity(%)", 0) as humidity_pct,
                COALESCE("Pressure(in)", 0) as pressure_in,
                COALESCE("Visibility(mi)", 10) as visibility_mi,
                COALESCE("Wind_Speed(mph)", 0) as wind_speed_mph,
                COALESCE("Precipitation(in)", 0) as precipitation_in,
                Weather_Condition as weather_condition,
                
                -- Road features (boolean)
                COALESCE(Amenity, false) as has_amenity,
                COALESCE(Bump, false) as has_bump,
                COALESCE(Crossing, false) as has_crossing,
                COALESCE(Give_Way, false) as has_give_way,
                COALESCE(Junction, false) as has_junction,
                COALESCE(No_Exit, false) as has_no_exit,
                COALESCE(Railway, false) as has_railway,
                COALESCE(Roundabout, false) as has_roundabout,
                COALESCE(Station, false) as has_station,
                COALESCE(Stop, false) as has_stop,
                COALESCE(Traffic_Calming, false) as has_traffic_calming,
                COALESCE(Traffic_Signal, false) as has_traffic_signal,
                
                -- Calculated fields
                DATEDIFF('minute', TRY_CAST(Start_Time AS TIMESTAMP), TRY_CAST(End_Time AS TIMESTAMP)) as duration_minutes,
                CASE 
                    WHEN Severity >= 3 THEN true 
                    ELSE false 
                END as is_high_severity
                
            FROM raw.accidents
            WHERE TRY_CAST(Start_Time AS TIMESTAMP) IS NOT NULL  -- Filter out records with invalid dates
            AND Start_Lat IS NOT NULL
            AND Start_Lng IS NOT NULL
        """)
        
        row_count = con.execute("SELECT COUNT(*) FROM staging.accidents").fetchone()[0]
        print(f"✅ Created staging.accidents: {row_count:,} rows\n")
        
        # 2. STAGING.LOCATION_SUMMARY - Aggregate by location
        print("📍 Creating staging.location_summary...")
        con.execute("""
            CREATE OR REPLACE TABLE staging.location_summary AS
            SELECT 
                State,
                City,
                COUNT(*) as total_accidents,
                AVG(Severity) as avg_severity,
                SUM(CASE WHEN Severity >= 3 THEN 1 ELSE 0 END) as high_severity_count,
                AVG(duration_minutes) as avg_duration_minutes,
                
                -- Weather averages
                AVG(temperature_f) as avg_temperature,
                AVG(visibility_mi) as avg_visibility,
                
                -- Road features prevalence
                SUM(CASE WHEN has_traffic_signal THEN 1 ELSE 0 END) as accidents_at_signals,
                SUM(CASE WHEN has_junction THEN 1 ELSE 0 END) as accidents_at_junctions,
                SUM(CASE WHEN has_crossing THEN 1 ELSE 0 END) as accidents_at_crossings,
                
                -- Temporal patterns
                SUM(CASE WHEN is_weekend THEN 1 ELSE 0 END) as weekend_accidents,
                SUM(CASE WHEN time_period = 'Morning Rush' THEN 1 ELSE 0 END) as morning_rush_accidents,
                SUM(CASE WHEN time_period = 'Evening Rush' THEN 1 ELSE 0 END) as evening_rush_accidents
                
            FROM staging.accidents
            GROUP BY State, City
            HAVING COUNT(*) >= 10  -- Only cities with at least 10 accidents
        """)
        
        row_count = con.execute("SELECT COUNT(*) FROM staging.location_summary").fetchone()[0]
        print(f"✅ Created staging.location_summary: {row_count:,} rows\n")
        
        # 3. STAGING.TEMPORAL_PATTERNS - Time-based aggregations
        print("⏰ Creating staging.temporal_patterns...")
        con.execute("""
            CREATE OR REPLACE TABLE staging.temporal_patterns AS
            SELECT 
                year,
                month,
                day_of_week,
                hour,
                time_period,
                is_weekend,
                
                COUNT(*) as accident_count,
                AVG(Severity) as avg_severity,
                SUM(CASE WHEN Severity >= 3 THEN 1 ELSE 0 END) as high_severity_count,
                AVG(duration_minutes) as avg_duration,
                
                -- Weather conditions
                AVG(temperature_f) as avg_temperature,
                AVG(visibility_mi) as avg_visibility,
                AVG(precipitation_in) as avg_precipitation
                
            FROM staging.accidents
            GROUP BY year, month, day_of_week, hour, time_period, is_weekend
        """)
        
        row_count = con.execute("SELECT COUNT(*) FROM staging.temporal_patterns").fetchone()[0]
        print(f"✅ Created staging.temporal_patterns: {row_count:,} rows\n")
        
        # 4. STAGING.WEATHER_IMPACT - Weather condition analysis
        print("🌦️  Creating staging.weather_impact...")
        con.execute("""
            CREATE OR REPLACE TABLE staging.weather_impact AS
            SELECT 
                COALESCE(weather_condition, 'Unknown') as weather_condition,
                
                COUNT(*) as accident_count,
                AVG(Severity) as avg_severity,
                SUM(CASE WHEN Severity >= 3 THEN 1 ELSE 0 END) as high_severity_count,
                
                AVG(visibility_mi) as avg_visibility,
                AVG(temperature_f) as avg_temperature,
                AVG(wind_speed_mph) as avg_wind_speed,
                AVG(precipitation_in) as avg_precipitation,
                
                AVG(duration_minutes) as avg_duration
                
            FROM staging.accidents
            GROUP BY weather_condition
            HAVING COUNT(*) >= 100  -- Only conditions with substantial data
            ORDER BY accident_count DESC
        """)
        
        row_count = con.execute("SELECT COUNT(*) FROM staging.weather_impact").fetchone()[0]
        print(f"✅ Created staging.weather_impact: {row_count:,} rows\n")
        
        # Create indexes on staging tables
        print("🔍 Creating indexes on staging tables...")
        con.execute("CREATE INDEX IF NOT EXISTS idx_staging_severity ON staging.accidents(Severity)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_staging_state ON staging.accidents(State)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_staging_time ON staging.accidents(start_time)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_staging_high_sev ON staging.accidents(is_high_severity)")
        print("✅ Indexes created\n")
        
        # Verification queries
        print("=" * 70)
        print("STAGING TABLES SUMMARY")
        print("=" * 70)
        
        # Show table sizes
        tables = ['staging.accidents', 'staging.location_summary', 
                 'staging.temporal_patterns', 'staging.weather_impact']
        
        for table in tables:
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"{table:30} {count:>15,} rows")
        
        print("\n" + "=" * 70)
        print("✅ STAGING TABLES COMPLETE!")
        print("=" * 70)
        print(f"\nEnd time: {datetime.now()}")
        
        con.close()
        return True
        
    except Exception as e:
        print(f"❌ Error creating staging tables: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_staging_tables()
    sys.exit(0 if success else 1)