"""
Setup DuckDB Data Warehouse
Creates database and loads parquet data into tables
"""

import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PARQUET_FILE = PROJECT_ROOT / "data" / "processed" / "accidents.parquet"
DB_PATH = PROJECT_ROOT / "data" / "safestreets.duckdb"

def setup_warehouse():
    """Create DuckDB warehouse and load data"""
    
    print("=" * 70)
    print("DUCKDB WAREHOUSE SETUP")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print(f"Database: {DB_PATH}")
    print(f"Source: {PARQUET_FILE}\n")
    
    try:
        # Connect to DuckDB (creates file if doesn't exist)
        print("🔌 Connecting to DuckDB...")
        con = duckdb.connect(str(DB_PATH))
        print("✅ Connected\n")
        
        # Create raw schema
        print("📁 Creating schemas...")
        con.execute("CREATE SCHEMA IF NOT EXISTS raw")
        con.execute("CREATE SCHEMA IF NOT EXISTS staging")
        con.execute("CREATE SCHEMA IF NOT EXISTS analytics")
        print("✅ Schemas created: raw, staging, analytics\n")
        
        # Load parquet directly into raw table
        print("📥 Loading data from Parquet...")
        con.execute(f"""
            CREATE OR REPLACE TABLE raw.accidents AS 
            SELECT * FROM read_parquet('{PARQUET_FILE}')
        """)
        print("✅ Data loaded into raw.accidents\n")
        
        # Get row count
        result = con.execute("SELECT COUNT(*) FROM raw.accidents").fetchone()
        row_count = result[0]
        print(f"📊 Total records: {row_count:,}\n")
        
        # Show table info
        print("📋 Table Schema (first 10 columns):")
        schema_info = con.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'raw' 
            AND table_name = 'accidents'
            LIMIT 10
        """).fetchall()
        
        for col_name, data_type in schema_info:
            print(f"  {col_name:25} {data_type}")
        
        print(f"\n  ... and {46 - 10} more columns")
        
        # Create indexes for better query performance
        print("\n🔍 Creating indexes...")
        con.execute("CREATE INDEX IF NOT EXISTS idx_severity ON raw.accidents(Severity)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_state ON raw.accidents(State)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_start_time ON raw.accidents(Start_Time)")
        print("✅ Indexes created\n")
        
        # Run test queries
        print("🧪 Running test queries...\n")
        
        # Test 1: Count by severity
        print("Test 1: Accidents by Severity")
        severity_results = con.execute("""
            SELECT Severity, COUNT(*) as count, 
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
            FROM raw.accidents
            GROUP BY Severity
            ORDER BY Severity
        """).fetchall()
        
        for severity, count, pct in severity_results:
            print(f"  Level {severity}: {count:>10,} ({pct:>5}%)")
        
        # Test 2: Top 5 states
        print("\nTest 2: Top 5 States")
        state_results = con.execute("""
            SELECT State, COUNT(*) as count
            FROM raw.accidents
            GROUP BY State
            ORDER BY count DESC
            LIMIT 5
        """).fetchall()
        
        for state, count in state_results:
            print(f"  {state}: {count:>10,}")
        
        # Test 3: Query performance
        print("\nTest 3: Query Performance")
        start = datetime.now()
        con.execute("SELECT * FROM raw.accidents WHERE Severity = 3 AND State = 'CA' LIMIT 1000").fetchall()
        elapsed = (datetime.now() - start).total_seconds()
        print(f"  Query time: {elapsed:.3f} seconds")
        
        print("\n" + "=" * 70)
        print("✅ WAREHOUSE SETUP COMPLETE!")
        print("=" * 70)
        print(f"\nDatabase location: {DB_PATH}")
        print(f"Size: {DB_PATH.stat().st_size / (1024**2):.2f} MB")
        print("\nYou can now query the database using:")
        print("  - Python: duckdb.connect('data/safestreets.duckdb')")
        print("  - CLI: duckdb data/safestreets.duckdb")
        print("\nAvailable schemas:")
        print("  - raw.accidents (original data)")
        print("  - staging.* (for cleaned data - coming next)")
        print("  - analytics.* (for aggregated views - coming next)")
        
        print(f"\nEnd time: {datetime.now()}")
        print("=" * 70)
        
        con.close()
        return True
        
    except Exception as e:
        print(f"❌ Error setting up warehouse: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = setup_warehouse()
    sys.exit(0 if success else 1)