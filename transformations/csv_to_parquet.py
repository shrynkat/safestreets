"""
Convert US Accidents CSV to Parquet format
- Processes data in chunks to handle large files
- Optimizes data types
- Properly handles datetime columns
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "US_Accidents_March23.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def convert_to_parquet():
    """Convert CSV to Parquet with optimizations"""
    
    print("=" * 70)
    print("CSV TO PARQUET CONVERSION")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print(f"Input: {RAW_DATA}")
    print(f"Output: {PROCESSED_DIR}")
    print()
    
    # Create output directory
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define optimized data types (excluding date columns)
    dtype_dict = {
        'ID': 'str',
        'Severity': 'int8',
        'Distance(mi)': 'float32',
        'Temperature(F)': 'float32',
        'Wind_Chill(F)': 'float32',
        'Humidity(%)': 'float32',
        'Pressure(in)': 'float32',
        'Visibility(mi)': 'float32',
        'Wind_Speed(mph)': 'float32',
        'Precipitation(in)': 'float32',
    }
    
    # Date columns to parse
    date_columns = ['Start_Time', 'End_Time', 'Weather_Timestamp']
    
    try:
        print("📖 Reading CSV file...")
        print("This will take 5-10 minutes for the full dataset...\n")
        
        # Read entire CSV with proper date parsing
        print("🔄 Loading and processing data...")
        df = pd.read_csv(
            RAW_DATA,
            dtype=dtype_dict,
            parse_dates=date_columns,
            low_memory=False
        )
        
        print(f"✅ Loaded {len(df):,} rows")
        print()
        
        # Convert boolean columns
        print("🔧 Converting boolean columns...")
        bool_columns = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 
                       'Junction', 'No_Exit', 'Railway', 'Roundabout',
                       'Station', 'Stop', 'Traffic_Calming', 
                       'Traffic_Signal', 'Turning_Loop']
        
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype('bool')
        
        print("✅ Boolean conversion complete")
        print()
        
        # Save to parquet
        parquet_file = PROCESSED_DIR / "accidents.parquet"
        
        print(f"💾 Writing to Parquet: {parquet_file}")
        df.to_parquet(
            parquet_file,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        print("✅ Write complete!")
        print()
        
        # Get file sizes
        csv_size = RAW_DATA.stat().st_size / (1024**3)  # GB
        parquet_size = parquet_file.stat().st_size / (1024**3)  # GB
        compression_ratio = csv_size / parquet_size
        
        print("📊 Results:")
        print(f"  CSV size:     {csv_size:.2f} GB")
        print(f"  Parquet size: {parquet_size:.2f} GB")
        print(f"  Compression:  {compression_ratio:.1f}x smaller")
        print(f"  Total rows:   {len(df):,}")
        print(f"  Total columns: {len(df.columns)}")
        print()
        
        # Verify the file
        print("🔍 Verifying Parquet file...")
        df_test = pd.read_parquet(parquet_file, engine='pyarrow')
        print(f"  Verified rows: {len(df_test):,}")
        print(f"  Verified columns: {len(df_test.columns)}")
        
        # Show data types
        print()
        print("📋 Column Types (first 10):")
        print(df_test.dtypes.head(10))
        
        print()
        print(f"End time: {datetime.now()}")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = convert_to_parquet()
    sys.exit(0 if success else 1)
