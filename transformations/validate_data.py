"""
Data Quality Validation for SafeStreets
Checks for:
- Missing critical fields
- Invalid data ranges
- Data type consistency
- Duplicate records
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PARQUET_FILE = PROJECT_ROOT / "data" / "processed" / "accidents.parquet"

class DataValidator:
    """Validates accident data quality"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.issues = []
        
    def load_data(self):
        """Load parquet file"""
        print("📥 Loading data...")
        self.df = pd.read_parquet(self.file_path)
        print(f"✅ Loaded {len(self.df):,} rows, {len(self.df.columns)} columns\n")
        
    def check_required_columns(self):
        """Ensure critical columns exist"""
        print("🔍 Checking required columns...")
        required = ['ID', 'Severity', 'Start_Time', 'Start_Lat', 'Start_Lng', 'State']
        missing = [col for col in required if col not in self.df.columns]
        
        if missing:
            self.issues.append(f"Missing required columns: {missing}")
            print(f"  ❌ Missing: {missing}")
        else:
            print(f"  ✅ All required columns present")
        print()
        
    def check_null_values(self):
        """Check for nulls in critical columns"""
        print("🔍 Checking null values in critical columns...")
        critical_cols = ['ID', 'Severity', 'Start_Time', 'Start_Lat', 'Start_Lng']
        
        for col in critical_cols:
            if col in self.df.columns:
                null_count = self.df[col].isnull().sum()
                null_pct = (null_count / len(self.df)) * 100
                
                if null_count > 0:
                    self.issues.append(f"{col} has {null_count:,} nulls ({null_pct:.2f}%)")
                    print(f"  ⚠️  {col}: {null_count:,} nulls ({null_pct:.2f}%)")
                else:
                    print(f"  ✅ {col}: No nulls")
        print()
        
    def check_severity_range(self):
        """Validate severity values are 1-4"""
        print("🔍 Checking Severity values...")
        valid_severities = [1, 2, 3, 4]
        invalid = self.df[~self.df['Severity'].isin(valid_severities)]
        
        if len(invalid) > 0:
            self.issues.append(f"Found {len(invalid)} records with invalid severity")
            print(f"  ❌ {len(invalid)} invalid severity values")
            print(f"     Invalid values: {invalid['Severity'].unique()}")
        else:
            print(f"  ✅ All severity values valid (1-4)")
            
        # Show distribution
        print(f"\n  Severity Distribution:")
        for sev, count in self.df['Severity'].value_counts().sort_index().items():
            pct = (count / len(self.df)) * 100
            print(f"    Level {sev}: {count:>10,} ({pct:>5.2f}%)")
        print()
        
    def check_coordinates(self):
        """Validate latitude/longitude ranges"""
        print("🔍 Checking coordinate ranges...")
        
        # US latitude roughly 24°N to 49°N
        # US longitude roughly -125°W to -66°W
        lat_issues = self.df[
            (self.df['Start_Lat'] < 24) | 
            (self.df['Start_Lat'] > 50)
        ]
        
        lng_issues = self.df[
            (self.df['Start_Lng'] < -130) | 
            (self.df['Start_Lng'] > -60)
        ]
        
        if len(lat_issues) > 0:
            self.issues.append(f"{len(lat_issues)} records with out-of-range latitude")
            print(f"  ⚠️  {len(lat_issues)} latitude values outside US range")
        else:
            print(f"  ✅ All latitudes in valid range")
            
        if len(lng_issues) > 0:
            self.issues.append(f"{len(lng_issues)} records with out-of-range longitude")
            print(f"  ⚠️  {len(lng_issues)} longitude values outside US range")
        else:
            print(f"  ✅ All longitudes in valid range")
        print()
        
    def check_duplicates(self):
        """Check for duplicate accident IDs"""
        print("🔍 Checking for duplicate IDs...")
        duplicates = self.df[self.df.duplicated(subset=['ID'], keep=False)]
        
        if len(duplicates) > 0:
            self.issues.append(f"Found {len(duplicates)} duplicate IDs")
            print(f"  ⚠️  {len(duplicates)} duplicate records found")
        else:
            print(f"  ✅ No duplicate IDs")
        print()
        
    def check_date_range(self):
        """Validate date ranges"""
        print("🔍 Checking date ranges...")
        
        # Convert Start_Time if it's not already datetime
        if self.df['Start_Time'].dtype == 'object':
            try:
                self.df['Start_Time'] = pd.to_datetime(self.df['Start_Time'], format='mixed', errors='coerce')
            except:
                self.df['Start_Time'] = pd.to_datetime(self.df['Start_Time'], errors='coerce')
        
        # Check how many dates failed to parse
        null_dates = self.df['Start_Time'].isnull().sum()
        if null_dates > 0:
            print(f"  ⚠️  {null_dates} dates could not be parsed")
        
        min_date = self.df['Start_Time'].min()
        max_date = self.df['Start_Time'].max()
        
        print(f"  📅 Date range: {min_date} to {max_date}")
        
        # Check for future dates
        future = self.df[self.df['Start_Time'] > pd.Timestamp.now()]
        if len(future) > 0:
            self.issues.append(f"{len(future)} records with future dates")
            print(f"  ⚠️  {len(future)} records have future dates")
        else:
            print(f"  ✅ No future dates")
        print()
        
    def check_state_distribution(self):
        """Show top states by accident count"""
        print("🔍 Top 10 States by Accident Count:")
        top_states = self.df['State'].value_counts().head(10)
        
        for state, count in top_states.items():
            pct = (count / len(self.df)) * 100
            print(f"  {state}: {count:>10,} ({pct:>5.2f}%)")
        print()
        
    def generate_report(self):
        """Generate validation summary"""
        print("=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        if len(self.issues) == 0:
            print("✅ All validation checks passed!")
            print("   Data is ready for warehouse loading.")
        else:
            print(f"⚠️  Found {len(self.issues)} issues:")
            for i, issue in enumerate(self.issues, 1):
                print(f"   {i}. {issue}")
            print("\n   ⚡ Some issues may need cleaning before production use.")
        
        print("=" * 70)
        
    def run_all_checks(self):
        """Run all validation checks"""
        print("=" * 70)
        print("DATA QUALITY VALIDATION")
        print("=" * 70)
        print(f"File: {self.file_path}")
        print(f"Start time: {datetime.now()}\n")
        
        self.load_data()
        self.check_required_columns()
        self.check_null_values()
        self.check_severity_range()
        self.check_coordinates()
        self.check_duplicates()
        self.check_date_range()
        self.check_state_distribution()
        self.generate_report()
        
        print(f"\nEnd time: {datetime.now()}")
        
        return len(self.issues) == 0

def main():
    validator = DataValidator(PARQUET_FILE)
    success = validator.run_all_checks()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()