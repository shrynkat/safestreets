"""
Fetch US Accidents dataset from Kaggle
Dataset: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
"""

import os
import sys
from pathlib import Path
import kaggle
from datetime import datetime

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

def download_accidents_data():
    """Download US Accidents dataset from Kaggle"""
    
    print("=" * 60)
    print("SAFESTREETS DATA ACQUISITION")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")
    print(f"Downloading to: {RAW_DATA_DIR}")
    print()
    
    # Create raw data directory if it doesn't exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Dataset identifier
    dataset_name = "sobhanmoosavi/us-accidents"
    
    try:
        print(f"📥 Downloading dataset: {dataset_name}")
        print("This may take several minutes (dataset is ~1.5 GB)...")
        print()
        
        # Download dataset
        kaggle.api.dataset_download_files(
            dataset_name,
            path=RAW_DATA_DIR,
            unzip=True,
            quiet=False
        )
        
        print()
        print("✅ Download complete!")
        print()
        
        # List downloaded files
        print("Downloaded files:")
        for file in RAW_DATA_DIR.glob("*"):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.2f} MB)")
        
        print()
        print(f"End time: {datetime.now()}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

if __name__ == "__main__":
    success = download_accidents_data()
    sys.exit(0 if success else 1)
