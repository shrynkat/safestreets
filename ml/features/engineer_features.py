"""
Feature Engineering for SafeStreets ML Pipeline

This script extracts and engineers features from the staging.accidents table
for training accident severity prediction models.

Features include:
- Temporal: hour, day_of_week, is_weekend, time_period
- Location: state, city (label encoded)
- Weather: temperature, visibility, precipitation, weather_condition
- Road: traffic signals, junctions, crossings, etc.
- Target: is_high_severity (Severity >= 3)
"""

import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "safestreets.duckdb"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ml_features.parquet"


def connect_to_database() -> duckdb.DuckDBPyConnection:
    """Establish connection to DuckDB database."""
    logger.info(f"Connecting to database: {DB_PATH}")

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    conn = duckdb.connect(str(DB_PATH), read_only=True)
    logger.info("Database connection established")
    return conn


def extract_raw_features(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Query staging.accidents and extract raw features."""
    logger.info("Extracting features from staging.accidents...")

    query = """
    SELECT
        -- Identifier
        ID as accident_id,

        -- Temporal features (already computed in staging)
        hour,
        day_of_week,
        is_weekend,
        time_period,
        month,
        year,

        -- Location features
        State as state,
        City as city,
        latitude,
        longitude,

        -- Weather features
        temperature_f,
        humidity_pct,
        pressure_in,
        visibility_mi,
        wind_speed_mph,
        precipitation_in,
        weather_condition,

        -- Road features
        has_amenity,
        has_bump,
        has_crossing,
        has_give_way,
        has_junction,
        has_no_exit,
        has_railway,
        has_roundabout,
        has_station,
        has_stop,
        has_traffic_calming,
        has_traffic_signal,

        -- Additional features
        distance_mi,
        duration_minutes,

        -- Target variable
        is_high_severity

    FROM staging.accidents
    WHERE is_high_severity IS NOT NULL
    """

    df = conn.execute(query).fetchdf()
    logger.info(f"Extracted {len(df):,} records from staging.accidents")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with appropriate strategies."""
    logger.info("Handling missing values...")

    initial_nulls = df.isnull().sum().sum()

    # Numerical features: fill with median
    numerical_cols = [
        'temperature_f', 'humidity_pct', 'pressure_in',
        'visibility_mi', 'wind_speed_mph', 'precipitation_in',
        'distance_mi', 'duration_minutes', 'latitude', 'longitude'
    ]

    for col in numerical_cols:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            null_count = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  {col}: filled {null_count:,} nulls with median ({median_val:.2f})")

    # Categorical features: fill with mode or 'Unknown'
    categorical_cols = ['state', 'city', 'weather_condition', 'time_period']

    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            null_count = df[col].isnull().sum()
            mode_val = df[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
            df[col] = df[col].fillna(fill_val)
            logger.info(f"  {col}: filled {null_count:,} nulls with '{fill_val}'")

    # Boolean features: fill with False (assume feature not present)
    boolean_cols = [col for col in df.columns if col.startswith('has_') or col == 'is_weekend']

    for col in boolean_cols:
        if col in df.columns and df[col].isnull().any():
            null_count = df[col].isnull().sum()
            df[col] = df[col].fillna(False)
            logger.info(f"  {col}: filled {null_count:,} nulls with False")

    # Temporal features: fill with mode
    temporal_cols = ['hour', 'day_of_week', 'month', 'year']

    for col in temporal_cols:
        if col in df.columns and df[col].isnull().any():
            null_count = df[col].isnull().sum()
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info(f"  {col}: filled {null_count:,} nulls with {mode_val}")

    final_nulls = df.isnull().sum().sum()
    logger.info(f"Missing values: {initial_nulls:,} -> {final_nulls:,}")

    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features using label encoding."""
    logger.info("Encoding categorical features...")

    encoders = {}

    # State encoding
    if 'state' in df.columns:
        le_state = LabelEncoder()
        df['state_encoded'] = le_state.fit_transform(df['state'].astype(str))
        encoders['state'] = le_state
        logger.info(f"  state: {len(le_state.classes_)} unique values encoded")

    # City encoding (high cardinality - use frequency-based encoding)
    if 'city' in df.columns:
        city_counts = df['city'].value_counts()
        # Map cities to their frequency rank (more frequent = lower rank number)
        city_rank = {city: rank for rank, city in enumerate(city_counts.index)}
        df['city_encoded'] = df['city'].map(city_rank)
        logger.info(f"  city: {len(city_rank)} unique values encoded (frequency-based)")

    # Weather condition encoding
    if 'weather_condition' in df.columns:
        le_weather = LabelEncoder()
        df['weather_encoded'] = le_weather.fit_transform(df['weather_condition'].astype(str))
        encoders['weather_condition'] = le_weather
        logger.info(f"  weather_condition: {len(le_weather.classes_)} unique values encoded")

    # Time period encoding (ordinal)
    if 'time_period' in df.columns:
        time_period_map = {
            'Night': 0,
            'Morning': 1,
            'Afternoon': 2,
            'Evening': 3
        }
        df['time_period_encoded'] = df['time_period'].map(time_period_map)
        # Handle any unmapped values
        df['time_period_encoded'] = df['time_period_encoded'].fillna(0).astype(int)
        logger.info(f"  time_period: ordinal encoding applied")

    return df


def create_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional derived features."""
    logger.info("Creating additional features...")

    # Road hazard count: sum of all road-related boolean features
    road_features = [
        'has_amenity', 'has_bump', 'has_crossing', 'has_give_way',
        'has_junction', 'has_no_exit', 'has_railway', 'has_roundabout',
        'has_station', 'has_stop', 'has_traffic_calming', 'has_traffic_signal'
    ]

    existing_road_features = [col for col in road_features if col in df.columns]
    if existing_road_features:
        df['road_feature_count'] = df[existing_road_features].sum(axis=1)
        logger.info(f"  road_feature_count: sum of {len(existing_road_features)} road features")

    # Is rush hour (7-9 AM or 4-7 PM)
    if 'hour' in df.columns:
        df['is_rush_hour'] = df['hour'].apply(
            lambda h: 1 if (7 <= h <= 9) or (16 <= h <= 19) else 0
        )
        logger.info("  is_rush_hour: created from hour")

    # Weather severity indicator
    if all(col in df.columns for col in ['visibility_mi', 'precipitation_in']):
        df['poor_weather'] = (
            (df['visibility_mi'] < 5) | (df['precipitation_in'] > 0)
        ).astype(int)
        logger.info("  poor_weather: created from visibility and precipitation")

    # Temperature extremes
    if 'temperature_f' in df.columns:
        df['extreme_temp'] = (
            (df['temperature_f'] < 32) | (df['temperature_f'] > 95)
        ).astype(int)
        logger.info("  extreme_temp: created from temperature_f")

    return df


def select_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order final feature set for ML."""
    logger.info("Selecting final feature set...")

    # Define feature groups
    id_cols = ['accident_id']

    temporal_features = [
        'hour', 'day_of_week', 'is_weekend', 'month',
        'time_period_encoded', 'is_rush_hour'
    ]

    location_features = [
        'state_encoded', 'city_encoded', 'latitude', 'longitude'
    ]

    weather_features = [
        'temperature_f', 'humidity_pct', 'pressure_in',
        'visibility_mi', 'wind_speed_mph', 'precipitation_in',
        'weather_encoded', 'poor_weather', 'extreme_temp'
    ]

    road_features = [
        'has_amenity', 'has_bump', 'has_crossing', 'has_give_way',
        'has_junction', 'has_no_exit', 'has_railway', 'has_roundabout',
        'has_station', 'has_stop', 'has_traffic_calming', 'has_traffic_signal',
        'road_feature_count', 'distance_mi'
    ]

    target = ['is_high_severity']

    # Combine all features
    all_features = id_cols + temporal_features + location_features + weather_features + road_features + target

    # Select only existing columns
    existing_features = [col for col in all_features if col in df.columns]
    missing_features = [col for col in all_features if col not in df.columns]

    if missing_features:
        logger.warning(f"Missing features (excluded): {missing_features}")

    df_final = df[existing_features].copy()

    # Convert boolean columns to int for ML compatibility
    bool_cols = df_final.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df_final[col] = df_final[col].astype(int)

    logger.info(f"Final feature set: {len(existing_features)} columns")
    return df_final


def save_features(df: pd.DataFrame, output_path: Path) -> None:
    """Save feature dataframe to parquet file."""
    logger.info(f"Saving features to {output_path}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, compression='snappy')

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {len(df):,} records ({file_size_mb:.2f} MB)")


def print_feature_summary(df: pd.DataFrame) -> None:
    """Print summary of engineered features."""
    logger.info("\n" + "="*60)
    logger.info("FEATURE ENGINEERING SUMMARY")
    logger.info("="*60)

    logger.info(f"\nDataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    logger.info(f"\nTarget distribution (is_high_severity):")
    target_counts = df['is_high_severity'].value_counts()
    for val, count in target_counts.items():
        pct = count / len(df) * 100
        logger.info(f"  {val}: {count:,} ({pct:.1f}%)")

    logger.info(f"\nFeature columns:")
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        logger.info(f"  {col}: {dtype} (nulls: {null_count})")

    logger.info("\n" + "="*60)


def main():
    """Main feature engineering pipeline."""
    logger.info("Starting feature engineering pipeline...")

    try:
        # Connect to database
        conn = connect_to_database()

        # Extract raw features
        df = extract_raw_features(conn)

        # Close database connection
        conn.close()
        logger.info("Database connection closed")

        # Handle missing values
        df = handle_missing_values(df)

        # Encode categorical features
        df = encode_categorical_features(df)

        # Create additional features
        df = create_additional_features(df)

        # Select final features
        df = select_final_features(df)

        # Print summary
        print_feature_summary(df)

        # Save to parquet
        save_features(df, OUTPUT_PATH)

        logger.info("Feature engineering pipeline completed successfully!")
        return df

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except duckdb.Error as e:
        logger.error(f"Database error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
