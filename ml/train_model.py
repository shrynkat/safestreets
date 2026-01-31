"""
XGBoost Model Training for SafeStreets Accident Risk Prediction

This script trains an XGBoost classifier to predict high-severity accidents
using engineered features from the data pipeline.

Features:
- Loads features from data/processed/ml_features.parquet
- Handles class imbalance using scale_pos_weight
- Trains XGBoost with optimized hyperparameters
- Evaluates model performance with multiple metrics
- Saves model, metrics, and feature importance plot
- Tracks experiments with MLflow
"""

import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

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
PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "ml_features.parquet"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
MODEL_PATH = MODELS_DIR / "risk_model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.png"

# MLflow settings
MLFLOW_EXPERIMENT_NAME = "accident_risk_prediction"
MLFLOW_TRACKING_URI = str(PROJECT_ROOT / "mlruns")


def setup_mlflow() -> None:
    """Configure MLflow tracking."""
    logger.info("Setting up MLflow tracking...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")


def load_features() -> pd.DataFrame:
    """Load engineered features from parquet file."""
    logger.info(f"Loading features from {FEATURES_PATH}...")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_PATH}")

    df = pd.read_parquet(FEATURES_PATH)
    logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns")

    return df


def prepare_data(df: pd.DataFrame) -> tuple:
    """
    Prepare data for training.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    logger.info("Preparing data for training...")

    # Separate features and target
    target_col = 'is_high_severity'
    id_col = 'accident_id'

    # Get feature columns (exclude id and target)
    feature_cols = [col for col in df.columns if col not in [target_col, id_col]]

    X = df[feature_cols]
    y = df[target_col]

    logger.info(f"Features: {len(feature_cols)} columns")
    logger.info(f"Target distribution:")
    for val, count in y.value_counts().items():
        pct = count / len(y) * 100
        logger.info(f"  Class {val}: {count:,} ({pct:.1f}%)")

    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logger.info(f"Train set: {len(X_train):,} samples")
    logger.info(f"Test set: {len(X_test):,} samples")

    return X_train, X_test, y_train, y_test, feature_cols


def calculate_scale_pos_weight(y: pd.Series) -> float:
    """
    Calculate scale_pos_weight for handling class imbalance.

    scale_pos_weight = count(negative) / count(positive)
    """
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight = neg_count / pos_count

    logger.info(f"Class imbalance ratio: {scale_pos_weight:.2f}")
    logger.info(f"  Negative (0): {neg_count:,}")
    logger.info(f"  Positive (1): {pos_count:,}")

    return scale_pos_weight


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float
) -> XGBClassifier:
    """
    Train XGBoost classifier with specified hyperparameters.
    """
    logger.info("Training XGBoost classifier...")

    # Model hyperparameters
    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'scale_pos_weight': scale_pos_weight,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 1
    }

    logger.info("Model parameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")

    # Initialize and train model
    model = XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False
    )

    logger.info("Model training completed")

    return model


def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Evaluate model on test set with multiple metrics.

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model on test set...")

    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'test_samples': int(len(y_test)),
        'positive_samples': int(y_test.sum()),
        'negative_samples': int((y_test == 0).sum()),
        'predicted_positive': int(y_pred.sum()),
        'predicted_negative': int((y_pred == 0).sum())
    }

    logger.info("\nEvaluation Metrics:")
    logger.info("="*50)
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    logger.info("="*50)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info("\nConfusion Matrix:")
    logger.info(f"  TN: {cm[0][0]:,}  FP: {cm[0][1]:,}")
    logger.info(f"  FN: {cm[1][0]:,}  TP: {cm[1][1]:,}")

    # Classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=['Low Severity', 'High Severity']))

    return metrics


def plot_feature_importance(
    model: XGBClassifier,
    feature_names: list,
    output_path: Path
) -> None:
    """
    Create and save feature importance plot.
    """
    logger.info("Creating feature importance plot...")

    # Get feature importances
    importances = model.feature_importances_

    # Create DataFrame for sorting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)

    # Plot top 20 features
    top_n = min(20, len(importance_df))
    plot_df = importance_df.tail(top_n)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    bars = ax.barh(plot_df['feature'], plot_df['importance'], color='steelblue')

    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Top Feature Importances - Accident Risk Prediction', fontsize=14, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, plot_df['importance']):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Feature importance plot saved to {output_path}")

    # Log top features
    logger.info("\nTop 10 Most Important Features:")
    for i, row in enumerate(importance_df.tail(10).iloc[::-1].itertuples(), 1):
        logger.info(f"  {i}. {row.feature}: {row.importance:.4f}")


def save_model(model: XGBClassifier, output_path: Path) -> None:
    """Save trained model to pickle file."""
    logger.info(f"Saving model to {output_path}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Model saved ({file_size_mb:.2f} MB)")


def save_metrics(metrics: dict, output_path: Path) -> None:
    """Save evaluation metrics to JSON file."""
    logger.info(f"Saving metrics to {output_path}...")

    # Add metadata
    metrics_with_meta = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'XGBClassifier',
        'metrics': metrics
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(metrics_with_meta, f, indent=2)

    logger.info("Metrics saved")


def log_to_mlflow(
    model: XGBClassifier,
    metrics: dict,
    params: dict,
    feature_names: list
) -> str:
    """
    Log experiment to MLflow.

    Returns:
        MLflow run ID
    """
    logger.info("Logging to MLflow...")

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics({
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics['roc_auc']
        })

        # Log model
        mlflow.xgboost.log_model(model, "model")

        # Log feature importance plot as artifact
        if FEATURE_IMPORTANCE_PATH.exists():
            mlflow.log_artifact(str(FEATURE_IMPORTANCE_PATH))

        # Log metrics JSON as artifact
        if METRICS_PATH.exists():
            mlflow.log_artifact(str(METRICS_PATH))

        # Log feature names
        mlflow.log_dict({'feature_names': feature_names}, 'feature_names.json')

        logger.info(f"MLflow run ID: {run_id}")

    return run_id


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("SafeStreets Accident Risk Prediction - Model Training")
    logger.info("="*60)

    try:
        # Setup MLflow
        setup_mlflow()

        # Load features
        df = load_features()

        # Prepare data
        X_train, X_test, y_train, y_test, feature_names = prepare_data(df)

        # Calculate class weight for imbalance handling
        scale_pos_weight = calculate_scale_pos_weight(y_train)

        # Define model parameters for logging
        model_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'scale_pos_weight': round(scale_pos_weight, 4),
            'objective': 'binary:logistic',
            'random_state': 42
        }

        # Train model
        model = train_model(X_train, y_train, scale_pos_weight)

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)

        # Create output directory
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Save feature importance plot
        plot_feature_importance(model, feature_names, FEATURE_IMPORTANCE_PATH)

        # Save model
        save_model(model, MODEL_PATH)

        # Save metrics
        save_metrics(metrics, METRICS_PATH)

        # Log to MLflow
        run_id = log_to_mlflow(model, metrics, model_params, feature_names)

        logger.info("\n" + "="*60)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Model saved: {MODEL_PATH}")
        logger.info(f"Metrics saved: {METRICS_PATH}")
        logger.info(f"Feature importance: {FEATURE_IMPORTANCE_PATH}")
        logger.info(f"MLflow run: {run_id}")
        logger.info("="*60)

        return model, metrics

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please run feature engineering first: python -m ml.features.engineer_features")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
