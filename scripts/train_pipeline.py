"""
Complete Training Pipeline

Orchestrates the entire training workflow from data ingestion to model saving.
"""

import sys
import argparse
from pathlib import Path
from loguru import logger
import pandas as pd
import json

# Add src to path
sys.path.append('.')

from src.data.ingestion import DataIngestion
from src.data.validation import DataValidator
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineer import FeatureEngineer
from src.models.demand_model import DemandModel
from src.evaluation.metrics import ModelEvaluator


def setup_logging(log_file: str = "logs/training.log"):
    """Setup logging configuration."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file, rotation="10 MB", level="DEBUG")
    
    logger.info("=" * 80)
    logger.info("FUEL PRICE OPTIMIZATION - TRAINING PIPELINE")
    logger.info("=" * 80)


def main(args):
    """Main training pipeline."""
    
    # Setup logging
    setup_logging(args.log_file)
    
    try:
        # Step 1: Data Ingestion
        logger.info("\n[STEP 1/7] Data Ingestion")
        logger.info("-" * 80)
        
        ingestion = DataIngestion(data_dir=args.data_dir)
        df = ingestion.load_historical_data(args.input_file)
        
        logger.info(f"Loaded {len(df)} records")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Step 2: Data Validation
        logger.info("\n[STEP 2/7] Data Validation")
        logger.info("-" * 80)
        
        validator = DataValidator()
        is_valid, errors = validator.validate_historical_data(df)
        
        if not is_valid:
            logger.error(f"Data validation failed: {errors}")
            if not args.skip_validation:
                raise ValueError("Data validation failed")
            logger.warning("Continuing despite validation errors (--skip-validation flag)")
        
        # Get data quality report
        quality_report = validator.get_data_quality_report(df)
        logger.info(f"Data Quality Report: {json.dumps(quality_report, indent=2, default=str)}")
        
        # Step 3: Data Preprocessing
        logger.info("\n[STEP 3/7] Data Preprocessing")
        logger.info("-" * 80)
        
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.preprocess_historical_data(df)
        
        logger.info(f"Processed data shape: {df_processed.shape}")
        logger.info(f"New columns created: {set(df_processed.columns) - set(df.columns)}")
        
        # Save processed data
        if args.save_processed:
            ingestion.save_processed_data(df_processed, "processed_historical.csv")
        
        # Step 4: Feature Engineering
        logger.info("\n[STEP 4/7] Feature Engineering")
        logger.info("-" * 80)
        
        engineer = FeatureEngineer(config={
            'lag_days': [1, 7, 14],
            'rolling_windows': [7, 14, 30]
        })
        
        df_features = engineer.create_features(df_processed)
        
        logger.info(f"Feature engineering complete. Final shape: {df_features.shape}")
        logger.info(f"Total features: {len(df_features.columns)}")
        
        # Save feature data
        if args.save_processed:
            ingestion.save_processed_data(df_features, "features_historical.csv")
        
        # Step 5: Model Training
        logger.info("\n[STEP 5/7] Model Training")
        logger.info("-" * 80)
        
        model_config = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'random_state': 42
        }
        
        model = DemandModel(config=model_config)
        training_results = model.train(
            df_features,
            test_size=0.2,
            validation_size=0.15
        )
        
        # Log training results
        logger.info("\nTraining Results:")
        for split, metrics in training_results.items():
            logger.info(f"\n{split.upper()}:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        # Step 6: Model Evaluation
        logger.info("\n[STEP 6/7] Model Evaluation")
        logger.info("-" * 80)
        
        evaluator = ModelEvaluator()
        
        # Get predictions for evaluation
        X, y = model.prepare_features(df_features)
        y_pred = model.predict(X)
        
        # Comprehensive evaluation
        eval_metrics = evaluator.evaluate_predictions(y, y_pred, "full_dataset")
        
        # Create evaluation report
        report = evaluator.create_evaluation_report(eval_metrics)
        logger.info(f"\n{report}")
        
        # Feature importance
        logger.info("\nTop 20 Important Features:")
        feature_importance = model.get_feature_importance(20)
        logger.info(f"\n{feature_importance.to_string()}")
        
        # Save feature importance
        if args.save_processed:
            feature_importance.to_csv('models/feature_importance.csv', index=False)
        
        # Step 7: Model Saving
        logger.info("\n[STEP 7/7] Saving Model")
        logger.info("-" * 80)
        
        model_path = Path(args.output_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model.save(args.output_model)
        logger.info(f"Model saved to {args.output_model}")
        
        # Save training metadata
        metadata = {
            'training_date': pd.Timestamp.now().isoformat(),
            'data_shape': df_features.shape,
            'n_features': len(model.feature_columns),
            'model_config': model_config,
            'training_metrics': {k: {mk: float(mv) for mk, mv in v.items()} 
                               for k, v in training_results.items()},
            'evaluation_metrics': {k: float(v) for k, v in eval_metrics.items()}
        }
        
        metadata_path = model_path.parent / 'training_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Training metadata saved to {metadata_path}")
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"\nModel Performance Summary:")
        logger.info(f"  Test R²: {training_results['test']['r2']:.4f}")
        logger.info(f"  Test RMSE: {training_results['test']['rmse']:.2f} liters")
        logger.info(f"  Test MAPE: {training_results['test']['mape']:.2f}%")
        logger.info(f"\nModel saved to: {args.output_model}")
        
    except Exception as e:
        logger.exception(f"Training pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fuel price optimization model")
    
    parser.add_argument(
        '--input-file',
        type=str,
        default='data/raw/oil_retail_history.csv',
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '--output-model',
        type=str,
        default='models/demand_model.pkl',
        help='Path to save trained model'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Directory containing data files'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='logs/training.log',
        help='Path to log file'
    )
    
    parser.add_argument(
        '--save-processed',
        action='store_true',
        help='Save processed data and features'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip data validation errors'
    )
    
    args = parser.parse_args()
    main(args)
