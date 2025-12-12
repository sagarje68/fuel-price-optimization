"""
Price Prediction Script

Makes price recommendations based on today's market data.
"""

import sys
import argparse
import json
from pathlib import Path
from loguru import logger
from datetime import datetime

# Add src to path
sys.path.append('.')

from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineer import FeatureEngineer
from src.models.demand_model import DemandModel
from src.models.optimizer import PriceOptimizer


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)


def load_historical_context():
    """Load historical data for context."""
    logger.info("Loading historical data for context...")
    
    ingestion = DataIngestion()
    preprocessor = DataPreprocessor()
    engineer = FeatureEngineer()
    
    df = ingestion.load_historical_data()
    df = preprocessor.preprocess_historical_data(df)
    df = engineer.create_features(df)
    
    logger.info(f"Loaded {len(df)} historical records")
    
    return df


def main(args):
    """Main prediction workflow."""
    
    setup_logging(args.verbose)
    
    logger.info("=" * 80)
    logger.info("FUEL PRICE OPTIMIZATION - PRICE RECOMMENDATION")
    logger.info("=" * 80)
    
    try:
        # Step 1: Load today's market data
        logger.info("\n[STEP 1/5] Loading Market Data")
        logger.info("-" * 80)
        
        ingestion = DataIngestion()
        
        if args.input:
            today_data = ingestion.load_today_data(args.input)
        else:
            # Use default example
            today_data = ingestion.load_today_data()
        
        logger.info(f"Market data loaded for date: {today_data['date']}")
        logger.info(f"Current price: ${today_data['price']:.2f}")
        logger.info(f"Cost: ${today_data['cost']:.2f}")
        logger.info(f"Competitor prices: ${today_data['comp1_price']:.2f}, "
                   f"${today_data['comp2_price']:.2f}, ${today_data['comp3_price']:.2f}")
        
        # Step 2: Load trained model
        logger.info("\n[STEP 2/5] Loading Trained Model")
        logger.info("-" * 80)
        
        model = DemandModel()
        model.load(args.model_path)
        logger.info(f"Model loaded from {args.model_path}")
        logger.info(f"Model uses {len(model.feature_columns)} features")
        
        # Step 3: Prepare features
        logger.info("\n[STEP 3/5] Preparing Features")
        logger.info("-" * 80)
        
        preprocessor = DataPreprocessor()
        engineer = FeatureEngineer()
        
        # Load historical data for context
        historical_df = load_historical_context()
        
        # Prepare today's data
        today_df = preprocessor.prepare_today_data(today_data, historical_df)
        today_features = engineer.create_prediction_features(today_df, historical_df)
        
        logger.info(f"Features prepared. Shape: {today_features.shape}")
        
        # Step 4: Optimize price
        logger.info("\n[STEP 4/5] Optimizing Price")
        logger.info("-" * 80)
        
        optimizer_config = {
            'max_price_change_pct': 5.0,
            'min_profit_margin_pct': 5.0,
            'price_search_range': 0.15
        }
        
        optimizer = PriceOptimizer(model, config=optimizer_config)
        
        competitor_prices = {
            'comp1_price': today_data['comp1_price'],
            'comp2_price': today_data['comp2_price'],
            'comp3_price': today_data['comp3_price']
        }
        
        recommendation = optimizer.optimize_price(
            today_features,
            today_data['price'],
            today_data['cost'],
            competitor_prices
        )
        
        # Step 5: Display results
        logger.info("\n[STEP 5/5] Price Recommendation")
        logger.info("=" * 80)
        
        print("\n" + "=" * 80)
        print("PRICE RECOMMENDATION SUMMARY")
        print("=" * 80)
        print(f"\nDate: {today_data['date']}")
        print(f"\nCurrent Market Conditions:")
        print(f"  Current Price:        ${today_data['price']:.2f}")
        print(f"  Cost:                 ${today_data['cost']:.2f}")
        print(f"  Competitor 1 Price:   ${today_data['comp1_price']:.2f}")
        print(f"  Competitor 2 Price:   ${today_data['comp2_price']:.2f}")
        print(f"  Competitor 3 Price:   ${today_data['comp3_price']:.2f}")
        
        print(f"\n{'RECOMMENDED PRICE':^80}")
        print(f"{'─' * 80}")
        print(f"\n  💰 Optimal Price:     ${recommendation['recommended_price']:.2f}")
        print(f"\n  Price Change:         ${recommendation['price_change_from_current']:+.2f} "
              f"({recommendation['price_change_pct']:+.2f}%)")
        
        print(f"\n{'EXPECTED OUTCOMES':^80}")
        print(f"{'─' * 80}")
        print(f"\n  📊 Expected Volume:   {recommendation['expected_volume']:.0f} liters")
        print(f"  💵 Expected Profit:   ${recommendation['expected_profit']:.2f}")
        print(f"  📈 Profit Margin:     {recommendation['profit_margin_pct']:.2f}%")
        
        print(f"\n{'COMPETITIVE ANALYSIS':^80}")
        print(f"{'─' * 80}")
        print(f"\n  Position:             {recommendation['competitive_position'].replace('_', ' ').title()}")
        print(f"  vs Competitor 1:      ${recommendation['price_vs_competitors']['comp1_diff']:+.2f}")
        print(f"  vs Competitor 2:      ${recommendation['price_vs_competitors']['comp2_diff']:+.2f}")
        print(f"  vs Competitor 3:      ${recommendation['price_vs_competitors']['comp3_diff']:+.2f}")
        print(f"  vs Average:           ${recommendation['price_vs_avg_comp']:+.2f}")
        
        print(f"\n{'CONFIDENCE':^80}")
        print(f"{'─' * 80}")
        print(f"\n  Confidence Score:     {recommendation['confidence_score']:.2%}")
        
        print("\n" + "=" * 80)
        
        # Save recommendation
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add timestamp
            recommendation['timestamp'] = datetime.now().isoformat()
            recommendation['input_data'] = today_data
            
            with open(output_path, 'w') as f:
                json.dump(recommendation, f, indent=2, default=str)
            
            logger.info(f"\nRecommendation saved to {output_path}")
        
        # Generate simulation if requested
        if args.simulate:
            logger.info("\n" + "=" * 80)
            logger.info("PROFIT SIMULATION")
            logger.info("=" * 80)
            
            simulation_df = optimizer.simulate_price_range(
                today_features,
                today_data['cost'],
                recommendation['optimization_details']['search_range'][0],
                recommendation['optimization_details']['search_range'][1],
                n_points=50
            )
            
            # Find top 5 prices by profit
            top_5 = simulation_df.nlargest(5, 'profit')
            
            print("\nTop 5 Price Points by Profit:")
            print("-" * 80)
            print(f"{'Rank':<6} {'Price':<10} {'Volume':<12} {'Profit':<12} {'Margin %':<10}")
            print("-" * 80)
            
            for idx, (i, row) in enumerate(top_5.iterrows(), 1):
                print(f"{idx:<6} ${row['price']:<9.2f} {row['volume']:<12.0f} "
                      f"${row['profit']:<11.2f} {row['profit_margin_pct']:<10.2f}")
            
            if args.output:
                sim_path = Path(args.output).parent / 'simulation.csv'
                simulation_df.to_csv(sim_path, index=False)
                logger.info(f"\nSimulation results saved to {sim_path}")
        
        logger.info("\n" + "=" * 80)
        logger.info("PREDICTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.exception(f"Prediction failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get price recommendation")
    
    parser.add_argument(
        '--input',
        type=str,
        help='Path to today\'s market data JSON file'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/demand_model.pkl',
        help='Path to trained model'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save recommendation JSON'
    )
    
    parser.add_argument(
        '--simulate',
        action='store_true',
        help='Generate profit simulation across price range'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    main(args)
