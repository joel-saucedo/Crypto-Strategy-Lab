"""
Data fetching script for cryptocurrency trading data.
"""

import asyncio
import argparse
import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.fetcher import DataFetcher
from data.preprocessor import DataPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def fetch_data_main():
    """Main data fetching function."""
    parser = argparse.ArgumentParser(description='Fetch cryptocurrency trading data')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'], 
                       help='Trading symbols to fetch')
    parser.add_argument('--exchange', default='binance', 
                       help='Exchange to fetch from (binance, coinbase)')
    parser.add_argument('--timeframe', default='1h', 
                       help='Data timeframe (1m, 5m, 1h, 1d)')
    parser.add_argument('--days', type=int, default=365, 
                       help='Number of days to fetch')
    parser.add_argument('--config', default='config/data_sources.yaml',
                       help='Data sources configuration file')
    parser.add_argument('--output-dir', default='data/raw',
                       help='Output directory for raw data')
    parser.add_argument('--preprocess', action='store_true',
                       help='Also run preprocessing')
    
    args = parser.parse_args()
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    logger.info(f"Fetching data for {args.symbols} from {args.exchange}")
    logger.info(f"Timeframe: {args.timeframe}, Period: {args.days} days")
    
    async with DataFetcher(args.config) as fetcher:
        results = await fetcher.fetch_multiple_symbols(
            symbols=args.symbols,
            exchange=args.exchange,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Save raw data
        saved_files = []
        for symbol, (data, quality_metrics) in results.items():
            logger.info(f"{symbol}: {len(data)} records, quality score: {quality_metrics['data_quality_score']:.3f}")
            
            filepath = fetcher.save_data(
                data=data,
                symbol=symbol,
                exchange=args.exchange,
                timeframe=args.timeframe,
                data_dir=args.output_dir
            )
            saved_files.append((filepath, symbol, data))
            
        # Preprocessing if requested
        if args.preprocess:
            logger.info("Starting preprocessing...")
            preprocessor = DataPreprocessor()
            
            processed_dir = Path(args.output_dir).parent / 'processed'
            processed_dir.mkdir(exist_ok=True)
            
            for filepath, symbol, raw_data in saved_files:
                try:
                    processed_data = preprocessor.prepare_features(raw_data)
                    
                    # Save processed data
                    processed_file = processed_dir / f"{symbol}_{args.exchange}_{args.timeframe}_processed.parquet"
                    processed_data.to_parquet(processed_file, index=False)
                    
                    logger.info(f"Processed {symbol}: {len(processed_data)} records with {processed_data.shape[1]} features")
                    
                except Exception as e:
                    logger.error(f"Failed to preprocess {symbol}: {e}")
        
        logger.info("Data fetching completed successfully!")

if __name__ == "__main__":
    asyncio.run(fetch_data_main())
