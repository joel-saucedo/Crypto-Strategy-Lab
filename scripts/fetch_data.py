"""
Generic data fetching script for cryptocurrency and financial data.
Supports multiple providers (FMP, Yahoo Finance, Exchange APIs).
"""

import asyncio
import argparse
import yaml
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.data_manager import DataManager
from data.providers.base_provider import DataRequest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_data_directories():
    """Ensure data directories exist."""
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / 'data' / 'raw'
    processed_dir = base_dir / 'data' / 'processed'
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    return raw_dir, processed_dir

def calculate_features(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical features from OHLCV data."""
    processed_data = data.copy()
    
    # Basic returns
    processed_data['daily_return'] = data['close'].pct_change()
    processed_data['log_return'] = np.log(data['close'] / data['close'].shift(1))
    
    # Technical indicators
    processed_data['sma_20'] = data['close'].rolling(window=20).mean()
    processed_data['sma_50'] = data['close'].rolling(window=50).mean()
    processed_data['volatility_20'] = processed_data['daily_return'].rolling(window=20).std()
    processed_data['high_low_range'] = (data['high'] - data['low']) / data['close']
    
    if 'volume' in data.columns:
        processed_data['volume_ma_20'] = data['volume'].rolling(window=20).mean()
    
    # Price momentum
    processed_data['price_momentum_5'] = data['close'] / data['close'].shift(5) - 1
    processed_data['price_momentum_10'] = data['close'] / data['close'].shift(10) - 1
    
    # RSI calculation
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    processed_data['rsi'] = 100 - (100 / (1 + rs))
    
    # Drop NaN values
    processed_data = processed_data.dropna()
    
    return processed_data

async def fetch_data_main():
    """Main data fetching function."""
    parser = argparse.ArgumentParser(description='Fetch financial data from various sources')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSD'], 
                       help='Symbols to fetch (e.g., BTCUSD, AAPL, TSLA)')
    parser.add_argument('--provider', default='fmp', 
                       help='Data provider (fmp, yfinance)')
    parser.add_argument('--asset-type', default='crypto', 
                       help='Asset type (crypto, stock, forex)')
    parser.add_argument('--timeframe', default='1d', 
                       help='Data timeframe (1d, 1h, etc.)')
    parser.add_argument('--days', type=int, default=365, 
                       help='Number of days to fetch')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date (YYYY-MM-DD) - overrides --days')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date (YYYY-MM-DD) - defaults to today')
    parser.add_argument('--max-history', action='store_true',
                       help='Fetch maximum available history (overrides date options)')
    
    args = parser.parse_args()
    
    # Setup directories
    raw_dir, processed_dir = setup_data_directories()
    
    # Calculate date range
    if args.max_history:
        # For crypto, start from early Bitcoin era
        # For stocks, start from a reasonable historical point
        if args.asset_type == 'crypto':
            start_date = datetime(2013, 1, 1)
        else:
            start_date = datetime(2010, 1, 1)
        end_date = datetime.now()
        logger.info(f"Fetching maximum available history from {start_date.date()}")
    else:
        if args.start_date:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        else:
            start_date = datetime.now() - timedelta(days=args.days)
        
        if args.end_date:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        else:
            end_date = datetime.now()
    
    logger.info(f"Fetching data for {args.symbols} from {args.provider}")
    logger.info(f"Asset type: {args.asset_type}, Timeframe: {args.timeframe}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    
    try:
        # Initialize data manager
        data_manager = DataManager({
            'providers': {
                args.provider: {},
            },
            'provider_priority': [args.provider],
            'enable_cache': True
        })
        
        # Create request
        request = DataRequest(
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=args.timeframe,
            asset_type=args.asset_type
        )
        
        # Fetch data
        data = await data_manager.fetch_data_direct(request)
        
        if data is not None and not data.empty:
            logger.info(f"✅ Successfully fetched {len(data)} rows of data")
            logger.info(f"📊 Columns: {list(data.columns)}")
            logger.info(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
            
            # Process each symbol
            for symbol in args.symbols:
                # For single symbol, use all data; for multi-symbol, extract by symbol
                if len(args.symbols) == 1:
                    symbol_data = data
                else:
                    # Handle multi-symbol data extraction if needed
                    symbol_data = data  # Simplified for now
                
                if not symbol_data.empty:
                    logger.info(f"\n📈 Processing {symbol}...")
                    
                    # Save raw data
                    raw_file = raw_dir / f"{symbol}_daily_ohlcv.csv"
                    symbol_data.to_csv(raw_file)
                    logger.info(f"📁 Raw data saved: {raw_file}")
                    
                    # Calculate features and save processed data
                    processed_data = calculate_features(symbol_data)
                    processed_file = processed_dir / f"{symbol}_daily_processed.parquet"
                    processed_data.to_parquet(processed_file)
                    logger.info(f"📁 Processed data saved: {processed_file}")
                    
                    # Show statistics
                    logger.info(f"📊 {symbol} Statistics:")
                    logger.info(f"   Trading days: {len(processed_data)}")
                    logger.info(f"   Price range: ${symbol_data['low'].min():,.2f} - ${symbol_data['high'].max():,.2f}")
                    logger.info(f"   Latest close: ${symbol_data['close'].iloc[-1]:,.2f}")
                    
                    if 'daily_return' in processed_data.columns:
                        returns = processed_data['daily_return']
                        logger.info(f"   Mean daily return: {returns.mean():.4f}")
                        logger.info(f"   Daily volatility: {returns.std():.4f}")
            
            logger.info("\n🎉 Data fetching completed successfully!")
            
        else:
            logger.error("❌ Failed to fetch data or received empty dataset")
            
    except Exception as e:
        logger.error(f"❌ Error during data fetching: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(fetch_data_main())
