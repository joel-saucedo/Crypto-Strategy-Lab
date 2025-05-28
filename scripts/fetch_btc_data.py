"""
Data fetcher for cryptocurrency data using FMP API.
Fetches maximum available historical data for Bitcoin (BTCUSD).
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.providers.fmp_provider import FMPProvider
from data.providers.base_provider import DataRequest

def setup_data_directories():
    """Ensure data directories exist."""
    base_dir = Path(__file__).parent
    raw_dir = base_dir / 'data' / 'raw'
    processed_dir = base_dir / 'data' / 'processed'
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    return raw_dir, processed_dir

def calculate_returns_and_features(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate returns and technical features from OHLCV data."""
    processed_data = data.copy()
    
    # Basic returns
    processed_data['daily_return'] = data['close'].pct_change()
    processed_data['log_return'] = np.log(data['close'] / data['close'].shift(1))
    
    # Technical indicators
    processed_data['sma_20'] = data['close'].rolling(window=20).mean()
    processed_data['sma_50'] = data['close'].rolling(window=50).mean()
    processed_data['volatility_20'] = processed_data['daily_return'].rolling(window=20).std()
    processed_data['high_low_range'] = (data['high'] - data['low']) / data['close']
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

def save_data(data: pd.DataFrame, symbol: str, raw_dir: Path, processed_dir: Path):
    """Save data in both raw and processed formats."""
    
    # Save raw data as CSV
    raw_file = raw_dir / f"{symbol}_daily_ohlcv.csv"
    data.to_csv(raw_file)
    print(f"📁 Raw data saved to: {raw_file}")
    
    # Calculate features and save processed data as Parquet
    processed_data = calculate_returns_and_features(data)
    processed_file = processed_dir / f"{symbol}_daily_processed.parquet"
    processed_data.to_parquet(processed_file)
    print(f"📁 Processed data saved to: {processed_file}")
    
    return processed_data

async def fetch_btc_data():
    """Fetch maximum available BTC historical data from FMP."""
    
    print("🚀 Fetching Bitcoin Historical Data from FMP")
    print("=" * 50)
    
    try:
        # Setup data directories
        raw_dir, processed_dir = setup_data_directories()
        print(f"📂 Data directories ready")
        
        # Create FMP provider
        provider = FMPProvider()
        
        # Debug API key loading
        api_key = os.getenv('FMP_API_KEY')
        print(f"✅ Environment API Key: {'Yes' if api_key else 'No'}")
        if api_key:
            print(f"✅ API Key length: {len(api_key)} chars")
        print(f"✅ Provider API Key: {'Yes' if provider.api_key else 'No'}")
        
        if not provider.api_key:
            print("❌ No API key found! Check .env file")
            return False
        
        # Bitcoin symbol for FMP
        symbol = 'BTCUSD'
        
        # Fetch maximum available data (FMP typically has data from ~2013)
        # Start from when Bitcoin data became widely available
        start_date = datetime(2013, 1, 1)
        end_date = datetime.now()
        
        print(f"📊 Fetching {symbol} data from FMP")
        print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
        print(f"📈 Requesting daily timeframe data...")
        
        # Create DataRequest
        request = DataRequest(
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date,
            timeframe='1d',
            asset_type='crypto'
        )
        
        # Fetch data
        response = await provider.fetch_data(request)
        
        if response.data is not None and not response.data.empty:
            print(f"\n✅ Successfully fetched {len(response.data)} days of data!")
            print(f"📊 Columns: {list(response.data.columns)}")
            print(f"📅 Data spans: {response.data.index[0]} to {response.data.index[-1]}")
            
            # Save and process data
            print(f"\n📈 Processing {symbol} data...")
            processed_data = save_data(response.data, symbol, raw_dir, processed_dir)
            
            # Show statistics
            print(f"\n📊 Data Statistics:")
            print(f"   Total trading days: {len(processed_data)}")
            print(f"   Price range: ${response.data['low'].min():,.2f} - ${response.data['high'].max():,.2f}")
            print(f"   Latest close: ${response.data['close'].iloc[-1]:,.2f}")
            
            returns = processed_data['daily_return']
            print(f"\n📈 Return Statistics:")
            print(f"   Mean daily return: {returns.mean():.4f} ({returns.mean()*252:.2f}% annualized)")
            print(f"   Daily volatility: {returns.std():.4f} ({returns.std()*np.sqrt(252):.2f}% annualized)")
            print(f"   Max single-day gain: {returns.max():.4f} ({returns.max()*100:.2f}%)")
            print(f"   Max single-day loss: {returns.min():.4f} ({returns.min()*100:.2f}%)")
            
            # Show data quality
            print(f"\n✅ Data Quality:")
            print(f"   No missing values: {not processed_data.isnull().any().any()}")
            print(f"   Features calculated: {len([col for col in processed_data.columns if col not in ['open', 'high', 'low', 'close', 'volume']])}")
            
            return True
            
        else:
            error_msg = response.metadata.get('error', 'No data returned') if hasattr(response, 'metadata') else 'Unknown error'
            print(f"❌ Failed to fetch data: {error_msg}")
            return False
            
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Starting Bitcoin data fetch from FMP...")
    success = asyncio.run(fetch_btc_data())
    
    if success:
        print("\n🎉 Bitcoin data fetch completed successfully!")
        print("💾 Data is now ready for backtesting")
        print("\n📁 Files created:")
        print("   - data/raw/BTCUSD_daily_ohlcv.csv (raw OHLCV data)")
        print("   - data/processed/BTCUSD_daily_processed.parquet (processed with features)")
    else:
        print("\n❌ Data fetch failed. Please check the errors above.")
