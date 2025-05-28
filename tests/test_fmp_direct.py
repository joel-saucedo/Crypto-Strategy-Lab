"""
Direct FMP API test without async to debug the hanging issue.
"""

import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def test_fmp_direct():
    """Test FMP API directly with requests."""
    
    api_key = os.getenv('FMP_API_KEY')
    print(f"🔑 API Key: {api_key[:10]}..." if api_key else "❌ No API key")
    
    if not api_key:
        return
    
    # Test BTCUSD endpoint
    symbol = 'BTCUSD'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
    params = {
        'apikey': api_key,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d')
    }
    
    print(f"🌐 Testing URL: {url}")
    print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ JSON received")
            
            if 'historical' in data:
                historical = data['historical']
                print(f"📊 Got {len(historical)} data points")
                
                if historical:
                    print("Sample data point:")
                    print(historical[0])
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(historical)
                    print(f"DataFrame shape: {df.shape}")
                    print(f"Columns: {list(df.columns)}")
                    
                    return df
            else:
                print(f"❌ No 'historical' key in response")
                print(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fmp_direct()
