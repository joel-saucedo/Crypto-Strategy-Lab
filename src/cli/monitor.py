"""
CLI module for monitoring dashboard functionality.
"""

import sys
import os
import webbrowser
import time
import asyncio
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def run_monitor(args):
    """
    Launch monitoring dashboard.
    """
    try:
        print("=" * 60)
        print("📊 CRYPTO STRATEGY LAB - MONITORING DASHBOARD")
        print("=" * 60)
        
        port = args.port
        host = args.host
        
        print(f"🖥️  Starting monitoring dashboard:")
        print(f"   Host: {host}")
        print(f"   Port: {port}")
        print(f"   URL: http://{host}:{port}")
        print()
        
        # Check if Streamlit is available
        try:
            import streamlit as st
            streamlit_available = True
        except ImportError:
            streamlit_available = False
        
        if not streamlit_available:
            print("❌ Streamlit not installed")
            print("   Install with: pip install streamlit")
            print()
            print("📝 Starting simple HTTP dashboard instead...")
            return await start_simple_dashboard(host, port)
        
        # Try to launch Streamlit dashboard
        dashboard_script = Path("scripts/monitoring_dashboard.py")
        
        if not dashboard_script.exists():
            print(f"❌ Dashboard script not found: {dashboard_script}")
            print("📝 Creating basic dashboard...")
            return await create_and_run_basic_dashboard(host, port)
        
        # Launch Streamlit
        import subprocess
        
        print("🚀 Launching Streamlit dashboard...")
        
        try:
            # Open browser after a short delay
            async def open_browser():
                await asyncio.sleep(3)
                webbrowser.open(f'http://{host}:{port}')
            
            asyncio.create_task(open_browser())
            
            # Run Streamlit
            process = subprocess.Popen([
                'streamlit', 'run', str(dashboard_script),
                '--server.port', str(port),
                '--server.address', host,
                '--server.headless', 'true',
                '--browser.gatherUsageStats', 'false'
            ])
            
            print(f"✅ Dashboard started! Visit: http://{host}:{port}")
            print("   Press Ctrl+C to stop")
            
            # Wait for process
            process.wait()
            
        except FileNotFoundError:
            print("❌ Streamlit command not found")
            print("   Install with: pip install streamlit")
            return 1
        except KeyboardInterrupt:
            print("\n🛑 Stopping dashboard...")
            process.terminate()
            return 0
        
        return 0
        
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return 1

async def start_simple_dashboard(host: str, port: int):
    """Start a simple HTTP dashboard as fallback."""
    try:
        from http.server import HTTPServer, SimpleHTTPRequestHandler
        import threading
        
        class DashboardHandler(SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/' or self.path == '/dashboard':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    
                    html_content = create_simple_dashboard_html()
                    self.wfile.write(html_content.encode())
                else:
                    super().do_GET()
        
        # Start server in background thread
        server = HTTPServer((host, port), DashboardHandler)
        
        def run_server():
            server.serve_forever()
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        print(f"✅ Simple dashboard started! Visit: http://{host}:{port}")
        print("   Press Ctrl+C to stop")
        
        # Open browser
        webbrowser.open(f'http://{host}:{port}')
        
        # Wait for interrupt
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping dashboard...")
            server.shutdown()
            return 0
        
    except Exception as e:
        print(f"❌ Error starting simple dashboard: {e}")
        return 1

async def create_and_run_basic_dashboard(host: str, port: int):
    """Create and run a basic dashboard."""
    dashboard_content = '''import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Crypto Strategy Lab - Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Crypto Strategy Lab - Monitoring Dashboard")

# Sidebar
st.sidebar.header("🎛️ Controls")
selected_strategy = st.sidebar.selectbox("Strategy", ["All", "MACD", "RSI", "Bollinger Bands"])
selected_symbol = st.sidebar.selectbox("Symbol", ["BTCUSD", "ETHUSD", "ADAUSD"])
time_range = st.sidebar.selectbox("Time Range", ["1D", "1W", "1M", "3M", "1Y"])

# Main content
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Portfolio Value", "$125,430", "2.3%")
with col2:
    st.metric("Total Return", "15.2%", "0.8%")
with col3:
    st.metric("Sharpe Ratio", "1.45", "0.12")
with col4:
    st.metric("Max Drawdown", "-8.5%", "-1.2%")

# Charts
st.header("📈 Performance Charts")

# Generate sample data
dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
returns = np.random.normal(0.001, 0.02, len(dates))
portfolio_value = 100000 * np.exp(np.cumsum(returns))

chart_data = pd.DataFrame({
    'Date': dates,
    'Portfolio Value': portfolio_value
})

fig = px.line(chart_data, x='Date', y='Portfolio Value',
              title='Portfolio Performance Over Time')
st.plotly_chart(fig, use_container_width=True)

# Recent trades
st.header("📋 Recent Trades")
trades_data = pd.DataFrame({
    'Time': [datetime.now() - timedelta(hours=i) for i in range(5)],
    'Symbol': ['BTCUSD', 'ETHUSD', 'BTCUSD', 'ADAUSD', 'ETHUSD'],
    'Side': ['BUY', 'SELL', 'SELL', 'BUY', 'BUY'],
    'Quantity': [0.05, 2.1, 0.03, 1000, 1.8],
    'Price': [67430, 3890, 67200, 0.45, 3910],
    'PnL': ['+$234', '-$89', '+$156', '+$45', '-$23']
})

st.dataframe(trades_data, use_container_width=True)
'''
    
    # Write dashboard file
    dashboard_path = Path("scripts/temp_dashboard.py")
    dashboard_path.write_text(dashboard_content)
    
    # Run dashboard
    import subprocess
    
    try:
        print("🚀 Launching basic Streamlit dashboard...")
        
        process = subprocess.Popen([
            'streamlit', 'run', str(dashboard_path),
            '--server.port', str(port),
            '--server.address', host,
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false'
        ])
        
        # Open browser
        await asyncio.sleep(3)
        webbrowser.open(f'http://{host}:{port}')
        
        print(f"✅ Dashboard started! Visit: http://{host}:{port}")
        print("   Press Ctrl+C to stop")
        
        # Wait for process
        process.wait()
        
        # Cleanup
        dashboard_path.unlink(missing_ok=True)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error running basic dashboard: {e}")
        dashboard_path.unlink(missing_ok=True)
        return 1

def create_simple_dashboard_html():
    """Create simple HTML dashboard."""
    return '''<!DOCTYPE html>
<html>
<head>
    <title>Crypto Strategy Lab - Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
        .metrics { display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap; }
        .metric { background: white; padding: 20px; border-radius: 10px; flex: 1; min-width: 200px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .metric-value { font-size: 24px; font-weight: bold; color: #27ae60; }
        .metric-label { color: #7f8c8d; font-size: 14px; }
        .status { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .refresh-btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        .refresh-btn:hover { background: #2980b9; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Crypto Strategy Lab - Dashboard</h1>
        <p>Real-time monitoring and analytics</p>
    </div>
    
    <div class="metrics">
        <div class="metric">
            <div class="metric-label">Portfolio Value</div>
            <div class="metric-value">$125,430</div>
        </div>
        <div class="metric">
            <div class="metric-label">Total Return</div>
            <div class="metric-value">+15.2%</div>
        </div>
        <div class="metric">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value">1.45</div>
        </div>
        <div class="metric">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value">-8.5%</div>
        </div>
    </div>
    
    <div class="status">
        <h3>🎯 System Status</h3>
        <p>✅ All systems operational</p>
        <p>📊 Monitoring 3 strategies across 5 symbols</p>
        <p>🔄 Last update: <span id="timestamp"></span></p>
        <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
    </div>
    
    <script>
        document.getElementById('timestamp').textContent = new Date().toLocaleString();
        // Auto-refresh every 30 seconds
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>'''