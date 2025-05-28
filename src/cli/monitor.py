"""
CLI module for monitoring dashboard functionality.
"""

import sys
import os
import webbrowser
import time
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_monitor(args):
    """
    Launch monitoring dashboard.
    """
    try:
        print("=" * 60)
        print("📊 CRYPTO STRATEGY LAB - MONITORING DASHBOARD")
        print("=" * 60)
        
        port = args.port
        
        print(f"🖥️  Starting monitoring dashboard:")
        print(f"   Port: {port}")
        print(f"   URL: http://localhost:{port}")
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
            print("📝 Alternative: Simple web dashboard")
            return start_simple_dashboard(port)
        
        # Try to launch Streamlit dashboard
        dashboard_script = Path("scripts/monitoring_dashboard.py")
        
        if not dashboard_script.exists():
            print(f"❌ Dashboard script not found: {dashboard_script}")
            print("📝 Creating basic dashboard...")
            return create_and_run_basic_dashboard(port)
        
        print("🚀 Launching Streamlit dashboard...")
        
        # Launch streamlit
        import subprocess
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_script),
            "--server.port", str(port),
            "--server.headless", "true"
        ]
        
        print(f"   Command: {' '.join(cmd)}")
        print()
        print("📱 Dashboard starting...")
        print("   Open your browser and navigate to the URL above")
        print("   Press Ctrl+C to stop the dashboard")
        print()
        
        # Start the process
        process = subprocess.Popen(cmd)
        
        # Wait a moment for startup
        time.sleep(3)
        
        # Try to open browser
        try:
            webbrowser.open(f"http://localhost:{port}")
            print("🌐 Browser opened automatically")
        except:
            print("⚠️  Could not open browser automatically")
        
        # Wait for process to finish
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping dashboard...")
            process.terminate()
            process.wait()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error starting monitoring dashboard: {e}")
        import traceback
        traceback.print_exc()
        return 1

def start_simple_dashboard(port: int):
    """Start a simple HTTP dashboard when Streamlit is not available."""
    try:
        import http.server
        import socketserver
        from threading import Thread
        
        print("🔧 Starting simple HTTP dashboard...")
        
        # Create a simple HTML dashboard
        html_content = create_simple_html_dashboard()
        
        # Write to temporary file
        temp_dir = Path("/tmp/crypto_dashboard")
        temp_dir.mkdir(exist_ok=True)
        
        html_file = temp_dir / "index.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        # Change to temp directory
        os.chdir(temp_dir)
        
        # Start HTTP server
        Handler = http.server.SimpleHTTPRequestHandler
        
        with socketserver.TCPServer(("", port), Handler) as httpd:
            print(f"✅ Simple dashboard running at http://localhost:{port}")
            print("   Press Ctrl+C to stop")
            
            # Try to open browser
            try:
                webbrowser.open(f"http://localhost:{port}")
            except:
                pass
            
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n🛑 Dashboard stopped")
                
        return 0
        
    except Exception as e:
        print(f"❌ Error starting simple dashboard: {e}")
        return 1

def create_simple_html_dashboard():
    """Create a simple HTML dashboard."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Crypto Strategy Lab - Monitoring Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .status-card {
            background: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .status-card h3 {
            margin: 0 0 10px 0;
            color: #34495e;
        }
        .status-value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
        .info-section {
            background: #e8f4fd;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        .warning {
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Crypto Strategy Lab - Monitoring Dashboard</h1>
        
        <div class="status-grid">
            <div class="status-card">
                <h3>System Status</h3>
                <div class="status-value">Online</div>
            </div>
            <div class="status-card">
                <h3>Active Strategies</h3>
                <div class="status-value">0</div>
            </div>
            <div class="status-card">
                <h3>Total P&L</h3>
                <div class="status-value">$0.00</div>
            </div>
            <div class="status-card">
                <h3>Open Positions</h3>
                <div class="status-value">0</div>
            </div>
        </div>
        
        <div class="info-section">
            <h3>📊 Dashboard Information</h3>
            <p>This is a simplified monitoring dashboard. For full functionality, install Streamlit:</p>
            <code>pip install streamlit</code>
            <p>Then restart the monitoring dashboard for advanced features including:</p>
            <ul>
                <li>Real-time performance charts</li>
                <li>Strategy analytics</li>
                <li>Risk metrics</li>
                <li>Trade history</li>
                <li>Interactive controls</li>
            </ul>
        </div>
        
        <div class="warning">
            <h4>⚠️ Development Version</h4>
            <p>This dashboard shows placeholder data. In production, it would display real-time trading metrics, performance analytics, and system status.</p>
        </div>
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(() => {
            location.reload();
        }, 30000);
        
        console.log('Crypto Strategy Lab Dashboard Loaded');
    </script>
</body>
</html>
    """

def create_and_run_basic_dashboard(port: int):
    """Create and run a basic Streamlit dashboard."""
    try:
        dashboard_content = create_basic_streamlit_dashboard()
        
        # Create dashboard file
        dashboard_path = Path("scripts/monitoring_dashboard.py")
        dashboard_path.parent.mkdir(exist_ok=True)
        
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_content)
        
        print(f"✅ Created basic dashboard: {dashboard_path}")
        
        # Try to run it
        import subprocess
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", str(port)
        ]
        
        subprocess.run(cmd)
        return 0
        
    except Exception as e:
        print(f"❌ Error creating basic dashboard: {e}")
        return 1

def create_basic_streamlit_dashboard():
    """Create basic Streamlit dashboard content."""
    return '''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Crypto Strategy Lab",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Crypto Strategy Lab - Monitoring Dashboard")

# Sidebar
st.sidebar.header("📊 Controls")
refresh_rate = st.sidebar.selectbox("Refresh Rate", ["30s", "1m", "5m", "15m"])
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("System Status", "Online", "✅")

with col2:
    st.metric("Active Strategies", "0", "📈")

with col3:
    st.metric("Total P&L", "$0.00", "💰")

with col4:
    st.metric("Open Positions", "0", "📊")

# Performance chart
st.subheader("📈 Portfolio Performance")

# Generate sample data
dates = pd.date_range(start=datetime.now()-timedelta(days=30), end=datetime.now(), freq='D')
returns = np.random.normal(0.001, 0.02, len(dates))
portfolio_value = 100000 * np.exp(np.cumsum(returns))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dates,
    y=portfolio_value,
    mode='lines',
    name='Portfolio Value',
    line=dict(color='#3498db', width=2)
))

fig.update_layout(
    title="Portfolio Value Over Time",
    xaxis_title="Date",
    yaxis_title="Value ($)",
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# Strategy table
st.subheader("📋 Strategy Overview")

# Sample strategy data
strategy_data = pd.DataFrame({
    'Strategy': ['Mean Reversion', 'Momentum', 'Arbitrage'],
    'Status': ['Active', 'Paused', 'Active'],
    'P&L': ['$1,234', '-$567', '$891'],
    'Win Rate': ['65%', '45%', '72%'],
    'Positions': [2, 0, 1]
})

st.dataframe(strategy_data, use_container_width=True)

# Info section
st.info("""
⚠️ **Development Dashboard**
This dashboard shows sample data. In production, it would display:
- Real-time trading metrics
- Live strategy performance
- Risk analytics
- Trade execution logs
- System health monitoring
""")

# Auto-refresh
if auto_refresh:
    import time
    time.sleep(30)
    st.experimental_rerun()
'''
