#!/usr/bin/env python
"""
Monitoring dashboard for Crypto-Strategy-Lab.
Run with: python -m streamlit run scripts/monitoring_dashboard.py
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Crypto-Strategy-Lab Monitor",
    page_icon="📈",
    layout="wide"
)

# Define paths
DATA_DIR = Path("./data")
MONITORING_DIR = DATA_DIR / "monitoring"
PAPER_TRADE_DIR = DATA_DIR / "paper_trade"
LIVE_DIR = DATA_DIR / "live"
DEPLOYMENT_HISTORY = DATA_DIR / "deployment_history.csv"

def load_deployment_history():
    """Load deployment history if available."""
    if DEPLOYMENT_HISTORY.exists():
        return pd.read_csv(DEPLOYMENT_HISTORY)
    return pd.DataFrame(columns=["timestamp", "strategy", "exchange", "capital", "git_commit", "dsr"])

def load_strategy_configs():
    """Load all active strategy monitoring configs."""
    configs = {}
    if MONITORING_DIR.exists():
        for config_file in MONITORING_DIR.glob("*_monitoring_config.yaml"):
            strategy_name = config_file.stem.replace("_monitoring_config", "")
            with open(config_file, "r") as f:
                try:
                    configs[strategy_name] = yaml.safe_load(f)
                except Exception as e:
                    st.error(f"Error loading config for {strategy_name}: {e}")
    return configs

def load_paper_trading_results(strategy_name):
    """Load paper trading results for a given strategy."""
    results = []
    if PAPER_TRADE_DIR.exists():
        for file in PAPER_TRADE_DIR.glob(f"{strategy_name}_paper_trade_*.csv"):
            try:
                df = pd.read_csv(file)
                df["Source"] = file.stem
                results.append(df)
            except Exception as e:
                st.warning(f"Error loading {file}: {e}")
    
    if results:
        return pd.concat(results)
    return pd.DataFrame()

def load_live_results(strategy_name):
    """Load live trading results for a given strategy."""
    results = []
    if LIVE_DIR.exists():
        for file in LIVE_DIR.glob(f"{strategy_name}_*.csv"):
            try:
                df = pd.read_csv(file)
                df["Source"] = file.stem
                results.append(df)
            except Exception as e:
                st.warning(f"Error loading {file}: {e}")
    
    if results:
        return pd.concat(results)
    return pd.DataFrame()

def calculate_metrics(df):
    """Calculate performance metrics from trading data."""
    if df.empty:
        return {}
    
    # Ensure proper columns exist
    required_cols = ["Date", "Portfolio"]
    if not all(col in df.columns for col in required_cols):
        return {}
    
    # Convert date if needed
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])
    
    # Sort by date
    df = df.sort_values("Date")
    
    # Calculate returns
    df["Returns"] = df["Portfolio"].pct_change()
    
    # Calculate metrics
    start_value = df["Portfolio"].iloc[0]
    end_value = df["Portfolio"].iloc[-1]
    total_return = (end_value / start_value) - 1
    
    # Calculate max drawdown
    df["Peak"] = df["Portfolio"].cummax()
    df["Drawdown"] = (df["Portfolio"] / df["Peak"]) - 1
    max_drawdown = df["Drawdown"].min()
    
    # Annualized metrics
    days = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days
    if days > 0:
        annualized_return = ((1 + total_return) ** (365 / days)) - 1
    else:
        annualized_return = 0
    
    daily_returns = df["Returns"].dropna()
    if len(daily_returns) > 1:
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe = 0 if volatility == 0 else annualized_return / volatility
    else:
        volatility = 0
        sharpe = 0
    
    return {
        "start_date": df["Date"].iloc[0],
        "end_date": df["Date"].iloc[-1],
        "days": days,
        "start_value": start_value,
        "end_value": end_value,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": len(daily_returns[daily_returns > 0]) / len(daily_returns) if len(daily_returns) > 0 else 0
    }

def plot_performance(df, title):
    """Create performance chart."""
    if df.empty:
        return None
    
    # Ensure proper columns exist
    required_cols = ["Date", "Portfolio"]
    if not all(col in df.columns for col in required_cols):
        return None
    
    # Convert date if needed
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])
    
    # Sort by date
    df = df.sort_values("Date")
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add portfolio value
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Portfolio"],
            name="Portfolio Value",
            line=dict(color="blue")
        ),
        secondary_y=False,
    )
    
    # Add price if available
    if "Price" in df.columns:
        # Normalize price to start at same point as portfolio for comparison
        price_normalized = df["Price"] * (df["Portfolio"].iloc[0] / df["Price"].iloc[0])
        
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=price_normalized,
                name="Underlying (Normalized)",
                line=dict(color="gray", dash="dot")
            ),
            secondary_y=False,
        )
    
    # Add signals if available
    if "Signal" in df.columns:
        # Create separate traces for different signal types
        for signal_value in [-1, 0, 1]:
            mask = df["Signal"] == signal_value
            if mask.any():
                signal_name = "Short" if signal_value == -1 else ("Flat" if signal_value == 0 else "Long")
                color = "red" if signal_value == -1 else ("gray" if signal_value == 0 else "green")
                
                fig.add_trace(
                    go.Scatter(
                        x=df.loc[mask, "Date"],
                        y=df.loc[mask, "Portfolio"],
                        mode="markers",
                        marker=dict(color=color, size=8, symbol="circle"),
                        name=f"{signal_name} Signal",
                        hoverinfo="text",
                        hovertext=[f"Date: {date}<br>Signal: {signal_name}<br>Portfolio: ${value:.2f}"
                                  for date, value in zip(df.loc[mask, "Date"].dt.strftime("%Y-%m-%d"),
                                                        df.loc[mask, "Portfolio"])]
                    ),
                    secondary_y=False,
                )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        template="plotly_white"
    )
    
    return fig

def render_dashboard():
    """Render the main dashboard."""
    st.title("🚀 Crypto-Strategy-Lab Monitoring Dashboard")
    
    # Load data
    deployment_history = load_deployment_history()
    configs = load_strategy_configs()
    
    # Dashboard overview
    st.markdown("## 📊 System Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        active_strategies = len(configs)
        st.metric("Active Strategies", active_strategies)
    
    with col2:
        if not deployment_history.empty:
            last_deployment = pd.to_datetime(deployment_history["timestamp"]).max()
            last_deploy_days = (datetime.now() - last_deployment).days
            st.metric("Last Deployment", f"{last_deploy_days} days ago")
        else:
            st.metric("Last Deployment", "None")
    
    with col3:
        if not deployment_history.empty and "dsr" in deployment_history.columns:
            avg_dsr = deployment_history["dsr"].mean()
            st.metric("Average DSR", f"{avg_dsr:.2f}")
        else:
            st.metric("Average DSR", "N/A")
    
    # Deployment history
    if not deployment_history.empty:
        st.markdown("### Recent Deployments")
        st.dataframe(deployment_history.sort_values("timestamp", ascending=False).head(10))
    
    # Strategy selector
    strategy_list = list(configs.keys())
    
    if not strategy_list:
        st.warning("No active strategies found. Deploy a strategy first.")
        return
    
    selected_strategy = st.selectbox("Select Strategy", strategy_list)
    
    if selected_strategy:
        st.markdown(f"## Strategy: {selected_strategy}")
        
        # Strategy details
        if selected_strategy in configs:
            config = configs[selected_strategy]
            
            st.markdown("### Configuration")
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                st.write(f"**Exchange:** {config.get('exchange', 'N/A')}")
                st.write(f"**Ticker:** {config.get('ticker', 'N/A')}")
                st.write(f"**Deployment Time:** {config.get('deployment_time', 'N/A')}")
            
            with config_col2:
                if "alert_thresholds" in config:
                    thresholds = config["alert_thresholds"]
                    st.write(f"**Max Drawdown Alert:** {thresholds.get('drawdown_percent', 'N/A')}%")
                    st.write(f"**Inactivity Alert:** {thresholds.get('inactivity_hours', 'N/A')} hours")
        
        # Paper trading results
        st.markdown("### Paper Trading Results")
        paper_df = load_paper_trading_results(selected_strategy)
        
        if not paper_df.empty:
            paper_metrics = calculate_metrics(paper_df)
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Total Return", f"{paper_metrics.get('total_return', 0):.2%}")
            
            with metric_col2:
                st.metric("Sharpe Ratio", f"{paper_metrics.get('sharpe_ratio', 0):.2f}")
            
            with metric_col3:
                st.metric("Max Drawdown", f"{paper_metrics.get('max_drawdown', 0):.2%}")
            
            with metric_col4:
                st.metric("Win Rate", f"{paper_metrics.get('win_rate', 0):.2%}")
            
            paper_chart = plot_performance(paper_df, "Paper Trading Performance")
            if paper_chart:
                st.plotly_chart(paper_chart, use_container_width=True)
        else:
            st.info("No paper trading data available for this strategy")
        
        # Live trading results
        st.markdown("### Live Trading Results")
        live_df = load_live_results(selected_strategy)
        
        if not live_df.empty:
            live_metrics = calculate_metrics(live_df)
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Total Return", f"{live_metrics.get('total_return', 0):.2%}")
            
            with metric_col2:
                st.metric("Sharpe Ratio", f"{live_metrics.get('sharpe_ratio', 0):.2f}")
            
            with metric_col3:
                st.metric("Max Drawdown", f"{live_metrics.get('max_drawdown', 0):.2%}")
            
            with metric_col4:
                st.metric("Win Rate", f"{live_metrics.get('win_rate', 0):.2%}")
            
            live_chart = plot_performance(live_df, "Live Trading Performance")
            if live_chart:
                st.plotly_chart(live_chart, use_container_width=True)
        else:
            st.info("No live trading data available for this strategy")
        
        # Monitoring logs
        st.markdown("### Monitoring Logs")
        log_file = MONITORING_DIR / f"{selected_strategy}_monitor.log"
        
        if log_file.exists():
            with open(log_file, "r") as f:
                logs = f.readlines()
            
            # Display last 20 log entries
            st.code("".join(logs[-20:]))
        else:
            st.info("No monitoring logs available for this strategy")

def main():
    """Main function."""
    render_dashboard()

if __name__ == "__main__":
    main()
