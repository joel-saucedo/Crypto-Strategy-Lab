"""
Advanced visualization and dashboard components for comprehensive strategy analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import io
import base64

try:
    from ..utils.consolidation_utils import calculate_comprehensive_metrics, analyze_drawdowns
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.consolidation_utils import calculate_comprehensive_metrics, analyze_drawdowns


class AdvancedVisualization:
    """
    Advanced visualization system for comprehensive strategy analysis.
    """
    
    def __init__(self, theme: str = "plotly_white"):
        self.theme = theme
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
    
    def create_strategy_dashboard(self, 
                                returns: pd.Series,
                                prices: Optional[pd.Series] = None,
                                signals: Optional[pd.Series] = None,
                                benchmark_returns: Optional[pd.Series] = None,
                                strategy_name: str = "Strategy") -> go.Figure:
        """
        Create comprehensive strategy dashboard.
        
        Args:
            returns: Strategy returns
            prices: Asset prices
            signals: Trading signals
            benchmark_returns: Benchmark returns for comparison
            strategy_name: Name of the strategy
            
        Returns:
            Plotly figure with comprehensive dashboard
        """
        # Calculate metrics
        metrics = calculate_comprehensive_metrics(returns, benchmark_returns)
        drawdown_analysis = analyze_drawdowns(returns)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Cumulative Performance',
                'Drawdown Analysis', 
                'Rolling Sharpe Ratio',
                'Return Distribution',
                'Monthly Returns Heatmap',
                'Risk-Return Scatter'
            ],
            specs=[
                [{"secondary_y": True}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
                [{"colspan": 2}, None]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. Cumulative Performance
        cumulative_returns = (1 + returns).cumprod()
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                name=strategy_name,
                line=dict(color=self.colors['primary'], width=2)
            ),
            row=1, col=1
        )
        
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=benchmark_cumulative.index,
                    y=benchmark_cumulative.values,
                    name='Benchmark',
                    line=dict(color=self.colors['secondary'], width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        # Add signals if provided
        if signals is not None and prices is not None:
            buy_signals = signals[signals == 1]
            sell_signals = signals[signals == -1]
            
            if len(buy_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=prices.loc[buy_signals.index],
                        mode='markers',
                        marker=dict(color='green', size=8, symbol='triangle-up'),
                        name='Buy Signals'
                    ),
                    row=1, col=1, secondary_y=True
                )
            
            if len(sell_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=prices.loc[sell_signals.index],
                        mode='markers',
                        marker=dict(color='red', size=8, symbol='triangle-down'),
                        name='Sell Signals'
                    ),
                    row=1, col=1, secondary_y=True
                )
        
        # 2. Drawdown Analysis
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        fig.add_trace(
            go.Scatter(
                x=drawdowns.index,
                y=drawdowns.values,
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.3)',
                line=dict(color='red'),
                name='Drawdown'
            ),
            row=1, col=2
        )
        
        # 3. Rolling Sharpe Ratio
        rolling_sharpe = self._calculate_rolling_sharpe(returns, window=252)
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                line=dict(color=self.colors['info']),
                name='Rolling Sharpe'
            ),
            row=2, col=1
        )
        
        # Add reference lines
        fig.add_hline(y=1.0, line=dict(color='green', dash='dash'), row=2, col=1)
        fig.add_hline(y=0.5, line=dict(color='orange', dash='dash'), row=2, col=1)
        fig.add_hline(y=0.0, line=dict(color='red', dash='dash'), row=2, col=1)
        
        # 4. Return Distribution
        fig.add_trace(
            go.Histogram(
                x=returns.values,
                nbinsx=50,
                name='Return Distribution',
                marker_color=self.colors['primary'],
                opacity=0.7
            ),
            row=2, col=2
        )
        
        # 5. Monthly Returns Heatmap
        monthly_returns = self._create_monthly_returns_table(returns)
        if not monthly_returns.empty:
            fig.add_trace(
                go.Heatmap(
                    z=monthly_returns.values,
                    x=monthly_returns.columns,
                    y=monthly_returns.index,
                    colorscale='RdYlGn',
                    zmid=0,
                    name='Monthly Returns'
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"{strategy_name} - Comprehensive Dashboard",
            height=900,
            showlegend=True,
            template=self.theme
        )
        
        # Add metrics annotation
        metrics_text = self._format_metrics_text(metrics, drawdown_analysis)
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=metrics_text,
            showarrow=False,
            font=dict(size=10, family="monospace"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        return fig
    
    def create_portfolio_comparison(self, 
                                  strategy_returns: Dict[str, pd.Series],
                                  benchmark_returns: Optional[pd.Series] = None) -> go.Figure:
        """
        Create portfolio comparison visualization.
        
        Args:
            strategy_returns: Dictionary of strategy returns
            benchmark_returns: Optional benchmark returns
            
        Returns:
            Plotly figure with portfolio comparison
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Cumulative Performance Comparison',
                'Risk-Return Scatter',
                'Correlation Heatmap',
                'Performance Metrics Comparison'
            ]
        )
        
        # 1. Cumulative Performance
        for name, returns in strategy_returns.items():
            cumulative = (1 + returns).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=cumulative.index,
                    y=cumulative.values,
                    name=name,
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=benchmark_cumulative.index,
                    y=benchmark_cumulative.values,
                    name='Benchmark',
                    line=dict(dash='dash', width=2, color='black')
                ),
                row=1, col=1
            )
        
        # 2. Risk-Return Scatter
        risk_return_data = []
        for name, returns in strategy_returns.items():
            metrics = calculate_comprehensive_metrics(returns, benchmark_returns)
            risk_return_data.append({
                'name': name,
                'return': metrics['annualized_return'],
                'risk': metrics['volatility'],
                'sharpe': metrics['sharpe_ratio']
            })
        
        scatter_df = pd.DataFrame(risk_return_data)
        fig.add_trace(
            go.Scatter(
                x=scatter_df['risk'],
                y=scatter_df['return'],
                mode='markers+text',
                text=scatter_df['name'],
                textposition="top center",
                marker=dict(
                    size=scatter_df['sharpe'] * 10,
                    color=scatter_df['sharpe'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio")
                ),
                name='Strategies'
            ),
            row=1, col=2
        )
        
        # 3. Correlation Heatmap
        if len(strategy_returns) > 1:
            returns_df = pd.DataFrame(strategy_returns)
            correlation_matrix = returns_df.corr()
            
            fig.add_trace(
                go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                    colorscale='RdBu',
                    zmid=0,
                    name='Correlation'
                ),
                row=2, col=1
            )
        
        # 4. Performance Metrics Bar Chart
        metrics_comparison = []
        for name, returns in strategy_returns.items():
            metrics = calculate_comprehensive_metrics(returns, benchmark_returns)
            metrics_comparison.append({
                'Strategy': name,
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Calmar Ratio': metrics['calmar_ratio'],
                'Max Drawdown': metrics['max_drawdown'],
                'Win Rate': metrics['win_rate']
            })
        
        metrics_df = pd.DataFrame(metrics_comparison)
        for metric in ['Sharpe Ratio', 'Calmar Ratio']:
            fig.add_trace(
                go.Bar(
                    x=metrics_df['Strategy'],
                    y=metrics_df[metric],
                    name=metric
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Portfolio Strategy Comparison",
            height=800,
            showlegend=True,
            template=self.theme
        )
        
        return fig
    
    def create_risk_analysis_dashboard(self, returns: pd.Series) -> go.Figure:
        """
        Create comprehensive risk analysis dashboard.
        
        Args:
            returns: Strategy returns
            
        Returns:
            Plotly figure with risk analysis
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Rolling Volatility',
                'VaR and CVaR',
                'Tail Risk Analysis',
                'Underwater Curve',
                'Return Autocorrelation',
                'Volatility Clustering'
            ]
        )
        
        # 1. Rolling Volatility
        rolling_vol = returns.rolling(252).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                name='Rolling Volatility (1Y)',
                line=dict(color=self.colors['warning'])
            ),
            row=1, col=1
        )
        
        # 2. VaR and CVaR
        var_95 = returns.rolling(252).quantile(0.05)
        cvar_95 = returns.rolling(252).apply(lambda x: x[x <= x.quantile(0.05)].mean())
        
        fig.add_trace(
            go.Scatter(
                x=var_95.index,
                y=var_95.values,
                name='VaR 95%',
                line=dict(color=self.colors['danger'])
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=cvar_95.index,
                y=cvar_95.values,
                name='CVaR 95%',
                line=dict(color=self.colors['danger'], dash='dash')
            ),
            row=1, col=2
        )
        
        # 3. Tail Risk (Q-Q plot approximation)
        sorted_returns = np.sort(returns.values)
        theoretical_quantiles = np.linspace(0.01, 0.99, len(sorted_returns))
        normal_quantiles = np.percentile(np.random.normal(0, returns.std(), 10000), 
                                       theoretical_quantiles * 100)
        
        fig.add_trace(
            go.Scatter(
                x=normal_quantiles,
                y=sorted_returns,
                mode='markers',
                name='Actual vs Normal',
                marker=dict(size=3, color=self.colors['info'])
            ),
            row=1, col=3
        )
        
        # Add 45-degree reference line
        min_val = min(min(normal_quantiles), min(sorted_returns))
        max_val = max(max(normal_quantiles), max(sorted_returns))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Normal Distribution',
                line=dict(dash='dash', color='gray')
            ),
            row=1, col=3
        )
        
        # 4. Underwater Curve (Drawdown)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        underwater = (cumulative - running_max) / running_max
        
        fig.add_trace(
            go.Scatter(
                x=underwater.index,
                y=underwater.values,
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.3)',
                line=dict(color='red'),
                name='Underwater Curve'
            ),
            row=2, col=1
        )
        
        # 5. Return Autocorrelation
        lags = range(1, 21)
        autocorr = [returns.autocorr(lag=lag) for lag in lags]
        
        fig.add_trace(
            go.Bar(
                x=list(lags),
                y=autocorr,
                name='Autocorrelation',
                marker_color=self.colors['primary']
            ),
            row=2, col=2
        )
        
        # 6. Volatility Clustering (Absolute returns autocorrelation)
        abs_returns = returns.abs()
        vol_autocorr = [abs_returns.autocorr(lag=lag) for lag in lags]
        
        fig.add_trace(
            go.Bar(
                x=list(lags),
                y=vol_autocorr,
                name='Volatility Clustering',
                marker_color=self.colors['warning']
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title="Risk Analysis Dashboard",
            height=800,
            showlegend=True,
            template=self.theme
        )
        
        return fig
    
    def create_live_monitoring_dashboard(self, 
                                       live_data: Dict[str, pd.Series],
                                       target_metrics: Dict[str, float]) -> go.Figure:
        """
        Create live monitoring dashboard for real-time strategy tracking.
        
        Args:
            live_data: Dictionary containing live performance data
            target_metrics: Target performance metrics for comparison
            
        Returns:
            Plotly figure with live monitoring dashboard
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Live Performance',
                'Real-time Metrics',
                'Alert Status',
                'Recent Trades',
                'Risk Monitoring',
                'System Health'
            ],
            specs=[
                [{"type": "xy"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"type": "xy"}, {"type": "xy"}, {"type": "indicator"}]
            ]
        )
        
        # Get returns data
        returns = live_data.get('returns', pd.Series())
        
        if len(returns) > 0:
            # 1. Live Performance
            cumulative = (1 + returns).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=cumulative.index,
                    y=cumulative.values,
                    name='Live Performance',
                    line=dict(color=self.colors['primary'], width=3)
                ),
                row=1, col=1
            )
            
            # Calculate current metrics
            current_metrics = calculate_comprehensive_metrics(returns)
            
            # 2. Real-time Metrics (Gauge)
            current_sharpe = current_metrics.get('sharpe_ratio', 0)
            target_sharpe = target_metrics.get('sharpe_ratio', 1.0)
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=current_sharpe,
                    delta={'reference': target_sharpe},
                    title={'text': "Sharpe Ratio"},
                    gauge={
                        'axis': {'range': [-2, 3]},
                        'bar': {'color': self.colors['primary']},
                        'steps': [
                            {'range': [-2, 0], 'color': "lightgray"},
                            {'range': [0, 1], 'color': "yellow"},
                            {'range': [1, 3], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': target_sharpe
                        }
                    }
                ),
                row=1, col=2
            )
            
            # 3. Alert Status
            max_dd = current_metrics.get('max_drawdown', 0)
            target_max_dd = target_metrics.get('max_drawdown', 0.2)
            
            alert_status = "NORMAL"
            alert_color = "green"
            if max_dd > target_max_dd:
                alert_status = "ALERT"
                alert_color = "red"
            elif max_dd > target_max_dd * 0.8:
                alert_status = "WARNING"
                alert_color = "orange"
            
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=max_dd,
                    title={'text': f"Max Drawdown<br><span style='color:{alert_color}'>{alert_status}</span>"},
                    number={'suffix': "%", 'font': {'color': alert_color}}
                ),
                row=1, col=3
            )
            
            # 4. Recent Trades
            if 'trades' in live_data:
                recent_trades = live_data['trades'].tail(10)
                fig.add_trace(
                    go.Scatter(
                        x=recent_trades.index,
                        y=recent_trades.values,
                        mode='markers+lines',
                        name='Recent P&L',
                        marker=dict(
                            color=np.where(recent_trades > 0, 'green', 'red'),
                            size=8
                        )
                    ),
                    row=2, col=1
                )
            
            # 5. Risk Monitoring
            if len(returns) >= 20:
                rolling_vol = returns.rolling(20).std() * np.sqrt(252)
                fig.add_trace(
                    go.Scatter(
                        x=rolling_vol.index,
                        y=rolling_vol.values,
                        name='Rolling Volatility',
                        line=dict(color=self.colors['warning'])
                    ),
                    row=2, col=2
                )
                
                # Add volatility target line
                target_vol = target_metrics.get('volatility', 0.15)
                fig.add_hline(
                    y=target_vol, 
                    line=dict(color='red', dash='dash'),
                    row=2, col=2
                )
            
            # 6. System Health
            system_health = live_data.get('system_health', 100)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=system_health,
                    title={'text': "System Health"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': self.colors['success']},
                        'steps': [
                            {'range': [0, 50], 'color': "red"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ]
                    }
                ),
                row=2, col=3
            )
        
        fig.update_layout(
            title="Live Strategy Monitoring Dashboard",
            height=800,
            showlegend=True,
            template=self.theme
        )
        
        return fig
    
    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        return (returns.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(252)
    
    def _create_monthly_returns_table(self, returns: pd.Series) -> pd.DataFrame:
        """Create monthly returns table for heatmap."""
        try:
            monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly.index = monthly.index.to_period('M')
            
            # Create pivot table
            monthly_table = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack()
            monthly_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            return monthly_table
        except:
            return pd.DataFrame()
    
    def _format_metrics_text(self, metrics: Dict[str, float], 
                           drawdown_analysis: Dict[str, Any]) -> str:
        """Format metrics for annotation."""
        text = f"""
<b>Key Metrics:</b>
Total Return: {metrics.get('total_return', 0):.1%}
Annualized Return: {metrics.get('annualized_return', 0):.1%}
Volatility: {metrics.get('volatility', 0):.1%}
Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}
Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}
Max Drawdown: {metrics.get('max_drawdown', 0):.1%}
Win Rate: {metrics.get('win_rate', 0):.1%}
VaR 95%: {metrics.get('var_95', 0):.2%}
CVaR 95%: {metrics.get('cvar_95', 0):.2%}
        """.strip()
        return text


# Export the main class
__all__ = ['AdvancedVisualization']
