"""
Visualization and analysis tools for backtest results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Import consolidated utilities for consistent calculations
try:
    from ..utils.consolidation_utils import (
        calculate_max_drawdown_optimized
    )
except ImportError:
    try:
        from utils.consolidation_utils import (
            calculate_max_drawdown_optimized
        )
    except ImportError:
        # Fallback for direct execution
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.consolidation_utils import (
            calculate_max_drawdown_optimized
        )

try:
    from ..core.backtest_engine import BacktestResult
except ImportError:
    # Handle direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.backtest_engine import BacktestResult

class BacktestAnalyzer:
    """Analyzer for backtest results with visualization capabilities."""
    
    def __init__(self, result: BacktestResult):
        self.result = result
        self.portfolio_history = result.portfolio_history
        self.trades = result.trades
        self.metrics = result.metrics
        
    def plot_portfolio_performance(self, include_benchmark: bool = True, 
                                 show_drawdown: bool = True) -> go.Figure:
        """
        Plot portfolio performance over time.
        
        Args:
            include_benchmark: Whether to include benchmark comparison
            show_drawdown: Whether to show drawdown subplot
            
        Returns:
            Plotly figure
        """
        # Create subplots
        subplot_titles = ['Portfolio Value']
        if show_drawdown:
            subplot_titles.append('Drawdown')
        
        fig = make_subplots(
            rows=2 if show_drawdown else 1,
            cols=1,
            subplot_titles=subplot_titles,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3] if show_drawdown else [1.0]
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=self.portfolio_history['timestamp'],
                y=self.portfolio_history['portfolio_value'],
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Benchmark comparison
        if include_benchmark and self.result.benchmark_data is not None:
            # Normalize benchmark to start at same value as portfolio
            benchmark_normalized = self.result.benchmark_data.copy()
            start_value = self.portfolio_history['portfolio_value'].iloc[0]
            benchmark_start = benchmark_normalized.iloc[0]
            benchmark_normalized = (benchmark_normalized / benchmark_start) * start_value
            
            fig.add_trace(
                go.Scatter(
                    x=benchmark_normalized.index,
                    y=benchmark_normalized.values,
                    name='Benchmark',
                    line=dict(color='orange', width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        # Drawdown
        if show_drawdown:
            portfolio_values = self.portfolio_history['portfolio_value'].values
            # Use consolidated utility for consistent drawdown calculation
            running_max = np.maximum.accumulate(portfolio_values)
            drawdown_values = (portfolio_values - running_max) / running_max * 100
            
            fig.add_trace(
                go.Scatter(
                    x=self.portfolio_history['timestamp'],
                    y=drawdown_values,
                    name='Drawdown %',
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.3)',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Portfolio Performance Analysis',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            height=600 if show_drawdown else 400
        )
        
        if show_drawdown:
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        return fig
    
    def plot_strategy_comparison(self) -> go.Figure:
        """Plot performance comparison between strategies."""
        strategy_stats = self.result.get_strategy_breakdown()
        
        if not strategy_stats:
            return go.Figure().add_annotation(text="No strategy data available")
        
        strategies = list(strategy_stats.keys())
        metrics = ['total_pnl', 'win_rate', 'total_trades']
        
        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=[m.replace('_', ' ').title() for m in metrics],
            horizontal_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set1[:len(strategies)]
        
        for i, metric in enumerate(metrics):
            values = [strategy_stats[s][metric] for s in strategies]
            
            fig.add_trace(
                go.Bar(
                    x=strategies,
                    y=values,
                    name=metric.replace('_', ' ').title(),
                    marker_color=colors,
                    showlegend=False
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title='Strategy Performance Comparison',
            height=400
        )
        
        return fig
    
    def plot_monthly_returns(self) -> go.Figure:
        """Plot monthly returns heatmap."""
        if self.portfolio_history.empty:
            return go.Figure().add_annotation(text="No portfolio history available")
        
        # Calculate monthly returns
        portfolio_history = self.portfolio_history.copy()
        portfolio_history['month'] = portfolio_history['timestamp'].dt.to_period('M')
        
        monthly_values = portfolio_history.groupby('month')['portfolio_value'].last()
        monthly_returns = monthly_values.pct_change() * 100
        
        # Create year-month matrix
        monthly_returns.index = pd.to_datetime(monthly_returns.index.astype(str))
        monthly_returns = monthly_returns.dropna()
        
        if len(monthly_returns) == 0:
            return go.Figure().add_annotation(text="Insufficient data for monthly returns")
        
        # Create pivot table for heatmap
        df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'returns': monthly_returns.values
        })
        
        pivot = df.pivot(index='year', columns='month', values='returns')
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            y=pivot.index,
            colorscale='RdYlGn',
            text=np.round(pivot.values, 2),
            texttemplate='%{text}%',
            textfont={"size": 10},
            colorbar=dict(title="Returns (%)")
        ))
        
        fig.update_layout(
            title='Monthly Returns Heatmap',
            xaxis_title='Month',
            yaxis_title='Year',
            height=400
        )
        
        return fig
    
    def plot_trade_analysis(self) -> go.Figure:
        """Plot trade analysis including PnL distribution and timing."""
        if not self.trades:
            return go.Figure().add_annotation(text="No trades available")
        
        trades_df = pd.DataFrame([{
            'pnl': trade.pnl,
            'pnl_pct': trade.pnl_pct,
            'symbol': trade.symbol,
            'strategy_id': trade.strategy_id,
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'duration_hours': (trade.exit_time - trade.entry_time).total_seconds() / 3600
        } for trade in self.trades])
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['PnL Distribution', 'PnL by Strategy', 
                          'Trade Duration', 'Cumulative PnL'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # PnL Distribution
        fig.add_trace(
            go.Histogram(
                x=trades_df['pnl'],
                nbinsx=30,
                name='PnL Distribution',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # PnL by Strategy
        strategy_pnl = trades_df.groupby('strategy_id')['pnl'].sum()
        fig.add_trace(
            go.Bar(
                x=strategy_pnl.index,
                y=strategy_pnl.values,
                name='Strategy PnL',
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
        
        # Trade Duration
        fig.add_trace(
            go.Histogram(
                x=trades_df['duration_hours'],
                nbinsx=20,
                name='Duration (hours)',
                marker_color='lightyellow'
            ),
            row=2, col=1
        )
        
        # Cumulative PnL
        trades_df = trades_df.sort_values('exit_time')
        cumulative_pnl = trades_df['pnl'].cumsum()
        fig.add_trace(
            go.Scatter(
                x=trades_df['exit_time'],
                y=cumulative_pnl,
                name='Cumulative PnL',
                line=dict(color='purple')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Trade Analysis',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive backtest report."""
        
        # Basic metrics
        report = {
            'summary': self.metrics,
            'strategy_breakdown': self.result.get_strategy_breakdown(),
            'symbol_breakdown': self.result.get_symbol_breakdown(),
        }
        
        # Risk metrics
        if not self.portfolio_history.empty:
            returns = self.portfolio_history['returns'].pct_change().dropna()
            
            report['risk_metrics'] = {
                'volatility': returns.std() * np.sqrt(252),  # Annualized
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                'max_consecutive_losses': self._calculate_max_consecutive_losses(),
                'profit_factor': self.metrics.get('profit_factor', 0),
                'recovery_factor': abs(self.metrics.get('total_return', 0) / self.metrics.get('max_drawdown', 1)) if self.metrics.get('max_drawdown', 0) > 0 else 0
            }
        
        # Trade statistics
        if self.trades:
            trade_pnls = [trade.pnl for trade in self.trades]
            winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
            losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
            
            report['trade_statistics'] = {
                'largest_win': max(trade_pnls),
                'largest_loss': min(trade_pnls),
                'avg_trade_pnl': np.mean(trade_pnls),
                'median_trade_pnl': np.median(trade_pnls),
                'win_loss_ratio': abs(np.mean(winning_trades) / np.mean(losing_trades)) if losing_trades else float('inf'),
                'expectancy': np.mean(trade_pnls)
            }
        
        return report
    
    def _calculate_max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losing trades."""
        if not self.trades:
            return 0
        
        consecutive_losses = 0
        max_consecutive = 0
        
        for trade in self.trades:
            if trade.pnl < 0:
                consecutive_losses += 1
                max_consecutive = max(max_consecutive, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return max_consecutive
    
    def export_results(self, filepath: str):
        """Export backtest results to file."""
        import json
        
        report = self.generate_report()
        
        # Convert datetime objects to strings for JSON serialization
        def serialize_datetime(obj):
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, (pd.Series, pd.DataFrame)):
                return obj.to_dict()
            return obj
        
        # Convert report to JSON-serializable format
        serializable_report = json.loads(json.dumps(report, default=serialize_datetime))
        
        with open(filepath, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        print(f"Backtest results exported to {filepath}")
