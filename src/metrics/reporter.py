"""
Metrics reporting and visualization module.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MetricsReporter:
    """
    Generate comprehensive reports and visualizations for trading strategy metrics.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize metrics reporter.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def generate_performance_report(
        self,
        metrics: Dict[str, Any],
        returns: pd.Series,
        positions: Optional[pd.Series] = None,
        benchmark_returns: Optional[pd.Series] = None,
        strategy_name: str = "Strategy",
        save_html: bool = True
    ) -> str:
        """
        Generate comprehensive performance report.
        
        Args:
            metrics: Calculated metrics dictionary
            returns: Strategy returns
            positions: Position series (optional)
            benchmark_returns: Benchmark returns (optional)
            strategy_name: Name of the strategy
            save_html: Whether to save HTML report
            
        Returns:
            HTML report string
        """
        report_html = self._create_html_report(
            metrics, returns, positions, benchmark_returns, strategy_name
        )
        
        if save_html:
            report_file = self.output_dir / f"{strategy_name}_performance_report.html"
            with open(report_file, 'w') as f:
                f.write(report_html)
            logger.info(f"Performance report saved to {report_file}")
            
        return report_html
    
    def create_performance_dashboard(
        self,
        metrics: Dict[str, Any],
        returns: pd.Series,
        positions: Optional[pd.Series] = None,
        benchmark_returns: Optional[pd.Series] = None,
        strategy_name: str = "Strategy"
    ) -> None:
        """
        Create comprehensive performance dashboard with multiple charts.
        
        Args:
            metrics: Calculated metrics dictionary
            returns: Strategy returns
            positions: Position series (optional)
            benchmark_returns: Benchmark returns (optional)
            strategy_name: Name of the strategy
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Cumulative returns
        ax1 = plt.subplot(3, 3, 1)
        self._plot_cumulative_returns(returns, benchmark_returns, ax1)
        
        # 2. Rolling Sharpe ratio
        ax2 = plt.subplot(3, 3, 2)
        self._plot_rolling_sharpe(returns, ax2)
        
        # 3. Drawdown
        ax3 = plt.subplot(3, 3, 3)
        self._plot_drawdown(returns, ax3)
        
        # 4. Return distribution
        ax4 = plt.subplot(3, 3, 4)
        self._plot_return_distribution(returns, ax4)
        
        # 5. Rolling volatility
        ax5 = plt.subplot(3, 3, 5)
        self._plot_rolling_volatility(returns, ax5)
        
        # 6. Monthly returns heatmap
        ax6 = plt.subplot(3, 3, 6)
        self._plot_monthly_returns_heatmap(returns, ax6)
        
        # 7. Risk-return scatter
        ax7 = plt.subplot(3, 3, 7)
        self._plot_risk_return_scatter(returns, benchmark_returns, ax7)
        
        # 8. Positions (if available)
        ax8 = plt.subplot(3, 3, 8)
        if positions is not None:
            self._plot_positions(positions, ax8)
        else:
            ax8.text(0.5, 0.5, 'No position data', ha='center', va='center', transform=ax8.transAxes)
            
        # 9. Key metrics table
        ax9 = plt.subplot(3, 3, 9)
        self._plot_metrics_table(metrics, ax9)
        
        plt.suptitle(f'{strategy_name} - Performance Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save dashboard
        dashboard_file = self.output_dir / f"{strategy_name}_dashboard.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance dashboard saved to {dashboard_file}")
    
    def create_risk_report(
        self,
        metrics: Dict[str, Any],
        returns: pd.Series,
        strategy_name: str = "Strategy"
    ) -> None:
        """
        Create detailed risk analysis report.
        
        Args:
            metrics: Calculated metrics dictionary
            returns: Strategy returns
            strategy_name: Name of the strategy
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # VaR and CVaR analysis
        self._plot_var_analysis(returns, axes[0, 0])
        
        # Tail risk analysis
        self._plot_tail_risk(returns, axes[0, 1])
        
        # Rolling risk metrics
        self._plot_rolling_risk_metrics(returns, axes[0, 2])
        
        # Return distribution vs normal
        self._plot_distribution_comparison(returns, axes[1, 0])
        
        # Stress testing
        self._plot_stress_scenarios(returns, axes[1, 1])
        
        # Risk decomposition
        self._plot_risk_decomposition(metrics, axes[1, 2])
        
        plt.suptitle(f'{strategy_name} - Risk Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        risk_file = self.output_dir / f"{strategy_name}_risk_analysis.png"
        plt.savefig(risk_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Risk analysis saved to {risk_file}")
    
    def _create_html_report(
        self,
        metrics: Dict[str, Any],
        returns: pd.Series,
        positions: Optional[pd.Series],
        benchmark_returns: Optional[pd.Series],
        strategy_name: str
    ) -> str:
        """Create HTML performance report."""
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{strategy_name} Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .warning {{ color: orange; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{strategy_name} Performance Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Analysis Period: {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')}</p>
                <p>Total Observations: {len(returns):,}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                {self._create_executive_summary(metrics)}
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                {self._create_metrics_table(metrics)}
            </div>
            
            <div class="section">
                <h2>Risk Analysis</h2>
                {self._create_risk_summary(metrics)}
            </div>
            
            <div class="section">
                <h2>Statistical Analysis</h2>
                {self._create_statistical_summary(metrics)}
            </div>
            
            {self._create_benchmark_comparison(metrics) if benchmark_returns is not None else ""}
            
            <div class="section">
                <h2>Recommendations</h2>
                {self._create_recommendations(metrics)}
            </div>
            
        </body>
        </html>
        """
        
        return html_template
    
    def _create_executive_summary(self, metrics: Dict[str, Any]) -> str:
        """Create executive summary section."""
        total_return = metrics.get('total_return', 0) * 100
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = metrics.get('max_drawdown', 0) * 100
        win_rate = metrics.get('win_rate', 0) * 100
        
        return f"""
        <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px;">
            <p><strong>Total Return:</strong> <span class="{'positive' if total_return > 0 else 'negative'}">{total_return:.2f}%</span></p>
            <p><strong>Sharpe Ratio:</strong> <span class="{'positive' if sharpe_ratio > 1 else 'warning' if sharpe_ratio > 0.5 else 'negative'}">{sharpe_ratio:.3f}</span></p>
            <p><strong>Maximum Drawdown:</strong> <span class="negative">{max_drawdown:.2f}%</span></p>
            <p><strong>Win Rate:</strong> {win_rate:.1f}%</p>
        </div>
        """
    
    def _create_metrics_table(self, metrics: Dict[str, Any]) -> str:
        """Create metrics table HTML."""
        
        # Group metrics by category
        categories = {
            'Return Metrics': ['total_return', 'annualized_return', 'cagr', 'mean_return'],
            'Risk Metrics': ['volatility', 'annualized_volatility', 'max_drawdown', 'var_5'],
            'Risk-Adjusted': ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'information_ratio'],
            'Trading Metrics': ['win_rate', 'profit_factor', 'expectancy', 'total_trades']
        }
        
        table_html = '<table class="metrics-table">'
        table_html += '<tr><th>Category</th><th>Metric</th><th>Value</th></tr>'
        
        for category, metric_names in categories.items():
            for i, metric in enumerate(metric_names):
                if metric in metrics:
                    value = metrics[metric]
                    if isinstance(value, float):
                        if metric in ['total_return', 'annualized_return', 'cagr', 'mean_return', 'win_rate']:
                            formatted_value = f"{value * 100:.2f}%"
                        elif metric in ['volatility', 'annualized_volatility', 'max_drawdown']:
                            formatted_value = f"{value * 100:.2f}%"
                        else:
                            formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    
                    category_cell = category if i == 0 else ""
                    table_html += f'<tr><td>{category_cell}</td><td>{metric.replace("_", " ").title()}</td><td>{formatted_value}</td></tr>'
        
        table_html += '</table>'
        return table_html
    
    def _create_risk_summary(self, metrics: Dict[str, Any]) -> str:
        """Create risk summary section."""
        risk_summary = "<ul>"
        
        # Risk level assessment
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown', 0)
        volatility = metrics.get('annualized_volatility', 0)
        
        if sharpe > 1.5:
            risk_summary += "<li><strong>Risk-Adjusted Performance:</strong> Excellent (Sharpe > 1.5)</li>"
        elif sharpe > 1.0:
            risk_summary += "<li><strong>Risk-Adjusted Performance:</strong> Good (Sharpe > 1.0)</li>"
        elif sharpe > 0.5:
            risk_summary += "<li><strong>Risk-Adjusted Performance:</strong> Moderate (Sharpe > 0.5)</li>"
        else:
            risk_summary += "<li><strong>Risk-Adjusted Performance:</strong> Poor (Sharpe ≤ 0.5)</li>"
            
        if max_dd < 0.05:
            risk_summary += "<li><strong>Drawdown Risk:</strong> Low (Max DD < 5%)</li>"
        elif max_dd < 0.15:
            risk_summary += "<li><strong>Drawdown Risk:</strong> Moderate (Max DD < 15%)</li>"
        else:
            risk_summary += "<li><strong>Drawdown Risk:</strong> High (Max DD ≥ 15%)</li>"
            
        if volatility < 0.10:
            risk_summary += "<li><strong>Volatility:</strong> Low (< 10% annual)</li>"
        elif volatility < 0.20:
            risk_summary += "<li><strong>Volatility:</strong> Moderate (< 20% annual)</li>"
        else:
            risk_summary += "<li><strong>Volatility:</strong> High (≥ 20% annual)</li>"
            
        risk_summary += "</ul>"
        return risk_summary
    
    def _create_statistical_summary(self, metrics: Dict[str, Any]) -> str:
        """Create statistical summary section."""
        stats_summary = "<ul>"
        
        # Statistical significance
        t_pvalue = metrics.get('t_test_pvalue', 1.0)
        if t_pvalue < 0.05:
            stats_summary += f"<li><strong>Return Significance:</strong> Statistically significant (p = {t_pvalue:.4f})</li>"
        else:
            stats_summary += f"<li><strong>Return Significance:</strong> Not statistically significant (p = {t_pvalue:.4f})</li>"
            
        # Distribution characteristics
        skewness = metrics.get('skewness', 0)
        kurtosis = metrics.get('kurtosis', 0)
        
        if abs(skewness) < 0.5:
            skew_desc = "approximately symmetric"
        elif skewness > 0:
            skew_desc = "positively skewed (right tail)"
        else:
            skew_desc = "negatively skewed (left tail)"
            
        stats_summary += f"<li><strong>Return Distribution:</strong> {skew_desc} (skewness = {skewness:.3f})</li>"
        
        if kurtosis > 3:
            kurt_desc = "fat-tailed (high kurtosis)"
        elif kurtosis < 3:
            kurt_desc = "thin-tailed (low kurtosis)"
        else:
            kurt_desc = "normal kurtosis"
            
        stats_summary += f"<li><strong>Tail Behavior:</strong> {kurt_desc} (kurtosis = {kurtosis:.3f})</li>"
        
        stats_summary += "</ul>"
        return stats_summary
    
    def _create_benchmark_comparison(self, metrics: Dict[str, Any]) -> str:
        """Create benchmark comparison section."""
        if 'information_ratio' not in metrics:
            return ""
            
        ir = metrics.get('information_ratio', 0)
        beta = metrics.get('beta', 1)
        
        comparison = f"""
        <div class="section">
            <h2>Benchmark Comparison</h2>
            <ul>
                <li><strong>Information Ratio:</strong> {ir:.3f}</li>
                <li><strong>Beta:</strong> {beta:.3f}</li>
                <li><strong>Alpha Generation:</strong> {'Positive' if ir > 0 else 'Negative'}</li>
            </ul>
        </div>
        """
        
        return comparison
    
    def _create_recommendations(self, metrics: Dict[str, Any]) -> str:
        """Create recommendations section."""
        recommendations = []
        
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown', 0)
        win_rate = metrics.get('win_rate', 0)
        
        if sharpe < 1.0:
            recommendations.append("Consider improving risk-adjusted returns by optimizing position sizing or entry/exit rules.")
            
        if max_dd > 0.20:
            recommendations.append("High maximum drawdown suggests implementing stronger risk management controls.")
            
        if win_rate < 0.45:
            recommendations.append("Low win rate may indicate need for better signal quality or entry timing.")
            
        if len(recommendations) == 0:
            recommendations.append("Strategy shows good performance characteristics. Consider scaling or diversification.")
            
        rec_html = "<ul>"
        for rec in recommendations:
            rec_html += f"<li>{rec}</li>"
        rec_html += "</ul>"
        
        return rec_html
    
    def _plot_cumulative_returns(self, returns: pd.Series, benchmark_returns: Optional[pd.Series], ax) -> None:
        """Plot cumulative returns."""
        cumulative = (1 + returns).cumprod()
        ax.plot(cumulative.index, cumulative.values, label='Strategy', linewidth=2)
        
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            ax.plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                   label='Benchmark', linewidth=2, alpha=0.7)
        
        ax.set_title('Cumulative Returns')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_rolling_sharpe(self, returns: pd.Series, ax, window: int = 252) -> None:
        """Plot rolling Sharpe ratio."""
        rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
        ax.plot(rolling_sharpe.index, rolling_sharpe.values)
        ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.7, label='Good (1.0)')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Fair (0.5)')
        ax.axhline(y=0.0, color='r', linestyle='--', alpha=0.7, label='Poor (0.0)')
        ax.set_title(f'Rolling Sharpe Ratio ({window}d)')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_drawdown(self, returns: pd.Series, ax) -> None:
        """Plot drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.7, color='red')
        ax.set_title('Drawdown')
        ax.set_ylabel('Drawdown')
        ax.grid(True, alpha=0.3)
    
    def _plot_return_distribution(self, returns: pd.Series, ax) -> None:
        """Plot return distribution."""
        ax.hist(returns, bins=50, alpha=0.7, density=True, label='Returns')
        
        # Overlay normal distribution
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, normal_dist, 'r-', label='Normal Dist')
        
        ax.set_title('Return Distribution')
        ax.set_xlabel('Return')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_rolling_volatility(self, returns: pd.Series, ax, window: int = 252) -> None:
        """Plot rolling volatility."""
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        ax.plot(rolling_vol.index, rolling_vol.values)
        ax.set_title(f'Rolling Volatility ({window}d)')
        ax.set_ylabel('Annualized Volatility')
        ax.grid(True, alpha=0.3)
    
    def _plot_monthly_returns_heatmap(self, returns: pd.Series, ax) -> None:
        """Plot monthly returns heatmap."""
        try:
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_table = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
            
            sns.heatmap(monthly_table, annot=True, fmt='.2%', cmap='RdYlGn', center=0, ax=ax)
            ax.set_title('Monthly Returns Heatmap')
        except:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_risk_return_scatter(self, returns: pd.Series, benchmark_returns: Optional[pd.Series], ax) -> None:
        """Plot risk-return scatter."""
        if benchmark_returns is not None:
            strategy_vol = returns.std() * np.sqrt(252)
            strategy_ret = returns.mean() * 252
            benchmark_vol = benchmark_returns.std() * np.sqrt(252)
            benchmark_ret = benchmark_returns.mean() * 252
            
            ax.scatter(strategy_vol, strategy_ret, s=100, label='Strategy', alpha=0.7)
            ax.scatter(benchmark_vol, benchmark_ret, s=100, label='Benchmark', alpha=0.7)
            ax.set_xlabel('Volatility (Annual)')
            ax.set_ylabel('Return (Annual)')
            ax.set_title('Risk-Return Profile')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No benchmark data', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_positions(self, positions: pd.Series, ax) -> None:
        """Plot position sizes."""
        ax.plot(positions.index, positions.values)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title('Position Sizes')
        ax.set_ylabel('Position')
        ax.grid(True, alpha=0.3)
    
    def _plot_metrics_table(self, metrics: Dict[str, Any], ax) -> None:
        """Plot key metrics as table."""
        key_metrics = [
            ('Total Return', f"{metrics.get('total_return', 0) * 100:.2f}%"),
            ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.3f}"),
            ('Max Drawdown', f"{metrics.get('max_drawdown', 0) * 100:.2f}%"),
            ('Win Rate', f"{metrics.get('win_rate', 0) * 100:.1f}%"),
            ('Volatility', f"{metrics.get('annualized_volatility', 0) * 100:.2f}%"),
            ('Calmar Ratio', f"{metrics.get('calmar_ratio', 0):.3f}")
        ]
        
        table_data = []
        for metric, value in key_metrics:
            table_data.append([metric, value])
        
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax.set_title('Key Metrics')
    
    def _plot_var_analysis(self, returns: pd.Series, ax) -> None:
        """Plot VaR analysis."""
        var_5 = np.percentile(returns, 5)
        var_1 = np.percentile(returns, 1)
        
        ax.hist(returns, bins=50, alpha=0.7, density=True)
        ax.axvline(var_5, color='orange', linestyle='--', label=f'5% VaR: {var_5:.4f}')
        ax.axvline(var_1, color='red', linestyle='--', label=f'1% VaR: {var_1:.4f}')
        ax.set_title('Value at Risk Analysis')
        ax.set_xlabel('Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_tail_risk(self, returns: pd.Series, ax) -> None:
        """Plot tail risk analysis."""
        # Q-Q plot against normal distribution
        stats.probplot(returns, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot vs Normal Distribution')
        ax.grid(True, alpha=0.3)
    
    def _plot_rolling_risk_metrics(self, returns: pd.Series, ax) -> None:
        """Plot rolling risk metrics."""
        window = min(252, len(returns) // 4)
        rolling_vol = returns.rolling(window).std()
        rolling_var = returns.rolling(window).quantile(0.05)
        
        ax2 = ax.twinx()
        ax.plot(rolling_vol.index, rolling_vol.values, label='Volatility', color='blue')
        ax2.plot(rolling_var.index, rolling_var.values, label='5% VaR', color='red')
        
        ax.set_ylabel('Volatility', color='blue')
        ax2.set_ylabel('VaR', color='red')
        ax.set_title('Rolling Risk Metrics')
        ax.grid(True, alpha=0.3)
    
    def _plot_distribution_comparison(self, returns: pd.Series, ax) -> None:
        """Plot distribution comparison."""
        ax.hist(returns, bins=30, alpha=0.7, density=True, label='Actual')
        
        # Normal distribution overlay
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', label='Normal')
        
        ax.set_title('Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_stress_scenarios(self, returns: pd.Series, ax) -> None:
        """Plot stress test scenarios."""
        # Calculate stressed returns
        scenarios = {
            'Normal': returns,
            '2x Volatility': returns * 2,
            '50% Correlation Break': returns * np.random.normal(0.5, 0.3, len(returns))
        }
        
        for name, scenario in scenarios.items():
            cumulative = (1 + scenario).cumprod()
            ax.plot(cumulative.index, cumulative.values, label=name, alpha=0.7)
        
        ax.set_title('Stress Test Scenarios')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_risk_decomposition(self, metrics: Dict[str, Any], ax) -> None:
        """Plot risk decomposition."""
        risk_components = {
            'Market Risk': metrics.get('beta', 0) * 0.15,  # Approximate
            'Specific Risk': metrics.get('annualized_volatility', 0) - metrics.get('beta', 0) * 0.15,
            'Tail Risk': max(0, metrics.get('annualized_volatility', 0) * 0.2)
        }
        
        labels = list(risk_components.keys())
        sizes = [max(0, v) for v in risk_components.values()]
        
        if sum(sizes) > 0:
            ax.pie(sizes, labels=labels, autopct='%1.1f%%')
        ax.set_title('Risk Decomposition')
    
    def save_metrics_to_json(self, metrics: Dict[str, Any], filename: str) -> None:
        """Save metrics to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        json_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                json_metrics[key] = value.tolist()
            elif isinstance(value, (np.int64, np.int32)):
                json_metrics[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                json_metrics[key] = float(value)
            else:
                json_metrics[key] = value
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(json_metrics, f, indent=2, default=str)
            
        logger.info(f"Metrics saved to {filepath}")
    
    def create_comparison_report(
        self,
        strategy_metrics: Dict[str, Dict[str, Any]],
        strategy_returns: Dict[str, pd.Series]
    ) -> None:
        """
        Create comparison report for multiple strategies.
        
        Args:
            strategy_metrics: Dictionary mapping strategy name to metrics
            strategy_returns: Dictionary mapping strategy name to returns series
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Cumulative returns comparison
        ax = axes[0, 0]
        for name, returns in strategy_returns.items():
            cumulative = (1 + returns).cumprod()
            ax.plot(cumulative.index, cumulative.values, label=name, linewidth=2)
        ax.set_title('Cumulative Returns Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Risk-return scatter
        ax = axes[0, 1]
        for name, metrics in strategy_metrics.items():
            vol = metrics.get('annualized_volatility', 0)
            ret = metrics.get('annualized_return', 0)
            ax.scatter(vol, ret, s=100, label=name, alpha=0.7)
        ax.set_xlabel('Volatility')
        ax.set_ylabel('Return')
        ax.set_title('Risk-Return Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Sharpe ratio comparison
        ax = axes[1, 0]
        names = list(strategy_metrics.keys())
        sharpe_ratios = [strategy_metrics[name].get('sharpe_ratio', 0) for name in names]
        bars = ax.bar(names, sharpe_ratios, alpha=0.7)
        ax.set_title('Sharpe Ratio Comparison')
        ax.set_ylabel('Sharpe Ratio')
        ax.grid(True, alpha=0.3)
        
        # Color bars based on performance
        for i, bar in enumerate(bars):
            if sharpe_ratios[i] > 1:
                bar.set_color('green')
            elif sharpe_ratios[i] > 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Drawdown comparison
        ax = axes[1, 1]
        max_drawdowns = [strategy_metrics[name].get('max_drawdown', 0) for name in names]
        ax.bar(names, max_drawdowns, alpha=0.7, color='red')
        ax.set_title('Maximum Drawdown Comparison')
        ax.set_ylabel('Max Drawdown')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        comparison_file = self.output_dir / "strategy_comparison.png"
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Strategy comparison saved to {comparison_file}")
