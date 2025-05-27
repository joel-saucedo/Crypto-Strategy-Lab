"""
Latency simulation utilities for realistic execution modeling.
"""

import numpy as np
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class LatencySimulator:
    """
    Advanced latency simulation for cryptocurrency trading.
    Models realistic network conditions, exchange-specific delays, and market conditions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize latency simulator with comprehensive parameters.
        
        Args:
            config: Latency simulation configuration
        """
        self.config = config
        self.setup_exchange_profiles()
        self.setup_market_conditions()
        
    def setup_exchange_profiles(self):
        """Setup exchange-specific latency profiles."""
        self.exchange_profiles = self.config.get('exchange_profiles', {
            'binance': {
                'base_latency': {'mean': 45, 'std': 12, 'min': 15, 'max': 150},
                'order_latency': {'mean': 25, 'std': 8, 'min': 10, 'max': 80},
                'market_data_latency': {'mean': 15, 'std': 5, 'min': 5, 'max': 50},
                'geographic_factor': 1.0,  # Assume optimal location
                'reliability': 0.995,  # 99.5% uptime
                'capacity_factor': 1.0
            },
            'coinbase': {
                'base_latency': {'mean': 65, 'std': 18, 'min': 25, 'max': 200},
                'order_latency': {'mean': 35, 'std': 12, 'min': 15, 'max': 120},
                'market_data_latency': {'mean': 20, 'std': 7, 'min': 8, 'max': 60},
                'geographic_factor': 1.2,
                'reliability': 0.992,
                'capacity_factor': 0.9
            },
            'kraken': {
                'base_latency': {'mean': 80, 'std': 22, 'min': 30, 'max': 250},
                'order_latency': {'mean': 45, 'std': 15, 'min': 20, 'max': 140},
                'market_data_latency': {'mean': 25, 'std': 8, 'min': 10, 'max': 70},
                'geographic_factor': 1.1,
                'reliability': 0.990,
                'capacity_factor': 0.85
            },
            'simulated': {
                'base_latency': {'mean': 1, 'std': 0.2, 'min': 0.5, 'max': 3},
                'order_latency': {'mean': 0.5, 'std': 0.1, 'min': 0.2, 'max': 1},
                'market_data_latency': {'mean': 0.2, 'std': 0.05, 'min': 0.1, 'max': 0.5},
                'geographic_factor': 1.0,
                'reliability': 1.0,
                'capacity_factor': 1.0
            }
        })
        
    def setup_market_conditions(self):
        """Setup market condition factors that affect latency."""
        self.market_conditions = {
            'volatility_factor': 1.0,  # Higher volatility = higher latency
            'volume_factor': 1.0,      # Higher volume = higher latency
            'time_factor': 1.0,        # Different times of day affect latency
            'congestion_factor': 1.0   # Network congestion
        }
        
    async def get_order_latency(self, exchange: str = 'simulated', order_type: str = 'market') -> float:
        """
        Get realistic order submission latency.
        
        Args:
            exchange: Exchange name
            order_type: Type of order ('market', 'limit', 'stop')
            
        Returns:
            Latency in milliseconds
        """
        profile = self.exchange_profiles.get(exchange, self.exchange_profiles['simulated'])
        
        # Base order latency
        latency_params = profile['order_latency']
        base_latency = self._generate_latency(latency_params)
        
        # Order type adjustment
        type_multiplier = {
            'market': 1.0,
            'limit': 0.8,   # Limit orders slightly faster
            'stop': 1.2,    # Stop orders slightly slower
            'stop_limit': 1.3
        }.get(order_type, 1.0)
        
        # Apply market conditions
        total_latency = base_latency * type_multiplier * self._get_market_factor()
        
        # Apply geographic and capacity factors
        total_latency *= profile['geographic_factor'] * (2 - profile['capacity_factor'])
        
        # Simulate network unreliability
        if np.random.random() > profile['reliability']:
            total_latency *= np.random.uniform(3, 10)  # Significant delay on failure
            
        return max(total_latency, 0.1)
    
    async def get_market_data_latency(self, exchange: str = 'simulated') -> float:
        """
        Get market data feed latency.
        
        Args:
            exchange: Exchange name
            
        Returns:
            Latency in milliseconds
        """
        profile = self.exchange_profiles.get(exchange, self.exchange_profiles['simulated'])
        
        latency_params = profile['market_data_latency']
        base_latency = self._generate_latency(latency_params)
        
        # Market data is less affected by market conditions
        market_factor = 1 + (self._get_market_factor() - 1) * 0.3
        
        total_latency = base_latency * market_factor * profile['geographic_factor']
        
        return max(total_latency, 0.05)
    
    async def get_latency(self, exchange: str = 'simulated', operation: str = 'order') -> float:
        """
        Get latency for any operation.
        
        Args:
            exchange: Exchange name
            operation: Operation type ('order', 'market_data', 'base')
            
        Returns:
            Latency in milliseconds
        """
        if operation == 'order':
            return await self.get_order_latency(exchange)
        elif operation == 'market_data':
            return await self.get_market_data_latency(exchange)
        else:
            return await self.get_base_latency(exchange)
    
    async def get_base_latency(self, exchange: str = 'simulated') -> float:
        """
        Get base network latency.
        
        Args:
            exchange: Exchange name
            
        Returns:
            Latency in milliseconds
        """
        profile = self.exchange_profiles.get(exchange, self.exchange_profiles['simulated'])
        
        latency_params = profile['base_latency']
        base_latency = self._generate_latency(latency_params)
        
        total_latency = base_latency * profile['geographic_factor']
        
        return max(total_latency, 0.1)
    
    def _generate_latency(self, params: Dict[str, float]) -> float:
        """
        Generate latency from parameters using realistic distribution.
        
        Args:
            params: Latency parameters (mean, std, min, max)
            
        Returns:
            Generated latency
        """
        # Use log-normal distribution for more realistic latency modeling
        mu = np.log(params['mean'])
        sigma = params['std'] / params['mean']  # Coefficient of variation
        
        latency = np.random.lognormal(mu, sigma)
        
        # Clip to realistic bounds
        latency = np.clip(latency, params['min'], params['max'])
        
        return latency
    
    def _get_market_factor(self) -> float:
        """
        Calculate combined market condition factor.
        
        Returns:
            Market factor multiplier
        """
        return (self.market_conditions['volatility_factor'] * 0.3 +
                self.market_conditions['volume_factor'] * 0.3 +
                self.market_conditions['time_factor'] * 0.2 +
                self.market_conditions['congestion_factor'] * 0.2)
    
    def update_market_conditions(
        self,
        volatility: Optional[float] = None,
        volume: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Update market conditions that affect latency.
        
        Args:
            volatility: Current market volatility (normalized)
            volume: Current volume (normalized)  
            timestamp: Current timestamp for time-based effects
        """
        if volatility is not None:
            # Higher volatility increases latency (1.0 to 2.0 range)
            self.market_conditions['volatility_factor'] = 1.0 + min(volatility, 1.0)
            
        if volume is not None:
            # Higher volume increases latency (1.0 to 1.5 range)
            self.market_conditions['volume_factor'] = 1.0 + min(volume, 0.5)
            
        if timestamp is not None:
            # Time-based effects (market sessions, weekends)
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Peak trading hours have higher latency
            if 13 <= hour <= 16:  # US market hours (peak)
                time_factor = 1.3
            elif 7 <= hour <= 11:  # EU market hours
                time_factor = 1.2
            elif 23 <= hour or hour <= 4:  # Asia market hours
                time_factor = 1.1
            else:
                time_factor = 1.0
                
            # Weekends have lower latency
            if day_of_week >= 5:  # Weekend
                time_factor *= 0.8
                
            self.market_conditions['time_factor'] = time_factor
    
    def simulate_network_congestion(self, congestion_level: float = 1.0):
        """
        Simulate network congestion effects.
        
        Args:
            congestion_level: Congestion level (1.0 = normal, >1.0 = congested)
        """
        self.market_conditions['congestion_factor'] = congestion_level
    
    async def simulate_realistic_execution_flow(
        self,
        exchange: str,
        num_orders: int = 1
    ) -> Dict[str, Any]:
        """
        Simulate a realistic execution flow with all latency components.
        
        Args:
            exchange: Exchange name
            num_orders: Number of orders to simulate
            
        Returns:
            Dictionary with execution timing details
        """
        results = {
            'total_time': 0.0,
            'market_data_latency': 0.0,
            'order_latencies': [],
            'execution_sequence': []
        }
        
        start_time = datetime.now()
        
        # Market data latency
        md_latency = await self.get_market_data_latency(exchange)
        results['market_data_latency'] = md_latency
        await asyncio.sleep(md_latency / 1000.0)
        
        # Order execution latencies
        for i in range(num_orders):
            order_latency = await self.get_order_latency(exchange)
            results['order_latencies'].append(order_latency)
            
            execution_step = {
                'order_id': i + 1,
                'latency': order_latency,
                'timestamp': datetime.now()
            }
            results['execution_sequence'].append(execution_step)
            
            await asyncio.sleep(order_latency / 1000.0)
        
        end_time = datetime.now()
        results['total_time'] = (end_time - start_time).total_seconds() * 1000
        
        return results
    
    def get_latency_statistics(self, exchange: str, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Generate latency statistics for an exchange.
        
        Args:
            exchange: Exchange name
            num_samples: Number of samples for statistics
            
        Returns:
            Dictionary with latency statistics
        """
        profile = self.exchange_profiles.get(exchange, self.exchange_profiles['simulated'])
        
        # Generate samples
        order_latencies = []
        market_data_latencies = []
        
        for _ in range(num_samples):
            order_lat = self._generate_latency(profile['order_latency'])
            md_lat = self._generate_latency(profile['market_data_latency'])
            
            order_latencies.append(order_lat)
            market_data_latencies.append(md_lat)
        
        return {
            'exchange': exchange,
            'order_latency': {
                'mean': np.mean(order_latencies),
                'std': np.std(order_latencies),
                'median': np.median(order_latencies),
                'p95': np.percentile(order_latencies, 95),
                'p99': np.percentile(order_latencies, 99),
                'min': np.min(order_latencies),
                'max': np.max(order_latencies)
            },
            'market_data_latency': {
                'mean': np.mean(market_data_latencies),
                'std': np.std(market_data_latencies),
                'median': np.median(market_data_latencies),
                'p95': np.percentile(market_data_latencies, 95),
                'p99': np.percentile(market_data_latencies, 99),
                'min': np.min(market_data_latencies),
                'max': np.max(market_data_latencies)
            },
            'reliability': profile['reliability'],
            'geographic_factor': profile['geographic_factor']
        }
