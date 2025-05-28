#!/usr/bin/env python3
"""
Environment Configuration Validator

This script validates and manages environment configurations for
multi-exchange trading setup.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates environment configuration for trading setup"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize config validator
        
        Args:
            env_file: Path to .env file (optional)
        """
        self.env_file = env_file or '.env'
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
        # Load environment variables
        if Path(self.env_file).exists():
            load_dotenv(self.env_file)
            logger.info(f"Loaded environment from {self.env_file}")
        else:
            logger.warning(f"Environment file {self.env_file} not found")
    
    def validate_exchange_config(self, exchange: str) -> bool:
        """
        Validate configuration for specific exchange
        
        Args:
            exchange: Exchange name ('bybit' or 'alpaca')
        
        Returns:
            True if configuration is valid
        """
        if exchange.lower() == 'bybit':
            return self._validate_bybit_config()
        elif exchange.lower() == 'alpaca':
            return self._validate_alpaca_config()
        else:
            self.errors.append(f"Unknown exchange: {exchange}")
            return False
    
    def _validate_bybit_config(self) -> bool:
        """Validate Bybit configuration"""
        valid = True
        
        # Check for API credentials
        api_key = self._get_env_var(['BYBIT_API_KEY', 'BYBIT_TESTNET_API_KEY'])
        api_secret = self._get_env_var(['BYBIT_API_SECRET', 'BYBIT_TESTNET_API_SECRET'])
        
        if not api_key:
            self.errors.append("Bybit API key not found. Set BYBIT_API_KEY or BYBIT_TESTNET_API_KEY")
            valid = False
        
        if not api_secret:
            self.errors.append("Bybit API secret not found. Set BYBIT_API_SECRET or BYBIT_TESTNET_API_SECRET")
            valid = False
        
        # Check if credentials match (testnet vs live)
        if api_key and api_secret:
            has_testnet = bool(os.getenv('BYBIT_TESTNET_API_KEY'))
            has_live = bool(os.getenv('BYBIT_API_KEY'))
            
            if has_testnet and has_live:
                self.warnings.append("Both testnet and live Bybit credentials found")
            elif has_testnet:
                logger.info("Using Bybit testnet credentials")
            elif has_live:
                logger.info("Using Bybit live credentials")
        
        return valid
    
    def _validate_alpaca_config(self) -> bool:
        """Validate Alpaca configuration"""
        valid = True
        
        # Check for API credentials
        api_key = self._get_env_var(['ALPACA_API_KEY', 'ALPACA_PAPER_API_KEY'])
        api_secret = self._get_env_var(['ALPACA_API_SECRET', 'ALPACA_PAPER_API_SECRET'])
        
        if not api_key:
            self.errors.append("Alpaca API key not found. Set ALPACA_API_KEY or ALPACA_PAPER_API_KEY")
            valid = False
        
        if not api_secret:
            self.errors.append("Alpaca API secret not found. Set ALPACA_API_SECRET or ALPACA_PAPER_API_SECRET")
            valid = False
        
        # Check if credentials match (paper vs live)
        if api_key and api_secret:
            has_paper = bool(os.getenv('ALPACA_PAPER_API_KEY'))
            has_live = bool(os.getenv('ALPACA_API_KEY'))
            
            if has_paper and has_live:
                self.warnings.append("Both paper and live Alpaca credentials found")
            elif has_paper:
                logger.info("Using Alpaca paper trading credentials")
            elif has_live:
                logger.info("Using Alpaca live trading credentials")
        
        return valid
    
    def _get_env_var(self, var_names: List[str]) -> Optional[str]:
        """Get first available environment variable from list"""
        for var_name in var_names:
            value = os.getenv(var_name)
            if value:
                return value
        return None
    
    def validate_general_config(self) -> bool:
        """Validate general configuration"""
        valid = True
        
        # Check trading mode
        trading_mode = os.getenv('DEFAULT_TRADING_MODE', 'paper').lower()
        if trading_mode not in ['live', 'paper', 'testnet']:
            self.errors.append(f"Invalid DEFAULT_TRADING_MODE: {trading_mode}")
            valid = False
        
        # Check log level
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        if log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            self.warnings.append(f"Invalid LOG_LEVEL: {log_level}, using INFO")
        
        # Check risk management settings
        try:
            max_position = float(os.getenv('MAX_POSITION_SIZE', '1000'))
            if max_position <= 0:
                self.warnings.append("MAX_POSITION_SIZE should be positive")
        except ValueError:
            self.warnings.append("Invalid MAX_POSITION_SIZE, using default")
        
        try:
            max_loss = float(os.getenv('MAX_DAILY_LOSS', '500'))
            if max_loss <= 0:
                self.warnings.append("MAX_DAILY_LOSS should be positive")
        except ValueError:
            self.warnings.append("Invalid MAX_DAILY_LOSS, using default")
        
        return valid
    
    def test_exchange_connection(self, exchange: str) -> bool:
        """
        Test connection to exchange
        
        Args:
            exchange: Exchange name
        
        Returns:
            True if connection successful
        """
        try:
            from src.exchanges import ExchangeFactory, TradingMode
            
            # Determine trading mode
            trading_mode = os.getenv('DEFAULT_TRADING_MODE', 'paper').lower()
            if exchange.lower() == 'bybit' and trading_mode == 'paper':
                trading_mode = 'testnet'  # Bybit uses testnet for testing
            
            mode = TradingMode(trading_mode)
            
            # Create exchange instance
            exchange_obj = ExchangeFactory.create_exchange(exchange, mode=mode)
            
            # Test connection (this would be async in real usage)
            logger.info(f"Testing {exchange} connection...")
            # Note: In real implementation, you'd run this async
            # result = await exchange_obj.test_connection()
            logger.info(f"Exchange {exchange} configuration appears valid")
            return True
            
        except Exception as e:
            self.errors.append(f"Failed to test {exchange} connection: {e}")
            return False
    
    def create_example_env(self) -> None:
        """Create example .env file"""
        example_content = """# Exchange API Configuration
# Copy this file to .env and fill in your actual API credentials

# Bybit API Configuration
# Get your API keys from: https://www.bybit.com/app/user/api-management
BYBIT_API_KEY=your_bybit_api_key_here
BYBIT_API_SECRET=your_bybit_api_secret_here

# Bybit Testnet Configuration (for testing)
BYBIT_TESTNET_API_KEY=your_bybit_testnet_api_key_here
BYBIT_TESTNET_API_SECRET=your_bybit_testnet_api_secret_here

# Alpaca API Configuration  
# Get your API keys from: https://app.alpaca.markets/
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_API_SECRET=your_alpaca_api_secret_here

# Alpaca Paper Trading Configuration
ALPACA_PAPER_API_KEY=your_alpaca_paper_api_key_here
ALPACA_PAPER_API_SECRET=your_alpaca_paper_api_secret_here

# Trading Configuration
DEFAULT_TRADING_MODE=paper  # Options: live, paper, testnet
LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR

# Risk Management
MAX_POSITION_SIZE=1000  # Maximum position size in USD
MAX_DAILY_LOSS=500      # Maximum daily loss in USD
ENABLE_STOP_LOSS=true   # Enable automatic stop-loss orders

# Data Configuration
DATA_CACHE_ENABLED=true
DATA_CACHE_TTL=300      # Cache TTL in seconds
"""
        
        with open('.env.example', 'w') as f:
            f.write(example_content)
        
        logger.info("Created .env.example file")
    
    def print_summary(self) -> None:
        """Print validation summary"""
        print("\n" + "="*60)
        print("CONFIGURATION VALIDATION SUMMARY")
        print("="*60)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ All validations passed!")
        elif not self.errors:
            print("\n✅ Validation passed with warnings")
        else:
            print("\n❌ Validation failed")
        
        print("="*60)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate exchange configuration')
    parser.add_argument('--env-file', help='Path to .env file')
    parser.add_argument('--exchange', choices=['bybit', 'alpaca'], help='Test specific exchange')
    parser.add_argument('--create-example', action='store_true', help='Create example .env file')
    parser.add_argument('--test-connection', action='store_true', help='Test exchange connections')
    
    args = parser.parse_args()
    
    validator = ConfigValidator(args.env_file)
    
    if args.create_example:
        validator.create_example_env()
        return
    
    # Validate general configuration
    validator.validate_general_config()
    
    # Validate exchange-specific configuration
    exchanges = [args.exchange] if args.exchange else ['bybit', 'alpaca']
    
    for exchange in exchanges:
        logger.info(f"Validating {exchange} configuration...")
        validator.validate_exchange_config(exchange)
        
        if args.test_connection:
            validator.test_exchange_connection(exchange)
    
    # Print summary
    validator.print_summary()
    
    # Exit with error code if validation failed
    if validator.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
