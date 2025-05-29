"""
CLI module for configuration validation functionality.
"""

import sys
import os
import yaml
import json
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def run_validation(args):
    """
    Validate configuration files and system setup.
    """
    try:
        print("=" * 60)
        print("🔍 CRYPTO STRATEGY LAB - CONFIGURATION VALIDATION")
        print("=" * 60)
        
        validation_passed = True
        
        # Validate specific config file if provided
        if hasattr(args, 'config') and args.config:
            print(f"📄 Validating config file: {args.config}")
            file_valid = validate_config_file(args.config)
            validation_passed = validation_passed and file_valid
            print()
        
        # Validate specific strategy if provided
        if hasattr(args, 'strategy') and args.strategy:
            print(f"📈 Validating strategy: {args.strategy}")
            strategy_valid = validate_specific_strategy(args.strategy)
            validation_passed = validation_passed and strategy_valid
            print()
        
        # Validate overall system configuration
        print("🔧 Validating system configuration...")
        system_valid = validate_system_config()
        validation_passed = validation_passed and system_valid
        print()
        
        # Validate data directories
        print("📁 Validating data directories...")
        data_valid = validate_data_structure()
        validation_passed = validation_passed and data_valid
        print()
        
        # Validate all strategies
        print("📈 Validating all strategies...")
        strategies_valid = validate_strategies()
        validation_passed = validation_passed and strategies_valid
        print()
        
        # Validate dependencies
        print("📦 Validating dependencies...")
        deps_valid = validate_dependencies()
        validation_passed = validation_passed and deps_valid
        print()
        
        # Overall result
        print("=" * 60)
        if validation_passed:
            print("✅ ALL VALIDATIONS PASSED")
            print("   System is ready for operation")
            return_code = 0
        else:
            print("❌ VALIDATION FAILURES DETECTED")
            print("   Fix the issues above before proceeding")
            return_code = 1
        
        print("=" * 60)
        return return_code
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return 1

def validate_config_file(config_path: str) -> bool:
    """Validate a specific configuration file."""
    try:
        config_file = Path(config_path)
        
        if not config_file.exists():
            print(f"   ❌ Config file not found: {config_path}")
            return False
        
        # Determine file type and validate
        if config_file.suffix.lower() in ['.yaml', '.yml']:
            return validate_yaml_config(config_file)
        elif config_file.suffix.lower() == '.json':
            return validate_json_config(config_file)
        else:
            print(f"   ⚠️  Unknown config file type: {config_file.suffix}")
            return True  # Don't fail for unknown types
        
    except Exception as e:
        print(f"   ❌ Error validating config file: {e}")
        return False

def validate_yaml_config(config_file: Path) -> bool:
    """Validate YAML configuration file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            print(f"   ❌ Empty YAML file: {config_file}")
            return False
        
        print(f"   ✅ Valid YAML: {config_file}")
        
        # Additional validation based on file location
        if 'strategies' in str(config_file):
            return validate_strategy_config(config, config_file)
        
        return True
        
    except yaml.YAMLError as e:
        print(f"   ❌ Invalid YAML in {config_file}: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error reading {config_file}: {e}")
        return False

def validate_json_config(config_file: Path) -> bool:
    """Validate JSON configuration file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"   ✅ Valid JSON: {config_file}")
        return True
        
    except json.JSONDecodeError as e:
        print(f"   ❌ Invalid JSON in {config_file}: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error reading {config_file}: {e}")
        return False

def validate_strategy_config(config: dict, config_file: Path) -> bool:
    """Validate strategy-specific configuration."""
    # Check for strategy section
    if 'strategy' not in config:
        print(f"   ❌ Missing 'strategy' section in {config_file}")
        return False
    
    strategy_section = config['strategy']
    required_fields = ['name', 'description']
    missing_fields = []
    
    for field in required_fields:
        if field not in strategy_section:
            missing_fields.append(field)
    
    # Check for parameters (either in param_grid, best_params, or top-level parameters)
    has_parameters = 'param_grid' in config or 'best_params' in config or 'parameters' in config
    if not has_parameters:
        missing_fields.append('parameters (param_grid, best_params, or parameters)')
    
    if missing_fields:
        print(f"   ❌ Missing required fields in {config_file}: {missing_fields}")
        return False
    
    print(f"   ✅ Strategy config valid: {strategy_section.get('name', 'Unknown')}")
    return True

def validate_system_config() -> bool:
    """Validate overall system configuration."""
    all_valid = True
    
    # Check main config files
    config_files = [
        'config/base.yaml',
        'config/data_sources.yaml'
    ]
    
    for config_file in config_files:
        config_path = Path(config_file)
        if config_path.exists():
            print(f"   ✅ Found: {config_file}")
        else:
            print(f"   ⚠️  Missing: {config_file}")
            # Don't fail for missing config files - they're optional
    
    # Check pyproject.toml
    if Path('pyproject.toml').exists():
        print("   ✅ Found: pyproject.toml")
    else:
        print("   ⚠️  Missing: pyproject.toml")
    
    # Check requirements.txt
    if Path('requirements.txt').exists():
        print("   ✅ Found: requirements.txt")
    else:
        print("   ❌ Missing: requirements.txt")
        all_valid = False
    
    return all_valid

def validate_data_structure() -> bool:
    """Validate data directory structure."""
    all_valid = True
    
    required_dirs = [
        'data/',
        'data/raw/',
        'data/processed/',
        'data/cache/'
    ]
    
    for dir_path in required_dirs:
        data_dir = Path(dir_path)
        if data_dir.exists():
            print(f"   ✅ Directory exists: {dir_path}")
        else:
            print(f"   ⚠️  Creating directory: {dir_path}")
            try:
                data_dir.mkdir(parents=True, exist_ok=True)
                print(f"   ✅ Created: {dir_path}")
            except Exception as e:
                print(f"   ❌ Failed to create {dir_path}: {e}")
                all_valid = False
    
    # Check for sample data
    sample_data_files = [
        'data/raw/BTCUSD_daily_ohlcv.csv',
        'data/processed/BTCUSD_daily_processed.parquet'
    ]
    
    for data_file in sample_data_files:
        if Path(data_file).exists():
            print(f"   ✅ Sample data found: {data_file}")
        else:
            print(f"   ⚠️  No sample data: {data_file}")
    
    return all_valid

def validate_strategies() -> bool:
    """Validate strategy implementations."""
    all_valid = True
    
    strategies_dir = Path('src/strategies')
    
    if not strategies_dir.exists():
        print("   ❌ Strategies directory not found")
        return False
    
    # Check base strategy
    base_strategy_file = strategies_dir / 'base_strategy.py'
    if base_strategy_file.exists():
        print("   ✅ Base strategy found")
    else:
        print("   ❌ Base strategy missing")
        all_valid = False
    
    # Find strategy implementations
    strategy_dirs = [
        d for d in strategies_dir.iterdir() 
        if (d.is_dir() and 
            not d.name.startswith('_') and 
            d.name != '__pycache__')
    ]
    
    if not strategy_dirs:
        print("   ⚠️  No strategy implementations found")
        return all_valid
    
    print(f"   📈 Found {len(strategy_dirs)} strategy implementations:")
    
    for strategy_dir in strategy_dirs:
        strategy_name = strategy_dir.name
        
        # Check for required files
        required_files = ['__init__.py', 'signal.py']
        strategy_valid = True
        
        for req_file in required_files:
            file_path = strategy_dir / req_file
            if file_path.exists():
                print(f"      ✅ {strategy_name}/{req_file}")
            else:
                print(f"      ❌ {strategy_name}/{req_file} missing")
                strategy_valid = False
        
        # Check for config file
        config_file = Path(f'config/strategies/{strategy_name}.yaml')
        if config_file.exists():
            print(f"      ✅ {strategy_name} config found")
            
            # Validate the config
            if not validate_config_file(str(config_file)):
                strategy_valid = False
        else:
            print(f"      ⚠️  {strategy_name} config missing")
        
        if not strategy_valid:
            all_valid = False
    
    return all_valid

def validate_dependencies() -> bool:
    """Validate Python dependencies."""
    all_valid = True
    
    # Core dependencies
    core_deps = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scipy',
        'sklearn'
    ]
    
    print("   📦 Checking core dependencies:")
    for dep in core_deps:
        try:
            __import__(dep)
            print(f"      ✅ {dep}")
        except ImportError:
            print(f"      ❌ {dep} not installed")
            all_valid = False
    
    # Optional dependencies
    optional_deps = [
        ('streamlit', 'monitoring dashboard'),
        ('plotly', 'interactive charts'),
        ('numba', 'performance optimization'),
        ('yfinance', 'data fetching'),
        ('ccxt', 'exchange connectivity')
    ]
    
    print("   📦 Checking optional dependencies:")
    for dep, purpose in optional_deps:
        try:
            __import__(dep)
            print(f"      ✅ {dep} ({purpose})")
        except ImportError:
            print(f"      ⚠️  {dep} not installed ({purpose})")
    
    return all_valid

def validate_specific_strategy(strategy_name: str) -> bool:
    """Validate a specific strategy implementation."""
    all_valid = True
    
    strategies_dir = Path('src/strategies')
    strategy_dir = strategies_dir / strategy_name
    
    if not strategy_dir.exists():
        print(f"   ❌ Strategy directory not found: {strategy_name}")
        return False
    
    print(f"   📂 Found strategy directory: {strategy_name}")
    
    # Check for required files
    required_files = ['__init__.py', 'signal.py']
    for req_file in required_files:
        file_path = strategy_dir / req_file
        if file_path.exists():
            print(f"   ✅ {strategy_name}/{req_file}")
        else:
            print(f"   ❌ {strategy_name}/{req_file} missing")
            all_valid = False
    
    # Check for config file
    config_file = Path(f'config/strategies/{strategy_name}.yaml')
    if config_file.exists():
        print(f"   ✅ {strategy_name} config found")
        
        # Validate the config
        if not validate_config_file(str(config_file)):
            all_valid = False
    else:
        print(f"   ⚠️  {strategy_name} config missing")
    
    # Try to import the strategy
    try:
        import sys
        import importlib
        sys.path.append('src')
        strategy_module = importlib.import_module(f'strategies.{strategy_name}.signal')
        print(f"   ✅ Successfully imported {strategy_name}")
        
        # Check for signal class (try different naming conventions)
        expected_class_names = [
            f"{''.join(word.capitalize() for word in strategy_name.split('_'))}Signal",
            f"{strategy_name.title().replace('_', '')}Signal",
            f"{strategy_name.upper()}Signal",
            "Signal"
        ]
        
        found_class = None
        for class_name in expected_class_names:
            if hasattr(strategy_module, class_name):
                found_class = class_name
                break
        
        if found_class:
            print(f"   ✅ Found signal class: {found_class}")
        else:
            print(f"   ⚠️  Signal class not found (tried: {expected_class_names})")
            # Don't fail validation for missing class names - the module imported successfully
            
    except ImportError as e:
        print(f"   ❌ Failed to import {strategy_name}: {e}")
        all_valid = False
    except Exception as e:
        print(f"   ❌ Error validating {strategy_name}: {e}")
        all_valid = False
    
    return all_valid
