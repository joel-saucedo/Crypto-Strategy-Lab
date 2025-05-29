#!/usr/bin/env python3
"""
Main entry point for executing the crypto-strategy-lab package as a module.
This allows running: python -m src
"""

import sys
import asyncio
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import directly from main_cli module
from main_cli import main

if __name__ == '__main__':
    asyncio.run(main())
