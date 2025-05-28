"""
Crypto Strategy Lab - Multi-Exchange Trading Framework
A production-ready quantitative trading system with 12 orthogonal strategies.
"""

__version__ = "2.0.0"
__author__ = "Joel Saucedo"
__email__ = "joel@crypto-strategy-lab.com"

# Make src a proper package
import sys
from pathlib import Path

# Add src to Python path
_src_path = Path(__file__).parent / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))
