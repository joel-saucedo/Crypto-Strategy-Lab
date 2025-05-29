"""
CLI module for live trading functionality.
"""

import sys
import os
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_live_trading(args):
    """
    Run live trading with real money.
    """
    print("=" * 60)
    print("⚠️  CRYPTO STRATEGY LAB - LIVE TRADING")
    print("=" * 60)
    
    print("🚨 DANGER: LIVE TRADING WITH REAL MONEY")
    print("=" * 60)
    
    if not args.confirm:
        print("❌ Live trading requires explicit confirmation")
        print("   Use --confirm flag to acknowledge the risks")
        print()
        print("⚠️  RISKS:")
        print("   • You can lose all your money")
        print("   • Markets are volatile and unpredictable")
        print("   • Software bugs can cause significant losses")
        print("   • Network issues can prevent order management")
        print("   • Exchange failures can lock your funds")
        print()
        print("💡 Consider paper trading first to test your strategy")
        return 1
    
    print(f"💰 Configuration:")
    print(f"   Strategy: {args.strategy}")
    print(f"   Starting Capital: ${args.capital:,.2f}")
    print(f"   Mode: LIVE TRADING (REAL MONEY)")
    print()
    
    # Additional safety check
    print("🛡️  FINAL SAFETY CHECK")
    print("   Type 'I UNDERSTAND THE RISKS' to proceed:")
    
    try:
        confirmation = input("   > ")
    except (EOFError, KeyboardInterrupt):
        print("\n❌ Live trading cancelled")
        return 1
    
    if confirmation.strip() != "I UNDERSTAND THE RISKS":
        print("❌ Incorrect confirmation. Live trading cancelled for safety.")
        return 1
    
    print()
    print("🚀 LIVE TRADING INITIALIZATION")
    print("=" * 60)
    print()
    print("⚠️  LIVE TRADING NOT FULLY IMPLEMENTED FOR SAFETY")
    print()
    print("📝 For complete live trading implementation, you need:")
    print("   1. ✅ Complete comprehensive testing with paper trading")
    print("   2. ✅ Implement proper risk management systems")
    print("   3. ❌ Add exchange connectivity and order management")
    print("   4. ❌ Implement monitoring and alerting systems")
    print("   5. ❌ Add circuit breakers and kill switches")
    print("   6. ❌ Complete legal and compliance requirements")
    print("   7. ❌ Set up proper API keys and authentication")
    print("   8. ❌ Implement position sizing and risk controls")
    print()
    print("💡 Continue with paper trading to test your strategies safely:")
    print(f"   python -m src.cli paper --strategy {args.strategy} --capital {args.capital}")
    print()
    print("🔧 When ready for live trading:")
    print("   1. Complete all implementation requirements above")
    print("   2. Remove safety checks in this module")
    print("   3. Add proper exchange integration")
    print("   4. Set up monitoring and alerting")
    print()
    
    return 1  # Return error to prevent accidental live trading
