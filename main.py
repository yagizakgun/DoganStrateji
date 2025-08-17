#!/usr/bin/env python3
"""
Binance Futures Trading Bot
===========================

A trading bot that implements a confluence-based strategy using:
- Price action analysis (Higher Highs / Lower Lows)
- Cumulative Volume Delta (CVD) divergence detection
- Funding rate extremes
- Open Interest trends

Usage:
    python main.py [--status] [--stop] [--force-signal] [--signal-table] [--close-all] [--reset-cvd]

Environment Variables:
    BINANCE_API_KEY     : Your Binance API key
    BINANCE_API_SECRET  : Your Binance API secret
    BINANCE_TESTNET     : Set to 'true' for testnet (default: true)
    TRADING_SYMBOL      : Trading symbol (default: BTCUSDT)

Configuration:
    Edit config/settings.json for detailed configuration options.
"""

import sys
import signal
import argparse
import time
import json
from datetime import datetime

from core.trading_bot import get_bot
from config.config import config
from utils.logger import get_logger

def signal_handler(signum, frame):
    """Handle termination signals gracefully"""
    from utils.ui_components import Colors, Emojis
    
    emojis = Emojis()
    print(f"\n{Colors.BRIGHT_YELLOW}{emojis.get('warning', '⚠️')} Received signal {signum}, shutting down...{Colors.RESET}")
    bot = get_bot()
    if bot.is_running:
        print(f"{Colors.BRIGHT_RED}{emojis.get('gear', '⚙️')} Emergency stop initiated...{Colors.RESET}")
        bot.emergency_stop()
    sys.exit(0)

def print_banner():
    """Print enhanced bot banner with colors and animations"""
    from utils.ui_components import Colors, Emojis, Animation, Box
    
    emojis = Emojis()
    
    # Animated loading for dramatic effect
    print(f"\n{Colors.BRIGHT_CYAN}Starting Binance Futures Trading Bot...{Colors.RESET}")
    Animation.loading_dots(1.5, f"{Colors.BRIGHT_BLUE}Initializing systems")
    
    # Main banner with colors (safe for all terminals)
    try:
        rocket_emoji = emojis.get('rocket', '[BOT]')
        chart_emoji = emojis.get('chart', '>')
        shield_emoji = emojis.get('shield', '>')
        clock_emoji = emojis.get('clock', '>')
        warning_emoji = emojis.get('warning', '!')
        
        banner = f"""
{Colors.BRIGHT_CYAN}{'='*80}{Colors.RESET}
{Colors.BOLD}{Colors.BRIGHT_WHITE}                    {rocket_emoji} BINANCE FUTURES TRADING BOT {rocket_emoji}                    {Colors.RESET}
{Colors.BRIGHT_YELLOW}                         CVD Divergence Strategy                               {Colors.RESET}
{Colors.BRIGHT_CYAN}{'='*80}{Colors.RESET}

{Colors.BRIGHT_GREEN}  {chart_emoji} Strategy:{Colors.RESET} Price Action + CVD Divergence + Funding Rate Confluence           
{Colors.BRIGHT_BLUE}  {shield_emoji} Risk Management:{Colors.RESET} Fixed % per trade with dynamic position sizing             
{Colors.BRIGHT_MAGENTA}  {clock_emoji} Timeframe:{Colors.RESET} Configurable (default: 15m)                                      
                                                                               
{Colors.BRIGHT_RED}  {warning_emoji} WARNING:{Colors.RESET} {Colors.YELLOW}This bot trades with real money on live markets!{Colors.RESET}               
{Colors.BRIGHT_YELLOW}      Use testnet mode for testing and paper trading.{Colors.RESET}                         
{Colors.BRIGHT_CYAN}{'='*80}{Colors.RESET}
"""
        print(banner)
    except UnicodeEncodeError:
        # Fallback banner without emojis
        banner = f"""
{Colors.BRIGHT_CYAN}{'='*80}{Colors.RESET}
{Colors.BOLD}{Colors.BRIGHT_WHITE}                       BINANCE FUTURES TRADING BOT                        {Colors.RESET}
{Colors.BRIGHT_YELLOW}                         CVD Divergence Strategy                               {Colors.RESET}
{Colors.BRIGHT_CYAN}{'='*80}{Colors.RESET}

{Colors.BRIGHT_GREEN}  > Strategy:{Colors.RESET} Price Action + CVD Divergence + Funding Rate Confluence           
{Colors.BRIGHT_BLUE}  > Risk Management:{Colors.RESET} Fixed % per trade with dynamic position sizing             
{Colors.BRIGHT_MAGENTA}  > Timeframe:{Colors.RESET} Configurable (default: 15m)                                      
                                                                               
{Colors.BRIGHT_RED}  ! WARNING:{Colors.RESET} {Colors.YELLOW}This bot trades with real money on live markets!{Colors.RESET}               
{Colors.BRIGHT_YELLOW}      Use testnet mode for testing and paper trading.{Colors.RESET}                         
{Colors.BRIGHT_CYAN}{'='*80}{Colors.RESET}
"""
        print(banner)

def print_configuration():
    """Print enhanced configuration with colors and layout"""
    from utils.ui_components import Colors, Box
    
    try:
        # Create configuration display
        config_lines = [
            f"{Colors.BRIGHT_WHITE}Trading Configuration:{Colors.RESET}",
            f"  {Colors.BRIGHT_GREEN}Symbol:{Colors.RESET} {config.symbol}",
            f"  {Colors.BRIGHT_BLUE}Timeframe:{Colors.RESET} {config.timeframe}", 
            f"  {Colors.BRIGHT_MAGENTA}Risk per Trade:{Colors.RESET} {config.risk_per_trade*100:.1f}%",
            f"  {Colors.BRIGHT_YELLOW}Risk/Reward Ratio:{Colors.RESET} 1:{config.risk_reward_ratio}",
            "",
            f"{Colors.BRIGHT_WHITE}Environment:{Colors.RESET}",
            f"  {Colors.BRIGHT_CYAN}Testnet:{Colors.RESET} {'Yes' if config.testnet else 'No'}",
            f"  {Colors.BRIGHT_RED if not config.paper_trading_enabled else Colors.BRIGHT_GREEN}Paper Trading:{Colors.RESET} {'Yes' if config.paper_trading_enabled else 'No'}",
            f"  {Colors.BRIGHT_BLUE}Verbose Console:{Colors.RESET} {'Yes' if config.verbose_console else 'No'}",
        ]
        
        print("\n".join(config_lines))
        print(f"\n{Colors.BRIGHT_CYAN}{'='*80}{Colors.RESET}\n")
        
    except Exception as e:
        # Fallback configuration display
        print(f"\nTrading Configuration:")
        print(f"  Symbol: {config.symbol}")
        print(f"  Timeframe: {config.timeframe}")
        print(f"  Risk per Trade: {config.risk_per_trade*100:.1f}%")
        print(f"  Risk/Reward Ratio: 1:{config.risk_reward_ratio}")
        print(f"  Testnet: {'Yes' if config.testnet else 'No'}")
        print(f"  Paper Trading: {'Yes' if config.paper_trading_enabled else 'No'}")
        print("\n" + "="*80 + "\n")

def main():
    """Main entry point"""
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Binance Futures Trading Bot')
    parser.add_argument('--status', action='store_true', help='Show bot status and exit')
    parser.add_argument('--stop', action='store_true', help='Stop the bot')
    parser.add_argument('--force-signal', action='store_true', help='Force signal check')
    parser.add_argument('--signal-table', action='store_true', help='Show signal analysis table')
    parser.add_argument('--close-all', action='store_true', help='Close all positions')
    parser.add_argument('--reset-cvd', action='store_true', help='Reset CVD calculator')
    
    args = parser.parse_args()
    
    # Initialize logger early
    logger = get_logger(config)
    
    try:
        # Get bot instance
        bot = get_bot()
        
        # Handle command line arguments
        if args.status:
            bot.show_status()
            return
        
        if args.stop:
            bot.stop()
            return
            
        if args.force_signal:
            bot.force_signal_check()
            return
            
        if args.signal_table:
            bot.show_signal_table()
            return
            
        if args.close_all:
            bot.close_all_positions()
            return
            
        if args.reset_cvd:
            bot.reset_cvd()
            return
        
        # Print banner and configuration
        print_banner()
        print_configuration()
        
        # Initialize and start bot
        if bot.initialize():
            logger.info("Bot initialized successfully. Starting main trading loop...")
            bot.run()
        else:
            logger.error("Failed to initialize bot")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        if 'bot' in locals():
            bot.stop()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if 'bot' in locals():
            bot.emergency_stop()
        sys.exit(1)

if __name__ == "__main__":
    main()