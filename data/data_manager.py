import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import time

from utils.logger import TradingBotLogger
from utils.helpers import timeframe_to_minutes, safe_float, safe_int, format_timestamp

class DataManager:
    def __init__(self, config, logger: TradingBotLogger):
        self.config = config
        self.logger = logger
        self.client = None
        self._initialize_client()
        
        # Cache for storing recent data
        self.klines_cache = {}
        self.funding_rate_cache = {}
        self.open_interest_cache = {}
        
    def _initialize_client(self):
        """Initialize Binance client"""
        try:
            if self.config.testnet:
                self.client = Client(
                    api_key=self.config.api_key,
                    api_secret=self.config.api_secret,
                    testnet=True
                )
                self.logger.info("Connected to Binance Testnet")
            else:
                self.client = Client(
                    api_key=self.config.api_key,
                    api_secret=self.config.api_secret
                )
                self.logger.info("Connected to Binance Live")
                
            # Test connection
            self.client.ping()
            account_info = self.client.futures_account()
            self.logger.info(f"Account balance: {account_info.get('totalWalletBalance', 'N/A')} USDT")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance client: {str(e)}")
            raise
    
    def get_historical_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """Fetch historical klines data"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{interval}_{limit}"
            current_time = datetime.now()
            
            if cache_key in self.klines_cache:
                cached_data, cache_time = self.klines_cache[cache_key]
                if (current_time - cache_time).seconds < 60:  # Cache for 1 minute
                    return cached_data
            
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to proper data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Cache the result
            self.klines_cache[cache_key] = (df, current_time)
            
            self.logger.debug(f"Fetched {len(df)} klines for {symbol} {interval}")
            return df
            
        except Exception as e:
            self.logger.log_api_error(f"get_historical_klines({symbol}, {interval})", e)
            raise
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            price = safe_float(ticker['price'])
            self.logger.debug(f"Current price for {symbol}: {price}")
            return price
        except Exception as e:
            self.logger.log_api_error(f"get_current_price({symbol})", e)
            raise
    
    def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """Get current funding rate"""
        try:
            # Check cache first
            current_time = datetime.now()
            if symbol in self.funding_rate_cache:
                cached_data, cache_time = self.funding_rate_cache[symbol]
                if (current_time - cache_time).seconds < 300:  # Cache for 5 minutes
                    return cached_data
            
            funding_info = self.client.futures_funding_rate(symbol=symbol, limit=1)
            
            if funding_info:
                latest = funding_info[0]
                result = {
                    'symbol': symbol,
                    'funding_rate': safe_float(latest['fundingRate']),
                    'funding_time': latest['fundingTime'],
                    'mark_price': safe_float(latest.get('markPrice', 0))
                }
                
                # Cache the result
                self.funding_rate_cache[symbol] = (result, current_time)
                
                self.logger.debug(f"Funding rate for {symbol}: {result['funding_rate']}")
                return result
            
            return {'symbol': symbol, 'funding_rate': 0, 'funding_time': 0, 'mark_price': 0}
            
        except Exception as e:
            self.logger.log_api_error(f"get_funding_rate({symbol})", e)
            raise
    
    def get_open_interest(self, symbol: str) -> Dict[str, Any]:
        """Get current open interest"""
        try:
            # Check cache first
            current_time = datetime.now()
            if symbol in self.open_interest_cache:
                cached_data, cache_time = self.open_interest_cache[symbol]
                if (current_time - cache_time).seconds < 60:  # Cache for 1 minute
                    return cached_data
            
            oi_info = self.client.futures_open_interest(symbol=symbol)
            
            result = {
                'symbol': symbol,
                'open_interest': safe_float(oi_info['openInterest']),
                'timestamp': safe_int(oi_info['time'])
            }
            
            # Cache the result
            self.open_interest_cache[symbol] = (result, current_time)
            
            self.logger.debug(f"Open Interest for {symbol}: {result['open_interest']}")
            return result
            
        except Exception as e:
            self.logger.log_api_error(f"get_open_interest({symbol})", e)
            raise
    
    def get_account_balance(self) -> float:
        """Get account balance in USDT"""
        try:
            account_info = self.client.futures_account()
            balance = safe_float(account_info.get('totalWalletBalance', 0))
            self.logger.debug(f"Account balance: {balance} USDT")
            return balance
        except Exception as e:
            self.logger.log_api_error("get_account_balance", e)
            raise
    
    def get_position_info(self, symbol: str) -> Dict[str, Any]:
        """Get current position information"""
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            
            if positions:
                position = positions[0]
                return {
                    'symbol': symbol,
                    'position_amount': safe_float(position['positionAmt']),
                    'entry_price': safe_float(position['entryPrice']),
                    'mark_price': safe_float(position['markPrice']),
                    'pnl': safe_float(position['unRealizedProfit']),
                    'percentage': safe_float(position['percentage'])
                }
            
            return {
                'symbol': symbol,
                'position_amount': 0,
                'entry_price': 0,
                'mark_price': 0,
                'pnl': 0,
                'percentage': 0
            }
            
        except Exception as e:
            self.logger.log_api_error(f"get_position_info({symbol})", e)
            raise
    
    def get_historical_funding_rates(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get historical funding rates"""
        try:
            funding_history = self.client.futures_funding_rate(symbol=symbol, limit=limit)
            
            df = pd.DataFrame(funding_history)
            df['funding_rate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
            df['funding_time'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df.set_index('funding_time', inplace=True)
            
            self.logger.debug(f"Fetched {len(df)} historical funding rates for {symbol}")
            return df
            
        except Exception as e:
            self.logger.log_api_error(f"get_historical_funding_rates({symbol})", e)
            raise
    
    def get_historical_open_interest(self, symbol: str, period: str = "5m", limit: int = 100) -> pd.DataFrame:
        """Get historical open interest data"""
        try:
            # Note: Binance doesn't provide historical OI directly
            # This is a placeholder for future implementation
            # You might need to collect this data over time or use a third-party service
            
            current_oi = self.get_open_interest(symbol)
            
            # Create a simple DataFrame with current data
            df = pd.DataFrame([current_oi])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.log_api_error(f"get_historical_open_interest({symbol})", e)
            raise
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists and is tradeable"""
        try:
            exchange_info = self.client.futures_exchange_info()
            symbols = [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING']
            return symbol in symbols
        except Exception as e:
            self.logger.log_api_error(f"validate_symbol({symbol})", e)
            return False
    
    def get_server_time(self) -> int:
        """Get server time"""
        try:
            return self.client.get_server_time()['serverTime']
        except Exception as e:
            self.logger.log_api_error("get_server_time", e)
            return int(time.time() * 1000)
    
    def clear_cache(self):
        """Clear all cached data"""
        self.klines_cache.clear()
        self.funding_rate_cache.clear()
        self.open_interest_cache.clear()
        self.logger.debug("Data cache cleared")