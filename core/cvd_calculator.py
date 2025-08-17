import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

from utils.logger import TradingBotLogger
from utils.helpers import timeframe_to_minutes, find_pivot_highs, find_pivot_lows

class CVDCalculator:
    """
    Cumulative Volume Delta (CVD) Calculator
    
    Processes real-time trade data to calculate CVD based on buyer/seller aggressor logic:
    - If isBuyerMaker = True: Seller was aggressor (subtract quantity)
    - If isBuyerMaker = False: Buyer was aggressor (add quantity)
    """
    
    def __init__(self, config, logger: TradingBotLogger):
        self.config = config
        self.logger = logger
        self.symbol = config.symbol
        self.timeframe = config.timeframe
        self.timeframe_minutes = timeframe_to_minutes(self.timeframe)
        
        # CVD data storage
        self.current_candle_delta = 0.0
        self.current_candle_start_time = None
        self.cvd_history = deque(maxlen=1000)  # Store last 1000 candles
        self.cumulative_cvd = 0.0
        
        # Temporary storage for current candle
        self.current_candle_data = {
            'open_time': None,
            'close_time': None,
            'delta': 0.0,
            'buy_volume': 0.0,
            'sell_volume': 0.0,
            'total_volume': 0.0,
            'trade_count': 0
        }
        
        # Pivot detection for divergence analysis
        self.cvd_pivots_high = deque(maxlen=50)
        self.cvd_pivots_low = deque(maxlen=50)
        self.lookback_period = config.cvd_lookback
        
        self.logger.info(f"CVD Calculator initialized for {self.symbol} {self.timeframe}")
    
    def process_trade(self, trade_data: Dict[str, Any]):
        """
        Process incoming trade data and update CVD
        
        Args:
            trade_data: Dictionary containing trade information
                - price: float
                - quantity: float
                - timestamp: int (milliseconds)
                - is_buyer_maker: bool
                - symbol: str
        """
        try:
            timestamp = trade_data['timestamp']
            quantity = trade_data['quantity']
            is_buyer_maker = trade_data['is_buyer_maker']
            
            # Calculate candle boundaries
            candle_start_time = self._get_candle_start_time(timestamp)
            
            # Check if we need to close the current candle
            if (self.current_candle_start_time is not None and 
                candle_start_time != self.current_candle_start_time):
                self._close_current_candle()
            
            # Initialize new candle if needed
            if self.current_candle_start_time != candle_start_time:
                self._initialize_new_candle(candle_start_time)
            
            # Update current candle data
            self.current_candle_data['trade_count'] += 1
            self.current_candle_data['total_volume'] += quantity
            
            # Calculate delta based on aggressor
            if is_buyer_maker:
                # Seller was aggressor - subtract from delta
                delta = -quantity
                self.current_candle_data['sell_volume'] += quantity
            else:
                # Buyer was aggressor - add to delta
                delta = quantity
                self.current_candle_data['buy_volume'] += quantity
            
            # Update current candle delta
            self.current_candle_data['delta'] += delta
            self.current_candle_delta += delta
            
            self.logger.debug(f"Trade processed: {quantity} @ {trade_data['price']}, "
                            f"Delta: {delta}, Cumulative: {self.current_candle_delta}")
            
        except Exception as e:
            self.logger.error(f"Error processing trade: {e}")
    
    def _get_candle_start_time(self, timestamp: int) -> int:
        """Calculate the start time of the candle for given timestamp"""
        timestamp_seconds = timestamp // 1000
        timeframe_seconds = self.timeframe_minutes * 60
        
        # Round down to nearest timeframe boundary
        candle_start_seconds = (timestamp_seconds // timeframe_seconds) * timeframe_seconds
        return candle_start_seconds * 1000
    
    def _initialize_new_candle(self, candle_start_time: int):
        """Initialize a new candle"""
        self.current_candle_start_time = candle_start_time
        self.current_candle_delta = 0.0
        
        self.current_candle_data = {
            'open_time': candle_start_time,
            'close_time': candle_start_time + (self.timeframe_minutes * 60 * 1000) - 1,
            'delta': 0.0,
            'buy_volume': 0.0,
            'sell_volume': 0.0,
            'total_volume': 0.0,
            'trade_count': 0
        }
        
        self.logger.debug(f"New candle initialized: {datetime.fromtimestamp(candle_start_time/1000)}")
    
    def _close_current_candle(self):
        """Close the current candle and update CVD history"""
        if self.current_candle_start_time is None:
            return
        
        # Add current candle delta to cumulative CVD
        self.cumulative_cvd += self.current_candle_delta
        
        # Create candle record
        candle_record = {
            'timestamp': self.current_candle_start_time,
            'delta': self.current_candle_delta,
            'cumulative_cvd': self.cumulative_cvd,
            'buy_volume': self.current_candle_data['buy_volume'],
            'sell_volume': self.current_candle_data['sell_volume'],
            'total_volume': self.current_candle_data['total_volume'],
            'trade_count': self.current_candle_data['trade_count'],
            'buy_ratio': (self.current_candle_data['buy_volume'] / 
                         max(self.current_candle_data['total_volume'], 1))
        }
        
        # Add to history
        self.cvd_history.append(candle_record)
        
        # Update pivot detection
        self._update_pivot_detection()
        
        self.logger.info(f"Candle closed: Delta={self.current_candle_delta:.2f}, "
                        f"CVD={self.cumulative_cvd:.2f}, "
                        f"Trades={self.current_candle_data['trade_count']}")
    
    def _update_pivot_detection(self):
        """Update pivot high/low detection for CVD"""
        if len(self.cvd_history) < self.lookback_period * 2:
            return
        
        # Get recent CVD values
        recent_cvd = [candle['cumulative_cvd'] for candle in list(self.cvd_history)[-50:]]
        cvd_series = pd.Series(recent_cvd)
        
        # Find pivot highs and lows
        pivot_highs = find_pivot_highs(cvd_series, self.lookback_period // 2)
        pivot_lows = find_pivot_lows(cvd_series, self.lookback_period // 2)
        
        # Update pivot storage (store with actual timestamps and values)
        for idx in pivot_highs:
            if idx < len(self.cvd_history):
                candle = list(self.cvd_history)[-(len(recent_cvd) - idx)]
                self.cvd_pivots_high.append({
                    'timestamp': candle['timestamp'],
                    'cvd_value': candle['cumulative_cvd'],
                    'index': len(self.cvd_history) - (len(recent_cvd) - idx)
                })
        
        for idx in pivot_lows:
            if idx < len(self.cvd_history):
                candle = list(self.cvd_history)[-(len(recent_cvd) - idx)]
                self.cvd_pivots_low.append({
                    'timestamp': candle['timestamp'],
                    'cvd_value': candle['cumulative_cvd'],
                    'index': len(self.cvd_history) - (len(recent_cvd) - idx)
                })
    
    def get_current_cvd(self) -> float:
        """Get current cumulative CVD including ongoing candle"""
        return self.cumulative_cvd + self.current_candle_delta
    
    def get_cvd_series(self, length: int = 100) -> pd.DataFrame:
        """Get CVD history as pandas DataFrame"""
        if not self.cvd_history:
            return pd.DataFrame()
        
        # Get last N candles
        recent_candles = list(self.cvd_history)[-length:]
        
        df = pd.DataFrame(recent_candles)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_pivot_highs(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent CVD pivot highs"""
        return list(self.cvd_pivots_high)[-count:] if self.cvd_pivots_high else []
    
    def get_pivot_lows(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent CVD pivot lows"""
        return list(self.cvd_pivots_low)[-count:] if self.cvd_pivots_low else []
    
    def detect_cvd_divergence(self, price_pivots: List[Tuple[int, float]], 
                             divergence_type: str = 'bearish') -> bool:
        """
        Detect divergence between price and CVD
        
        Args:
            price_pivots: List of (timestamp, price) tuples for price pivots
            divergence_type: 'bearish' or 'bullish'
        
        Returns:
            bool: True if divergence detected
        """
        try:
            if divergence_type == 'bearish':
                cvd_pivots = self.get_pivot_highs(5)
            else:
                cvd_pivots = self.get_pivot_lows(5)
            
            if len(price_pivots) < 2 or len(cvd_pivots) < 2:
                return False
            
            # Get the two most recent pivots
            price_pivot1, price_pivot2 = price_pivots[-2], price_pivots[-1]
            cvd_pivot1, cvd_pivot2 = cvd_pivots[-2], cvd_pivots[-1]
            
            if divergence_type == 'bearish':
                # Price makes higher high, CVD makes lower high
                price_higher_high = price_pivot2[1] > price_pivot1[1]
                cvd_lower_high = cvd_pivot2['cvd_value'] < cvd_pivot1['cvd_value']
                divergence = price_higher_high and cvd_lower_high
            else:
                # Price makes lower low, CVD makes higher low
                price_lower_low = price_pivot2[1] < price_pivot1[1]
                cvd_higher_low = cvd_pivot2['cvd_value'] > cvd_pivot1['cvd_value']
                divergence = price_lower_low and cvd_higher_low
            
            if divergence:
                self.logger.info(f"CVD {divergence_type} divergence detected!")
                self.logger.info(f"Price: {price_pivot1[1]} -> {price_pivot2[1]}")
                self.logger.info(f"CVD: {cvd_pivot1['cvd_value']:.2f} -> {cvd_pivot2['cvd_value']:.2f}")
            
            return divergence
            
        except Exception as e:
            self.logger.error(f"Error detecting CVD divergence: {e}")
            return False
    
    def get_current_candle_stats(self) -> Dict[str, Any]:
        """Get current candle statistics"""
        total_volume = max(self.current_candle_data['total_volume'], 1)
        
        return {
            'delta': self.current_candle_delta,
            'buy_volume': self.current_candle_data['buy_volume'],
            'sell_volume': self.current_candle_data['sell_volume'],
            'total_volume': total_volume,
            'buy_ratio': self.current_candle_data['buy_volume'] / total_volume,
            'sell_ratio': self.current_candle_data['sell_volume'] / total_volume,
            'trade_count': self.current_candle_data['trade_count'],
            'cumulative_cvd': self.get_current_cvd()
        }
    
    def reset(self):
        """Reset CVD calculator"""
        self.current_candle_delta = 0.0
        self.current_candle_start_time = None
        self.cvd_history.clear()
        self.cumulative_cvd = 0.0
        self.cvd_pivots_high.clear()
        self.cvd_pivots_low.clear()
        
        self.logger.info("CVD Calculator reset")
    
    def get_cvd_strength(self) -> str:
        """Get CVD strength assessment"""
        if len(self.cvd_history) < 5:
            return "insufficient_data"
        
        recent_deltas = [candle['delta'] for candle in list(self.cvd_history)[-5:]]
        avg_delta = sum(recent_deltas) / len(recent_deltas)
        
        if avg_delta > 1000:
            return "strong_bullish"
        elif avg_delta > 100:
            return "bullish"
        elif avg_delta < -1000:
            return "strong_bearish"
        elif avg_delta < -100:
            return "bearish"
        else:
            return "neutral"