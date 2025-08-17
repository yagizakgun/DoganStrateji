import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta

from utils.logger import TradingBotLogger
from utils.helpers import (
    find_pivot_highs, find_pivot_lows, detect_divergence, 
    is_funding_rate_extreme, safe_float
)
from core.cvd_calculator import CVDCalculator
from data.data_manager import DataManager

class SignalDetector:
    """
    Detects trading signals based on confluence of:
    1. Price action (HH/LL)
    2. CVD divergence
    3. Funding rate extremes
    4. Open Interest trends
    """
    
    def __init__(self, config, logger: TradingBotLogger, 
                 data_manager: DataManager, cvd_calculator: CVDCalculator):
        self.config = config
        self.logger = logger
        self.data_manager = data_manager
        self.cvd_calculator = cvd_calculator
        
        # Signal parameters
        self.symbol = config.symbol
        self.lookback_period = config.lookback_period
        self.high_funding_threshold = config.high_funding_threshold
        self.low_funding_threshold = config.low_funding_threshold
        self.confirmation_period = config.get('signals.confirmation_period', 3)
        
        # Price data cache
        self.price_data = pd.DataFrame()
        self.last_update = None
        
        # Pivot tracking
        self.price_pivots_high = []
        self.price_pivots_low = []
        
        self.logger.info("SignalDetector initialized")
    
    def update_price_data(self):
        """Update price data and detect new pivots"""
        try:
            # Fetch latest price data
            current_time = datetime.now()
            
            # Update every minute or if no data exists
            if (self.last_update is None or 
                (current_time - self.last_update).seconds >= 60):
                
                new_data = self.data_manager.get_historical_klines(
                    self.symbol, 
                    self.config.timeframe, 
                    limit=self.lookback_period + 20
                )
                
                self.price_data = new_data
                self.last_update = current_time
                
                # Update pivot detection
                self._update_price_pivots()
                
                self.logger.debug(f"Price data updated: {len(self.price_data)} candles")
        
        except Exception as e:
            self.logger.error(f"Failed to update price data: {e}")
    
    def _update_price_pivots(self):
        """Update price pivot detection"""
        if len(self.price_data) < self.lookback_period:
            return
        
        # Find pivot highs and lows
        highs = self.price_data['high']
        lows = self.price_data['low']
        
        pivot_high_indices = find_pivot_highs(highs, self.lookback_period // 2)
        pivot_low_indices = find_pivot_lows(lows, self.lookback_period // 2)
        
        # Store pivots with timestamps and values
        self.price_pivots_high = []
        for idx in pivot_high_indices:
            if idx < len(self.price_data):
                timestamp = self.price_data.index[idx]
                price = self.price_data.iloc[idx]['high']
                self.price_pivots_high.append((int(timestamp.timestamp() * 1000), price))
        
        self.price_pivots_low = []
        for idx in pivot_low_indices:
            if idx < len(self.price_data):
                timestamp = self.price_data.index[idx]
                price = self.price_data.iloc[idx]['low']
                self.price_pivots_low.append((int(timestamp.timestamp() * 1000), price))
        
        # Keep only recent pivots
        self.price_pivots_high = self.price_pivots_high[-10:]
        self.price_pivots_low = self.price_pivots_low[-10:]
    
    def check_bearish_signal(self) -> Dict[str, Any]:
        """
        Check for bearish (short) entry signal
        
        Conditions:
        1. Price makes Higher High (HH)
        2. CVD makes Lower High (LH) - divergence
        3. High funding rate (positive, suggesting over-leverage)
        4. Optional: Rising Open Interest with weak longs
        """
        signal = {
            'type': 'bearish',
            'valid': False,
            'confidence': 0,
            'conditions': {},
            'details': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Update data
            self.update_price_data()
            
            # 1. Check for Price Higher High
            price_hh = self._check_price_higher_high()
            signal['conditions']['price_higher_high'] = price_hh['valid']
            signal['details']['price_analysis'] = price_hh
            
            if not price_hh['valid']:
                return signal
            
            # 2. Check for CVD Divergence (Lower High)
            cvd_divergence = self.cvd_calculator.detect_cvd_divergence(
                self.price_pivots_high, 'bearish'
            )
            signal['conditions']['cvd_divergence'] = cvd_divergence
            
            if not cvd_divergence:
                return signal
            
            # 3. Check Funding Rate
            funding_analysis = self._check_funding_rate('high')
            signal['conditions']['funding_rate_extreme'] = funding_analysis['extreme']
            signal['details']['funding_analysis'] = funding_analysis
            
            # 4. Check Open Interest (contextual)
            oi_analysis = self._check_open_interest_trend()
            signal['conditions']['open_interest_rising'] = oi_analysis.get('rising', False)
            signal['details']['oi_analysis'] = oi_analysis
            
            # 5. Check confirmation
            confirmation = self._check_price_confirmation('bearish')
            signal['conditions']['confirmation'] = confirmation['valid']
            signal['details']['confirmation'] = confirmation
            
            # Calculate confidence score
            confidence_score = 0
            if price_hh['valid']:
                confidence_score += 25
            if cvd_divergence:
                confidence_score += 35  # Primary signal
            if funding_analysis['extreme']:
                confidence_score += 20
            if oi_analysis.get('rising', False):
                confidence_score += 10
            if confirmation['valid']:
                confidence_score += 10
            
            signal['confidence'] = confidence_score
            
            # Signal is valid if we have the core conditions
            signal['valid'] = (
                price_hh['valid'] and 
                cvd_divergence and 
                confidence_score >= 60
            )
            
            if signal['valid']:
                current_price = self.data_manager.get_current_price(self.symbol)
                
                # Regular logging
                self.logger.log_trade_signal(
                    'BEARISH', self.symbol, current_price,
                    signal['details'], confidence_score
                )
                
                # LLM logging with detailed analysis
                if self.logger.llm_logger:
                    self.logger.llm_logger.log_signal_detected(
                        signal_type='bearish',
                        symbol=self.symbol,
                        confidence=confidence_score,
                        entry_price=current_price,
                        conditions=list(signal['conditions'].keys()),
                        analysis_data={
                            'signal_details': signal['details'],
                            'market_conditions': {
                                'funding_rate': signal['conditions'].get('high_funding_rate', {}).get('rate', 0),
                                'cvd_divergence': cvd_divergence,
                                'price_action': 'higher_high'
                            },
                            'confidence_breakdown': {
                                'core_conditions': 60,
                                'additional_factors': confidence_score - 60
                            }
                        }
                    )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error checking bearish signal: {e}")
            signal['error'] = str(e)
            return signal
    
    def check_bullish_signal(self) -> Dict[str, Any]:
        """
        Check for bullish (long) entry signal
        
        Conditions:
        1. Price makes Lower Low (LL)
        2. CVD makes Higher Low (HL) - divergence
        3. Negative funding rate (suggesting short bias)
        4. Optional: Decreasing Open Interest (selling exhaustion)
        """
        signal = {
            'type': 'bullish',
            'valid': False,
            'confidence': 0,
            'conditions': {},
            'details': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Update data
            self.update_price_data()
            
            # 1. Check for Price Lower Low
            price_ll = self._check_price_lower_low()
            signal['conditions']['price_lower_low'] = price_ll['valid']
            signal['details']['price_analysis'] = price_ll
            
            if not price_ll['valid']:
                return signal
            
            # 2. Check for CVD Divergence (Higher Low)
            cvd_divergence = self.cvd_calculator.detect_cvd_divergence(
                self.price_pivots_low, 'bullish'
            )
            signal['conditions']['cvd_divergence'] = cvd_divergence
            
            if not cvd_divergence:
                return signal
            
            # 3. Check Funding Rate
            funding_analysis = self._check_funding_rate('low')
            signal['conditions']['funding_rate_extreme'] = funding_analysis['extreme']
            signal['details']['funding_analysis'] = funding_analysis
            
            # 4. Check Open Interest (contextual)
            oi_analysis = self._check_open_interest_trend()
            signal['conditions']['open_interest_falling'] = oi_analysis.get('falling', False)
            signal['details']['oi_analysis'] = oi_analysis
            
            # 5. Check confirmation
            confirmation = self._check_price_confirmation('bullish')
            signal['conditions']['confirmation'] = confirmation['valid']
            signal['details']['confirmation'] = confirmation
            
            # Calculate confidence score
            confidence_score = 0
            if price_ll['valid']:
                confidence_score += 25
            if cvd_divergence:
                confidence_score += 35  # Primary signal
            if funding_analysis['extreme']:
                confidence_score += 20
            if oi_analysis.get('falling', False):
                confidence_score += 10
            if confirmation['valid']:
                confidence_score += 10
            
            signal['confidence'] = confidence_score
            
            # Signal is valid if we have the core conditions
            signal['valid'] = (
                price_ll['valid'] and 
                cvd_divergence and 
                confidence_score >= 60
            )
            
            if signal['valid']:
                current_price = self.data_manager.get_current_price(self.symbol)
                
                # Regular logging
                self.logger.log_trade_signal(
                    'BULLISH', self.symbol, current_price,
                    signal['details'], confidence_score
                )
                
                # LLM logging with detailed analysis
                if self.logger.llm_logger:
                    self.logger.llm_logger.log_signal_detected(
                        signal_type='bullish',
                        symbol=self.symbol,
                        confidence=confidence_score,
                        entry_price=current_price,
                        conditions=list(signal['conditions'].keys()),
                        analysis_data={
                            'signal_details': signal['details'],
                            'market_conditions': {
                                'funding_rate': signal['conditions'].get('low_funding_rate', {}).get('rate', 0),
                                'cvd_divergence': cvd_divergence,
                                'price_action': 'lower_low'
                            },
                            'confidence_breakdown': {
                                'core_conditions': 60,
                                'additional_factors': confidence_score - 60
                            }
                        }
                    )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error checking bullish signal: {e}")
            signal['error'] = str(e)
            return signal
    
    def _check_price_higher_high(self) -> Dict[str, Any]:
        """Check if price has made a recent higher high"""
        result = {'valid': False, 'details': ''}
        
        if len(self.price_pivots_high) < 2:
            result['details'] = 'Insufficient pivot highs'
            return result
        
        # Compare last two pivot highs
        prev_high = self.price_pivots_high[-2][1]
        recent_high = self.price_pivots_high[-1][1]
        
        if recent_high > prev_high:
            result['valid'] = True
            result['details'] = f'Higher High: {prev_high:.2f} -> {recent_high:.2f}'
            result['prev_high'] = prev_high
            result['recent_high'] = recent_high
        else:
            result['details'] = f'No Higher High: {prev_high:.2f} -> {recent_high:.2f}'
        
        return result
    
    def _check_price_lower_low(self) -> Dict[str, Any]:
        """Check if price has made a recent lower low"""
        result = {'valid': False, 'details': ''}
        
        if len(self.price_pivots_low) < 2:
            result['details'] = 'Insufficient pivot lows'
            return result
        
        # Compare last two pivot lows
        prev_low = self.price_pivots_low[-2][1]
        recent_low = self.price_pivots_low[-1][1]
        
        if recent_low < prev_low:
            result['valid'] = True
            result['details'] = f'Lower Low: {prev_low:.2f} -> {recent_low:.2f}'
            result['prev_low'] = prev_low
            result['recent_low'] = recent_low
        else:
            result['details'] = f'No Lower Low: {prev_low:.2f} -> {recent_low:.2f}'
        
        return result
    
    def _check_funding_rate(self, direction: str) -> Dict[str, Any]:
        """Check funding rate extremes"""
        try:
            funding_info = self.data_manager.get_funding_rate(self.symbol)
            funding_rate = funding_info['funding_rate']
            
            extreme_type = is_funding_rate_extreme(
                funding_rate, 
                self.high_funding_threshold,
                self.low_funding_threshold
            )
            
            result = {
                'funding_rate': funding_rate,
                'extreme': False,
                'type': extreme_type,
                'threshold_high': self.high_funding_threshold,
                'threshold_low': self.low_funding_threshold
            }
            
            if direction == 'high' and extreme_type == 'high':
                result['extreme'] = True
            elif direction == 'low' and extreme_type == 'low':
                result['extreme'] = True
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error checking funding rate: {e}")
            return {'funding_rate': 0, 'extreme': False, 'error': str(e)}
    
    def _check_open_interest_trend(self) -> Dict[str, Any]:
        """Check open interest trend"""
        try:
            current_oi = self.data_manager.get_open_interest(self.symbol)
            
            # Basit trend analizi - OI değişimini kabul et
            # Gerçek implementasyonda geçmiş OI verilerini takip edersiniz
            result = {
                'current_oi': current_oi['open_interest'],
                'rising': True,   # Geçici olarak True - trend analizi aktif
                'falling': False, # Tam trend analizi için geçmiş veri gerekir
                'trend': 'rising'  # Basit varsayım
            }
            
            # Not: Daha gelişmiş implementasyon için OI geçmişi saklanmalı
            self.logger.debug(f"OI Analysis: {result['current_oi']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error checking open interest: {e}")
            return {'current_oi': 0, 'rising': False, 'falling': False, 'error': str(e)}
    
    def _check_price_confirmation(self, signal_type: str) -> Dict[str, Any]:
        """Check for price confirmation of signal"""
        result = {'valid': False, 'details': ''}
        
        try:
            if len(self.price_data) < 2:
                result['details'] = 'Insufficient price data'
                return result
            
            current_price = self.data_manager.get_current_price(self.symbol)
            recent_candles = self.price_data.tail(self.confirmation_period)
            
            if signal_type == 'bearish':
                # Look for initial weakness
                recent_high = recent_candles['high'].max()
                if current_price < recent_high * 0.999:  # Below recent high
                    result['valid'] = True
                    result['details'] = f'Price showing weakness: {current_price} < {recent_high}'
                else:
                    result['details'] = f'No price weakness: {current_price} >= {recent_high}'
            
            elif signal_type == 'bullish':
                # Look for initial strength
                recent_low = recent_candles['low'].min()
                if current_price > recent_low * 1.001:  # Above recent low
                    result['valid'] = True
                    result['details'] = f'Price showing strength: {current_price} > {recent_low}'
                else:
                    result['details'] = f'No price strength: {current_price} <= {recent_low}'
            
            return result
            
        except Exception as e:
            result['details'] = f'Error checking confirmation: {e}'
            return result
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of current signal conditions"""
        try:
            current_price = self.data_manager.get_current_price(self.symbol)
            funding_info = self.data_manager.get_funding_rate(self.symbol)
            oi_info = self.data_manager.get_open_interest(self.symbol)
            cvd_stats = self.cvd_calculator.get_current_candle_stats()
            
            return {
                'symbol': self.symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'funding_rate': funding_info['funding_rate'],
                'open_interest': oi_info['open_interest'],
                'cvd_current': cvd_stats['cumulative_cvd'],
                'cvd_delta': cvd_stats['delta'],
                'price_pivots_high': len(self.price_pivots_high),
                'price_pivots_low': len(self.price_pivots_low),
                'cvd_pivots_high': len(self.cvd_calculator.get_pivot_highs()),
                'cvd_pivots_low': len(self.cvd_calculator.get_pivot_lows()),
                'cvd_strength': self.cvd_calculator.get_cvd_strength()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting signal summary: {e}")
            return {'error': str(e)}