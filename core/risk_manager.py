import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from utils.logger import TradingBotLogger
from utils.helpers import calculate_position_size, calculate_stop_loss_take_profit, safe_float
from data.data_manager import DataManager

class RiskManager:
    """
    Manages risk for trading operations including:
    - Position sizing based on account balance and risk percentage
    - Stop loss and take profit calculation
    - Maximum position limits
    - Risk-to-reward ratio enforcement
    """
    
    def __init__(self, config, logger: TradingBotLogger, data_manager: DataManager):
        self.config = config
        self.logger = logger
        self.data_manager = data_manager
        
        # Risk parameters
        self.risk_per_trade = config.risk_per_trade  # % of account to risk per trade
        self.risk_reward_ratio = config.risk_reward_ratio  # R:R ratio
        self.max_positions = config.get('trading.max_positions', 1)
        
        # Position tracking
        self.active_positions = {}
        self.trade_history = []
        
        self.logger.info(f"RiskManager initialized - Risk: {self.risk_per_trade*100}%, R:R: {self.risk_reward_ratio}")
    
    def calculate_position_parameters(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Calculate position parameters based on signal and risk management rules
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            Dictionary with position parameters or None if position not allowed
        """
        try:
            # Check if we can open a new position
            if not self._can_open_position():
                self.logger.warning("Cannot open position: Maximum positions reached")
                return None
            
            # Get account balance
            account_balance = self.data_manager.get_account_balance()
            if account_balance <= 0:
                self.logger.error("Invalid account balance")
                return None
            
            # Get current price
            current_price = self.data_manager.get_current_price(self.config.symbol)
            if current_price <= 0:
                self.logger.error("Invalid current price")
                return None
            
            # Determine trade direction
            side = 'BUY' if signal['type'] == 'bullish' else 'SELL'
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_sl_tp(current_price, side)
            
            # Calculate position size
            position_size = self._calculate_position_size(
                account_balance, current_price, stop_loss
            )
            
            if position_size <= 0:
                self.logger.warning("Calculated position size is zero or negative")
                return None
            
            # Calculate potential PnL
            risk_amount = account_balance * self.risk_per_trade
            reward_amount = risk_amount * self.risk_reward_ratio
            
            position_params = {
                'symbol': self.config.symbol,
                'side': side,
                'position_size': position_size,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_amount': risk_amount,
                'reward_amount': reward_amount,
                'risk_reward_ratio': self.risk_reward_ratio,
                'account_balance': account_balance,
                'risk_percentage': self.risk_per_trade * 100,
                'signal_confidence': signal.get('confidence', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Position parameters calculated:")
            self.logger.info(f"  Side: {side}, Size: {position_size}")
            self.logger.info(f"  Entry: {current_price}, SL: {stop_loss}, TP: {take_profit}")
            self.logger.info(f"  Risk: ${risk_amount:.2f}, Reward: ${reward_amount:.2f}")
            
            return position_params
            
        except Exception as e:
            self.logger.error(f"Error calculating position parameters: {e}")
            return None
    
    def _can_open_position(self) -> bool:
        """Check if we can open a new position"""
        # Check existing positions for the symbol
        existing_position = self.data_manager.get_position_info(self.config.symbol)
        if abs(existing_position['position_amount']) > 0:
            self.logger.warning(f"Position already exists for {self.config.symbol}")
            return False
        
        # Check maximum position limit
        active_count = len([p for p in self.active_positions.values() if p.get('status') == 'active'])
        if active_count >= self.max_positions:
            self.logger.warning(f"Maximum positions reached: {active_count}/{self.max_positions}")
            return False
        
        return True
    
    def _calculate_sl_tp(self, entry_price: float, side: str) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        # Base stop loss percentage (can be made configurable)
        base_sl_pct = 0.02  # 2%
        
        # Adjust SL based on volatility (simplified)
        try:
            # Get recent price data for volatility calculation
            recent_data = self.data_manager.get_historical_klines(
                self.config.symbol, self.config.timeframe, 20
            )
            
            if len(recent_data) > 1:
                # Calculate ATR-like volatility
                high_low = recent_data['high'] - recent_data['low']
                avg_range = high_low.rolling(window=10).mean().iloc[-1]
                
                # Adjust SL based on volatility
                volatility_factor = avg_range / entry_price
                adjusted_sl_pct = max(base_sl_pct, volatility_factor * 1.5)
            else:
                adjusted_sl_pct = base_sl_pct
        
        except Exception as e:
            self.logger.warning(f"Could not calculate volatility-adjusted SL: {e}")
            adjusted_sl_pct = base_sl_pct
        
        # Calculate SL and TP
        stop_loss, take_profit = calculate_stop_loss_take_profit(
            entry_price, side, self.risk_reward_ratio, adjusted_sl_pct
        )
        
        return stop_loss, take_profit
    
    def _calculate_position_size(self, account_balance: float, 
                               entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management"""
        position_size = calculate_position_size(
            account_balance, self.risk_per_trade, entry_price, stop_loss
        )
        
        # Apply minimum and maximum position size limits
        min_position_size = self._get_min_position_size()
        max_position_size = self._get_max_position_size(account_balance)
        
        position_size = max(min_position_size, min(position_size, max_position_size))
        
        return position_size
    
    def _get_min_position_size(self) -> float:
        """Get minimum position size for the symbol"""
        # This should be fetched from exchange info
        # For now, use a conservative minimum
        return 0.001  # Minimum for most crypto pairs
    
    def _get_max_position_size(self, account_balance: float) -> float:
        """Get maximum position size based on account balance"""
        # Limit position to maximum % of account
        max_position_pct = 0.5  # 50% of account max
        max_position_value = account_balance * max_position_pct
        
        current_price = self.data_manager.get_current_price(self.config.symbol)
        if current_price > 0:
            return max_position_value / current_price
        
        return float('inf')
    
    def validate_position(self, position_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate position parameters before execution"""
        validation = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check position size
            if position_params['position_size'] <= 0:
                validation['errors'].append("Position size must be positive")
                validation['valid'] = False
            
            # Check price levels
            entry = position_params['entry_price']
            sl = position_params['stop_loss']
            tp = position_params['take_profit']
            side = position_params['side']
            
            if side == 'BUY':
                if sl >= entry:
                    validation['errors'].append("Stop loss should be below entry for long position")
                    validation['valid'] = False
                if tp <= entry:
                    validation['errors'].append("Take profit should be above entry for long position")
                    validation['valid'] = False
            else:  # SELL
                if sl <= entry:
                    validation['errors'].append("Stop loss should be above entry for short position")
                    validation['valid'] = False
                if tp >= entry:
                    validation['errors'].append("Take profit should be below entry for short position")
                    validation['valid'] = False
            
            # Check risk amount
            risk_amount = position_params['risk_amount']
            account_balance = position_params['account_balance']
            
            if risk_amount > account_balance * 0.1:  # More than 10%
                validation['warnings'].append(f"High risk amount: ${risk_amount:.2f} ({risk_amount/account_balance*100:.1f}%)")
            
            # Check confidence
            confidence = position_params.get('signal_confidence', 0)
            if confidence < 60:
                validation['warnings'].append(f"Low signal confidence: {confidence}%")
            
            return validation
            
        except Exception as e:
            validation['errors'].append(f"Validation error: {e}")
            validation['valid'] = False
            return validation
    
    def register_position(self, position_id: str, position_params: Dict[str, Any]):
        """Register a new position for tracking"""
        self.active_positions[position_id] = {
            **position_params,
            'position_id': position_id,
            'status': 'active',
            'open_time': datetime.now(),
            'unrealized_pnl': 0.0
        }
        
        self.logger.info(f"Position registered: {position_id}")
    
    def update_position_pnl(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Update position PnL based on current market price"""
        if position_id not in self.active_positions:
            return None
        
        try:
            position = self.active_positions[position_id]
            current_price = self.data_manager.get_current_price(position['symbol'])
            
            entry_price = position['entry_price']
            position_size = position['position_size']
            side = position['side']
            
            if side == 'BUY':
                unrealized_pnl = (current_price - entry_price) * position_size
            else:  # SELL
                unrealized_pnl = (entry_price - current_price) * position_size
            
            position['unrealized_pnl'] = unrealized_pnl
            position['current_price'] = current_price
            position['last_update'] = datetime.now()
            
            return {
                'position_id': position_id,
                'unrealized_pnl': unrealized_pnl,
                'current_price': current_price,
                'entry_price': entry_price,
                'pnl_percentage': (unrealized_pnl / (entry_price * position_size)) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error updating position PnL: {e}")
            return None
    
    def check_stop_loss_take_profit(self, position_id: str) -> Optional[str]:
        """Check if position should be closed due to SL/TP"""
        if position_id not in self.active_positions:
            return None
        
        try:
            position = self.active_positions[position_id]
            current_price = self.data_manager.get_current_price(position['symbol'])
            
            side = position['side']
            sl = position['stop_loss']
            tp = position['take_profit']
            
            if side == 'BUY':
                if current_price <= sl:
                    return 'stop_loss'
                elif current_price >= tp:
                    return 'take_profit'
            else:  # SELL
                if current_price >= sl:
                    return 'stop_loss'
                elif current_price <= tp:
                    return 'take_profit'
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking SL/TP: {e}")
            return None
    
    def close_position(self, position_id: str, reason: str, exit_price: float):
        """Close a position and calculate final PnL"""
        if position_id not in self.active_positions:
            self.logger.warning(f"Position {position_id} not found for closing")
            return
        
        try:
            position = self.active_positions[position_id]
            
            # Calculate final PnL
            entry_price = position['entry_price']
            position_size = position['position_size']
            side = position['side']
            
            if side == 'BUY':
                realized_pnl = (exit_price - entry_price) * position_size
            else:  # SELL
                realized_pnl = (entry_price - exit_price) * position_size
            
            # Update position record
            position['status'] = 'closed'
            position['exit_price'] = exit_price
            position['realized_pnl'] = realized_pnl
            position['close_time'] = datetime.now()
            position['close_reason'] = reason
            
            # Add to trade history
            self.trade_history.append(position.copy())
            
            # Remove from active positions
            del self.active_positions[position_id]
            
            # Log the closure
            self.logger.log_position_closed(
                position['symbol'], side, position_size,
                entry_price, exit_price, realized_pnl, reason
            )
            
            return realized_pnl
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return None
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        try:
            account_balance = self.data_manager.get_account_balance()
            active_count = len(self.active_positions)
            
            # Calculate total unrealized PnL
            total_unrealized_pnl = 0
            for position in self.active_positions.values():
                self.update_position_pnl(position['position_id'])
                total_unrealized_pnl += position.get('unrealized_pnl', 0)
            
            # Calculate trade statistics
            if self.trade_history:
                realized_pnls = [trade['realized_pnl'] for trade in self.trade_history]
                win_count = len([pnl for pnl in realized_pnls if pnl > 0])
                loss_count = len([pnl for pnl in realized_pnls if pnl < 0])
                win_rate = win_count / len(realized_pnls) if realized_pnls else 0
                
                avg_win = np.mean([pnl for pnl in realized_pnls if pnl > 0]) if win_count > 0 else 0
                avg_loss = np.mean([pnl for pnl in realized_pnls if pnl < 0]) if loss_count > 0 else 0
                
                total_realized_pnl = sum(realized_pnls)
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                total_realized_pnl = 0
            
            return {
                'account_balance': account_balance,
                'active_positions': active_count,
                'max_positions': self.max_positions,
                'risk_per_trade': self.risk_per_trade * 100,
                'risk_reward_ratio': self.risk_reward_ratio,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_realized_pnl': total_realized_pnl,
                'total_trades': len(self.trade_history),
                'win_rate': win_rate * 100,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def get_active_positions(self) -> Dict[str, Any]:
        """Get all active positions with current PnL"""
        positions = {}
        
        for position_id, position in self.active_positions.items():
            self.update_position_pnl(position_id)
            positions[position_id] = position.copy()
        
        return positions