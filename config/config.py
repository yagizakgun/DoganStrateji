import json
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self, config_file: str = "config/settings.json"):
        self.config_file = config_file
        self._config = self._load_config()
        self._override_with_env()
    
    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {self.config_file} not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file {self.config_file}")
    
    def _override_with_env(self):
        # API credentials MUST come from environment variables for security
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        # Set API credentials from environment (required)
        self._config['api']['api_key'] = api_key or ''
        self._config['api']['api_secret'] = api_secret or ''
        
        testnet = os.getenv('BINANCE_TESTNET')
        if testnet is not None:
            self._config['api']['testnet'] = testnet.lower() == 'true'
        
        symbol = os.getenv('TRADING_SYMBOL')
        if symbol:
            self._config['trading']['symbol'] = symbol
            
        # Paper trading configuration
        paper_trading = os.getenv('PAPER_TRADING')
        if paper_trading is not None:
            self._config['paper_trading']['enabled'] = paper_trading.lower() == 'true'
            
        paper_balance = os.getenv('PAPER_VIRTUAL_BALANCE')
        if paper_balance:
            try:
                self._config['paper_trading']['virtual_balance'] = float(paper_balance)
            except ValueError:
                pass
    
    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        keys = key.split('.')
        config_section = self._config
        
        for k in keys[:-1]:
            if k not in config_section:
                config_section[k] = {}
            config_section = config_section[k]
        
        config_section[keys[-1]] = value
    
    def save(self):
        with open(self.config_file, 'w') as f:
            json.dump(self._config, f, indent=4)
    
    @property
    def api_key(self) -> str:
        return self.get('api.api_key')
    
    @property
    def api_secret(self) -> str:
        return self.get('api.api_secret')
    
    @property
    def testnet(self) -> bool:
        return self.get('api.testnet', True)
    
    @property
    def symbol(self) -> str:
        return self.get('trading.symbol', 'BTCUSDT')
    
    @property
    def timeframe(self) -> str:
        return self.get('trading.timeframe', '15m')
    
    @property
    def lookback_period(self) -> int:
        return self.get('trading.lookback_period', 20)
    
    @property
    def risk_per_trade(self) -> float:
        return self.get('trading.risk_per_trade', 0.01)
    
    @property
    def risk_reward_ratio(self) -> float:
        return self.get('trading.risk_reward_ratio', 1.5)
    
    @property
    def high_funding_threshold(self) -> float:
        return self.get('signals.high_funding_threshold', 0.01)
    
    @property
    def low_funding_threshold(self) -> float:
        return self.get('signals.low_funding_threshold', -0.01)
    
    @property
    def cvd_lookback(self) -> int:
        return self.get('signals.cvd_lookback', 10)
    
    @property
    def paper_trading_enabled(self) -> bool:
        return self.get('paper_trading.enabled', False)
    
    @property
    def paper_virtual_balance(self) -> float:
        return self.get('paper_trading.virtual_balance', 10000.0)
    
    @property
    def paper_commission_rate(self) -> float:
        return self.get('paper_trading.commission_rate', 0.0004)
    
    @property
    def paper_slippage_bps(self) -> int:
        return self.get('paper_trading.slippage_bps', 2)
    
    @property
    def paper_fill_delay_ms(self) -> int:
        return self.get('paper_trading.fill_delay_ms', 50)
    
    @property
    def verbose_console(self) -> bool:
        return self.get('logging.verbose_console', True)
    
    @property
    def show_heartbeat(self) -> bool:
        return self.get('logging.show_heartbeat', True)
    
    @property
    def heartbeat_interval(self) -> int:
        return self.get('logging.heartbeat_interval', 30)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the current configuration."""
        return {
            'api': {
                'testnet': self.testnet,
                'has_credentials': bool(self.api_key and self.api_secret)
            },
            'trading': {
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'lookback_period': self.lookback_period,
                'risk_per_trade': self.risk_per_trade,
                'risk_reward_ratio': self.risk_reward_ratio
            },
            'signals': {
                'high_funding_threshold': self.high_funding_threshold,
                'low_funding_threshold': self.low_funding_threshold,
                'cvd_lookback': self.cvd_lookback
            },
            'paper_trading': {
                'enabled': self.paper_trading_enabled,
                'virtual_balance': self.paper_virtual_balance,
                'commission_rate': self.paper_commission_rate,
                'slippage_bps': self.paper_slippage_bps,
                'fill_delay_ms': self.paper_fill_delay_ms
            },
            'logging': {
                'verbose_console': self.verbose_console,
                'show_heartbeat': self.show_heartbeat,
                'heartbeat_interval': self.heartbeat_interval
            }
        }

config = Config()