import os
import time
import yfinance as yf
import pandas as pd
import ta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from dotenv import load_dotenv
import logging
import requests
import numpy as np
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Custom Telegram Handler for Logging
class TelegramHandler(logging.Handler):
    def __init__(self, bot_token, chat_id):
        super().__init__()
        self.bot_token = bot_token
        self.chat_id = chat_id

    def emit(self, record):
        log_entry = self.format(record)
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = {"chat_id": self.chat_id, "text": log_entry, "parse_mode": "HTML"}
        try:
            response = requests.post(url, data=data, timeout=10)
            if response.status_code != 200:
                print(f"Failed to send Telegram message: {response.status_code}")
        except Exception as e:
            print(f"Telegram send error: {e}")

# Setup logging with Telegram integration
load_dotenv()
telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detailed_trading_log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Add Telegram handler if credentials are available
if telegram_bot_token and telegram_chat_id:
    telegram_handler = TelegramHandler(telegram_bot_token, telegram_chat_id)
    telegram_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(telegram_handler)

class TradingLogger:
    def __init__(self):
        self.trades = []
        self.daily_summary = {}
    
    def log_trade(self, trade_type, price, shares, profit=None, reason=None, timestamp=None, balance=None, indicators=None):
        trade_info = {
            'timestamp': timestamp or datetime.now(),
            'type': trade_type,
            'price': price,
            'shares': shares,
            'profit': profit,
            'reason': reason,
            'balance': balance,
            'indicators': indicators or {}
        }
        self.trades.append(trade_info)
        
        # Log trade with detailed format
        if trade_type == 'buy':
            msg = f"🟢 BUY  | {timestamp} | {shares:>6.0f} shares @ Rp{price:>8.0f} | Balance: Rp{balance:>12,.0f}"
            if indicators:
                msg += f" | RSI:{indicators.get('rsi', 0):.1f} MACD:{indicators.get('macd', 0):.3f} Prob:{indicators.get('probability', 0):.2f}"
        else:
            profit_pct = (profit / (shares * price - profit)) * 100 if shares and price and (shares * price - profit) != 0 else 0
            msg = f"🔴 SELL | {timestamp} | {shares:>6.0f} shares @ Rp{price:>8.0f} | Profit: Rp{profit:>8,.0f} ({profit_pct:>5.1f}%) | {reason} | Balance: Rp{balance:>12,.0f}"
        
        print(msg)
        logging.info(msg)
    
    def get_trade_summary(self):
        if not self.trades:
            return "No trades executed"
        
        buy_trades = [t for t in self.trades if t['type'] == 'buy']
        sell_trades = [t for t in self.trades if t['type'] == 'sell' and t['profit'] is not None]
        
        total_trades = len(sell_trades)
        profitable_trades = len([t for t in sell_trades if t['profit'] > 0])
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = sum(t['profit'] for t in sell_trades)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        
        best_trade = max(sell_trades, key=lambda x: x['profit']) if sell_trades else None
        worst_trade = min(sell_trades, key=lambda x: x['profit']) if sell_trades else None
        
        summary = f"""
═══════════════════════════════════════════════════════════════
                         TRADING SUMMARY
═══════════════════════════════════════════════════════════════
📊 Total Transactions: {total_trades}
🎯 Win Rate: {win_rate:.1f}% ({profitable_trades}/{total_trades})
💰 Total Profit: Rp {total_profit:,.0f}
📈 Average Profit/Trade: Rp {avg_profit:,.0f}
"""
        
        if best_trade:
            summary += f"🚀 Best Trade: Rp {best_trade['profit']:,.0f} on {best_trade['timestamp'].strftime('%Y-%m-%d %H:%M')}\n"
        if worst_trade:
            summary += f"📉 Worst Trade: Rp {worst_trade['profit']:,.0f} on {worst_trade['timestamp'].strftime('%Y-%m-%d %H:%M')}\n"
        
        sell_reasons = {}
        for trade in sell_trades:
            reason = trade['reason']
            if reason not in sell_reasons:
                sell_reasons[reason] = {'count': 0, 'profit': 0}
            sell_reasons[reason]['count'] += 1
            sell_reasons[reason]['profit'] += trade['profit']
        
        summary += "\n📋 Sell Reasons Analysis:\n"
        for reason, data in sell_reasons.items():
            avg_profit_reason = data['profit'] / data['count']
            summary += f"   {reason}: {data['count']} trades, Avg: Rp {avg_profit_reason:.0f}\n"
        summary += "═══════════════════════════════════════"
        
        logging.info(summary)
        return summary

# Initialize logger
trade_logger = TradingLogger()

def get_stock_data(symbol, interval, period):
    try:
        print(f"📥 Downloading data for {symbol} with interval {interval} and period {period}...")
        
        valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        if interval not in valid_intervals:
            logging.error(f"Invalid interval: {interval}")
            return pd.DataFrame()
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                data = yf.download(symbol, interval=interval, period=period, auto_adjust=True, progress=False)
                if not data.empty:
                    break
                time.sleep(2)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(5)
        
        if data.empty:
            logging.warning(f"Empty data for {symbol} interval {interval}")
            return pd.DataFrame()
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logging.error(f"Missing columns: {missing_columns}")
            return pd.DataFrame()
        
        if hasattr(data.index, 'tz') and data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        data = data.dropna()
        
        if len(data) < 50:
            logging.warning(f"Insufficient data: only {len(data)} rows")
            return pd.DataFrame()
        
        logging.info(f"✅ Successfully retrieved {len(data)} rows for {symbol} {interval}")
        return data
        
    except Exception as e:
        logging.error(f"Error retrieving data for {symbol} {interval}: {e}")
        print(f"Error retrieving data for {symbol} {interval}: {e}")
        return pd.DataFrame()

def add_indicators(df):
    if df.empty or len(df) < 50:
        logging.warning("DataFrame empty or insufficient data for indicators")
        return df
    
    try:
        df = df.copy()
        
        # RSI Indicators
        df['rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        df['rsi_fast'] = ta.momentum.RSIIndicator(close=df['Close'], window=7).rsi()
        df['rsi_momentum_ratio'] = df['rsi'] / df['rsi_fast'].replace(0, np.nan)
        df['rsi_slow'] = ta.momentum.RSIIndicator(close=df['Close'], window=21).rsi()
        
        # Moving Averages
        df['ma5'] = df['Close'].rolling(window=5).mean()
        df['ma10'] = df['Close'].rolling(window=10).mean()
        df['ma20'] = df['Close'].rolling(window=20).mean()
        df['ma50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        
        # Exponential Moving Averages
        df['ema12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=df['Close'], window=20)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # MACD
        macd = ta.trend.MACD(close=df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Volume Indicators
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma'].replace(0, np.nan)
        
        # VWAP
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        df['vwap'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['vwap_distance'] = (df['Close'] - df['vwap']) / df['vwap']
        
        # ATR and Volatility
        df['atr'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
        df['atr_percent'] = df['atr'] / df['Close']
        df['volatility'] = df['Close'].rolling(window=20).std()
        
        # Price Changes and Momentum
        df['price_change'] = df['Close'].pct_change()
        df['price_change_5'] = df['Close'].pct_change(periods=5)
        df['price_momentum'] = (df['Close'] / df['Close'].shift(10) - 1).fillna(0)
        
        # Support and Resistance
        df['support'] = df['Low'].rolling(window=20).min()
        df['resistance'] = df['High'].rolling(window=20).max()
        df['support_distance'] = (df['Close'] - df['support']) / df['support']
        df['resistance_distance'] = (df['resistance'] - df['Close']) / df['resistance']
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticIndicator(high=df['High'], low=df['Low'], close=df['Close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close']).williams_r()
        
        # CCI
        df['cci'] = ta.trend.CCIIndicator(high=df['High'], low=df['Low'], close=df['Close']).cci()
        
        # MFI
        df['mfi'] = ta.volume.MoneyFlowIndexIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).money_flow_index()
        
        # Cross Signals
        df['golden_cross'] = (df['ma5'] > df['ma20']) & (df['ma5'].shift(1) <= df['ma20'].shift(1))
        df['death_cross'] = (df['ma5'] < df['ma20']) & (df['ma5'].shift(1) >= df['ma20'].shift(1))
        df['rsi_oversold'] = df['rsi'] <= 30
        df['rsi_overbought'] = df['rsi'] >= 70
        df['volume_spike'] = df['volume_ratio'] >= 2
        
        # Fill NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logging.info(f"✅ Technical indicators calculated successfully. Shape: {df.shape}")
        return df
        
    except Exception as e:
        logging.error(f"Error calculating indicators: {e}")
        return df

def prepare_multi_timeframe_data(symbol='BBCA.JK'):
    try:
        print(f"🔄 Preparing multi-timeframe data for {symbol}...")
        
        timeframe_configs = [
            {'base': ('5m', '7d'), 'medium': ('15m', '7d'), 'long': ('1h', '30d')},
            {'base': ('15m', '7d'), 'medium': ('1h', '30d'), 'long': ('1d', '60d')},
            {'base': ('1h', '30d'), 'medium': ('1d', '60d'), 'long': ('1wk', '1y')},
            {'base': ('1d', '1y'), 'medium': ('1wk', '2y'), 'long': ('1mo', '5y')}
        ]
        
        df_multi = None
        
        for i, config in enumerate(timeframe_configs):
            try:
                print(f"🔍 Trying configuration {i+1}: {config['base'][0]} base timeframe")
                
                df_base = get_stock_data(symbol, config['base'][0], config['base'][1])
                
                if df_base.empty:
                    continue
                
                df_base = add_indicators(df_base)
                
                if df_base.empty or len(df_base) < 100:
                    continue
                
                df_medium = get_stock_data(symbol, config['medium'][0], config['medium'][1])
                df_long = get_stock_data(symbol, config['long'][0], config['long'][1])
                
                if not df_medium.empty:
                    df_medium = add_indicators(df_medium)
                if not df_long.empty:
                    df_long = add_indicators(df_long)
                
                df = df_base.copy()
                
                if not df_medium.empty and len(df_medium) > 20:
                    medium_cols = ['rsi', 'macd', 'bb_position', 'volume_ratio']
                    for col in medium_cols:
                        if col in df_medium.columns:
                            medium_resampled = df_medium[col].reindex(df.index, method='ffill')
                            df[f'{col}_medium'] = medium_resampled
                        else:
                            df[f'{col}_medium'] = df[col] if col in df.columns else 0
                
                if not df_long.empty and len(df_long) > 20:
                    long_cols = ['rsi', 'macd', 'bb_position', 'volume_ratio']
                    for col in long_cols:
                        if col in df_long.columns:
                            long_resampled = df_long[col].reindex(df.index, method='ffill')
                            df[f'{col}_long'] = long_resampled
                        else:
                            df[f'{col}_long'] = df[col] if col in df.columns else 0
                
                # Filter trading hours for Jakarta stocks
                if symbol.endswith('.JK') and config['base'][0] != '1d':
                    df = df.between_time('09:00', '16:00')
                elif not symbol.endswith('.JK') and config['base'][0] != '1d':
                    df = df.between_time('09:30', '16:00')
                
                df = df.dropna()
                
                if len(df) >= 100:
                    df_multi = df
                    print(f"✅ Successfully prepared multi-timeframe data with {len(df)} rows using {config['base'][0]} base")
                    logging.info(f"Multi-timeframe data prepared: {len(df)} rows, base timeframe: {config['base'][0]}")
                    break
                
                print(f"Failed to prepare sufficient data with config {i+1}")
                    
            except Exception as e:
                logging.warning(f"Failed to prepare data with config {config}: {e}")
                continue
        
        return df_multi if df_multi is not None else pd.DataFrame()
        
    except Exception as e:
        logging.error(f"Error in prepare_multi_timeframe_data: {e}")
        print(f"Error preparing multi-timeframe_data: {e}")
        return pd.DataFrame()

def prepare_dataset(df):
    if df.empty:
        logging.error("Empty dataframe for model preparation")
        return None, None, None, None
    
    try:
        # Define target variables
        df['target_conservative'] = (df['Close'].shift(-1) > df['Close'] * 1.005).astype(int)
        df['target_aggressive'] = (df['Close'].shift(-1) > df['Close'] * 1.01).astype(int)
        df['target'] = df['target_conservative']
        
        df = df[:-1]  # Remove last row due to shift
        
        base_features = [
            'rsi', 'rsi_fast', 'rsi_slow', 'ma5', 'ma10', 'ma20', 'ma50',
            'ema12', 'ema26', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
            'macd', 'macd_signal', 'macd_diff', 'volume_ratio',
            'vwap', 'vwap_distance', 'atr_percent', 'volatility',
            'price_change', 'price_change_5', 'price_momentum',
            'support_distance', 'resistance_distance', 'stoch_k', 'stoch_d',
            'williams_r', 'cci', 'golden_cross', 'death_cross', 'rsi_oversold',
            'rsi_overbought', 'volume_spike'
        ]
        
        multi_features = [
            'rsi_medium', 'rsi_long', 'macd_medium', 'macd_long',
            'bb_position_medium', 'bb_position_long', 'volume_ratio_medium', 'volume_ratio_long'
        ]
        
        available_features = [col for col in base_features + multi_features if col in df.columns]
        
        if len(available_features) < 10:
            logging.error(f"Insufficient features available: {len(available_features)}")
            return None, None, None, None
        
        print(f"📊 Using {len(available_features)} features for model training")
        
        X = df[available_features].copy()
        y = df['target'].copy()
        
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Add interaction features
        if 'rsi' in X.columns and 'macd' in X.columns:
            X['rsi_macd_interaction'] = X['rsi'] * X['macd']
        if 'volume_ratio' in X.columns and 'price_momentum' in X.columns:
            X['volume_price_momentum'] = X['volume_ratio'] * X['price_momentum']
        if 'bb_position' in X.columns and 'rsi' in X.columns:
            X['bb_rsi_signal'] = X['bb_position'] * X['rsi']
        
        print(f"Target distribution:\n{y.value_counts()}")
        logging.info(f"Target distribution:\n{y.value_counts()}")
        
        if len(y.unique()) > 1 and len(y[y == 1]) > 5:
            smote = SMOTE(random_state=42, k_neighbors=min(5, len(y[y == 1]) - 1))
            X_resampled, y_resampled = smote.fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_resampled)
        
        logging.info(f"Dataset prepared successfully: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        return X_scaled, y_resampled, scaler, X.columns.tolist()
        
    except Exception as e:
        logging.error(f"Error preparing dataset: {e}")
        print(f"Error preparing dataset: {e}")
        return None, None, None, None

def train_model(X, y):
    try:
        if X is None or y is None:
            logging.error("No data available for model training")
            return None, None
            
        tscv = TimeSeriesSplit(n_splits=3)
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5]
        }
        
        pos_weight = len(y[y == 0]) / len(y[y == 1]) if len(y[y == 1]) > 0 else 1
        
        model = XGBClassifier(
            random_state=42,
            scale_pos_weight=pos_weight,
            n_jobs=-1,
            objective='binary:logistic',
            eval_metric='logloss'
        )
        
        print("✅ Training model with GridSearchCV...")
        grid_search = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=tscv, scoring='f1', n_jobs=-1, verbose=1
        )
        
        try:
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            print(f"🎯 Model Performance:")
            print(f"   Accuracy:  {accuracy:.3f}")
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall:    {recall:.3f}")
            print(f"   F1-Score:  {f1:.3f}")
            print(f"   Best parameters: {grid_search.best_params_}")
            
            logging.info(f"Model trained successfully - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            logging.info(f"Best parameters: {grid_search.best_params_}")
            
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 Top 10 Most Important Features:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
            
            return best_model, y_test
            
        except Exception as e:
            logging.error(f"Error during grid search or prediction: {e}")
            return None, None
        
    except Exception as e:
        logging.error(f"Error in train_model: {e}")
        print(f"Error in train_model: {e}")
        return None, None

def backtest_trading(df_test, model, scaler, features, initial_balance=100000000, fee_rate=0.001):
    try:
        if model is None or scaler is None:
            logging.error("Model or scaler is None")
            return initial_balance, [initial_balance], []
        
        print("📈 Running advanced backtest...")
        
        df_test = df_test.copy()
        
        # Add interaction features
        if 'rsi' in df_test.columns and 'macd' in df_test.columns:
            df_test['rsi_macd_interaction'] = df_test['rsi'] * df_test['macd']
        if 'volume_ratio' in df_test.columns and 'price_momentum' in df_test.columns:
            df_test['volume_price_momentum'] = df_test['volume_ratio'] * df_test['price_momentum']
        if 'bb_position' in df_test.columns and 'rsi' in df_test.columns:
            df_test['bb_rsi_signal'] = df_test['bb_position'] * df_test['rsi']
        
        X_test = df_test[features].fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], 0)
        X_test_scaled = scaler.transform(X_test)
        
        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)[:, 1]
        
        balance = initial_balance
        shares = 0
        balance_history = [initial_balance]
        entry_price = None
        entry_timestamp = None
        
        max_risk_per_trade = 0.02
        take_profit_pct = 0.015
        stop_loss_pct = 0.008
        max_positions = 3
        consecutive_losses = 0
        max_consecutive_losses = 3
        
        high_confidence_threshold = 0.75
        medium_confidence_threshold = 0.65
        
        current_positions = 0
        
        logging.info(f"Starting backtest with initial balance: Rp {initial_balance:,.0f}")
        print(f"💰 Starting backtest with balance: Rp {balance:,.0f}")
        print("Log trade:")
        print("=" * 80)
        
        for i in range(len(df_test)):
            try:
                current_price = df_test.iloc[i]['Close']
                current_time = df_test.index[i]
                pred = predictions[i]
                prob = probabilities[i]
                
                current_indicators = {
                    'rsi': df_test.iloc[i].get('rsi', 0),
                    'macd': df_test.iloc[i].get('macd', 0),
                    'bb_position': df_test.iloc[i].get('bb_position', 0),
                    'volume_ratio': df_test.iloc[i].get('volume_ratio', 0),
                    'probability': prob
                }
                
                if shares > 0 and entry_price is not None and entry_timestamp is not None:
                    current_return = (current_price - entry_price) / entry_price
                    
                    if current_return >= take_profit_pct:
                        profit = shares * (current_price - entry_price) - (shares * current_price * fee_rate)
                        balance += shares * current_price - (shares * current_price * fee_rate)
                        trade_logger.log_trade('sell', current_price, shares, profit, 'Take Profit', 
                                             current_time, balance, current_indicators)
                        shares = 0
                        entry_price = None
                        entry_timestamp = None
                        current_positions -= 1
                        consecutive_losses = 0
                        balance_history.append(balance)
                        continue
                    
                    elif current_return <= -stop_loss_pct:
                        profit = shares * (current_price - entry_price) - (shares * current_price * fee_rate)
                        balance += shares * current_price - (shares * current_price * fee_rate)
                        trade_logger.log_trade('sell', current_price, shares, profit, 'Stop Loss', 
                                             current_time, balance, current_indicators)
                        shares = 0
                        entry_price = None
                        entry_timestamp = None
                        current_positions -= 1
                        consecutive_losses += 1
                        balance_history.append(balance)
                        continue
                    
                    elif (current_time - entry_timestamp).total_seconds() / 60 >= 30:  # 30-minute hold limit
                        profit = shares * (current_price - entry_price) - (shares * current_price * fee_rate)
                        balance += shares * current_price - (shares * current_price * fee_rate)
                        trade_logger.log_trade('sell', current_price, shares, profit, 'Time Exit', 
                                             current_time, balance, current_indicators)
                        shares = 0
                        entry_price = None
                        entry_timestamp = None
                        current_positions -= 1
                        if profit < 0:
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0
                        balance_history.append(balance)
                        continue
                
                if shares == 0 and pred == 1 and prob >= medium_confidence_threshold and \
                   current_positions < max_positions and consecutive_losses < max_consecutive_losses and \
                   balance > 1000000:
                    rsi_ok = 20 <= current_indicators['rsi'] <= 80
                    volume_ok = current_indicators['volume_ratio'] >= 1.2
                    
                    if rsi_ok and volume_ok:
                        risk_amount = balance * max_risk_per_trade
                        position_size = risk_amount / (current_price * stop_loss_pct)
                        max_position_value = balance * 0.3
                        max_shares_by_value = max_position_value / current_price
                        
                        target_shares = min(position_size, max_shares_by_value)
                        target_shares = max(100, int(target_shares / 100) * 100)
                        
                        total_cost = target_shares * current_price * (1 + fee_rate)
                        
                        if total_cost <= balance:
                            balance -= total_cost
                            shares = target_shares
                            entry_price = current_price
                            entry_timestamp = current_time
                            current_positions += 1
                            
                            trade_logger.log_trade('buy', current_price, shares, None, None, 
                                                 current_time, balance, current_indicators)
                            balance_history.append(balance)
                
                balance_history.append(balance)
                
            except Exception as e:
                logging.warning(f"Error in backtest iteration: {e}")
                continue
        
        if shares > 0:
            final_value = shares * df_test['Close'].iloc[-1]
            profit = shares * (df_test['Close'].iloc[-1] - entry_price) - (final_value * fee_rate)
            balance += final_value - (final_value * fee_rate)
            trade_logger.log_trade('sell', df_test['Close'].iloc[-1], shares, profit, 'Final Close', 
                                 df_test.index[-1], balance, current_indicators)
            balance_history.append(balance)
        
        final_balance = balance
        total_return = (final_balance - initial_balance) / initial_balance * 100
        
        print(f"\n💰 Backtest Results:")
        print(f"   Initial Balance: Rp {initial_balance:,.0f}")
        print(f"   Final Balance: Rp {final_balance:,.0f}")
        print(f"   Total Return: {total_return:.2f}%")
        
        logging.info(f"Backtest completed - Final Balance: Rp {final_balance:,.0f}, Return: {total_return:.2f}%")
        
        return final_balance, balance_history, trade_logger.trades
        
    except Exception as e:
        logging.error(f"Error in backtest: {e}")
        print(f"Error in backtest: {e}")
        return initial_balance, [initial_balance], []

def live_trading(symbol='BBCA.JK', model=None, scaler=None, features=None, initial_balance=100000000):
    try:
        if model is None or scaler is None or features is None:
            logging.error("No model, scaler, or features provided for live trading")
            return None
        
        print(f"🔴 LIVE Trading for {symbol}")
        print("⚠️ WARNING: This is LIVE trading with real money implications!")
        print("=" * 80)
        
        balance = initial_balance
        shares = 0
        entry_price = None
        entry_time = None
        consecutive_losses = 0
        
        max_risk = 0.015
        take_profit = 0.012
        stop_loss = 0.006
        max_consecutive_losses = 3
        confidence_threshold = 0.7
        
        logging.info(f"Live Trading Started - Symbol: {symbol}, Initial Balance: Rp {initial_balance:,.0f}")
        
        while True:
            try:
                current_data = get_stock_data(symbol, '5m', '1d')
                if current_data.empty:
                    print("⚠️ Market data not available, waiting...")
                    time.sleep(60)
                    continue
                
                current_data = add_indicators(current_data)
                if current_data.empty or len(current_data) <= 10:
                    print("⚠️ Insufficient data for analysis...")
                    time.sleep(60)
                    continue
                
                latest = current_data.iloc[-1]
                current_price = latest['Close']
                current_time = current_data.index[-1]
                
                feature_data = [latest.get(feature, 0) for feature in features]
                X_current = np.array(feature_data).reshape(1, -1)
                X_current = np.nan_to_num(X_current, nan=0.0)
                X_scaled = scaler.transform(X_current)
                
                prediction = model.predict(X_scaled)[0]
                probability = model.predict_proba(X_scaled)[0][1]
                
                indicators = {
                    'rsi': latest.get('rsi', 0),
                    'macd': latest.get('macd', 0),
                    'bb_position': latest.get('bb_position', 0),
                    'volume_ratio': latest.get('volume_ratio', 0),
                    'probability': probability
                }
                
                print(f"📊 {current_time.strftime('%Y-%m-%d %H:%M:%S')} | Price: {current_price:,.0f} | "
                      f"RSI: {indicators['rsi']:.1f} | Prob: {probability:.3f} | Pred: {prediction}")
                
                if shares > 0 and entry_price is not None:
                    current_return = (current_price - entry_price) / entry_price
                    
                    if current_return >= take_profit:
                        profit = shares * (current_price - entry_price) - (shares * current_price * 0.001)
                        balance += shares * current_price - (shares * current_price * 0.001)
                        trade_logger.log_trade('sell', current_price, shares, profit, 'Take Profit',
                                             current_time, balance, indicators)
                        shares = 0
                        entry_price = None
                        entry_time = None
                        consecutive_losses = 0
                        continue
                    
                    elif current_return <= -stop_loss:
                        profit = shares * (current_price - entry_price) - (shares * current_price * 0.001)
                        balance += shares * current_price - (shares * current_price * 0.001)
                        trade_logger.log_trade('sell', current_price, shares, profit, 'Stop Loss',
                                             current_time, balance, indicators)
                        shares = 0
                        entry_price = None
                        entry_time = None
                        consecutive_losses += 1
                        continue
                
                if shares == 0 and prediction == 1 and probability >= confidence_threshold and \
                   consecutive_losses < max_consecutive_losses and \
                   20 <= indicators['rsi'] <= 80 and indicators['volume_ratio'] >= 1.2:
                    risk_amount = balance * max_risk
                    position_size = risk_amount / (current_price * stop_loss)
                    target_shares = max(100, int(position_size / 100) * 100)
                    
                    total_cost = target_shares * current_price * (1 + 0.001)
                    
                    if total_cost <= balance * 0.9:
                        balance -= total_cost
                        shares = target_shares
                        entry_price = current_price
                        entry_time = current_time
                        trade_logger.log_trade('buy', current_price, shares, None, None,
                                             current_time, balance, indicators)
                
                time.sleep(300)  # Wait 5 minutes before next iteration
                
            except KeyboardInterrupt:
                print("\n🛑 Live trading stopped by user")
                break
                
            except Exception as e:
                logging.error(f"Error in live trading loop: {e}")
                time.sleep(60)
                continue
        
        final_summary = trade_logger.get_trade_summary()
        print(f"\nFinal Summary:\n{final_summary}")
        logging.info(f"Live Trading Stopped\n{final_summary}")
        
    except Exception as e:
        logging.error(f"Error in live trading: {e}")
        print(f"Error in live trading: {e}")

def analyze_performance(balance_history, initial_balance, trades):
    try:
        if not balance_history or len(balance_history) < 2:
            print("Insufficient data for performance analysis")
            return
        
        final_balance = balance_history[-1]
        total_return = (final_balance - initial_balance) / initial_balance * 100
        
        balance_series = pd.Series(balance_history)
        returns = balance_series.pct_change().dropna()
        
        volatility = returns.std() * np.sqrt(len(returns)) * 100
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(len(returns)) if returns.std() > 0 else 0
        max_drawdown = ((balance_series.cummax() - balance_series) / balance_series.cummax()).max() * 100
        
        profitable_trades = len([t for t in trades if t.get('profit', 0) > 0])
        total_trades = len([t for t in trades if t.get('profit') is not None])
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        print(f"\n📈 PERFORMANCE ANALYSIS\n{'=' * 50}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Volatility: {volatility:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Trades: {total_trades}")
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(balance_history)
        plt.title('Portfolio Value Over Time')
        plt.ylabel('Balance (Rp)')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.hist(returns, bins=30, alpha=0.7)
        plt.title('Return Distribution')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        cumulative_returns = (balance_series / initial_balance - 1) * 100
        plt.plot(cumulative_returns)
        plt.title('Cumulative Returns (%)')
        plt.ylabel('Returns (%)')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        drawdown = (balance_series.cummax() - balance_series) / balance_series.cummax() * 100
        plt.fill_between(range(len(drawdown)), drawdown, alpha=0.7, color='red')
        plt.title('Drawdown')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('trading_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        logging.error(f"Error in performance analysis: {e}")
        print(f"Error in performance analysis: {e}")

def main():
    try:
        print("🚀 Starting Advanced Trading Bot...")
        print("=" * 80)
        
        SYMBOL = 'BBCA.JK'
        INITIAL_BALANCE = 100000000
        
        print("📊 Step 1: Preparing multi-timeframe data...")
        df = prepare_multi_timeframe_data(SYMBOL)
        
        if df.empty:
            print("❌ No data available. Exiting...")
            return
        
        print(f"✅ Data prepared: {len(df)} rows")
        
        print("\n🔧 Step 2: Preparing dataset for training...")
        X, y, scaler, features = prepare_dataset(df)
        
        if X is None:
            print("❌ Dataset preparation failed. Exiting...")
            return
        
        print("\n🤖 Step 3: Training machine learning model...")
        model, y_test = train_model(X, y)
        
        if model is None:
            print("❌ Model training failed. Exiting...")
            return
        
        print("\n📈 Step 4: Running backtest...")
        final_balance, balance_history, trades = backtest_trading(
            df.tail(len(y_test)), model, scaler, features, INITIAL_BALANCE
        )
        
        print("\n📊 Step 5: Analyzing performance...")
        analyze_performance(balance_history, INITIAL_BALANCE, trades)
        
        print("\n" + trade_logger.get_trade_summary())
        
        print("\n" + "="*80)
        choice = input("🔴 Do you want to start LIVE trading? (yes/no): ").lower().strip()
        
        if choice in ['yes', 'y']:
            print("⚠️ Starting live trading in 10 seconds...")
            print("⚠️ Use Ctrl+C to stop trading")
            time.sleep(10)
            live_trading(SYMBOL, model, scaler, features, INITIAL_BALANCE)
        else:
            print("✅ Backtest completed. Live trading skipped.")
        
    except KeyboardInterrupt:
        print("\n🛑 Trading bot stopped by user")
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        print(f"❌ Error in main execution: {e}")
    finally:
        print("\n🏁 Trading bot session ended")

if __name__ == "__main__":
    main()