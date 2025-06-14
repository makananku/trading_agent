import os
import time
import ccxt
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(
    filename='trading_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load .env untuk API Key
load_dotenv()
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
if not api_key or not api_secret:
    raise ValueError("API Key atau Secret Key tidak ditemukan di file .env!")

# Inisialisasi Binance Live dengan timeout lebih besar
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'timeout': 30000,  # Timeout 30 detik
    'options': {'defaultType': 'spot'},
    'urls': {
        'api': {
            'public': 'https://api1.binance.com/api',  # Endpoint alternatif
            'private': 'https://api1.binance.com/sapi'
        }
    }
})

# Preload markets dengan retry
def load_markets_with_retry(max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            logging.info("Memuat markets...")
            exchange.load_markets()
            logging.info("Markets berhasil dimuat.")
            return True
        except ccxt.RequestTimeout as e:
            logging.warning(f"Timeout saat load markets (percobaan {attempt+1}/{max_retries}): {e}")
            print(f"Timeout saat load markets, mencoba lagi dalam {delay} detik...")
            time.sleep(delay)
        except ccxt.AuthenticationError:
            logging.error("Autentikasi gagal. Periksa API Key dan Secret Key.")
            raise ValueError("Autentikasi gagal. Periksa API Key dan Secret Key di .env.")
        except ccxt.NetworkError as e:
            logging.error(f"Kesalahan jaringan saat load markets: {e}")
            print(f"Kesalahan jaringan: {e}")
            time.sleep(delay)
        except ccxt.BaseError as e:
            logging.error(f"Error saat load markets: {e}")
            print(f"Error saat load markets: {e}")
            time.sleep(delay)
    # Fallback: set markets secara manual untuk BTC/USDT
    exchange.markets = {'BTC/USDT': {'symbol': 'BTC/USDT', 'active': True, 'precision': {'amount': 8, 'price': 8}}}
    logging.warning("Gagal memuat markets, menggunakan fallback untuk BTC/USDT.")
    return False

load_markets_with_retry()

# Fungsi cek saldo
def check_balance(symbol='BTC/USDT'):
    try:
        balance = exchange.fetch_balance()
        usdt_free = balance['USDT']['free']
        btc_free = balance['BTC']['free']
        logging.info(f"Saldo USDT: {usdt_free}, Saldo BTC: {btc_free}")
        print(f"Saldo USDT: {usdt_free}, Saldo BTC: {btc_free}")
        return usdt_free, btc_free
    except Exception as e:
        logging.error(f"Error saat cek saldo: {e}")
        print(f"Error saat cek saldo: {e}")
        return 0, 0

# Fungsi ambil data dengan retry untuk rate limit dan timeout
def get_ohlcv(symbol, timeframe, limit=100, max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            logging.info(f"Berhasil mengambil data {symbol} timeframe {timeframe}")
            return df
        except ccxt.RateLimitExceeded:
            logging.warning(f"Rate limit terlampaui, mencoba lagi dalam {delay} detik...")
            print(f"Rate limit terlampaui, mencoba lagi dalam {delay} detik...")
            time.sleep(delay)
        except ccxt.RequestTimeout:
            logging.warning(f"Timeout saat fetch OHLCV (percobaan {attempt+1}/{max_retries}), mencoba lagi dalam {delay} detik...")
            print(f"Timeout saat fetch OHLCV, mencoba lagi dalam {delay} detik...")
            time.sleep(delay)
        except ccxt.AuthenticationError:
            logging.error("Autentikasi gagal saat fetch OHLCV. Periksa API Key.")
            raise ValueError("Autentikasi gagal saat fetch OHLCV. Periksa API Key.")
        except ccxt.BaseError as e:
            logging.error(f"Error saat fetch OHLCV: {e}")
            print(f"Error saat fetch OHLCV: {e}")
            time.sleep(delay)
    raise Exception(f"Gagal mengambil data {symbol} setelah {max_retries} percobaan.")

# Tambah indikator teknikal
def add_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['bb_upper'] = ta.volatility.BollingerBands(close=df['close']).bollinger_hband()
    df['macd'] = ta.trend.MACD(close=df['close']).macd()
    df['volume'] = df['volume']
    return df

# Ambil dan gabungkan data multi-timeframe
def prepare_multi_timeframe_data(symbol='BTC/USDT'):
    df_1m = add_indicators(get_ohlcv(symbol, '1m', limit=500))
    df_15m = add_indicators(get_ohlcv(symbol, '15m', limit=500))
    df_1h = add_indicators(get_ohlcv(symbol, '1h', limit=500))
    df_1d = add_indicators(get_ohlcv(symbol, '1d', limit=500))

    # Set index untuk merge
    df_1m.set_index('timestamp', inplace=True)
    df_15m.set_index('timestamp', inplace=True)
    df_1h.set_index('timestamp', inplace=True)
    df_1d.set_index('timestamp', inplace=True)

    # Merge data
    df = df_1m.copy()
    df = df.merge(df_15m[['rsi']], left_index=True, right_index=True, how='left', suffixes=('', '_15m'))
    df = df.merge(df_1h[['rsi']], left_index=True, right_index=True, how='left', suffixes=('', '_1h'))
    df = df.merge(df_1d[['rsi']], left_index=True, right_index=True, how='left', suffixes=('', '_1d'))
    df.dropna(inplace=True)
    return df

# Siapkan dataset untuk machine learning
def prepare_dataset(df):
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    features = ['rsi', 'ma20', 'ma50', 'bb_upper', 'macd', 'volume', 'rsi_15m', 'rsi_1h', 'rsi_1d']
    X = df[features]
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, features

# Training model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Akurasi: {accuracy_score(y_test, y_pred):.2f}")
    logging.info(f"Akurasi: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    logging.info(f"Precision: {precision_score(y_test, y_pred):.2f}")
    return model, X_test, y_test, y_pred

# Backtest dengan biaya trading, stop-loss, dan take-profit
def backtest(df_test, model, scaler, features, initial_balance=1000, fee_rate=0.001, stop_loss=0.02, take_profit=0.03):
    df_test['predicted'] = model.predict(scaler.transform(df_test[features]))
    balance = initial_balance
    btc = 0
    balance_history = []
    trades = []
    entry_price = None

    for i in range(len(df_test)):
        price = df_test.iloc[i]['close']
        pred = df_test.iloc[i]['predicted']
        if pred == 1 and balance > 0:
            btc = (balance * (1 - fee_rate)) / price
            balance = 0
            entry_price = price
            trades.append({'type': 'buy', 'price': price})
            logging.info(f"Backtest: Buy at ${price}")
        elif btc > 0:
            # Cek stop-loss dan take-profit
            if price <= entry_price * (1 - stop_loss):
                balance = btc * price * (1 - fee_rate)
                btc = 0
                trades.append({'type': 'sell', 'price': price, 'reason': 'stop_loss'})
                logging.info(f"Backtest: Sell at ${price} (Stop-Loss)")
            elif price >= entry_price * (1 + take_profit):
                balance = btc * price * (1 - fee_rate)
                btc = 0
                trades.append({'type': 'sell', 'price': price, 'reason': 'take_profit'})
                logging.info(f"Backtest: Sell at ${price} (Take-Profit)")
            elif pred == 0:
                balance = btc * price * (1 - fee_rate)
                btc = 0
                trades.append({'type': 'sell', 'price': price, 'reason': 'signal'})
                logging.info(f"Backtest: Sell at ${price} (Signal)")
        balance_history.append(balance + btc * price if btc > 0 else balance)

    if btc > 0:
        balance = btc * df_test.iloc[-1]['close'] * (1 - fee_rate)
        trades.append({'type': 'sell', 'price': df_test.iloc[-1]['close'], 'reason': 'end'})
        logging.info(f"Backtest: Sell at ${df_test.iloc[-1]['close']} (End)")
    return balance, balance_history, trades

# Plot equity curve
def plot_equity_curve(balance_history):
    plt.figure(figsize=(12, 6))
    plt.plot(balance_history, label='Equity Curve')
    plt.title('Backtest Equity Curve')
    plt.xlabel('Trade')
    plt.ylabel('Balance (USDT)')
    plt.grid()
    plt.legend()
    plt.show()

# Plot indikator
def plot_indicators(df):
    df[['close', 'ma20', 'ma50']].tail(200).plot(figsize=(12, 6))
    plt.title("BTC/USDT + MA20 + MA50")
    plt.grid()
    plt.show()

    df[['rsi', 'rsi_15m', 'rsi_1h', 'rsi_1d']].tail(200).plot(figsize=(12, 6))
    plt.title("Multi-timeframe RSI")
    plt.grid()
    plt.show()

# Fungsi untuk place order
def place_order(exchange, symbol, side, amount):
    try:
        if side == 'buy':
            order = exchange.create_market_buy_order(symbol, amount)
        else:
            order = exchange.create_market_sell_order(symbol, amount)
        print(f"Order {side} berhasil: {order}")
        logging.info(f"Order {side} berhasil: {order}")
        return order
    except ccxt.InsufficientFunds:
        print("Saldo tidak cukup!")
        logging.error("Saldo tidak cukup!")
    except ccxt.InvalidOrder as e:
        print(f"Order gagal: {e}")
        logging.error(f"Order gagal: {e}")
    except Exception as e:
        print(f"Error tak terduga: {e}")
        logging.error(f"Error tak terduga: {e}")
    return None

# Live trading dengan stop-loss dan take-profit
def live_trading(symbol='BTC/USDT', timeframe='1m', amount=0.001, model=None, scaler=None, features=None, stop_loss=0.02, take_profit=0.03):
    entry_price = None
    position_open = False
    # Cek saldo awal
    usdt_free, btc_free = check_balance()
    if usdt_free < amount * get_ohlcv(symbol, timeframe).iloc[-1]['close']:
        logging.error("Saldo USDT tidak cukup untuk trading!")
        raise ValueError("Saldo USDT tidak cukup untuk trading!")
    if btc_free < amount:
        logging.warning(f"Saldo BTC: {btc_free}. Mungkin tidak cukup untuk sell.")
    while True:
        try:
            df = prepare_multi_timeframe_data(symbol)
            latest_data = scaler.transform(df.tail(1)[features])
            prediction = model.predict(latest_data)[0]
            price = df.iloc[-1]['close']
            
            if prediction == 1 and not position_open:
                print(f"Buy {amount} BTC at ${price}")
                logging.info(f"Buy {amount} BTC at ${price}")
                order = place_order(exchange, symbol, 'buy', amount)
                if order:
                    entry_price = price
                    position_open = True
            elif position_open:
                # Cek stop-loss dan take-profit
                if price <= entry_price * (1 - stop_loss):
                    print(f"Sell {amount} BTC at ${price} (Stop-Loss)")
                    logging.info(f"Sell {amount} BTC at ${price} (Stop-Loss)")
                    order = place_order(exchange, symbol, 'sell', amount)
                    if order:
                        position_open = False
                elif price >= entry_price * (1 + take_profit):
                    print(f"Sell {amount} BTC at ${price} (Take-Profit)")
                    logging.info(f"Sell {amount} BTC at ${price} (Take-Profit)")
                    order = place_order(exchange, symbol, 'sell', amount)
                    if order:
                        position_open = False
                elif prediction == 0:
                    print(f"Sell {amount} BTC at ${price} (Signal)")
                    logging.info(f"Sell {amount} BTC at ${price} (Signal)")
                    order = place_order(exchange, symbol, 'sell', amount)
                    if order:
                        position_open = False
            time.sleep(60)  # Tunggu 1 menit untuk timeframe 1m
        except Exception as e:
            print(f"Error: {e}")
            logging.error(f"Error: {e}")
            time.sleep(5)

# Main execution
if __name__ == "__main__":
    try:
        # Cek saldo awal
        check_balance()
        # Siapkan data
        df = prepare_multi_timeframe_data()
        X, y, scaler, features = prepare_dataset(df)

        # Train model
        model, X_test, y_test, y_pred = train_model(X, y)

        # Backtest
        df_test = df.iloc[-len(y_test):].copy()
        final_balance, balance_history, trades = backtest(df_test, model, scaler, features)
        print(f"Hasil akhir simulasi: ${final_balance:.2f}")
        logging.info(f"Hasil akhir simulasi: ${final_balance:.2f}")
        print(f"Jumlah trade: {len(trades)}")
        logging.info(f"Jumlah trade: {len(trades)}")

        # Visualisasi
        plot_equity_curve(balance_history)
        plot_indicators(df)

        # Simpan data (opsional)
        df.to_csv("multi_timeframe_btc_data.csv")

        # Uncomment untuk live trading
        # live_trading(symbol='BTC/USDT', timeframe='1m', amount=0.001, model=model, scaler=scaler, features=features)
    except Exception as e:
        print(f"Error di main execution: {e}")
        logging.error(f"Error di main execution: {e}")