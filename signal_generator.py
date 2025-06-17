import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance as yf
import joblib
import ssl
import matplotlib.pyplot as plt

# Отключение SSL-проверки для yfinance
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

MODEL_FILE = 'ml_model_final_fix.joblib'
PREDICTION_THRESHOLD = 0.1
LOOKBACK_PERIOD = 20
TIMEFRAME = '5m'
SL_RATIO = 0.004
TP_RATIO = 0.01


def flatten_multiindex_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def load_data(period="2d", interval="5m"):
    eurusd = yf.download('EURUSD=X', period=period, interval=interval, auto_adjust=True)
    dxy = yf.download('DX-Y.NYB', period=period, interval=interval, auto_adjust=True)
    eurusd = flatten_multiindex_columns(eurusd)
    dxy = flatten_multiindex_columns(dxy)
    if eurusd.empty or dxy.empty:
        raise ValueError('Нет данных для EURUSD или DXY')
    eurusd.ta.rsi(length=14, append=True)
    eurusd.ta.macd(fast=12, slow=26, signal=9, append=True)
    eurusd.ta.atr(length=14, append=True)
    eurusd.rename(columns={'RSI_14':'RSI', 'MACD_12_26_9':'MACD', 'MACDh_12_26_9':'MACD_hist', 'MACDs_12_26_9':'MACD_signal', 'ATRr_14':'ATR'}, inplace=True)
    dxy = dxy.rename(columns={'Low': 'DXY_Low'})
    data = pd.concat([eurusd, dxy['DXY_Low']], axis=1)
    data.dropna(inplace=True)
    data.index = data.index.tz_convert('UTC')
    return data

def get_last_signal():
    data = load_data()
    if len(data) < LOOKBACK_PERIOD:
        return None, None, None, None
    last = data.iloc[-1]
    features = [last['RSI'], last['MACD'], last['MACD_hist'], last['MACD_signal'], last['ATR']]
    if any(np.isnan(features)):
        return None, None, None, None
    model = joblib.load(MODEL_FILE)
    win_prob = model.predict_proba([features])[0][1]
    signal = win_prob >= PREDICTION_THRESHOLD
    entry = last['Open']
    sl = entry * (1 + SL_RATIO)
    tp = entry * (1 - TP_RATIO)
    # Генерация графика, если есть сигнал
    plot_path = None
    if signal:
        plt.figure(figsize=(10, 5))
        candles = data[-100:]
        plt.plot(candles.index, candles['Close'], label='Close', color='black')
        plt.axhline(entry, color='blue', linestyle='--', label='Entry')
        plt.axhline(sl, color='red', linestyle='--', label='Stop Loss')
        plt.axhline(tp, color='green', linestyle='--', label='Take Profit')
        plt.scatter([last.name], [entry], color='blue', marker='v', s=100, label='Sell Entry')
        plt.legend()
        plt.title(f'SELL EURUSD ({TIMEFRAME})')
        plt.tight_layout()
        plot_path = 'signal.png'
        plt.savefig(plot_path)
        plt.close()
    return signal, entry, sl, tp, last, plot_path

if __name__ == "__main__":
    signal, entry, sl, tp, last, plot_path = get_last_signal()
    if signal is None:
        print("Нет сигнала (недостаточно данных или NaN)")
    elif signal:
        print(f"СИГНАЛ: SELL EURUSD ({TIMEFRAME})\nВремя: {last.name}\nEntry: {entry:.5f}\nStop Loss: {sl:.5f}\nTake Profit: {tp:.5f}")
        if plot_path:
            print(f"GRAPH_PATH: {plot_path}")
    else:
        print(f"Нет сигнала | Время: {last.name}") 
