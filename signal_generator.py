import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance as yf
import joblib
import ssl
import matplotlib.pyplot as plt
from signal_core import generate_signal_and_plot_30m

# Отключение SSL-проверки для yfinance
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

MODEL_FILE = 'ml_model_final_fix.joblib'
PREDICTION_THRESHOLD = 0.55
LOOKBACK_PERIOD = 34
SL_RATIO = 0.004
TP_RATIO = 0.01
TIMEFRAME_30M = '30m'

def flatten_multiindex_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def load_data(period="7d", interval="30m"):
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
        return None, None, None, None, None, None
    
    last_candle = data.iloc[-2]  # Анализируем предыдущую закрытую свечу
    current_hour = last_candle.name.hour
    
    # Ограничение по времени для 30м сигнала
    if not (13 <= current_hour <= 17):
        return None, None, None, None, None, None
    
    start_index = len(data) - LOOKBACK_PERIOD - 2
    end_index = len(data) - 2
    
    if start_index < 0 or end_index <= start_index:
        return None, None, None, None, None, None
    
    eurusd_judas_swing = last_candle['High'] > data['High'].iloc[start_index:end_index].max()
    dxy_raid = last_candle['DXY_Low'] < data['DXY_Low'].iloc[start_index:end_index].min()
    
    signal = False
    entry, sl, tp, plot_path = None, None, None, None
    
    if eurusd_judas_swing and dxy_raid:
        features = [last_candle[col] for col in ['RSI', 'MACD', 'MACD_hist', 'MACD_signal', 'ATR']]
        if any(np.isnan(features)):
            return None, None, None, None, None, None
        
        model = joblib.load(MODEL_FILE)
        win_prob = model.predict_proba([features])[0][1]
        
        if win_prob >= PREDICTION_THRESHOLD:
            signal = True
            entry = last_candle['Open']
            sl = entry * (1 + SL_RATIO)
            tp = entry * (1 - TP_RATIO)
            
            # Генерация графика, если есть сигнал
            try:
                plt.figure(figsize=(10, 5))
                candles = data[-60:]
                plt.plot(candles.index, candles['Close'], label='Close', color='black')
                plt.axhline(entry, color='blue', linestyle='--', label='Entry')
                plt.axhline(sl, color='red', linestyle='--', label='Stop Loss')
                plt.axhline(tp, color='green', linestyle='--', label='Take Profit')
                plt.scatter([last_candle.name], [entry], color='blue', marker='v', s=100, label='Sell Entry')
                plt.legend()
                plt.title(f'SELL EURUSD ({TIMEFRAME_30M})')
                plt.tight_layout()
                plot_path = 'signal_30m.png'
                plt.savefig(plot_path)
                plt.close()
            except Exception as e:
                print(f"Ошибка при создании графика: {e}")
                plot_path = None
    
    return signal, entry, sl, tp, last_candle, plot_path

if __name__ == "__main__":
    signal, entry, sl, tp, last_candle, plot_path, timeframe, status = generate_signal_and_plot_30m()
    if signal is None:
        print("Нет сигнала (недостаточно данных или NaN)")
    elif signal:
        print(f"СИГНАЛ: SELL EURUSD ({timeframe})\nВремя: {last_candle.name}\nEntry: {entry:.5f}\nStop Loss: {sl:.5f}\nTake Profit: {tp:.5f}")
        if plot_path:
            print(f"GRAPH_PATH: {plot_path}")
    else:
        print(f"Нет сигнала | Время: {last_candle.name if last_candle is not None else 'N/A'}") 
