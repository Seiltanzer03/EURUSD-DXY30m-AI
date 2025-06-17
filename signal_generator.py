import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance as yf
import joblib
import ssl

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
        return None, None, None
    last = data.iloc[-1]
    features = [last['RSI'], last['MACD'], last['MACD_hist'], last['MACD_signal'], last['ATR']]
    if any(np.isnan(features)):
        return None, None, None
    model = joblib.load(MODEL_FILE)
    win_prob = model.predict_proba([features])[0][1]
    signal = win_prob >= PREDICTION_THRESHOLD
    return signal, win_prob, last

if __name__ == "__main__":
    signal, win_prob, last = get_last_signal()
    if signal is None:
        print("Нет сигнала (недостаточно данных или NaN)")
    elif signal:
        print(f"СИГНАЛ: SELL EURUSD | Вероятность: {win_prob:.2%} | Время: {last.name}")
    else:
        print(f"Нет сигнала | Вероятность: {win_prob:.2%} | Время: {last.name}") 