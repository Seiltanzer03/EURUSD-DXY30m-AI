import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance as yf
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
import warnings

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

def generate_signal_and_plot():
    data = load_data()
    if len(data) < LOOKBACK_PERIOD:
        return None, None, None, None, None, None
    last = data.iloc[-1]
    features = [last['RSI'], last['MACD'], last['MACD_hist'], last['MACD_signal'], last['ATR']]
    if any(np.isnan(features)):
        return None, None, None, None, None, None
    model = joblib.load(MODEL_FILE)
    win_prob = model.predict_proba([features])[0][1]
    signal = win_prob >= PREDICTION_THRESHOLD
    entry = last['Open']
    sl = entry * (1 + SL_RATIO)
    tp = entry * (1 - TP_RATIO)
    plot_path = None
    if signal:
        candles = data[~data.index.duplicated(keep='last')].tail(50).copy()
        if len(candles) < 10:
            warnings.warn('Недостаточно данных для построения графика (меньше 10 свечей)')
            plot_path = None
        else:
            candles.index.name = 'Date'
            addplots = [
                mpf.make_addplot([entry]*len(candles), color='blue', linestyle='--', width=1, label='Entry'),
                mpf.make_addplot([sl]*len(candles), color='red', linestyle='--', width=1, label='Stop Loss'),
                mpf.make_addplot([tp]*len(candles), color='green', linestyle='--', width=1, label='Take Profit'),
            ]
            fig, axlist = mpf.plot(
                candles,
                type='candle',
                style='charles',
                addplot=addplots,
                returnfig=True,
                title=f'SELL EURUSD ({TIMEFRAME})',
                ylabel='Price',
                figsize=(10, 5)
            )
            ax = axlist[0]
            ax.scatter([candles.index[-1]], [entry], color='blue', marker='v', s=100, label='Sell Entry')
            fig.tight_layout()
            plot_path = 'signal.png'
            fig.savefig(plot_path)
            plt.close(fig)
    return signal, entry, sl, tp, last, plot_path 
