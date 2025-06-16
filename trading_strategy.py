import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
import joblib
import os
import numpy as np
import yfinance as yf
import time
import matplotlib
matplotlib.use('Agg')

# --- МОДЕЛЬ СТРАТЕГИИ (без изменений) ---
class SMCStrategy(Strategy):
    lookback_period = 20
    sl_ratio = 0.004
    tp_ratio = 0.01
    risk_percent = 0.01
    start_hour = 13
    end_hour = 17
    
    ml_model = None
    prediction_threshold = 0.67

    def init(self):
        self.signal_to_trade = 0

    def next(self):
        if self.signal_to_trade == -1 and not self.position:
            entry_price = self.data.Open[-1] 
            sl_price = entry_price * (1 + self.sl_ratio)
            tp_price = entry_price * (1 - self.tp_ratio)

            if not (np.isfinite(entry_price) and np.isfinite(sl_price) and np.isfinite(tp_price) and tp_price < entry_price < sl_price):
                self.signal_to_trade = 0
                return

            stop_distance_per_unit = sl_price - entry_price
            if stop_distance_per_unit > 0:
                units_to_trade = (self.equity * self.risk_percent) / stop_distance_per_unit
                if units_to_trade > 0:
                    try:
                        self.sell(size=int(units_to_trade), sl=sl_price, tp=tp_price)
                    except Exception:
                        self.signal_to_trade = 0
                        return
            self.signal_to_trade = 0
            return

        if self.position:
            self.signal_to_trade = 0
            return

        current_hour = self.data.index[-1].hour
        if not (self.start_hour <= current_hour <= self.end_hour) or self.ml_model is None:
            return

        current_index = len(self.data.Close) - 1
        if current_index < self.lookback_period:
            return

        start_index = current_index - self.lookback_period
        recent_dxy_low = self.data.DXY_Low[start_index:current_index].min()
        dxy_raid = self.data.DXY_Low[-1] < recent_dxy_low
        recent_eurusd_high = self.data.High[start_index:current_index].max()
        eurusd_judas_swing = self.data.High[-1] > recent_eurusd_high
        
        if dxy_raid and eurusd_judas_swing:
            current_features = np.array([
                self.data.RSI[-1], self.data.MACD[-1], self.data.MACD_hist[-1],
                self.data.MACD_signal[-1], self.data.ATR[-1]
            ]).reshape(1, -1)
            
            if np.isnan(current_features).any():
                return
            
            win_probability = self.ml_model.predict_proba(current_features)[0][1]
            if win_probability >= self.prediction_threshold:
                self.signal_to_trade = -1

# --- ФУНКЦИЯ ЗАГРУЗКИ ДАННЫХ ---
def load_data_from_yfinance(ticker, period="2y", interval="30m"):
    print(f"Загрузка данных для {ticker} из Yahoo Finance за период {period}...")
    for i in range(3):
        try:
            df = yf.download(tickers=ticker, period=period, interval=interval)
            if not df.empty:
                df.index = df.index.tz_convert('UTC')
                print(f"Данные для {ticker} успешно загружены.")
                return df
        except Exception as e:
            print(f"Ошибка при загрузке {ticker} (попытка {i+1}/3): {e}. Повтор через 5 секунд...")
            time.sleep(5)
    print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить данные для {ticker} после 3 попыток. !!!")
    return None

# --- ОСНОВНАЯ ФУНКЦИЯ ДЛЯ ЗАПУСКА БЭКТЕСТА ---
def run_backtest(prediction_threshold=0.67):
    """
    Запускает полный процесс бэктестинга и возвращает результаты.
    """
    print(f"--- ЗАПУСК БЭКТЕСТА (порог={prediction_threshold}) ---")
    
    # 1. Загрузка данных
    eurusd_data = load_data_from_yfinance('EURUSD=X')
    dxy_data = load_data_from_yfinance('DX-Y.NYB')

    if eurusd_data is None or dxy_data is None:
        return "Ошибка: Не удалось загрузить данные из Yahoo Finance.", None

    # 2. Подготовка данных
    eurusd_data.ta.rsi(length=14, append=True)
    eurusd_data.ta.macd(fast=12, slow=26, signal=9, append=True)
    eurusd_data.ta.atr(length=14, append=True)
    eurusd_data.rename(columns={'RSI_14':'RSI', 'MACD_12_26_9':'MACD', 'MACDh_12_26_9':'MACD_hist', 'MACDs_12_26_9':'MACD_signal', 'ATRr_14':'ATR'}, inplace=True)
    
    dxy_data_renamed = dxy_data.rename(columns={'Low': 'DXY_Low'})
    data = pd.concat([eurusd_data, dxy_data_renamed['DXY_Low']], axis=1)
    data.dropna(inplace=True)
    
    # 3. Загрузка модели
    MODEL_FILE = 'ml_model_final_fix.joblib'
    if not os.path.exists(MODEL_FILE):
        return f"Ошибка: Файл модели {MODEL_FILE} не найден.", None
    model = joblib.load(MODEL_FILE)
    
    # 4. Запуск бэктеста
    SMCStrategy.prediction_threshold = prediction_threshold
    SMCStrategy.ml_model = model

    bt = Backtest(data, SMCStrategy, cash=10000, commission=.0002, margin=0.05)
    
    try:
        stats = bt.run()
        plot_filename = f"backtest_plot_{int(time.time())}.png"
        bt.plot(filename=plot_filename, plot_drawdown=True, plot_equity=True)
        print("--- БЭКТЕСТ ЗАВЕРШЕН ---")
        return stats.to_string(), plot_filename
    except Exception as e:
        print(f"ОШИБКА ВО ВРЕМЯ БЭКТЕСТА: {e}")
        return f"Ошибка во время выполнения бэктеста: {e}", None
