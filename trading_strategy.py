import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
import joblib
import os
import numpy as np
import yfinance as yf
import time
import requests

# Определяем класс стратегии, чтобы он был доступен для импорта
class SMCStrategy(Strategy):
    lookback_period = 20
    sl_ratio = 0.004
    tp_ratio = 0.01
    risk_percent = 0.01
    start_hour = 13
    end_hour = 17
    
    # Эти параметры будут устанавливаться динамически
    ml_model = None
    prediction_threshold = 0.5

    def init(self):
        self.signal_to_trade = 0

    def next(self):
        # Логика генерации сигнала
        current_hour = self.data.index[-1].hour
        is_trading_time = self.start_hour <= current_hour <= self.end_hour
        
        if not is_trading_time or self.position or self.ml_model is None:
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
        else:
            self.signal_to_trade = 0

        # Логика исполнения сделки
        if self.signal_to_trade == -1 and not self.position:
            entry_price = self.data.Open[-1] 
            sl_price = entry_price * (1 + self.sl_ratio)
            tp_price = entry_price * (1 - self.tp_ratio)

            if not (np.isfinite(entry_price) and np.isfinite(sl_price) and np.isfinite(tp_price) and (tp_price < entry_price < sl_price)):
                self.signal_to_trade = 0
                return

            stop_distance_per_unit = sl_price - entry_price
            if stop_distance_per_unit > 0:
                initial_equity = 10000 # Используем фиксированное значение для расчета риска
                fixed_risk_amount = initial_equity * self.risk_percent
                units_to_trade = fixed_risk_amount / stop_distance_per_unit
                if units_to_trade > 0:
                    try:
                        self.sell(size=int(units_to_trade), sl=sl_price, tp=tp_price)
                    except Exception:
                        pass # Игнорируем ошибки исполнения
            self.signal_to_trade = 0

def load_data_from_yfinance(ticker, period="2mo", interval="30m"):
    """Загружает данные из Yahoo Finance с User-Agent."""
    print(f"Загрузка {period} данных для {ticker}...")
    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    try:
        df = yf.download(tickers=ticker, period=period, interval=interval, auto_adjust=True, session=session)
        if df.empty:
            raise ValueError(f"Нет данных для {ticker}. Рынок может быть закрыт.")
        df.index = df.index.tz_convert('UTC')
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"Данные для {ticker} успешно загружены.")
        return df
    except Exception as e:
        print(f"Критическая ошибка при загрузке {ticker}: {e}")
        raise

def run_backtest(threshold=0.67):
    """Основная функция для запуска бэктеста."""
    print("--- Запуск бэктеста ---")
    
    # 1. Загрузка данных
    try:
        eurusd_data = load_data_from_yfinance('EURUSD=X')
        dxy_data = load_data_from_yfinance('DX-Y.NYB')
    except Exception as e:
        return f"Ошибка загрузки данных: {e}", None

    # 2. Подготовка данных
    eurusd_data.ta.rsi(length=14, append=True)
    eurusd_data.ta.macd(fast=12, slow=26, signal=9, append=True)
    eurusd_data.ta.atr(length=14, append=True)
    eurusd_data.rename(columns={'RSI_14':'RSI', 'MACD_12_26_9':'MACD', 'MACDh_12_26_9':'MACD_hist', 'MACDs_12_26_9':'MACD_signal', 'ATRr_14':'ATR'}, inplace=True)
    
    dxy_data_renamed = dxy_data.rename(columns={'Low': 'DXY_Low'})
    data = pd.concat([eurusd_data, dxy_data_renamed['DXY_Low']], axis=1)
    data.dropna(inplace=True)

    # 3. Загрузка модели
    model_file = 'ml_model_final_fix.joblib'
    if not os.path.exists(model_file):
        return "Файл модели не найден!", None
    model = joblib.load(model_file)

    # 4. Запуск бэктеста
    SMCStrategy.ml_model = model
    SMCStrategy.prediction_threshold = threshold
    
    bt = Backtest(data, SMCStrategy, cash=10000, commission=.0002, margin=0.05)
    stats = bt.run()
    
    # 5. Сохранение результатов
    plot_filename = f"backtest_report_{threshold}_{int(time.time())}.html"
    bt.plot(filename=plot_filename, open_browser=False)
    
    print("--- Бэктест завершен ---")
    return stats, plot_filename

# Этот блок больше не нужен, так как запуск будет из бота
# if __name__ == "__main__":
#     run_backtest()
