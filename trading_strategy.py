import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
import joblib
import os
import numpy as np
import yfinance as yf
import time
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import weasyprint

def flatten_multiindex_columns(df):
    """
    Проверяет, являются ли колонки MultiIndex, и если да, 'выпрямляет' их,
    оставляя только верхний уровень (e.g., 'Open', 'Close').
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

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

def load_data_from_yfinance(ticker, period="7d", interval="30m"):
    """Загружает данные из Yahoo Finance и обрабатывает возможный MultiIndex."""
    print(f"Загрузка {period} данных для {ticker}...")
    try:
        df = yf.download(tickers=ticker, period=period, interval=interval, auto_adjust=True)
        df = flatten_multiindex_columns(df)

        if df.empty:
            raise ValueError(f"Нет данных для {ticker}. Рынок может быть закрыт.")
        df.index = df.index.tz_convert('UTC')
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"Данные для {ticker} успешно загружены.")
        return df
    except Exception as e:
        print(f"Критическая ошибка при загрузке {ticker}: {e}")
        raise

def run_backtest(threshold=0.55):
    """Основная функция для запуска бэктеста на данных Yahoo."""
    print("--- Запуск бэктеста на данных Yahoo ---")
    
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
    if not os.path.exists(MODEL_FILE):
        return f"Файл модели не найден: {MODEL_FILE}", None
    model = joblib.load(MODEL_FILE)

    # 4. Запуск бэктеста
    SMCStrategy.ml_model = model
    SMCStrategy.prediction_threshold = threshold
    
    bt = Backtest(data, SMCStrategy, cash=10000, commission=.0002)
    stats = bt.run()
    
    # 5. Сохранение результатов в виде PDF
    html_filename = f"backtest_report_{threshold}_{int(time.time())}.html"
    bt.plot(filename=html_filename, open_browser=False)

    pdf_filename = html_filename.replace('.html', '.pdf')
    try:
        print(f"Конвертация {html_filename} в {pdf_filename}...")
        weasyprint.HTML(html_filename).write_pdf(pdf_filename)
        print("Конвертация завершена.")
        os.remove(html_filename) # Удаляем временный HTML
        return stats, pdf_filename
    except Exception as e:
        print(f"Ошибка при конвертации HTML в PDF: {e}")
        # Если не вышло, возвращаем хотя бы HTML
        return stats, html_filename

def run_backtest_local(eurusd_file, dxy_file, threshold):
    """Запускает бэктест на локальных CSV-файлах."""
    print("--- Запуск ЛОКАЛЬНОГО бэктеста ---")
    
    try:
        # 1. Загрузка и подготовка данных
        data = generate_features_for_backtest(eurusd_file, dxy_file)
    except Exception as e:
        return f"Ошибка подготовки данных: {e}", None

    # 2. Загрузка модели
    if not os.path.exists(MODEL_FILE):
        return f"Файл модели не найден: {MODEL_FILE}", None
    model = joblib.load(MODEL_FILE)

    # 3. Запуск бэктеста
    SMCStrategy.ml_model = model
    SMCStrategy.prediction_threshold = threshold
    
    bt = Backtest(data, SMCStrategy, cash=10000, commission=.0002)
    try:
        stats = bt.run()
    except Exception as e:
        return f"Ошибка во время выполнения бэктеста: {e}", None
        
    # 4. Сохранение результатов в виде PDF
    html_filename = f"backtest_local_report_{threshold}_{int(time.time())}.html"
    bt.plot(filename=html_filename, open_browser=False)

    pdf_filename = html_filename.replace('.html', '.pdf')
    try:
        print(f"Конвертация {html_filename} в {pdf_filename}...")
        weasyprint.HTML(html_filename).write_pdf(pdf_filename)
        print("Конвертация завершена.")
        os.remove(html_filename)
        return stats, pdf_filename
    except Exception as e:
        print(f"Ошибка при конвертации HTML в PDF: {e}")
        return stats, html_filename

def generate_features_for_backtest(eurusd_csv, dxy_csv):
    """Общая функция для загрузки и подготовки данных для бэктестов."""
    try:
        eurusd_data = pd.read_csv(eurusd_csv, parse_dates=['Gmt time'])
        eurusd_data.rename(columns={
            'Gmt time': 'Datetime', 'Open': 'Open', 'High': 'High',
            'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume',
        }, inplace=True)
        eurusd_data.set_index('Datetime', inplace=True)
        # Указываем правильный формат, если он отличается
        try:
            eurusd_data.index = pd.to_datetime(eurusd_data.index, format='%d.%m.%Y %H:%M:%S.%f')
        except (ValueError, TypeError):
             eurusd_data.index = pd.to_datetime(eurusd_data.index)
        eurusd_data.index = eurusd_data.index.tz_localize('UTC')

        dxy_data = pd.read_csv(dxy_csv, parse_dates=['Gmt time'])
        dxy_data.rename(columns={'Gmt time': 'Datetime', 'Low': 'DXY_Low'}, inplace=True)
        dxy_data.set_index('Datetime', inplace=True)
        try:
            dxy_data.index = pd.to_datetime(dxy_data.index, format='%d.%m.%Y %H:%M:%S.%f')
        except (ValueError, TypeError):
            dxy_data.index = pd.to_datetime(dxy_data.index)

        dxy_data.index = dxy_data.index.tz_localize('UTC')
    except Exception as e:
        raise ValueError(f"Ошибка при чтении или обработке CSV: {e}")

    # Расчет индикаторов
    eurusd_data.ta.rsi(length=14, append=True)
    eurusd_data.ta.macd(fast=12, slow=26, signal=9, append=True)
    eurusd_data.ta.atr(length=14, append=True)
    eurusd_data.rename(columns={
        'RSI_14':'RSI', 'MACD_12_26_9':'MACD', 'MACDh_12_26_9':'MACD_hist',
        'MACDs_12_26_9':'MACD_signal', 'ATRr_14':'ATR'
    }, inplace=True)

    # Объединение
    data = pd.concat([eurusd_data, dxy_data[['DXY_Low']]], axis=1)
    data.dropna(inplace=True)
    
    return data

# Этот блок больше не нужен, так как запуск будет из бота
# if __name__ == "__main__":
#     run_backtest()
