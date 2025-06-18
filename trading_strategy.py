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
import mplfinance as mpf

def flatten_multiindex_columns(df):
    """
    Проверяет, являются ли колонки MultiIndex, и если да, 'выпрямляет' их,
    оставляя только верхний уровень (e.g., 'Open', 'Close').
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# Создаем кастомный стиль для свечей в стиле dark mode
mc = mpf.make_marketcolors(
    up='#26a69a',  # Зеленый
    down='#ef5350', # Красный
    edge='inherit',
    wick={'up':'#26a69a', 'down':'#ef5350'},
    volume='inherit'
)
s = mpf.make_mpf_style(
    base_mpf_style='nightclouds', # Темная тема
    marketcolors=mc,
    gridstyle=':',
    gridcolor='gray',
    gridaxis='both',
    y_on_right=False
)

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

def plot_backtest_results_to_pdf(stats, data, filename, title='Результаты бэктеста'):
    """
    Создает и сохраняет кастомный PDF-отчет бэктеста со свечным графиком,
    линиями сделок и встроенной статистикой.
    """
    try:
        if data.empty:
            print("Ошибка: Нет данных для построения графика.")
            return None
            
        trades = stats['_trades']
        equity_curve = stats['_equity_curve']
        
        # --- 1. Подготовка текста со статистикой ---
        stats_text = (
            f"Start: {stats['Start']}\n"
            f"End: {stats['End']}\n"
            f"Duration: {stats['Duration']}\n"
            f"Exposure Time [%]: {stats['Exposure Time [%]']:.2f}\n"
            f"Equity Final [$]: {stats['Equity Final [$]']:,.2f}\n"
            f"Return [%]: {stats['Return [%]']:.2f}\n"
            f"Max. Drawdown [%]: {stats['Max. Drawdown [%]']:.2f}\n"
            f"Win Rate [%]: {stats['Win Rate [%]']:.2f}\n"
            f"# Trades: {stats['# Trades']}\n"
            f"Profit Factor: {stats['Profit Factor']:.2f}\n"
            f"SQN: {stats['SQN']:.2f}\n"
            f"Expectancy [%]: {stats['Expectancy [%]']:.2f}"
        )

        # --- 2. Создание графика ---
        fig, axes = mpf.plot(
            data,
            type='candle',
            style=s,
            ylabel='Цена EURUSD',
            figsize=(25, 18),
            returnfig=True,
            panel_ratios=(8, 2), # 8 частей для цены, 2 для equity
            addplot=[mpf.make_addplot(equity_curve['Equity'], panel=1, color='cyan', ylabel='Equity ($)')]
        )
        ax_main = axes[0]

        # --- 3. Нанесение линий сделок на график ---
        if not trades.empty:
            for _, trade in trades.iterrows():
                ax_main.plot(
                    [trade.EntryTime, trade.ExitTime],
                    [trade.EntryPrice, trade.ExitPrice],
                    'c--', # Cyan, dashed line
                    linewidth=1.0
                )
        
        # --- 4. Добавление текста и заголовка ---
        fig.text(0.02, 0.98, stats_text, 
                 ha='left', va='top', fontsize=12,
                 fontfamily='monospace', # Моноширинный шрифт для аккуратности
                 bbox=dict(boxstyle='round', facecolor='#2E2E2E', alpha=0.8))
        
        fig.suptitle(title, fontsize=20, y=0.99)
        
        # --- 5. Сохранение ---
        fig.savefig(filename, bbox_inches='tight', format='pdf', dpi=200)
        plt.close(fig)
        
        print(f"PDF-отчет бэктеста успешно сохранен в {filename}")
        return filename
    except Exception as e:
        print(f"Критическая ошибка при создании PDF-отчета: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_backtest(threshold=0.55):
    """
    Основная функция для запуска бэктеста.
    Возвращает статистику и путь к файлу с PDF-отчетом.
    """
    print("--- Запуск бэктеста ---")
    
    # 1. Загрузка данных за 2 месяца (60 дней)
    try:
        eurusd_data = load_data_from_yfinance('EURUSD=X', period='60d', interval='30m')
        dxy_data = load_data_from_yfinance('DX-Y.NYB', period='60d', interval='30m')
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
    
    # 5. Сохранение результатов в виде PDF
    plot_filename = f"backtest_report_{threshold}_{int(time.time())}.pdf"
    plot_backtest_results_to_pdf(stats, data, plot_filename, title=f"Бэктест M30 (60 дней) | Порог: {threshold}")
    
    print("--- Бэктест завершен ---")
    # Возвращаем статистику и путь к файлу отчета
    return stats, plot_filename

def run_backtest_local(eurusd_csv='EURUSD_Candlestick_30_m_BID_18.06.2022-18.06.2025 (2).csv', dxy_csv='DOLLAR.IDXUSD_Candlestick_30_m_BID_18.06.2022-18.06.2025 (2).csv', threshold=0.55):
    """
    Запуск бэктеста на локальных csv-файлах котировок.
    Возвращает статистику и путь к PDF-отчету.
    """
    print("--- Запуск ЛОКАЛЬНОГО бэктеста ---")
    # 1. Загрузка данных из CSV
    try:
        eurusd_data = pd.read_csv(eurusd_csv, parse_dates=['Gmt time'], dayfirst=True)
        eurusd_data.rename(columns={
            'Gmt time': 'Datetime',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume',
        }, inplace=True)
        eurusd_data.set_index('Datetime', inplace=True)
        eurusd_data.index = pd.to_datetime(eurusd_data.index) #, format='%d.%m.%Y %H:%M:%S.%f')
        eurusd_data.index = eurusd_data.index.tz_localize('UTC')

        dxy_data = pd.read_csv(dxy_csv, parse_dates=['Gmt time'], dayfirst=True)
        dxy_data.rename(columns={
            'Gmt time': 'Datetime',
            'Low': 'DXY_Low',
        }, inplace=True)
        dxy_data.set_index('Datetime', inplace=True)
        dxy_data.index = pd.to_datetime(dxy_data.index) #, format='%d.%m.%Y %H:%M:%S.%f')
        dxy_data.index = dxy_data.index.tz_localize('UTC')
    except Exception as e:
        return f"Ошибка загрузки локальных данных: {e}", None

    # 2. Подготовка данных (индикаторы)
    eurusd_data.ta.rsi(length=14, append=True)
    eurusd_data.ta.macd(fast=12, slow=26, signal=9, append=True)
    eurusd_data.ta.atr(length=14, append=True)
    eurusd_data.rename(columns={'RSI_14':'RSI', 'MACD_12_26_9':'MACD', 'MACDh_12_26_9':'MACD_hist', 'MACDs_12_26_9':'MACD_signal', 'ATRr_14':'ATR'}, inplace=True)

    # 3. Объединение с DXY
    dxy_data_renamed = dxy_data[['DXY_Low']]
    data = pd.concat([eurusd_data, dxy_data_renamed], axis=1)
    data.dropna(inplace=True)

    # 4. Загрузка модели
    model_file = 'ml_model_final_fix.joblib'
    if not os.path.exists(model_file):
        return "Файл модели не найден!", None
    model = joblib.load(model_file)

    # 5. Запуск бэктеста
    SMCStrategy.ml_model = model
    SMCStrategy.prediction_threshold = threshold
    
    bt = Backtest(data, SMCStrategy, cash=10000, commission=.0002, margin=0.05)
    stats = bt.run()
    
    # 6. Сохранение результатов в виде интерактивного PDF
    plot_filename = f"backtest_local_report_{threshold}_{int(time.time())}.pdf"
    plot_backtest_results_to_pdf(stats, data, plot_filename, title=f"Полный бэктест M30 | Порог: {threshold}")
    
    print("--- Локальный бэктест завершен ---")
    return stats, plot_filename

def run_backtest_local_no_ml(eurusd_csv='eurusd_data_2y.csv', dxy_csv='dxy_data_2y.csv'):
    """Запуск бэктеста на локальных CSV без ML-модели (только SMC)."""
    print("--- Запуск ЛОКАЛЬНОГО бэктеста (только SMC) ---")
    # 1. Загрузка данных из CSV
    try:
        eurusd_data = pd.read_csv(eurusd_csv, parse_dates=['Gmt time'])
        eurusd_data.rename(columns={
            'Gmt time': 'Datetime',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume',
        }, inplace=True)
        eurusd_data.set_index('Datetime', inplace=True)
        eurusd_data.index = pd.to_datetime(eurusd_data.index, format='%d.%m.%Y %H:%M:%S.%f')
        eurusd_data.index = eurusd_data.index.tz_localize('UTC')

        dxy_data = pd.read_csv(dxy_csv, parse_dates=['Gmt time'])
        dxy_data.rename(columns={
            'Gmt time': 'Datetime',
            'Low': 'DXY_Low',
        }, inplace=True)
        dxy_data.set_index('Datetime', inplace=True)
        dxy_data.index = pd.to_datetime(dxy_data.index, format='%d.%m.%Y %H:%M:%S.%f')
        dxy_data.index = dxy_data.index.tz_localize('UTC')
    except Exception as e:
        return f"Ошибка загрузки локальных данных: {e}", None

    # 2. Подготовка данных (индикаторы)
    eurusd_data.ta.rsi(length=14, append=True)
    eurusd_data.ta.macd(fast=12, slow=26, signal=9, append=True)
    eurusd_data.ta.atr(length=14, append=True)
    eurusd_data.rename(columns={'RSI_14':'RSI', 'MACD_12_26_9':'MACD', 'MACDh_12_26_9':'MACD_hist', 'MACDs_12_26_9':'MACD_signal', 'ATRr_14':'ATR'}, inplace=True)

    # 3. Объединение с DXY
    dxy_data_renamed = dxy_data[['DXY_Low']]
    data = pd.concat([eurusd_data, dxy_data_renamed], axis=1)
    data.dropna(inplace=True)

    # 4. Переопределяем SMCStrategy так, чтобы игнорировать ML-фильтр
    class SMCStrategyNoML(SMCStrategy):
        def next(self):
            current_hour = self.data.index[-1].hour
            is_trading_time = self.start_hour <= current_hour <= self.end_hour
            if not is_trading_time or self.position:
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
                self.signal_to_trade = -1
            else:
                self.signal_to_trade = 0
            if self.signal_to_trade == -1 and not self.position:
                entry_price = self.data.Open[-1]
                sl_price = entry_price * (1 + self.sl_ratio)
                tp_price = entry_price * (1 - self.tp_ratio)
                if not (np.isfinite(entry_price) and np.isfinite(sl_price) and np.isfinite(tp_price) and (tp_price < entry_price < sl_price)):
                    self.signal_to_trade = 0
                    return
                stop_distance_per_unit = sl_price - entry_price
                if stop_distance_per_unit > 0:
                    initial_equity = 10000
                    fixed_risk_amount = initial_equity * self.risk_percent
                    units_to_trade = fixed_risk_amount / stop_distance_per_unit
                    if units_to_trade > 0:
                        try:
                            self.sell(size=int(units_to_trade), sl=sl_price, tp=tp_price)
                        except Exception:
                            pass
                self.signal_to_trade = 0

    bt = Backtest(data, SMCStrategyNoML, cash=10000, commission=.0002, margin=0.05)
    stats = bt.run()
    
    # 5. Сохранение результатов в виде PDF
    plot_filename = f"backtest_no_ml_report_{int(time.time())}.pdf"
    plot_backtest_results_to_pdf(stats, data, plot_filename)

    print("--- Локальный бэктест БЕЗ ML-фильтра завершен ---")
    return stats, plot_filename

# Этот блок больше не нужен, так как запуск будет из бота
# if __name__ == "__main__":
#     run_backtest()
