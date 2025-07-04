import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
import joblib
import os
import numpy as np
import yfinance as yf
import time
import requests
import re

def disable_pan_tool_in_html(html_file):
    """
    Модифицирует HTML-файл отчета бэктеста, чтобы полностью удалить инструменты Pan (x-axis) и Wheel Zoom (x-axis).
    
    Параметры:
    - html_file: путь к HTML-файлу отчета
    """
    try:
        # Читаем содержимое HTML-файла
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Добавляем JavaScript-код для удаления инструментов Pan (x-axis) и Wheel Zoom (x-axis)
        # Этот подход более радикальный - мы полностью удаляем эти инструменты из панели
        disable_tools_js = """
        <script type="text/javascript">
        document.addEventListener("DOMContentLoaded", function() {
            // Ждем загрузку Bokeh
            setTimeout(function() {
                // Находим все элементы панели инструментов
                var toolbars = document.querySelectorAll(".bk-toolbar-box");
                
                // Для каждой панели инструментов
                toolbars.forEach(function(toolbar) {
                    // Находим и удаляем кнопки Pan (x-axis)
                    var panButtons = toolbar.querySelectorAll("button.bk-tool-icon-pan");
                    panButtons.forEach(function(btn) {
                        btn.parentNode.removeChild(btn);
                    });
                    
                    // Находим и удаляем кнопки Wheel Zoom (x-axis)
                    var wheelZoomButtons = toolbar.querySelectorAll("button.bk-tool-icon-wheel-zoom");
                    wheelZoomButtons.forEach(function(btn) {
                        btn.parentNode.removeChild(btn);
                    });
                    
                    // Активируем другой инструмент по умолчанию (например, box zoom)
                    var boxZoomButtons = toolbar.querySelectorAll("button.bk-tool-icon-box-zoom");
                    if (boxZoomButtons.length > 0 && !boxZoomButtons[0].classList.contains("bk-active")) {
                        boxZoomButtons[0].click();
                    }
                });
            }, 1000); // Задержка для уверенности, что Bokeh полностью загрузился
        });
        </script>
        """
        
        # Вставляем JavaScript-код перед закрывающим тегом </body>
        modified_html = re.sub('</body>', disable_tools_js + '</body>', html_content)
        
        # Также попробуем удалить определения этих инструментов из конфигурации Bokeh
        # Это более сложный подход, но может быть более эффективным
        # Ищем определения инструментов в конфигурации
        pan_tool_pattern = r'(\{"type":"PanTool"[^}]*\})'
        wheel_zoom_pattern = r'(\{"type":"WheelZoomTool"[^}]*\})'
        
        # Удаляем определения этих инструментов
        modified_html = re.sub(pan_tool_pattern, '', modified_html)
        modified_html = re.sub(wheel_zoom_pattern, '', modified_html)
        
        # Записываем модифицированный HTML обратно в файл
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(modified_html)
            
        print(f"HTML-файл {html_file} успешно модифицирован для удаления Pan (x-axis) и Wheel Zoom (x-axis)")
        return True
    except Exception as e:
        print(f"Ошибка при модификации HTML-файла: {e}")
        return False

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
    print(f"Загрузка {period} данных для {ticker} с интервалом {interval}...")
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
    """Основная функция для запуска бэктеста (30-минутный ТФ)."""
    print("--- Запуск бэктеста (30m) ---")
    
    # 1. Загрузка данных
    try:
        eurusd_data = load_data_from_yfinance('EURUSD=X', period='59d', interval='30m')
        dxy_data = load_data_from_yfinance('DX-Y.NYB', period='59d', interval='30m')
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
    
    # 5. Сохранение результатов с ресамплингом до 2-часовых свечей для отображения
    plot_filename = f"backtest_report_30m_{threshold}_{int(time.time())}.html"
    bt.plot(filename=plot_filename, open_browser=False, resample='2H')
    
    # Модифицируем HTML-файл, чтобы отключить инструменты Pan (x-axis) и Wheel Zoom (x-axis)
    disable_pan_tool_in_html(plot_filename)
    
    print("--- Бэктест (30m) завершен, отображение в режиме 2H ---")
    return stats, plot_filename

def run_full_backtest(threshold=0.55):
    """Запускает бэктест на основе локальных CSV-файлов."""
    print("--- Запуск полного бэктеста на CSV ---")
    
    # 1. Загрузка данных
    try:
        eurusd_file = 'EURUSD_Candlestick_30_m_BID_18.06.2022-18.06.2025 (2).csv'
        dxy_file = 'DOLLAR.IDXUSD_Candlestick_30_m_BID_18.06.2022-18.06.2025 (2).csv'

        # Загрузка и парсинг дат с правильным именем колонки
        eurusd_data = pd.read_csv(eurusd_file, parse_dates=['Gmt time'], dayfirst=True)
        dxy_data = pd.read_csv(dxy_file, parse_dates=['Gmt time'], dayfirst=True)
        
        # Переименование колонок
        eurusd_data.rename(columns={'Gmt time': 'Datetime', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}, inplace=True)
        dxy_data.rename(columns={'Gmt time': 'Datetime', 'Low': 'DXY_Low'}, inplace=True)
        
        # Установка индекса
        eurusd_data.set_index('Datetime', inplace=True)
        dxy_data.set_index('Datetime', inplace=True)

    except FileNotFoundError as e:
        return f"Ошибка: Файл не найден - {e}. Убедитесь, что файлы котировок находятся в корневой папке проекта.", None
    except Exception as e:
        return f"Ошибка загрузки или обработки CSV-файлов: {e}", None

    # 2. Подготовка данных
    eurusd_data.ta.rsi(length=14, append=True)
    eurusd_data.ta.macd(fast=12, slow=26, signal=9, append=True)
    eurusd_data.ta.atr(length=14, append=True)
    eurusd_data.rename(columns={'RSI_14':'RSI', 'MACD_12_26_9':'MACD', 'MACDh_12_26_9':'MACD_hist', 'MACDs_12_26_9':'MACD_signal', 'ATRr_14':'ATR'}, inplace=True)
    
    data = pd.concat([eurusd_data, dxy_data['DXY_Low']], axis=1)
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
    
    # 5. Сохранение результатов с ресамплингом до 2-часовых свечей для отображения
    plot_filename = f"full_backtest_report_{threshold}_{int(time.time())}.html"
    bt.plot(filename=plot_filename, open_browser=False, resample='2H')
    
    # Модифицируем HTML-файл, чтобы отключить инструменты Pan (x-axis) и Wheel Zoom (x-axis)
    disable_pan_tool_in_html(plot_filename)
    
    print("--- Полный бэктест завершен, отображение в режиме 2H ---")
    return stats, plot_filename

# Этот блок больше не нужен, так как запуск будет из бота
# if __name__ == "__main__":
#     run_backtest()
