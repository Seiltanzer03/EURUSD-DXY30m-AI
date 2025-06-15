import pandas as pd
import pandas_ta as ta
import joblib
import yfinance as yf
import numpy as np
import time

# --- 1. Конфигурация ---
MODEL_FILE = 'ml_model_final_fix.joblib'
PREDICTION_THRESHOLD = 0.55
LOOKBACK_PERIOD = 20

# --- 2. Загрузка и подготовка данных ---
def get_historical_data(period='2mo', interval='30m'):
    """Загружает исторические данные за указанный период с повторными попытками."""
    print(f"Загрузка данных за последние {period}...")
    for attempt in range(3): # 3 попытки
        try:
            eurusd_data = yf.download(tickers='EURUSD=X', period=period, interval=interval, progress=False)
            if eurusd_data.empty:
                raise ValueError("Данные по EUR/USD пустые.")
            
            dxy_data = yf.download(tickers='DX-Y.NYB', period=period, interval=interval, progress=False)
            if dxy_data.empty:
                raise ValueError("Данные по DXY пустые.")

            # Если обе загрузки успешны, выходим из цикла
            break
        except Exception as e:
            print(f"Попытка {attempt + 1} не удалась: {e}")
            if attempt < 2:
                print("Пауза 5 секунд перед следующей попыткой...")
                time.sleep(5)
            else:
                print("Все попытки загрузки не удались.")
                return None

    # Обработка данных (аналогично trading_bot.py)
    eurusd_data.ta.rsi(length=14, append=True)
    eurusd_data.ta.macd(fast=12, slow=26, signal=9, append=True)
    eurusd_data.ta.atr(length=14, append=True)
    eurusd_data.rename(columns={'RSI_14':'RSI', 'MACD_12_26_9':'MACD', 'MACDh_12_26_9':'MACD_hist', 'MACDs_12_26_9':'MACD_signal', 'ATRr_14':'ATR'}, inplace=True)
    
    dxy_data_renamed = dxy_data.rename(columns={'Low': 'DXY_Low'})
    
    # Используем merge для правильного совмещения данных по времени
    data = pd.merge(eurusd_data, dxy_data_renamed['DXY_Low'], left_index=True, right_index=True, how='inner')
    data.dropna(inplace=True)
    
    print("Данные успешно загружены и обработаны.")
    return data

# --- 3. Логика бектеста ---
def run_backtest():
    """Запускает бектест на исторических данных."""
    print("--- Начало бектеста ---")
    try:
        ml_model = joblib.load(MODEL_FILE)
        print(f"Модель '{MODEL_FILE}' успешно загружена.")
    except FileNotFoundError:
        print(f"ОШИБКА: Файл модели {MODEL_FILE} не найден! Прерывание.")
        return

    data = get_historical_data()
    if data is None:
        print("Не удалось получить данные, бектест прерван.")
        return

    signals_found = 0
    print(f"\\nАнализ {len(data)} свечей. Период анализа: {data.index[0]} по {data.index[-1]}")
    print("--------------------------------------------------")

    # Итерируемся по данным, пропуская первый период, нужный для ретроспективы
    for i in range(LOOKBACK_PERIOD, len(data)):
        
        current_candle = data.iloc[i]
        
        # Проверка торговых часов (13:00 - 17:00 UTC)
        current_hour = current_candle.name.hour
        if not (13 <= current_hour <= 17):
            continue # Пропускаем, если вне торгового времени

        # Определяем диапазон для поиска максимумов/минимумов
        start_index = i - LOOKBACK_PERIOD
        end_index = i 
        lookback_data = data.iloc[start_index:end_index]

        # Условия сетапа
        eurusd_judas_swing = current_candle['High'] > lookback_data['High'].max()
        dxy_raid = current_candle['DXY_Low'] < lookback_data['DXY_Low'].min()
        
        if eurusd_judas_swing and dxy_raid:
            print(f"\\nНайден потенциальный сетап в {current_candle.name}...")
            
            features = [current_candle[col] for col in ['RSI', 'MACD', 'MACD_hist', 'MACD_signal', 'ATR']]
            if np.isnan(features).any():
                print("Пропуск: в данных для модели есть NaN значения.")
                continue

            win_prob = ml_model.predict_proba([features])[0][1]
            print(f"Вероятность по модели: {win_prob:.2%}")

            if win_prob >= PREDICTION_THRESHOLD:
                signals_found += 1
                print(">>> !!! СИГНАЛ СГЕНЕРИРОВАН !!! <<<")
                print(
                    f"🚨 СИГНАЛ НА ПРОДАЖУ (SELL) EUR/USD 🚨\\n"
                    f"  - Время (UTC): {current_candle.name}\\n"
                    f"  - Вероятность успеха: {win_prob:.2%}\\n"
                )
    
    print("\\n--- Бекест завершен ---")
    if signals_found > 0:
        print(f"✅ Найдено сигналов: {signals_found}")
    else:
        print("❌ За последние 2 месяца сигналов, соответствующих критериям, не найдено.")

if __name__ == "__main__":
    run_backtest() 