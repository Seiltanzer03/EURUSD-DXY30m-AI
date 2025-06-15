import pandas as pd
import pandas_ta as ta
import joblib
import yfinance as yf
import numpy as np
import time
import io
import matplotlib.pyplot as plt

# --- 1. Конфигурация ---
MODEL_FILE = 'ml_model_final_fix.joblib'
PREDICTION_THRESHOLD = 0.55
LOOKBACK_PERIOD = 20

# --- 2. Функции ---
def get_historical_data(period='2mo', interval='30m'):
    """Загружает исторические данные за указанный период с повторными попытками."""
    print(f"Загрузка данных за последние {period}...")
    for attempt in range(3):
        try:
            eurusd_data = yf.download(tickers='EURUSD=X', period=period, interval=interval, progress=False)
            if eurusd_data.empty:
                raise ValueError("Данные по EUR/USD пустые.")
            dxy_data = yf.download(tickers='DX-Y.NYB', period=period, interval=interval, progress=False)
            if dxy_data.empty:
                raise ValueError("Данные по DXY пустые.")
            break
        except Exception as e:
            print(f"Попытка {attempt + 1} не удалась: {e}")
            if attempt < 2:
                print("Пауза 5 секунд перед следующей попыткой...")
                time.sleep(5)
            else:
                print("Все попытки загрузки не удались.")
                return None
    
    eurusd_data.ta.rsi(length=14, append=True)
    eurusd_data.ta.macd(fast=12, slow=26, signal=9, append=True)
    eurusd_data.ta.atr(length=14, append=True)
    eurusd_data.rename(columns={'RSI_14':'RSI', 'MACD_12_26_9':'MACD', 'MACDh_12_26_9':'MACD_hist', 'MACDs_12_26_9':'MACD_signal', 'ATRr_14':'ATR'}, inplace=True)
    dxy_data_renamed = dxy_data.rename(columns={'Low': 'DXY_Low'})
    data = pd.merge(eurusd_data, dxy_data_renamed['DXY_Low'], left_index=True, right_index=True, how='inner')
    data.dropna(inplace=True)
    print("Данные успешно загружены и обработаны.")
    return data

def plot_signal(data_window, signal_index_in_window, lookback_period, win_prob):
    """Создает график для найденного сигнала и возвращает его как buffer."""
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # EUR/USD
    ax1.set_xlabel('Время', color='white')
    ax1.set_ylabel('EUR/USD', color='cyan')
    ax1.plot(data_window.index, data_window['Close'], color='cyan', label='EUR/USD Close')
    ax1.tick_params(axis='y', labelcolor='cyan')
    ax1.tick_params(axis='x', labelcolor='white')

    # DXY
    ax2 = ax1.twinx()
    ax2.set_ylabel('DXY Low', color='lime')
    ax2.plot(data_window.index, data_window['DXY_Low'], color='lime', label='DXY Low')
    ax2.tick_params(axis='y', labelcolor='lime')

    # Ретроспектива
    lookback_start_time = data_window.index[signal_index_in_window - lookback_period]
    lookback_end_time = data_window.index[signal_index_in_window - 1]
    ax1.axvspan(lookback_start_time, lookback_end_time, color='orange', alpha=0.2, label='Период ретроспективы')

    # Сигнальная свеча
    signal_candle = data_window.iloc[signal_index_in_window]
    ax1.scatter(signal_candle.name, signal_candle['High'], color='red', s=200, marker='v', zorder=5, label='Сигнал (Judas Swing)')
    ax2.scatter(signal_candle.name, signal_candle['DXY_Low'], color='magenta', s=200, marker='^', zorder=5, label='Рейд DXY')

    fig.suptitle(f'Сигнал на продажу EUR/USD @ {signal_candle.name.strftime("%Y-%m-%d %H:%M")}\nВероятность успеха: {win_prob:.2%}', fontsize=16, color='white')
    fig.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf

def run_backtest():
    """Запускает бектест и возвращает список буферов с графиками."""
    print("--- Начало бектеста ---")
    try:
        ml_model = joblib.load(MODEL_FILE)
        print(f"Модель '{MODEL_FILE}' успешно загружена.")
    except FileNotFoundError:
        print(f"ОШИБКА: Файл модели {MODEL_FILE} не найден!")
        return []

    data = get_historical_data()
    if data is None:
        print("Не удалось получить данные, бектест прерван.")
        return []

    signals_found = 0
    image_buffers = []
    print(f"\nАнализ {len(data)} свечей. Период: {data.index[0]} по {data.index[-1]}")
    print("--------------------------------------------------")

    for i in range(LOOKBACK_PERIOD, len(data)):
        current_candle = data.iloc[i]
        
        if not (13 <= current_candle.name.hour <= 17):
            continue

        start_index = i - LOOKBACK_PERIOD
        end_index = i 
        lookback_data = data.iloc[start_index:end_index]

        eurusd_judas_swing = current_candle['High'] > lookback_data['High'].max()
        dxy_raid = current_candle['DXY_Low'] < lookback_data['DXY_Low'].min()
        
        if eurusd_judas_swing and dxy_raid:
            print(f"\nНайден потенциальный сетап в {current_candle.name}...")
            features = [current_candle[col] for col in ['RSI', 'MACD', 'MACD_hist', 'MACD_signal', 'ATR']]
            if np.isnan(features).any():
                print("Пропуск: в данных для модели есть NaN значения.")
                continue

            win_prob = ml_model.predict_proba([features])[0][1]
            print(f"Вероятность по модели: {win_prob:.2%}")

            if win_prob >= PREDICTION_THRESHOLD:
                signals_found += 1
                print(">>> !!! СИГНАЛ СГЕНЕРИРОВАН !!! <<<")
                print(f"  - Время (UTC): {current_candle.name}\n  - Вероятность успеха: {win_prob:.2%}\n")
                
                plot_start_index = max(0, i - LOOKBACK_PERIOD - 20)
                plot_end_index = min(len(data), i + 20)
                data_window = data.iloc[plot_start_index:plot_end_index]
                signal_index_in_window = i - plot_start_index
                
                plot_buffer = plot_signal(data_window, signal_index_in_window, LOOKBACK_PERIOD, win_prob)
                image_buffers.append(plot_buffer)
    
    print("\n--- Бекест завершен ---")
    if signals_found > 0:
        print(f"✅ Найдено сигналов: {signals_found}")
    else:
        print("❌ За последние 2 месяца сигналов, соответствующих критериям, не найдено.")

    return image_buffers

if __name__ == "__main__":
    run_backtest() 
