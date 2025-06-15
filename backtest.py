import pandas as pd
import pandas_ta as ta
import joblib
import numpy as np
import time
import io
import os
import matplotlib.pyplot as plt
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.timeseries import TimeSeries

# --- 1. Конфигурация ---
MODEL_FILE = 'ml_model_final_fix.joblib'
PREDICTION_THRESHOLD = 0.55
LOOKBACK_PERIOD = 20
# Загружаем ключ из переменных окружения
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')

# --- 2. Функции ---

def get_historical_data_av(output_size='full'):
    """Загружает исторические данные с помощью Alpha Vantage."""
    print("Загрузка данных из Alpha Vantage...")
    if not ALPHA_VANTAGE_API_KEY:
        raise ValueError("API-ключ для Alpha Vantage не найден. Установите переменную окружения ALPHA_VANTAGE_API_KEY.")

    try:
        # --- Загрузка EUR/USD ---
        print("Загрузка данных для EUR/USD...")
        fx = ForeignExchange(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        eurusd_data, _ = fx.get_currency_exchange_intraday('EUR', 'USD', interval='30min', outputsize=output_size)
        eurusd_data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close'
        }, inplace=True)
        # Alpha Vantage не предоставляет объем для FX, создаем пустую колонку
        eurusd_data['Volume'] = 0 
        eurusd_data.index = pd.to_datetime(eurusd_data.index)
        print(f"Загружено {len(eurusd_data)} записей для EUR/USD.")
        
        # Пауза между запросами, чтобы не превышать лимит API
        time.sleep(15)

        # --- Загрузка DXY ---
        print("Загрузка данных для DXY...")
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        # Используем тикер 'DXY' - если не сработает, нужно будет искать замену
        dxy_data, _ = ts.get_intraday(symbol='DXY', interval='30min', outputsize=output_size)
        dxy_data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }, inplace=True)
        dxy_data.index = pd.to_datetime(dxy_data.index)
        print(f"Загружено {len(dxy_data)} записей для DXY.")

        return eurusd_data, dxy_data

    except Exception as e:
        print(f"Ошибка при загрузке данных из Alpha Vantage: {e}")
        # Проверим, содержит ли ошибка 'Invalid API call' - это может значить, что DXY не поддерживается
        if 'Invalid API call' in str(e):
            print("Похоже, тикер 'DXY' не поддерживается в Intraday. Попробуйте найти альтернативный тикер.")
        return None, None


def create_plot(data, signal_index, lookback_period):
    """Создает график для визуализации найденного сигнала."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    start_index = signal_index - lookback_period
    end_index = signal_index + 1
    plot_data = data.iloc[start_index:end_index]

    # График EUR/USD
    ax1.plot(plot_data.index, plot_data['eurusd_Close'], label='EUR/USD Close', color='blue')
    ax1.scatter(plot_data.index[lookback_period], plot_data['eurusd_Close'].iloc[lookback_period], color='red', s=100, zorder=5, label='Сигнальная свеча')
    ax1.set_title('EUR/USD')
    ax1.set_ylabel('Цена')
    ax1.legend()
    ax1.grid(True)
    
    # Закрашиваем область ретроспективы
    ax1.axvspan(plot_data.index[0], plot_data.index[lookback_period], color='gray', alpha=0.2, label='Период ретроспективы')

    # График DXY
    ax2.plot(plot_data.index, plot_data['dxy_Close'], label='DXY Close', color='green')
    ax2.scatter(plot_data.index[lookback_period], plot_data['dxy_Close'].iloc[lookback_period], color='red', s=100, zorder=5)
    ax2.set_title('DXY (Индекс доллара)')
    ax2.set_xlabel('Дата и время')
    ax2.set_ylabel('Цена')
    ax2.legend()
    ax2.grid(True)
    ax2.axvspan(plot_data.index[0], plot_data.index[lookback_period], color='gray', alpha=0.2)

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Сохраняем график в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf


def run_backtest():
    """Основная функция для запуска бектеста."""
    print("--- Запуск бектеста ---")
    
    eurusd_data, dxy_data = get_historical_data_av(output_size='full')
    
    if eurusd_data is None or dxy_data is None:
        print("Не удалось загрузить данные. Бектест прерван.")
        return []

    print("Объединение и обработка данных...")
    data = pd.merge(eurusd_data, dxy_data, on='date', suffixes=('_eurusd', '_dxy'))
    data.rename(columns={
        'Open_eurusd': 'eurusd_Open', 'High_eurusd': 'eurusd_High', 'Low_eurusd': 'eurusd_Low', 'Close_eurusd': 'eurusd_Close', 'Volume_eurusd': 'eurusd_Volume',
        'Open_dxy': 'dxy_Open', 'High_dxy': 'dxy_High', 'Low_dxy': 'dxy_Low', 'Close_dxy': 'dxy_Close', 'Volume_dxy': 'dxy_Volume'
    }, inplace=True)
    
    print("Расчет технических индикаторов...")
    data.ta.rsi(length=14, append=True, col_names=('EURUSD_RSI_14'))
    data.ta.rsi(length=14, close=data['dxy_Close'], append=True, col_names=('DXY_RSI_14'))
    
    print("Загрузка модели...")
    model = joblib.load(MODEL_FILE)
    
    image_buffers = []
    print("\n--- Поиск сигналов ---")
    
    for i in range(LOOKBACK_PERIOD, len(data)):
        segment = data.iloc[i-LOOKBACK_PERIOD:i]
        
        input_features = np.array([
            segment['EURUSD_RSI_14'].values,
            segment['DXY_RSI_14'].values
        ]).flatten().reshape(1, -1)
        
        try:
            prediction = model.predict_proba(input_features)[0][1]
            
            if prediction > PREDICTION_THRESHOLD:
                signal_time = data.index[i]
                print(f"🔥 Найден потенциальный сигнал!")
                print(f"   - Время: {signal_time}")
                print(f"   - Вероятность: {prediction:.2f}")
                
                plot_buffer = create_plot(data, i, LOOKBACK_PERIOD)
                image_buffers.append(plot_buffer)

        except Exception as e:
            print(f"Ошибка при предсказании на сегменте {i}: {e}")
            continue
            
    if not image_buffers:
        print("\nЗавершено. Сигналов, удовлетворяющих условиям, не найдено.")
    else:
        print(f"\nЗавершено. Найдено сигналов: {len(image_buffers)}.")
        
    return image_buffers

# Для возможности запуска файла напрямую
if __name__ == '__main__':
    run_backtest() 
