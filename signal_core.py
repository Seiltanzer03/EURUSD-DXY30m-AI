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

# Настраиваем стиль графиков для лучшей визуализации
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

# Создаем кастомный стиль для свечей
mc = mpf.make_marketcolors(
    up='green',
    down='red',
    edge='inherit',
    wick={'up':'green', 'down':'red'},
    volume='inherit'
)
s = mpf.make_mpf_style(
    base_mpf_style='charles',
    marketcolors=mc,
    gridstyle=':',
    gridcolor='gray',
    gridaxis='both',
    y_on_right=False,
    facecolor='white'
)

# --- Константы для таймфреймов и параметров ---
TIMEFRAME_5M = '5m'
TIMEFRAME_30M = '30m'
LOOKBACK_PERIOD_5M = 5
LOOKBACK_PERIOD_30M = 34  # ~17 часов
SL_RATIO_5M = 0.003
TP_RATIO_5M = 0.008
SL_RATIO_30M = 0.004
TP_RATIO_30M = 0.01
MODEL_FILE = 'ml_model_final_fix.joblib'
PREDICTION_THRESHOLD = 0.55

def flatten_multiindex_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def load_data(period="2d", interval="5m"):
    # 1. Загрузка данных
    eurusd = yf.download('EURUSD=X', period=period, interval=interval, auto_adjust=True)
    dxy = yf.download('DX-Y.NYB', period=period, interval=interval, auto_adjust=True)
    
    print(f"Загружено свечей EURUSD: {len(eurusd)}, DXY: {len(dxy)}")
    
    eurusd = flatten_multiindex_columns(eurusd)
    dxy = flatten_multiindex_columns(dxy)
    if eurusd.empty or dxy.empty:
        raise ValueError('Нет данных для EURUSD или DXY')

    # 2. Подготовка к объединению
    # Сбрасываем индекс, чтобы временная метка стала обычным столбцом
    eurusd.reset_index(inplace=True)
    dxy.reset_index(inplace=True)
    
    # Убеждаемся, что столбцы с датой/временем называются одинаково
    eurusd_date_col = next((col for col in ['Datetime', 'Date', 'index'] if col in eurusd.columns), None)
    dxy_date_col = next((col for col in ['Datetime', 'Date', 'index'] if col in dxy.columns), None)
    if not eurusd_date_col or not dxy_date_col:
        raise ValueError("Не удалось найти столбец с датой/временем в данных yfinance.")
    eurusd.rename(columns={eurusd_date_col: 'Datetime'}, inplace=True)
    dxy.rename(columns={dxy_date_col: 'Datetime'}, inplace=True)

    # 3. Надежное объединение данных по ближайшему времени
    # Это решает проблему неточного совпадения временных меток
    data = pd.merge_asof(
        eurusd.sort_values('Datetime'),
        dxy[['Datetime', 'Low']].rename(columns={'Low': 'DXY_Low'}).sort_values('Datetime'),
        on='Datetime',
        direction='backward'  # Используем последнее известное значение DXY
    )
    
    print(f"После объединения: {len(data)} строк")
    
    # 4. Установка индекса и расчет индикаторов
    data.set_index('Datetime', inplace=True)
    
    data.ta.rsi(length=14, append=True)
    data.ta.macd(fast=12, slow=26, signal=9, append=True)
    data.ta.atr(length=14, append=True)
    data.rename(columns={'RSI_14':'RSI', 'MACD_12_26_9':'MACD', 'MACDh_12_26_9':'MACD_hist', 'MACDs_12_26_9':'MACD_signal', 'ATRr_14':'ATR'}, inplace=True)
    
    # 5. Очистка и обработка временной зоны
    data.dropna(inplace=True)
    
    print(f"После удаления NaN: {len(data)} строк")
    
    # Убедимся, что индекс имеет правильный тип для mplfinance
    if not isinstance(data.index, pd.DatetimeIndex):
        print("Индекс не является DatetimeIndex, преобразуем...")
        data.index = pd.to_datetime(data.index)
    
    # Обработка временной зоны
    if data.index.tz is None:
        print("Устанавливаем временную зону UTC...")
        data = data.tz_localize('UTC')
    else:
        print(f"Преобразуем временную зону из {data.index.tz} в UTC...")
        data = data.tz_convert('UTC')
    
    # Проверка на наличие данных после всех преобразований
    if len(data) < 20:
        print(f"ПРЕДУПРЕЖДЕНИЕ: После всех преобразований осталось мало данных: {len(data)} строк")
    
    print(f"Финальный набор данных: {len(data)} строк, временной диапазон: {data.index[0]} - {data.index[-1]}")
    
    return data

def generate_signal_and_plot():
    """
    ВРЕМЕННО: Всегда возвращает сигнал для теста live-рассылки каждые 5 минут.
    """
    status_message = "Тестовый сигнал 5m (выдается всегда, для проверки live-рассылки)"
    try:
        data = load_data(period="3d", interval=TIMEFRAME_5M)
    except ValueError as e:
        status_message = f"Ошибка загрузки данных для 5м: {e}"
        print(status_message)
        return True, None, None, None, None, None, TIMEFRAME_5M, status_message

    if len(data) < LOOKBACK_PERIOD_5M + 2:
        status_message = f"Недостаточно данных для 5м сигнала: {len(data)} < {LOOKBACK_PERIOD_5M + 2}"
        print(status_message)
        return True, None, None, None, None, None, TIMEFRAME_5M, status_message

    last_candle = data.iloc[-2]
    entry = last_candle['Open']
    sl = entry * (1 + SL_RATIO_5M)
    tp = entry * (1 - TP_RATIO_5M)
    plot_path = None

    # Визуализация сигнала (оставляем как есть)
    try:
        candles = data.tail(60).copy()
        if candles.index.duplicated().any():
            candles = candles.loc[~candles.index.duplicated(keep='last')]
        candles = candles.reset_index()
        date_col = next((col for col in ['Datetime', 'Date', 'index'] if col in candles.columns), None)
        candles = candles.rename(columns={date_col: 'Date'})
        candles['Date'] = pd.to_datetime(candles['Date']).dt.tz_localize(None)
        candles = candles.set_index('Date')
        required_columns = ['Open', 'High', 'Low', 'Close']
        for col in required_columns:
            candles[col] = pd.to_numeric(candles[col], errors='coerce')
        candles = candles.dropna(subset=required_columns)
        if len(candles) >= 10:
            apds = [
                mpf.make_addplot([entry] * len(candles), type='line', color='blue', width=1, linestyle='--', panel=0),
                mpf.make_addplot([sl] * len(candles), type='line', color='red', width=1, linestyle='--', panel=0),
                mpf.make_addplot([tp] * len(candles), type='line', color='green', width=1, linestyle='--', panel=0)
            ]
            fig, axes = mpf.plot(
                candles,
                type='candle',
                style=s,
                title=f'TEST SELL EURUSD ({TIMEFRAME_5M})',
                ylabel='Price',
                addplot=apds,
                figsize=(12, 9),
                returnfig=True,
                panels=1
            )
            ax = axes[0]
            ax.scatter([len(candles)-1], [entry], color='blue', marker='v', s=120, label='Sell Entry')
            ax.legend(['Entry', 'Stop Loss', 'Take Profit'], loc='upper left')
            plot_path = 'signal_5m.png'
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
    except Exception as e:
        print(f"Ошибка при построении тестового графика: {e}")
        plot_path = None

    return True, entry, sl, tp, last_candle, plot_path, TIMEFRAME_5M, status_message

def generate_signal_and_plot_30m():
    """
    Генерирует сигнал для 30-минутного таймфрейма на основе паттерна
    (Judas Swing + DXY Raid) с применением ML-фильтра.
    """
    status_message = "Неизвестная ошибка"
    try:
        data = load_data(period="7d", interval=TIMEFRAME_30M)
    except ValueError as e:
        status_message = f"Ошибка при загрузке данных для 30M: {e}"
        print(status_message)
        return None, None, None, None, None, None, TIMEFRAME_30M, status_message

    if len(data) < LOOKBACK_PERIOD_30M + 2:
        status_message = f"Недостаточно данных для генерации сигнала 30м: {len(data)} < {LOOKBACK_PERIOD_30M + 2}"
        print(status_message)
        return None, None, None, None, None, None, TIMEFRAME_30M, status_message

    last_candle = data.iloc[-2] # Анализируем предыдущую закрытую свечу
    current_hour = last_candle.name.hour

    # Ограничение по времени для 30м сигнала
    if not (13 <= current_hour <= 17):
        status_message = f"Сигнал 30м: вне торгового времени (час UTC: {current_hour})."
        print(status_message)
        return None, None, None, None, None, None, TIMEFRAME_30M, status_message

    start_index = len(data) - LOOKBACK_PERIOD_30M - 2
    end_index = len(data) - 2
    
    if start_index < 0 or end_index <= start_index:
        print("Ошибка в индексах для поиска паттерна 30м.")
        return None, None, None, None, None, None, TIMEFRAME_30M, status_message

    eurusd_judas_swing = last_candle['High'] > data['High'].iloc[start_index:end_index].max()
    dxy_raid = last_candle['DXY_Low'] < data['DXY_Low'].iloc[start_index:end_index].min()

    signal = False
    entry, sl, tp, plot_path, win_prob = None, None, None, None, 0

    if eurusd_judas_swing and dxy_raid:
        status_message = "Паттерн 30м найден. Проверка ML-фильтра..."
        print(status_message)
        features = [last_candle[col] for col in ['RSI', 'MACD', 'MACD_hist', 'MACD_signal', 'ATR']]
        
        if any(np.isnan(features)):
            status_message = "ОШИБКА: Обнаружен NaN в признаках для 30м сигнала."
            print(status_message)
            return None, None, None, None, None, None, TIMEFRAME_30M, status_message
            
        model = joblib.load(MODEL_FILE)
        win_prob = model.predict_proba([features])[0][1]
        
        if win_prob >= PREDICTION_THRESHOLD:
            status_message = f"Сигнал 30м прошел ML-фильтр с вероятностью {win_prob:.2%}."
            print(status_message)
            signal = True
            entry = last_candle['Open']
            sl = entry * (1 + SL_RATIO_30M)
            tp = entry * (1 + TP_RATIO_30M)
        else:
            status_message = f"Паттерн 30м найден, но не прошел ML-фильтр. Вероятность: {win_prob:.2%}"
            print(status_message)
            return None, None, None, None, None, None, TIMEFRAME_30M, status_message # Сигнала нет
    
    if not signal:
        status_message = "Активных 30м сигналов по паттерну нет."
        print(status_message)

    return signal, entry, sl, tp, last_candle, plot_path, TIMEFRAME_30M, status_message

def find_signals_in_period(minutes=60, timeframe='5m'):
    """
    Ищет сигналы за указанный период времени (в минутах) и проверяет, достигли ли они SL или TP.
    
    Args:
        minutes (int): Период в минутах для поиска сигналов
        timeframe (str): Таймфрейм для анализа ('5m' или '30m')
    
    Returns:
        list: Список найденных сигналов с информацией о статусе (активен/закрыт по SL/TP)
    """
    print(f"Поиск сигналов за последние {minutes} минут на таймфрейме {timeframe}")
    
    # Определяем период загрузки данных в зависимости от запрошенного периода
    load_period = "3d"  # Загружаем больше данных, чем нужно для анализа
    
    try:
        # Загружаем данные
        data = load_data(period=load_period, interval=timeframe)
        
        if len(data) < 20:  # Минимальное количество свечей для анализа
            print(f"Недостаточно данных для анализа: {len(data)} < 20")
            return []
        
        # Определяем параметры в зависимости от таймфрейма
        if timeframe == TIMEFRAME_5M:
            lookback_period = LOOKBACK_PERIOD_5M
            sl_ratio = SL_RATIO_5M
            tp_ratio = TP_RATIO_5M
        else:  # 30m
            lookback_period = LOOKBACK_PERIOD_30M
            sl_ratio = SL_RATIO_30M
            tp_ratio = TP_RATIO_30M
        
        # Определяем временной диапазон для поиска сигналов
        now = pd.Timestamp.now(tz='UTC')
        start_time = now - pd.Timedelta(minutes=minutes)
        
        # Фильтруем данные по времени
        period_data = data[data.index >= start_time]
        
        if len(period_data) < 2:
            print(f"Недостаточно данных в запрошенном периоде: {len(period_data)} < 2")
            return []
        
        # Список для хранения найденных сигналов
        signals = []
        
        # Проходим по всем свечам в периоде, кроме последней (текущей)
        for i in range(len(period_data) - 1):
            candle = period_data.iloc[i]
            candle_time = candle.name
            
            # Проверяем наличие сигнала для данной свечи
            signal = False
            
            # Для таймфрейма 5m всегда генерируем сигнал (как в текущей реализации)
            if timeframe == TIMEFRAME_5M:
                signal = True
                entry = candle['Open']
                sl = entry * (1 + sl_ratio)
                tp = entry * (1 - tp_ratio)
            else:
                # Для 30m применяем логику с паттерном и ML-фильтром
                # Получаем данные для анализа паттерна
                candle_index = data.index.get_loc(candle_time)
                if candle_index < lookback_period:
                    continue  # Пропускаем, если недостаточно исторических данных
                
                start_index = candle_index - lookback_period
                end_index = candle_index
                
                # Проверяем паттерн
                eurusd_judas_swing = candle['High'] > data['High'].iloc[start_index:end_index].max()
                dxy_raid = candle['DXY_Low'] < data['DXY_Low'].iloc[start_index:end_index].min()
                
                if eurusd_judas_swing and dxy_raid:
                    # Проверяем ML-фильтр
                    features = [candle[col] for col in ['RSI', 'MACD', 'MACD_hist', 'MACD_signal', 'ATR']]
                    if any(np.isnan(features)):
                        continue
                    
                    model = joblib.load(MODEL_FILE)
                    win_prob = model.predict_proba([features])[0][1]
                    
                    if win_prob >= PREDICTION_THRESHOLD:
                        signal = True
                        entry = candle['Open']
                        sl = entry * (1 + sl_ratio)
                        tp = entry * (1 - tp_ratio)
            
            # Если сигнал найден, анализируем его результат
            if signal:
                # Получаем все свечи после сигнала
                future_candles = data.loc[data.index > candle_time]
                
                # Проверяем, достигла ли цена уровней SL или TP
                hit_sl = False
                hit_tp = False
                
                for _, future_candle in future_candles.iterrows():
                    # Для SELL сигнала: SL - выше входа, TP - ниже входа
                    if future_candle['High'] >= sl:
                        hit_sl = True
                        break
                    elif future_candle['Low'] <= tp:
                        hit_tp = True
                        break
                
                # Определяем статус сигнала
                if hit_sl:
                    status = "Сделка закрыта по стоп-лоссу"
                elif hit_tp:
                    status = "Сделка закрыта по тейк-профиту"
                else:
                    status = "Сделка активна"
                
                # Создаем график для сигнала
                plot_path = None
                try:
                    # Берем некоторое количество свечей до и после сигнала для графика
                    start_idx = max(0, data.index.get_loc(candle_time) - 30)
                    end_idx = min(len(data), data.index.get_loc(candle_time) + 30)
                    chart_data = data.iloc[start_idx:end_idx].copy()
                    
                    if len(chart_data) >= 10:
                        if chart_data.index.duplicated().any():
                            chart_data = chart_data.loc[~chart_data.index.duplicated(keep='last')]
                        chart_data = chart_data.reset_index()
                        date_col = next((col for col in ['Datetime', 'Date', 'index'] if col in chart_data.columns), None)
                        chart_data = chart_data.rename(columns={date_col: 'Date'})
                        chart_data['Date'] = pd.to_datetime(chart_data['Date']).dt.tz_localize(None)
                        chart_data = chart_data.set_index('Date')
                        
                        required_columns = ['Open', 'High', 'Low', 'Close']
                        for col in required_columns:
                            chart_data[col] = pd.to_numeric(chart_data[col], errors='coerce')
                        chart_data = chart_data.dropna(subset=required_columns)
                        
                        # Определяем индекс сигнальной свечи на графике
                        signal_idx = chart_data.reset_index()['Date'].dt.tz_localize('UTC').searchsorted(candle_time)
                        
                        # Создаем линии для уровней
                        apds = [
                            mpf.make_addplot([entry] * len(chart_data), type='line', color='blue', width=1, linestyle='--', panel=0),
                            mpf.make_addplot([sl] * len(chart_data), type='line', color='red', width=1, linestyle='--', panel=0),
                            mpf.make_addplot([tp] * len(chart_data), type='line', color='green', width=1, linestyle='--', panel=0)
                        ]
                        
                        # Добавляем заголовок с информацией о статусе сделки
                        title = f'SELL EURUSD ({timeframe}) - {status}'
                        
                        fig, axes = mpf.plot(
                            chart_data,
                            type='candle',
                            style=s,
                            title=title,
                            ylabel='Price',
                            addplot=apds,
                            figsize=(12, 9),
                            returnfig=True,
                            panels=1
                        )
                        
                        ax = axes[0]
                        # Отмечаем точку входа
                        if signal_idx < len(chart_data):
                            ax.scatter([signal_idx], [entry], color='blue', marker='v', s=120, label='Sell Entry')
                        
                        # Добавляем отметки для SL/TP, если они были достигнуты
                        if hit_sl or hit_tp:
                            for i, fc in enumerate(future_candles.iterrows()):
                                idx, future_candle = fc
                                chart_idx = chart_data.reset_index()['Date'].dt.tz_localize('UTC').searchsorted(idx)
                                if chart_idx >= len(chart_data):
                                    continue
                                
                                if hit_sl and future_candle['High'] >= sl:
                                    ax.scatter([chart_idx], [sl], color='red', marker='x', s=150, label='Stop Loss Hit')
                                    break
                                elif hit_tp and future_candle['Low'] <= tp:
                                    ax.scatter([chart_idx], [tp], color='green', marker='o', s=150, label='Take Profit Hit')
                                    break
                        
                        ax.legend(['Entry', 'Stop Loss', 'Take Profit'], loc='upper left')
                        
                        # Сохраняем график
                        import uuid
                        plot_path = f'signal_{timeframe}_{uuid.uuid4().hex[:8]}.png'
                        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                except Exception as e:
                    print(f"Ошибка при построении графика для сигнала: {e}")
                    plot_path = None
                
                # Добавляем сигнал в список
                signals.append({
                    'time': candle_time,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'status': status,
                    'timeframe': timeframe,
                    'plot_path': plot_path
                })
        
        return signals
    
    except Exception as e:
        print(f"Ошибка при поиске сигналов за период: {e}")
        return []
