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
    data = load_data()
    if len(data) < LOOKBACK_PERIOD:
        print(f"Недостаточно данных для генерации сигнала: {len(data)} < {LOOKBACK_PERIOD}")
        return None, None, None, None, None, None, None
    last = data.iloc[-1]
    features = [last['RSI'], last['MACD'], last['MACD_hist'], last['MACD_signal'], last['ATR']]
    if any(np.isnan(features)):
        return None, None, None, None, None, None, None
    model = joblib.load(MODEL_FILE)
    win_prob = model.predict_proba([features])[0][1]
    signal = win_prob >= PREDICTION_THRESHOLD
    entry = last['Open']
    sl = entry * (1 + SL_RATIO)
    tp = entry * (1 - TP_RATIO)
    plot_path = None
    if signal:
        try:
            # Берем последние 50 свечей для графика
            candles = data.tail(50).copy()
            
            # Отладочная информация
            print(f"Количество свечей для графика: {len(candles)}")
            print(f"Временной диапазон: {candles.index[0]} - {candles.index[-1]}")
            
            # Проверка на дубликаты индексов
            if candles.index.duplicated().any():
                print("Обнаружены дубликаты в индексе, удаляем...")
                candles = candles.loc[~candles.index.duplicated(keep='last')]
            
            if len(candles) < 10:
                warnings.warn(f'Недостаточно данных для построения графика ({len(candles)} < 10)')
                return signal, entry, sl, tp, last, None, TIMEFRAME
            
            # Подготовка данных для mplfinance
            # 1. Сбрасываем индекс в обычный столбец
            candles = candles.reset_index()
            
            # 2. Убеждаемся, что столбец с датой имеет правильное имя
            date_col = next((col for col in ['Datetime', 'Date', 'index'] if col in candles.columns), None)
            if not date_col:
                print("ОШИБКА: Не найден столбец с датой/временем")
                return signal, entry, sl, tp, last, None, TIMEFRAME
            
            # 3. Переименовываем столбец с датой в 'Date'
            candles = candles.rename(columns={date_col: 'Date'})
            
            # 4. Преобразуем дату в формат без временной зоны
            candles['Date'] = pd.to_datetime(candles['Date']).dt.tz_localize(None)
            
            # 5. Устанавливаем 'Date' как индекс
            candles = candles.set_index('Date')
            
            # Проверяем наличие обязательных столбцов OHLC
            required_columns = ['Open', 'High', 'Low', 'Close']
            if not all(col in candles.columns for col in required_columns):
                print(f"ОШИБКА: Отсутствуют обязательные столбцы: {[col for col in required_columns if col not in candles.columns]}")
                return signal, entry, sl, tp, last, None, TIMEFRAME
            
            # Преобразуем числовые столбцы
            for col in required_columns:
                candles[col] = pd.to_numeric(candles[col], errors='coerce')
            
            # Удаляем строки с NaN в OHLC
            candles = candles.dropna(subset=required_columns)
            
            if len(candles) < 10:
                print(f"ОШИБКА: После очистки осталось мало данных: {len(candles)}")
                return signal, entry, sl, tp, last, None, TIMEFRAME
            
            # Создаем линии для уровней входа, SL и TP
            # Используем списки для совместимости с mplfinance
            dates = candles.index.tolist()
            entry_line = [entry] * len(dates)
            sl_line = [sl] * len(dates)
            tp_line = [tp] * len(dates)
            
            # Создаем дополнительные графики для линий
            apds = [
                mpf.make_addplot([entry] * len(candles), type='line', color='blue', width=1, linestyle='--', label='Entry'),
                mpf.make_addplot([sl] * len(candles), type='line', color='red', width=1, linestyle='--', label='Stop Loss'),
                mpf.make_addplot([tp] * len(candles), type='line', color='green', width=1, linestyle='--', label='Take Profit')
            ]
            
            # Настраиваем стиль графика
            mc = mpf.make_marketcolors(
                up='green', down='red',
                edge={'up':'green', 'down':'red'},
                wick={'up':'green', 'down':'red'},
                volume='blue'
            )
            
            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                gridcolor='#E0E0E0',
                gridaxis='both',
                y_on_right=False,
                facecolor='white',
                figcolor='white',
                edgecolor='black'
            )
            
            # Создаем график
            fig, axes = mpf.plot(
                candles,
                type='candle',
                style=s,
                title=f'SELL EURUSD ({TIMEFRAME})',
                ylabel='Price',
                addplot=apds,
                figsize=(12, 8),
                returnfig=True
            )
            
            # Добавляем маркер точки входа
            ax = axes[0]
            ax.scatter([len(candles)-1], [entry], color='blue', marker='v', s=120, label='Sell Entry')
            
            # Добавляем легенду
            ax.legend(loc='upper left')
            
            # Сохраняем график
            plot_path = 'signal.png'
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"График успешно сохранен в {plot_path}")
            
        except Exception as e:
            print(f"Ошибка при построении графика: {e}")
            import traceback
            traceback.print_exc()
            plot_path = None
    
    return signal, entry, sl, tp, last, plot_path, TIMEFRAME

def generate_signal_and_plot_30m():
    interval = '30m'
    period = '4d'
    timeframe = '30m'
    
    try:
        data = load_data(period=period, interval=interval)
    except ValueError as e:
        print(f"Ошибка при загрузке данных для 30M: {e}")
        return None, None, None, None, None, None, timeframe

    if len(data) < LOOKBACK_PERIOD:
        print(f"[30M] Недостаточно данных для генерации сигнала: {len(data)} < {LOOKBACK_PERIOD}")
        return None, None, None, None, None, None, timeframe
        
    last = data.iloc[-1]
    
    # --- SMCStrategy фильтры ---
    current_hour = last.name.hour
    is_trading_time = 13 <= current_hour <= 17
    
    current_index = len(data) - 1
    start_index = current_index - LOOKBACK_PERIOD
    recent_dxy_low = data['DXY_Low'].iloc[start_index:current_index].min()
    dxy_raid = last['DXY_Low'] < recent_dxy_low
    recent_eurusd_high = data['High'].iloc[start_index:current_index].max()
    eurusd_judas_swing = last['High'] > recent_eurusd_high
    
    # Если фильтры не пройдены, возвращаем False (нет сигнала), а не None (ошибка)
    if not (is_trading_time and dxy_raid and eurusd_judas_swing):
        print(f"[30M] Фильтры не пройдены: is_trading_time={is_trading_time}, dxy_raid={dxy_raid}, eurusd_judas_swing={eurusd_judas_swing}")
        return False, None, None, None, last, None, timeframe
        
    features = [last['RSI'], last['MACD'], last['MACD_hist'], last['MACD_signal'], last['ATR']]
    # Если NaN в фичах - это ошибка данных, возвращаем None
    if any(np.isnan(features)):
        print(f"[30M] Обнаружены NaN в признаках: {features}")
        return None, None, None, None, last, None, timeframe
        
    model = joblib.load(MODEL_FILE)
    win_prob = model.predict_proba([features])[0][1]
    signal = win_prob >= 0.67
    
    entry = last['Open']
    sl = entry * (1 + SL_RATIO)
    tp = entry * (1 - TP_RATIO)
    plot_path = None
    
    if signal:
        try:
            # Берем последние 50 свечей для графика
            candles = data.tail(50).copy()
            
            # Отладочная информация
            print(f"[30M] Количество свечей для графика: {len(candles)}")
            print(f"[30M] Временной диапазон: {candles.index[0]} - {candles.index[-1]}")
            
            # Проверка на дубликаты индексов
            if candles.index.duplicated().any():
                print("[30M] Обнаружены дубликаты в индексе, удаляем...")
                candles = candles.loc[~candles.index.duplicated(keep='last')]
            
            if len(candles) < 10:
                print(f"[30M] Недостаточно данных для построения графика: {len(candles)} < 10")
                return signal, entry, sl, tp, last, None, timeframe
            
            # Подготовка данных для mplfinance
            # 1. Сбрасываем индекс в обычный столбец
            candles = candles.reset_index()
            
            # 2. Убеждаемся, что столбец с датой имеет правильное имя
            date_col = next((col for col in ['Datetime', 'Date', 'index'] if col in candles.columns), None)
            if not date_col:
                print("[30M] ОШИБКА: Не найден столбец с датой/временем")
                return signal, entry, sl, tp, last, None, timeframe
            
            # 3. Переименовываем столбец с датой в 'Date'
            candles = candles.rename(columns={date_col: 'Date'})
            
            # 4. Преобразуем дату в формат без временной зоны
            candles['Date'] = pd.to_datetime(candles['Date']).dt.tz_localize(None)
            
            # 5. Устанавливаем 'Date' как индекс
            candles = candles.set_index('Date')
            
            # Проверяем наличие обязательных столбцов OHLC
            required_columns = ['Open', 'High', 'Low', 'Close']
            if not all(col in candles.columns for col in required_columns):
                print(f"[30M] ОШИБКА: Отсутствуют обязательные столбцы: {[col for col in required_columns if col not in candles.columns]}")
                return signal, entry, sl, tp, last, None, timeframe
            
            # Преобразуем числовые столбцы
            for col in required_columns:
                candles[col] = pd.to_numeric(candles[col], errors='coerce')
            
            # Удаляем строки с NaN в OHLC
            candles = candles.dropna(subset=required_columns)
            
            if len(candles) < 10:
                print(f"[30M] ОШИБКА: После очистки осталось мало данных: {len(candles)}")
                return signal, entry, sl, tp, last, None, timeframe
            
            # Создаем линии для уровней входа, SL и TP
            # Используем списки для совместимости с mplfinance
            dates = candles.index.tolist()
            entry_line = [entry] * len(dates)
            sl_line = [sl] * len(dates)
            tp_line = [tp] * len(dates)
            
            # Создаем дополнительные графики для линий
            apds = [
                mpf.make_addplot([entry] * len(candles), type='line', color='blue', width=1, linestyle='--', label='Entry'),
                mpf.make_addplot([sl] * len(candles), type='line', color='red', width=1, linestyle='--', label='Stop Loss'),
                mpf.make_addplot([tp] * len(candles), type='line', color='green', width=1, linestyle='--', label='Take Profit')
            ]
            
            # Настраиваем стиль графика
            mc = mpf.make_marketcolors(
                up='green', down='red',
                edge={'up':'green', 'down':'red'},
                wick={'up':'green', 'down':'red'},
                volume='blue'
            )
            
            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                gridcolor='#E0E0E0',
                gridaxis='both',
                y_on_right=False,
                facecolor='white',
                figcolor='white',
                edgecolor='black'
            )
            
            # Создаем график
            fig, axes = mpf.plot(
                candles,
                type='candle',
                style=s,
                title=f'SELL EURUSD ({timeframe})',
                ylabel='Price',
                addplot=apds,
                figsize=(12, 8),
                returnfig=True
            )
            
            # Добавляем маркер точки входа
            ax = axes[0]
            ax.scatter([len(candles)-1], [entry], color='blue', marker='v', s=120, label='Sell Entry')
            
            # Добавляем легенду
            ax.legend(loc='upper left')
            
            # Сохраняем график
            plot_path = 'signal_30m.png'
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"[30M] График успешно сохранен в {plot_path}")
            
        except Exception as e:
            print(f"[30M] Ошибка при построении графика: {e}")
            import traceback
            traceback.print_exc()
            plot_path = None
    
    # Возвращаем False если сигнал не прошел порог вероятности
    if not signal:
        print(f"[30M] Сигнал не прошел порог вероятности: {win_prob} < 0.67")
        return False, entry, sl, tp, last, None, timeframe

    return signal, entry, sl, tp, last, plot_path, timeframe 
