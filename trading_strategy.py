import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import numpy as np

# --- 1. Загрузка и подготовка данных ---

def load_data_from_dukascopy_csv(filepath):
    """
    Загружает и форматирует данные из CSV-файла, скачанного с Dukascopy.
    """
    df = pd.read_csv(
        filepath,
        parse_dates=[0],
        dayfirst=True # Указываем, что день идет первым в дате
    )
    df.rename(columns={df.columns[0]: 'Time'}, inplace=True)
    df.set_index('Time', inplace=True)
    df.index = df.index.tz_localize('UTC')
    column_mapping = {
        df.columns[0]: 'Open', df.columns[1]: 'High',
        df.columns[2]: 'Low', df.columns[3]: 'Close',
        df.columns[4]: 'Volume'
    }
    df.rename(columns=column_mapping, inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df

print("Скрипт запущен")

# --- Загрузка M30 данных ---
try:
    print("Загрузка 30-минутных данных...")
    eurusd_data_m30 = load_data_from_dukascopy_csv('eurusd_data_2y.csv')
    dxy_data_m30 = load_data_from_dukascopy_csv('dxy_data_2y.csv')
    print("30-минутные данные успешно загружены.")
except FileNotFoundError:
    print("\n!!! ОШИБКА: Файлы 30-минутных данных не найдены. !!!")
    exit()

# --- Добавляем индикаторы для ML модели ---
eurusd_data_m30.ta.rsi(length=14, append=True)
eurusd_data_m30.ta.macd(fast=12, slow=26, signal=9, append=True)
eurusd_data_m30.ta.atr(length=14, append=True)
eurusd_data_m30.rename(columns={'RSI_14':'RSI', 'MACD_12_26_9':'MACD', 'MACDh_12_26_9':'MACD_hist', 'MACDs_12_26_9':'MACD_signal', 'ATRr_14':'ATR'}, inplace=True)

# --- Объединение всех данных ---
dxy_data_m30_renamed = dxy_data_m30.rename(columns={'Low': 'DXY_Low'})
data = pd.concat([eurusd_data_m30, dxy_data_m30_renamed['DXY_Low']], axis=1)

data.dropna(inplace=True)


# --- 2. Подготовка данных и обучение ML модели (С ФИНАЛЬНЫМ ИСПРАВЛЕНИЕМ) ---
MODEL_FILE = 'ml_model_final_fix.joblib'
FEATURES = ['RSI', 'MACD', 'MACD_hist', 'MACD_signal', 'ATR']

def get_ml_dataset(data, lookback_period, sl_ratio, tp_ratio):
    features_list = []
    labels_list = []

    # Итерируем до предпоследнего элемента, чтобы безопасно смотреть на i+1
    for i in range(lookback_period, len(data) - 1):
        # --- Логика сигнала (на баре i) ---
        current_hour = data.index[i].hour
        is_trading_time = 13 <= current_hour <= 17
        
        if not is_trading_time:
            continue

        start_index = i - lookback_period
        recent_dxy_low = data['DXY_Low'].iloc[start_index:i].min()
        dxy_raid = data['DXY_Low'].iloc[i] < recent_dxy_low
        recent_eurusd_high = data['High'].iloc[start_index:i].max()
        eurusd_judas_swing = data['High'].iloc[i] > recent_eurusd_high

        if not (dxy_raid and eurusd_judas_swing):
            continue

        # --- Если сигнал есть, определяем исход сделки с ОТКРЫТИЯ СЛЕДУЮЩЕГО БАРА (i+1) ---
        entry_price = data['Open'].iloc[i+1] # РЕАЛИСТИЧНАЯ цена входа
        sl_price = entry_price * (1 + sl_ratio)
        tp_price = entry_price * (1 - tp_ratio)
        
        outcome = 0 # 0 = loss
        # Проверяем исход, начиная со свечи входа (i+1)
        for j in range(i + 1, len(data)):
            future_low = data['Low'].iloc[j]
            future_high = data['High'].iloc[j]
            
            # Важно: сначала проверяем стоп-лосс. Если на одной свече цена
            # коснется и SL и TP, консервативно считаем это убытком.
            if future_high >= sl_price:
                break # Stop Loss
            if future_low <= tp_price:
                outcome = 1 # Take Profit
                break
        
        # --- Сохраняем фичи (с бара i) и результат (для входа на i+1) ---
        current_features = data[FEATURES].iloc[i].values
        if not np.isnan(current_features).any():
            features_list.append(current_features)
            labels_list.append(outcome)

    return np.array(features_list), np.array(labels_list)

if not os.path.exists(MODEL_FILE):
    print("Модель не найдена. Начинаю сбор данных и обучение (финальная версия)...")
    X, y = get_ml_dataset(data, 20, 0.004, 0.01)
    
    if len(X) > 10:
        # ПРАВИЛЬНОЕ РАЗДЕЛЕНИЕ ДАННЫХ (хронологическое)
        # Убираем перемешивание, чтобы избежать заглядывания в будущее
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        print(f"Данные разделены хронологически: {len(X_train)} для обучения, {len(X_test)} для теста.")

        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        print(f"Модель обучена. Точность на тестовых данных (из будущего): {model.score(X_test, y_test):.2f}")
        joblib.dump(model, MODEL_FILE)
        print(f"Модель сохранена в файл: {MODEL_FILE}")
    else:
        print("Недостаточно данных для обучения модели.")
        model = None
else:
    print(f"Загрузка существующей модели из файла: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)

# --- 3. Стратегия без H4 фильтра, с ML фильтром (ИСПРАВЛЕННАЯ ЛОГИКА ВХОДА) ---

class SMCStrategy(Strategy):
    lookback_period = 20
    sl_ratio = 0.004
    tp_ratio = 0.01
    risk_percent = 0.01
    start_hour = 13
    end_hour = 17
    
    ml_model = model
    prediction_threshold = 0.55 # Порог уверенности модели для входа

    def init(self):
        print(f"Тест с ML фильтром (ФИНАЛЬНАЯ ВЕРСИЯ, порог={self.prediction_threshold})")
        self.initial_equity = self.equity
        self.fixed_risk_amount = self.initial_equity * self.risk_percent
        
        # Переменная для хранения сигнала между барами
        self.signal_to_trade = 0 # 0 = нет сигнала, -1 = продажа

    def next(self):
        # --- Часть 1: Исполнение отложенного сигнала ---
        # Если есть сигнал с предыдущего бара, исполняем его на открытии текущего бара
        if self.signal_to_trade == -1 and not self.position:
            self.sl_moved_to_be = False
            # Входим по цене открытия текущего бара
            entry_price = self.data.Open[-1] 
            sl_price = entry_price * (1 + self.sl_ratio)
            tp_price = entry_price * (1 - self.tp_ratio)

            stop_distance_per_unit = sl_price - entry_price
            if stop_distance_per_unit > 0:
                units_to_trade = self.fixed_risk_amount / stop_distance_per_unit
                if units_to_trade > 0:
                    self.active_trade_entry_price = entry_price
                    self.risk_per_share_on_trade = stop_distance_per_unit
                    self.sell(size=int(units_to_trade), sl=sl_price, tp=tp_price)

            self.signal_to_trade = 0 # Сбрасываем сигнал после попытки входа
            return # Выходим, чтобы не принимать новых решений на этом же баре

        # Сбрасываем сигнал, если мы уже в позиции
        if self.position:
            self.signal_to_trade = 0

        # --- Часть 2: Генерация нового сигнала ---
        # Логика принятия решения остается на цене закрытия, но теперь она только выставляет флаг
        current_hour = self.data.index[-1].hour
        is_trading_time = self.start_hour <= current_hour <= self.end_hour
        
        # Сначала сбрасываем сигнал
        self.signal_to_trade = 0

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
                # Вместо немедленного входа, выставляем сигнал на следующий бар
                self.signal_to_trade = -1
                

# --- 4. Запуск бэктестинга ---
if model is not None:
    print("Модель успешно загружена/обучена. Подготовка к бэктестингу...")
    
    # --- Запуск теста стратегии с ML фильтром ---
    bt_ml = Backtest(data, SMCStrategy, cash=10000, commission=.0002, margin=0.05)
    print("\nЗапуск бэктестинга (финальная, исправленная версия)...")
    stats_ml = bt_ml.run()
    print("Бэктестинг завершен.")
    print("\n--- Результаты теста (финальная, исправленная версия) ---")
    print(stats_ml)
    print("\nГенерация графика результатов для финальной стратегии...")
    bt_ml.plot(filename="SMCStrategy_ML_FINAL_FIX.html")
    print("График сохранен в файл SMCStrategy_ML_FINAL_FIX.html")

else:
    print("\nБэктестинг не запущен, так как модель не была обучена.")