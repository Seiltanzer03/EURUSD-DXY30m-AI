import os
import json
import pandas as pd
import pandas_ta as ta
import joblib
import yfinance as yf
import telegram
from flask import Flask, request
from datetime import datetime, timedelta
import requests

# --- 1. Конфигурация и Инициализация ---
app = Flask(__name__)

# --- Глобальная сессия для yfinance ---
# Создаем сессию с User-Agent, чтобы yfinance не блокировали
yf_session = requests.Session()
yf_session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Загрузка секретов из переменных окружения
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID') # ID администратора для отладки
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

# Настройки стратегии
MODEL_FILE = 'ml_model_final_fix.joblib'
PREDICTION_THRESHOLD = 0.55
LOOKBACK_PERIOD = 20
SUBSCRIBERS_FILE = 'subscribers.json'
HISTORY_FILE = 'signals_history.json' # Файл для истории сигналов

# --- 2. Управление Подписчиками и Историей ---
def get_subscribers():
    """Читает ID подписчиков из файла."""
    if not os.path.exists(SUBSCRIBERS_FILE):
        return []
    with open(SUBSCRIBERS_FILE, 'r') as f:
        return json.load(f)

def add_subscriber(chat_id):
    """Добавляет нового подписчика."""
    subscribers = get_subscribers()
    if chat_id not in subscribers:
        subscribers.append(chat_id)
        with open(SUBSCRIBERS_FILE, 'w') as f:
            json.dump(subscribers, f)
        return True
    return False

def remove_subscriber(chat_id):
    """Удаляет подписчика."""
    subscribers = get_subscribers()
    if chat_id in subscribers:
        subscribers.remove(chat_id)
        with open(SUBSCRIBERS_FILE, 'w') as f:
            json.dump(subscribers, f)
        return True
    return False

def log_signal(signal_data):
    """Записывает найденный сигнал в историю."""
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    history.append(signal_data)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def get_stats():
    """Читает историю сигналов."""
    if not os.path.exists(HISTORY_FILE):
        return "История сигналов пуста."
    with open(HISTORY_FILE, 'r') as f:
        history = json.load(f)
    
    if not history:
        return "История сигналов пуста."

    report = "📊 **История сигналов** 📊\n\n"
    for signal in history:
        report += (
            f"🔹 **Тип:** {signal['type']}\n"
            f"   **Дата:** {signal['timestamp']}\n"
            f"   **Вероятность:** {signal['probability']:.2%}\n\n"
        )
    return report

# --- 3. Основная логика стратегии ---
def get_data(end_date=None):
    """
    Загружает и обрабатывает данные.
    Если end_date is None, загружает 'живые' данные.
    Если end_date - строка, загружает исторические данные до этой даты.
    """
    print(f"Загрузка данных. Режим: {'Исторический' if end_date else 'Live'}")
    try:
        eurusd_ticker = yf.Ticker('EURUSD=X', session=yf_session)
        dxy_ticker = yf.Ticker('DX-Y.NYB', session=yf_session)

        if end_date:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            # yfinance 'end' is exclusive, so add a day to include the target date
            start_dt = end_dt - timedelta(days=10)
            end_dt_inclusive = end_dt + timedelta(days=1)
            eurusd_data = eurusd_ticker.history(start=start_dt, end=end_dt_inclusive, interval='30m')
            dxy_data = dxy_ticker.history(start=start_dt, end=end_dt_inclusive, interval='30m')
        else:
            # For live data, get the last 5 days
            eurusd_data = eurusd_ticker.history(period='5d', interval='30m')
            dxy_data = dxy_ticker.history(period='5d', interval='30m')

        if eurusd_data.empty or dxy_data.empty:
            print("Данные по одному из активов отсутствуют.")
            return None

        # .history() returns index as Datetime, so we reset it to match previous logic
        eurusd_data.reset_index(inplace=True)
        dxy_data.reset_index(inplace=True)
        
        # Ensure column name is 'Datetime'
        date_col = next(col for col in eurusd_data.columns if 'date' in col.lower())
        eurusd_data.rename(columns={date_col: 'Datetime'}, inplace=True)
        date_col = next(col for col in dxy_data.columns if 'date' in col.lower())
        dxy_data.rename(columns={date_col: 'Datetime'}, inplace=True)
        
        eurusd_data.ta.rsi(length=14, append=True)
        eurusd_data.ta.macd(fast=12, slow=26, signal=9, append=True)
        eurusd_data.ta.atr(length=14, append=True)
        eurusd_data.rename(columns={'RSI_14':'RSI', 'MACD_12_26_9':'MACD', 'MACDh_12_26_9':'MACD_hist', 'MACDs_12_26_9':'MACD_signal', 'ATRr_14':'ATR'}, inplace=True)
        
        eurusd_data.set_index('Datetime', inplace=True)
        dxy_data.set_index('Datetime', inplace=True)
        
        dxy_data_renamed = dxy_data.rename(columns={'Low': 'DXY_Low'})
        data = pd.concat([eurusd_data, dxy_data_renamed['DXY_Low']], axis=1)
        data.dropna(inplace=True)
        
        print("Данные успешно обработаны.")
        return data
    except Exception as e:
        print(f"Критическая ошибка при загрузке или обработке данных: {e}")
        return None

def check_for_signal(end_date=None):
    """
    Проверяет сигнал. Если end_date указан, работает в режиме бэктеста.
    Возвращает кортеж (сообщение_для_пользователя, данные_для_лога) или (сообщение, None).
    """
    try:
        ml_model = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        return f"ОШИБКА: Файл модели {MODEL_FILE} не найден!", None

    data = get_data(end_date)
    if data is None: 
        return "Рынок закрыт или данные недоступны, проверка отменена.", None

    # Для бэктеста мы проверяем все свечи за указанный день. Для live - только последнюю.
    candles_to_check = data.iloc[-1:] if end_date is None else data[data.index.strftime('%Y-%m-%d') == end_date]

    if candles_to_check.empty:
         return f"Нет данных для проверки за {end_date}.", None

    for i in range(len(candles_to_check)):
        last_candle = candles_to_check.iloc[i]
        
        # Находим корректный срез данных для поиска Judas Swing
        try:
            candle_position = data.index.get_loc(last_candle.name)
            if candle_position < LOOKBACK_PERIOD: continue
            lookback_data = data.iloc[candle_position - LOOKBACK_PERIOD : candle_position]
        except Exception as e:
            print(f"Ошибка при срезе данных для свечи {last_candle.name}: {e}")
            continue

        current_hour = last_candle.name.hour
        if not (13 <= current_hour <= 17):
            continue # Пропускаем свечи вне торгового времени

        eurusd_judas_swing = last_candle['High'] > lookback_data['High'].max()
        dxy_raid = last_candle['DXY_Low'] < lookback_data['DXY_Low'].min()

        if eurusd_judas_swing and dxy_raid:
            features = [last_candle[col] for col in ['RSI', 'MACD', 'MACD_hist', 'MACD_signal', 'ATR']]
            win_prob = ml_model.predict_proba([features])[0][1]
            
            if win_prob >= PREDICTION_THRESHOLD:
                signal_msg = (
                    f"🚨 СИГНАЛ НА ПРОДАЖУ (SELL) EUR/USD 🚨\n\n"
                    f"Вероятность успеха: *{win_prob:.2%}*\n"
                    f"Время сетапа (UTC): `{last_candle.name}`"
                )
                signal_log_data = {
                    "type": "Backtest" if end_date else "Live",
                    "timestamp": last_candle.name.strftime('%Y-%m-%d %H:%M:%S'),
                    "probability": win_prob
                }
                return signal_msg, signal_log_data
                
    return "Активных сигналов нет.", None

# --- 4. Веб-сервер и Роуты ---
@app.route('/webhook', methods=['POST'])
def webhook():
    """Обрабатывает команды от пользователей Telegram."""
    update = telegram.Update.de_json(request.get_json(force=True), bot)
    if not update.message: return 'ok' # Игнорируем обновления без сообщений
    
    chat_id = update.message.chat.id
    text = update.message.text.strip()
    command_parts = text.split()
    command = command_parts[0]

    if command == '/start':
        bot.send_message(chat_id, "Добро пожаловать! Используйте:\n- /subscribe для подписки\n- /unsubscribe для отписки\n- /test ГГГГ-ММ-ДД для теста на дате\n- /stats для просмотра истории")
    elif command == '/subscribe':
        if add_subscriber(chat_id):
            bot.send_message(chat_id, "Вы успешно подписались на сигналы!")
        else:
            bot.send_message(chat_id, "Вы уже подписаны.")
    elif command == '/unsubscribe':
        if remove_subscriber(chat_id):
            bot.send_message(chat_id, "Вы успешно отписались от сигналов.")
        else:
            bot.send_message(chat_id, "Вы не были подписаны.")
    elif command == '/stats':
        stats_report = get_stats()
        bot.send_message(chat_id, stats_report, parse_mode='Markdown')
    elif command == '/test':
        if len(command_parts) < 2:
            bot.send_message(chat_id, "Пожалуйста, укажите дату в формате: /test ГГГГ-ММ-ДД")
            return 'ok'
        try:
            date_to_test = command_parts[1]
            datetime.strptime(date_to_test, '%Y-%m-%d') # Валидация формата
            bot.send_message(chat_id, f"Запускаю проверку за {date_to_test}, это может занять до минуты...")
            
            message, log_data = check_for_signal(end_date=date_to_test)
            
            if log_data:
                log_signal(log_data) # Логируем тестовый сигнал

            bot.send_message(chat_id, message, parse_mode='Markdown')
        except ValueError:
            bot.send_message(chat_id, "Неверный формат даты. Используйте: /test ГГГГ-ММ-ДД")
        except Exception as e:
            bot.send_message(chat_id, f"Произошла ошибка при тестировании: {e}")

    return 'ok'

@app.route('/check', methods=['GET'])
def check_route():
    """Запускает проверку сигнала (для UptimeRobot)."""
    print("Получен запрос на /check от планировщика.")
    message, log_data = check_for_signal()
    
    if log_data: # Если найден реальный сигнал
        print(f"Найден сигнал, рассылаю подписчикам...")
        log_signal(log_data) # Логируем реальный сигнал
        subscribers = get_subscribers()
        for sub_id in subscribers:
            try:
                bot.send_message(sub_id, message, parse_mode='Markdown')
            except Exception as e:
                print(f"Не удалось отправить сообщение подписчику {sub_id}: {e}")
    else:
        print(message) # Выводим в лог "Нет сигналов" или "Рынок закрыт"
        
    return message

@app.route('/')
def index():
    """Стартовая страница для проверки, что сервис жив."""
    return "Telegram Bot is running!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 
