import os
import json
import pandas as pd
import pandas_ta as ta
import joblib
import yfinance as yf
import telegram
from flask import Flask, request
import asyncio
import threading
from trading_strategy import run_backtest

# --- 1. Конфигурация и Инициализация ---
app = Flask(__name__)

# Загрузка секретов из переменных окружения
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID') # ID администратора для отладки
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

# Настройки стратегии
MODEL_FILE = 'ml_model_final_fix.joblib'
PREDICTION_THRESHOLD = 0.55
LOOKBACK_PERIOD = 20
SUBSCRIBERS_FILE = 'subscribers.json'

# --- 2. Управление Подписчиками ---
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

# --- НОВАЯ ФУНКЦИЯ ДЛЯ ЗАПУСКА БЭКТЕСТА В ПОТОКЕ ---
def run_backtest_and_send(chat_id, threshold):
    """Функция для запуска бэктеста в отдельном потоке и отправки результатов."""
    try:
        bot.send_message(chat_id, "▶️ Шаг 1/3: Загрузка 2-х летних данных из Yahoo Finance...")
        
        stats_str, plot_file = run_backtest(prediction_threshold=threshold)
        
        if plot_file and os.path.exists(plot_file):
            bot.send_message(chat_id, "✅ Шаг 2/3: Бэктест завершен. Формирую отчет...")
            bot.send_message(chat_id, f"<pre>\n{stats_str}\n</pre>", parse_mode='HTML')
            with open(plot_file, 'rb') as photo:
                bot.send_photo(chat_id, photo=photo, caption=f"График бэктеста для порога {threshold}")
            os.remove(plot_file)
            bot.send_message(chat_id, "🏁 Шаг 3/3: Готово!")
        else:
            bot.send_message(chat_id, f"❌ Не удалось выполнить бэктест. Ошибка:\n{stats_str}")
            
    except Exception as e:
        bot.send_message(chat_id, f"❌ Произошла критическая ошибка во время выполнения бэктеста: {e}")

# --- 3. Основная логика стратегии ---
def get_live_data():
    """'Пуленепробиваемая' загрузка данных."""
    print("Загрузка свежих данных...")
    try:
        eurusd_data = yf.download(tickers='EURUSD=X', period='5d', interval='30m')
        if eurusd_data.empty:
            print("Рынок EUR/USD закрыт или данные недоступны.")
            return None
            
        dxy_data = yf.download(tickers='DX-Y.NYB', period='5d', interval='30m')
        if dxy_data.empty:
            print("Рынок DXY закрыт или данные недоступны.")
            return None

        eurusd_data.reset_index(inplace=True)
        dxy_data.reset_index(inplace=True)

        date_col = next(col for col in ['Datetime', 'Date', 'index'] if col in eurusd_data.columns)
        eurusd_data.rename(columns={date_col: 'Datetime'}, inplace=True)
        date_col = next(col for col in ['Datetime', 'Date', 'index'] if col in dxy_data.columns)
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
        print(f"Критическая ошибка при загрузке данных: {e}")
        return None

def check_for_signal():
    """Проверяет сигнал и возвращает сообщение или None."""
    try:
        ml_model = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        return f"ОШИБКА: Файл модели {MODEL_FILE} не найден!"

    data = get_live_data()
    if data is None: return "Рынок закрыт, проверка отменена."

    last_candle_time = data.index[-1].tz_convert('UTC')
    if (pd.Timestamp.now(tz='UTC') - last_candle_time).total_seconds() > 3600 * 4:
        return f"Данные старые (последняя свеча: {last_candle_time}), рынок закрыт."

    last_candle = data.iloc[-2]
    current_hour = last_candle.name.hour

    if not (13 <= current_hour <= 17):
        return f"Вне торгового времени (час UTC: {current_hour})."
    
    start_index = len(data) - LOOKBACK_PERIOD - 2
    end_index = len(data) - 2
    
    eurusd_judas_swing = last_candle['High'] > data['High'].iloc[start_index:end_index].max()
    dxy_raid = last_candle['DXY_Low'] < data['DXY_Low'].iloc[start_index:end_index].min()

    if eurusd_judas_swing and dxy_raid:
        features = [last_candle[col] for col in ['RSI', 'MACD', 'MACD_hist', 'MACD_signal', 'ATR']]
        win_prob = ml_model.predict_proba([features])[0][1]
        
        if win_prob >= PREDICTION_THRESHOLD:
            return (
                f"🚨 СИГНАЛ НА ПРОДАЖУ (SELL) EUR/USD 🚨\n\n"
                f"Вероятность успеха: *{win_prob:.2%}*\n"
                f"Время сетапа (UTC): `{last_candle.name}`"
            )
    return "Активных сигналов нет."

# --- 4. Веб-сервер и Роуты ---
@app.route('/webhook', methods=['POST'])
def webhook():
    """Обрабатывает команды от пользователей Telegram."""
    update = telegram.Update.de_json(request.get_json(force=True), bot)
    if not update.message or not update.message.text:
        return 'ok'

    chat_id = update.message.chat.id
    text = update.message.text

    if text.startswith('/start'):
        bot.send_message(chat_id, "Добро пожаловать! Этот бот присылает торговые сигналы по стратегии SMC+AI.\n\nИспользуйте /subscribe для подписки.\n\nДля запуска бэктеста используйте команду: `/backtest [порог]`, например: `/backtest 0.67`", parse_mode='Markdown')
    elif text.startswith('/subscribe'):
        if add_subscriber(chat_id):
            bot.send_message(chat_id, "Вы успешно подписались на сигналы!")
        else:
            bot.send_message(chat_id, "Вы уже подписаны.")
    elif text.startswith('/unsubscribe'):
        if remove_subscriber(chat_id):
            bot.send_message(chat_id, "Вы успешно отписались от сигналов.")
        else:
            bot.send_message(chat_id, "Вы не были подписаны.")
    
    elif text.startswith('/backtest'):
        try:
            parts = text.split()
            if len(parts) > 1:
                threshold = float(parts[1].replace(',', '.'))
            else:
                threshold = 0.67 # Значение по умолчанию

            bot.send_message(chat_id, f"✅ Принято! Запускаю бэктест с порогом {threshold}.\n\nЭто может занять 1-2 минуты, пожалуйста, подождите...")
            
            thread = threading.Thread(target=run_backtest_and_send, args=(chat_id, threshold))
            thread.start()
            
        except (ValueError, IndexError):
            bot.send_message(chat_id, "❌ Ошибка! Неверный формат.\nИспользуйте: `/backtest [число]`, например: `/backtest 0.67`", parse_mode='Markdown')
        except Exception as e:
            bot.send_message(chat_id, f"❌ Ошибка при запуске команды: {e}")

    return 'ok'

@app.route('/check', methods=['GET'])
def check_route():
    """Запускает проверку сигнала (для UptimeRobot)."""
    print("Получен запрос на /check от планировщика.")
    message = check_for_signal()
    
    if "СИГНАЛ НА ПРОДАЖУ" in message:
        print(f"Найден сигнал, рассылаю подписчикам...")
        subscribers = get_subscribers()
        for sub_id in subscribers:
            try:
                bot.send_message(sub_id, message, parse_mode='Markdown')
            except Exception as e:
                print(f"Не удалось отправить сообщение подписчику {sub_id}: {e}")
    else:
        print(message) # Выводим в лог "Нет сигналов" или "Рынок закрыт"
        
    return message # Возвращаем статус для UptimeRobot

@app.route('/')
def index():
    """Стартовая страница для проверки, что сервис жив."""
    return "Telegram Bot is running!"

if __name__ == "__main__":
    # Локальный запуск для отладки. На Render будет использоваться gunicorn.
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 
