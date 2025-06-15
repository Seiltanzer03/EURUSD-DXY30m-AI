import os
import json
import pandas as pd
import pandas_ta as ta
import joblib
import yfinance as yf
import telegram
from flask import Flask, request
import asyncio
import io
import sys
import contextlib
from backtest import run_backtest
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.timeseries import TimeSeries
import time
import numpy as np

# --- 1. Конфигурация и Инициализация ---
app = Flask(__name__)

# Загрузка секретов из переменных окружения
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID') # ID администратора для отладки
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY') # Ключ для Alpha Vantage
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
    """
    Проверяет наличие торгового сигнала с использованием Alpha Vantage.
    """
    print("Начало проверки сигнала...")
    if not ALPHA_VANTAGE_API_KEY:
        print("Ошибка: API-ключ для Alpha Vantage не задан.")
        return

    try:
        # --- Загрузка EUR/USD ---
        fx = ForeignExchange(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        eurusd_data, _ = fx.get_currency_exchange_intraday('EUR', 'USD', interval='30min', outputsize='compact') # compact для скорости
        eurusd_data.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close'}, inplace=True)
        eurusd_data.index = pd.to_datetime(eurusd_data.index)
        
        # Пауза
        time.sleep(15)

        # --- Загрузка DXY ---
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        dxy_data, _ = ts.get_intraday(symbol='DXY', interval='30min', outputsize='compact')
        dxy_data.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'}, inplace=True)
        dxy_data.index = pd.to_datetime(dxy_data.index)
        
        # Объединение и обработка
        data = pd.merge(eurusd_data, dxy_data, on='date', suffixes=('_eurusd', '_dxy'), how='inner')
        # ... (дальнейший код аналогичен backtest.py, но работает с последними данными)

        # Расчет индикаторов
        data.ta.rsi(length=14, append=True, col_names=('EURUSD_RSI_14'))
        data.ta.rsi(length=14, close=data['dxy_Close'], append=True, col_names=('DXY_RSI_14'))
        data.dropna(inplace=True)

        if len(data) < LOOKBACK_PERIOD + 1:
            print("Недостаточно данных для анализа.")
            return

        # Берем последние данные для анализа
        latest_segment = data.iloc[-(LOOKBACK_PERIOD+1):-1]
        
        input_features = np.array([
            latest_segment['EURUSD_RSI_14'].values,
            latest_segment['DXY_RSI_14'].values
        ]).flatten().reshape(1, -1)

        model = joblib.load(MODEL_FILE)
        prediction = model.predict_proba(input_features)[0][1]
        
        print(f"Проверка завершена. Вероятность сигнала: {prediction:.2f}")

        if prediction > PREDICTION_THRESHOLD:
            message = f"🚨 ВНИМАНИЕ! Обнаружен торговый сигнал! 🚨\n\n" \
                      f"Инструмент: EUR/USD\n" \
                      f"Таймфрейм: 30 минут\n" \
                      f"Вероятность отработки: {prediction:.2%}\n\n" \
                      f"Рекомендуется проверить график и принять решение."
            
            # Рассылка подписчикам
            subscribers = get_subscribers()
            for chat_id in subscribers:
                try:
                    bot.send_message(chat_id, message)
                except Exception as e:
                    print(f"Не удалось отправить сообщение пользователю {chat_id}: {e}")

    except Exception as e:
        print(f"Ошибка при проверке сигнала: {e}")
        # Отправка ошибки администратору для отладки
        if TELEGRAM_CHAT_ID:
            error_message = f"Произошла ошибка в `check_for_signal`:\n\n{e}"
            bot.send_message(TELEGRAM_CHAT_ID, error_message)

# --- 4. Веб-сервер и Роуты ---
@app.route('/webhook', methods=['POST'])
def webhook():
    """Обрабатывает команды от пользователей Telegram."""
    update = telegram.Update.de_json(request.get_json(force=True), bot)
    chat_id = update.message.chat.id
    text = update.message.text

    try:
        admin_chat_id = int(TELEGRAM_CHAT_ID)
    except (ValueError, TypeError):
        admin_chat_id = None

    if text == '/start':
        bot.send_message(chat_id, "Добро пожаловать! Этот бот присылает торговые сигналы по стратегии SMC+AI. Используйте /subscribe для подписки и /unsubscribe для отписки.")
    elif text == '/subscribe':
        if add_subscriber(chat_id):
            bot.send_message(chat_id, "Вы успешно подписались на сигналы!")
        else:
            bot.send_message(chat_id, "Вы уже подписаны.")
    elif text == '/unsubscribe':
        if remove_subscriber(chat_id):
            bot.send_message(chat_id, "Вы успешно отписались от сигналов.")
        else:
            bot.send_message(chat_id, "Вы не были подписаны.")
    elif text == '/runbacktest':
        if admin_chat_id and chat_id == admin_chat_id:
            bot.send_message(chat_id, "✅ Принято! Запускаю бектест с визуализацией... Это может занять несколько минут.")
            
            log_stream = io.StringIO()
            image_buffers = []
            results_log = ""
            
            try:
                # Перехватываем текстовый вывод (print) из run_backtest
                with contextlib.redirect_stdout(log_stream):
                    image_buffers = run_backtest() # Теперь функция возвращает буферы с картинками
                results_log = log_stream.getvalue()
            except Exception as e:
                results_log = f"Произошла критическая ошибка при выполнении бектеста:\\n{e}"
            
            if not results_log:
                results_log = "Бектест завершился без текстового вывода."

            # Отправляем текстовый лог
            max_length = 4000
            for i in range(0, len(results_log), max_length):
                chunk = results_log[i:i + max_length]
                bot.send_message(chat_id, f"<pre>{chunk}</pre>", parse_mode='HTML')
            
            # Отправляем графики
            if image_buffers:
                bot.send_message(chat_id, f"Отправляю {len(image_buffers)} графика(ов) найденных сигналов...")
                for img_buf in image_buffers:
                    try:
                        # Важно! Перемещаем курсор в начало буфера перед отправкой
                        img_buf.seek(0)
                        bot.send_photo(chat_id, photo=img_buf)
                    except Exception as e:
                        bot.send_message(chat_id, f"Не удалось отправить график: {e}")
            elif "СИГНАЛ СГЕНЕРИРОВАН" in results_log:
                 bot.send_message(chat_id, "Сигналы были найдены, но не удалось создать графики.")

        else:
            bot.send_message(chat_id, "⛔️ У вас нет прав для выполнения этой команды.")
            
    return 'ok'

@app.route('/check')
def scheduled_check():
    """Эндпоинт для UptimeRobot, запускает проверку сигнала."""
    # Запускаем в фоновом режиме, чтобы не блокировать ответ
    asyncio.run(asyncio.to_thread(check_for_signal))
    return "Проверка сигнала запущена.", 200

@app.route('/')
def index():
    """Стартовая страница для проверки, что сервис жив."""
    return "Telegram Bot is running!"

if __name__ == "__main__":
    # Локальный запуск для отладки. На Render будет использоваться gunicorn.
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 
