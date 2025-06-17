import os
import json
import pandas as pd
import pandas_ta as ta
import joblib
import yfinance as yf
import telegram
from flask import Flask, request
import asyncio
from trading_strategy import run_backtest
import threading
import logging
import subprocess
import re
from signal_core import generate_signal_and_plot, TIMEFRAME

# --- 1. Конфигурация и Инициализация ---

# Настройка логирования для отладки
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Глобальные переменные для "ленивой" инициализации.
# Они будут созданы только один раз для каждого рабочего процесса Gunicorn.
_background_loop = None
_loop_thread = None
_thread_lock = threading.Lock()

def get_background_loop():
    """Лениво создает и запускает event loop в фоновом потоке."""
    global _background_loop, _loop_thread
    with _thread_lock:
        if _loop_thread is None:
            logging.info("Initializing background loop and thread for the first time in this worker...")
            _background_loop = asyncio.new_event_loop()
            
            def start_loop(loop):
                asyncio.set_event_loop(loop)
                loop.run_forever()

            _loop_thread = threading.Thread(target=start_loop, args=(_background_loop,), daemon=True)
            _loop_thread.start()
            logging.info("Background loop and thread have been started.")
    return _background_loop

# Хранилище для фоновых задач, чтобы их не удалил сборщик мусора
background_tasks = set()

app = Flask(__name__)

# Загрузка секретов из переменных окружения
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID') # ID администратора для отладки
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

# Настройки стратегии
MODEL_FILE = 'ml_model_final_fix.joblib'
PREDICTION_THRESHOLD = 0.67 # Оптимальный порог для live-сигналов
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
        if isinstance(eurusd_data.columns, pd.MultiIndex):
            eurusd_data.columns = eurusd_data.columns.get_level_values(0)

        if eurusd_data.empty:
            print("Рынок EUR/USD закрыт или данные недоступны.")
            return None
            
        dxy_data = yf.download(tickers='DX-Y.NYB', period='5d', interval='30m')
        if isinstance(dxy_data.columns, pd.MultiIndex):
            dxy_data.columns = dxy_data.columns.get_level_values(0)

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

async def run_backtest_async(chat_id, threshold):
    """Асинхронная функция для запуска бэктеста."""
    logging.info(f"Executing run_backtest_async for chat_id {chat_id} with threshold {threshold}.")
    try:
        # 1. Уведомляем пользователя о начале
        await bot.send_message(chat_id, f"✅ Запускаю бэктест с фильтром {threshold}. Это может занять несколько минут...")
        
        # 2. Запускаем ресурсоемкую функцию бэктеста в отдельном потоке, не блокируя event loop
        stats, plot_file = await asyncio.to_thread(run_backtest, threshold)
        
        # 3. Отправляем результаты
        if plot_file:
            # Отправляем статистику
            await bot.send_message(chat_id, f"📊 Результаты бэктеста:\n\n<pre>{stats}</pre>", parse_mode='HTML')
            
            # Отправляем HTML-отчет
            with open(plot_file, 'rb') as f:
                await bot.send_document(chat_id, document=f, caption=f"Подробный отчет по бэктесту с фильтром {threshold}")
            os.remove(plot_file) # Удаляем файл после отправки
        else:
            # Если бэктест не удался, stats содержит текст ошибки
            await bot.send_message(chat_id, f"❌ Ошибка во время бэктеста: {stats}")
            
    except Exception as e:
        await bot.send_message(chat_id, f"❌ Критическая ошибка в задаче бэктеста: {e}")

# Вспомогательная функция для парсинга текста и пути к графику
def parse_signal_output(output):
    lines = output.strip().split('\n')
    message_lines = []
    image_path = None
    for line in lines:
        if line.startswith('GRAPH_PATH:'):
            image_path = line.split(':', 1)[1].strip()
        else:
            message_lines.append(line)
    message = '\n'.join(message_lines)
    return message, image_path

async def handle_update(update):
    """Асинхронно обрабатывает входящие сообщения."""
    try:
        if not update.message or not update.message.text:
            logging.warning("Update received without a message or text, ignoring.")
            return

        chat_id = update.message.chat.id
        text = update.message.text
        logging.info(f"Received message from chat_id {chat_id}: {text}")

        if text == '/start':
            await bot.send_message(chat_id, "Добро пожаловать! Этот бот присылает торговые сигналы по стратегии SMC+AI. Используйте /subscribe для подписки и /unsubscribe для отписки.")
        elif text == '/subscribe':
            if add_subscriber(chat_id):
                await bot.send_message(chat_id, "Вы успешно подписались на сигналы!")
            else:
                await bot.send_message(chat_id, "Вы уже подписаны.")
        elif text == '/unsubscribe':
            if remove_subscriber(chat_id):
                await bot.send_message(chat_id, "Вы успешно отписались от сигналов.")
            else:
                await bot.send_message(chat_id, "Вы не были подписаны.")
        elif text.startswith('/backtest'):
            logging.info(f"'/backtest' command recognized for chat_id {chat_id}.")
            try:
                threshold = 0.67
                parts = text.split()
                if len(parts) > 1:
                    threshold = float(parts[1])
                logging.info(f"Creating backtest task with threshold {threshold} for chat_id {chat_id}.")
                task = asyncio.create_task(run_backtest_async(chat_id, threshold))
                background_tasks.add(task)
                task.add_done_callback(background_tasks.discard)
                logging.info(f"Backtest task for chat_id {chat_id} has been created and stored.")
            except (ValueError, IndexError):
                logging.error("Failed to parse /backtest command.", exc_info=True)
                await bot.send_message(chat_id, "Неверный формат. Используйте: /backtest [уровень_фильтра], например: /backtest 0.67")
        elif text == '/check':
            try:
                signal, entry, sl, tp, last, image_path = generate_signal_and_plot()
                if signal is None:
                    message = "Нет сигнала (недостаточно данных или NaN)"
                elif signal:
                    message = f"СИГНАЛ: SELL EURUSD ({TIMEFRAME})\nВремя: {last.name}\nEntry: {entry:.5f}\nStop Loss: {sl:.5f}\nTake Profit: {tp:.5f}"
                else:
                    message = f"Нет сигнала | Время: {last.name}"
            except Exception as e:
                message = f"Ошибка при генерации сигнала: {e}"
                image_path = None
            if image_path:
                with open(image_path, 'rb') as img:
                    await bot.send_photo(chat_id, photo=img, caption=message)
            else:
                await bot.send_message(chat_id, message)
        else:
            logging.info(f"Command '{text}' not recognized by any handler.")

    except Exception:
        logging.error("An unhandled exception occurred in handle_update.", exc_info=True)

# --- 4. Веб-сервер и Роуты ---
@app.route('/webhook', methods=['POST'])
def webhook():
    """Обрабатывает вебхуки от Telegram, отправляя задачу в фоновый event loop."""
    try:
        update_data = request.get_json(force=True)
        logging.info(f"Webhook received: {update_data}")
        update = telegram.Update.de_json(update_data, bot)
        
        # Получаем (или создаем) фоновый цикл
        loop = get_background_loop()
        
        # Отправляем coroutine на выполнение в фоновый поток (fire-and-forget)
        asyncio.run_coroutine_threadsafe(handle_update(update), loop)
        logging.info("handle_update task scheduled successfully.")
        
    except Exception:
        logging.error("An error occurred in the webhook handler.", exc_info=True)
    
    return 'ok'

@app.route('/check', methods=['GET'])
def check_route():
    print("Получен запрос на /check от планировщика.")
    try:
        signal, entry, sl, tp, last, image_path = generate_signal_and_plot()
        if signal is None:
            message = "Нет сигнала (недостаточно данных или NaN)"
        elif signal:
            message = f"СИГНАЛ: SELL EURUSD ({TIMEFRAME})\nВремя: {last.name}\nEntry: {entry:.5f}\nStop Loss: {sl:.5f}\nTake Profit: {tp:.5f}"
        else:
            message = f"Нет сигнала | Время: {last.name}"
    except Exception as e:
        message = f"Ошибка при генерации сигнала: {e}"
        image_path = None

    if "СИГНАЛ" in message:
        print(f"Найден сигнал, рассылаю подписчикам...")
        subscribers = get_subscribers()
        async def send_signals():
            for sub_id in subscribers:
                try:
                    if image_path:
                        with open(image_path, 'rb') as img:
                            await bot.send_photo(sub_id, photo=img, caption=message)
                    else:
                        await bot.send_message(sub_id, message, parse_mode='Markdown')
                except Exception as e:
                    print(f"Не удалось отправить сообщение подписчику {sub_id}: {e}")
        loop = get_background_loop()
        asyncio.run_coroutine_threadsafe(send_signals(), loop)
    else:
        print(message)
    return message

@app.route('/')
def index():
    """Стартовая страница для проверки, что сервис жив."""
    return "Telegram Bot is running!"

if __name__ == "__main__":
    # Локальный запуск для отладки. На Render будет использоваться gunicorn.
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 
