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
from signal_core import generate_signal_and_plot, generate_signal_and_plot_30m, TIMEFRAME

# Импортируем модули для демо-счета
from telegram_demo_account import add_demo_account_handlers
from webapp import register_demo_blueprint

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

# Регистрируем Blueprint для веб-интерфейса демо-счета
register_demo_blueprint(app)

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
                # 5-минутный таймфрейм
                signal_5m, entry_5m, sl_5m, tp_5m, last_5m, image_path_5m, timeframe_5m = generate_signal_and_plot()
                if signal_5m:
                    message_5m = (
                        f"🚨 СИГНАЛ (M5) 🚨\n"
                        f"SELL EURUSD\n"
                        f"Time: {last_5m.name.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                        f"Entry: {entry_5m:.5f}\n"
                        f"SL: {sl_5m:.5f}\n"
                        f"TP: {tp_5m:.5f}"
                    )
                    if image_path_5m and os.path.exists(image_path_5m):
                        with open(image_path_5m, 'rb') as img:
                            await bot.send_photo(chat_id, photo=img, caption=message_5m)
                    else:
                        await bot.send_message(chat_id, message_5m)
                else:
                    message_5m = f"Нет сигнала на M5. Время: {last_5m.name.strftime('%Y-%m-%d %H:%M:%S UTC') if last_5m is not None else 'N/A'}"
                    await bot.send_message(chat_id, message_5m)

                # 30-минутный таймфрейм
                signal_30m, entry_30m, sl_30m, tp_30m, last_30m, image_path_30m, tf_30m = generate_signal_and_plot_30m()
                if signal_30m:
                    message_30m = (
                        f"🚨 СИГНАЛ (M30) 🚨\n"
                        f"SELL EURUSD\n"
                        f"Time: {last_30m.name.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                        f"Entry: {entry_30m:.5f}\n"
                        f"SL: {sl_30m:.5f}\n"
                        f"TP: {tp_30m:.5f}"
                    )
                    if image_path_30m and os.path.exists(image_path_30m):
                        with open(image_path_30m, 'rb') as img:
                            await bot.send_photo(chat_id, photo=img, caption=message_30m)
                    else:
                        await bot.send_message(chat_id, message_30m)
                else:
                    message_30m = f"Нет сигнала на M30. Время: {last_30m.name.strftime('%Y-%m-%d %H:%M:%S UTC') if last_30m is not None else 'N/A'}"
                    await bot.send_message(chat_id, message_30m)

            except Exception as e:
                await bot.send_message(chat_id, f"Ошибка при генерации сигнала: {e}")
        else:
            logging.info(f"Command '{text}' not recognized by any handler.")

    except Exception as e:
        logging.error(f"An error occurred in handle_update: {e}", exc_info=True)

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
    """Эндпоинт для проверки сигнала по расписанию (UptimeRobot)."""
    print("Получен запрос на /check от планировщика.")
    try:
        # Получаем оба сигнала
        signal_5m, entry_5m, sl_5m, tp_5m, last_5m, image_path_5m, timeframe_5m = generate_signal_and_plot()
        
        signal_30m, entry_30m, sl_30m, tp_30m, last_30m, image_path_30m, tf_30m = generate_signal_and_plot_30m()

        # Создаем асинхронную задачу для отправки
        loop = get_background_loop()
        task = asyncio.run_coroutine_threadsafe(
            send_signals(signal_5m, entry_5m, sl_5m, tp_5m, last_5m, image_path_5m,
                         signal_30m, entry_30m, sl_30m, tp_30m, last_30m, image_path_30m, tf_30m), 
            loop
        )
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

        return "Проверка инициирована.", 200
    except Exception as e:
        print(f"Ошибка при генерации сигнала: {e}")
        # Можно отправить уведомление администратору об ошибке
        # asyncio.run(bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"Ошибка в /check: {e}"))
        return f"Ошибка: {e}", 500

async def send_signals(signal_5m, entry_5m, sl_5m, tp_5m, last_5m, image_path_5m,
                       signal_30m, entry_30m, sl_30m, tp_30m, last_30m, image_path_30m, timeframe_30m):
    """Асинхронно рассылает сигналы подписчикам."""
    subscribers = get_subscribers()
    if not subscribers:
        print("Нет подписчиков для рассылки.")
        return

    for sub_id in subscribers:
        # Рассылка сигнала M5
        if signal_5m:
            message_5m = (
                f"🚨 СИГНАЛ (M5) 🚨\n"
                f"SELL EURUSD\n"
                f"Time: {last_5m.name.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                f"Entry: {entry_5m:.5f}\n"
                f"SL: {sl_5m:.5f}\n"
                f"TP: {tp_5m:.5f}"
            )
            try:
                if image_path_5m and os.path.exists(image_path_5m):
                    with open(image_path_5m, 'rb') as img:
                        await bot.send_photo(sub_id, photo=img, caption=message_5m)
                else:
                    await bot.send_message(sub_id, message_5m)  # Отправка без картинки если что-то не так
            except Exception as e:
                logging.error(f"Не удалось отправить M5 сигнал подписчику {sub_id}: {e}")

        # Рассылка сигнала M30
        if signal_30m:
            message_30m = (
                f"🚨 СИГНАЛ (M30) 🚨\n"
                f"SELL EURUSD\n"
                f"Time: {last_30m.name.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                f"Entry: {entry_30m:.5f}\n"
                f"SL: {sl_30m:.5f}\n"
                f"TP: {tp_30m:.5f}"
            )
            try:
                if image_path_30m and os.path.exists(image_path_30m):
                    with open(image_path_30m, 'rb') as img:
                        await bot.send_photo(sub_id, photo=img, caption=message_30m)
                else:
                    await bot.send_message(sub_id, message_30m)  # Отправка без картинки если что-то не так
            except Exception as e:
                logging.error(f"Не удалось отправить M30 сигнал подписчику {sub_id}: {e}")

@app.route('/')
def index():
    """Стартовая страница для проверки, что сервис жив."""
    return "Telegram Bot is running with Demo Account integration!"

# Инициализация демо-аккаунта при запуске
def init_demo_account():
    """
    Инициализирует демо-аккаунт и добавляет обработчики команд к существующему боту.
    Эта функция должна вызываться при запуске приложения.
    """
    try:
        # Импортируем здесь, чтобы избежать циклических импортов
        from telegram.ext import Application
        
        logging.info("Инициализация демо-аккаунта...")
        
        # Создаем приложение с тем же токеном, что и основной бот
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        
        # Добавляем обработчики команд демо-счета
        add_demo_account_handlers(application)
        
        logging.info("Демо-аккаунт успешно инициализирован")
        
        # Запускаем приложение в режиме polling (в отдельном потоке)
        def start_polling():
            application.run_polling(allowed_updates=["message", "callback_query"])
        
        # Запускаем в отдельном потоке
        threading.Thread(target=start_polling, daemon=True).start()
        
        return True
    except Exception as e:
        logging.error(f"Ошибка при инициализации демо-аккаунта: {e}")
        return False

# Инициализируем демо-аккаунт при импорте модуля
demo_account_initialized = False

def initialize_on_startup():
    """Инициализирует демо-аккаунт при запуске сервера."""
    global demo_account_initialized
    if not demo_account_initialized:
        demo_account_initialized = init_demo_account()
        if demo_account_initialized:
            logging.info("Демо-аккаунт успешно инициализирован при запуске")
        else:
            logging.error("Не удалось инициализировать демо-аккаунт при запуске")

# Регистрируем функцию, которая будет вызвана после запуска Flask
@app.before_first_request
def before_first_request():
    """Вызывается перед первым запросом к Flask-приложению."""
    initialize_on_startup()

if __name__ == "__main__":
    # Инициализируем демо-аккаунт перед запуском сервера
    initialize_on_startup()
    
    # Локальный запуск для отладки. На Render будет использоваться gunicorn.
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 
