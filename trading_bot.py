import os
import json
import pandas as pd
import pandas_ta as ta
import joblib
import yfinance as yf
import telegram
from flask import Flask, request, abort, Response
import asyncio
from trading_strategy import run_backtest, run_full_backtest, run_backtest_m5
import threading
import logging
import subprocess
import re
from signal_core import generate_signal_and_plot, generate_signal_and_plot_30m
import uuid
import requests
import time

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
PREDICTION_THRESHOLD = 0.55 # Оптимальный порог для live-сигналов
LOOKBACK_PERIOD = 20
SUBSCRIBERS_FILE = 'subscribers.json'

# --- Flask-сервер для HTML-отчётов (Telegram Game) ---
app_reports = Flask('reports')
reports = {}  # {token: (html, expire_time)}

def cleanup_reports():
    while True:
        now = time.time()
        to_delete = [token for token, (_, exp) in reports.items() if exp < now]
        for token in to_delete:
            del reports[token]
        time.sleep(60)

@app_reports.route('/game_report')
def game_report():
    token = request.args.get('start') or request.args.get('token')
    if not token or token not in reports:
        return abort(404, 'Report not found')
    html, _ = reports[token]
    return Response(html, mimetype='text/html')

def start_reports_server():
    threading.Thread(target=lambda: app_reports.run(host='0.0.0.0', port=8080), daemon=True).start()
    threading.Thread(target=cleanup_reports, daemon=True).start()

# Запуск сервера при старте бота
start_reports_server()

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
        await bot.send_message(chat_id, f"✅ Запускаю бэктест с фильтром {threshold}. Это может занять несколько минут...")
        stats, plot_file = await asyncio.to_thread(run_backtest, threshold)
        if plot_file:
            await bot.send_message(chat_id, f"📊 Результаты бэктеста:\n\n<pre>{stats}</pre>", parse_mode='HTML')
            with open(plot_file, 'r', encoding='utf-8') as f:
                html = f.read()
            token = str(uuid.uuid4())
            expire_time = time.time() + 1800
            reports[token] = (html, expire_time)
            await bot.send_game(chat_id, game_short_name='backtest_report', start_parameter=token)
            os.remove(plot_file)
        else:
            await bot.send_message(chat_id, f"❌ Ошибка во время бэктеста: {stats}")
    except Exception as e:
        await bot.send_message(chat_id, f"❌ Критическая ошибка в задаче бэктеста: {e}")

async def run_full_backtest_async(chat_id, threshold):
    """Асинхронная функция для запуска полного бэктеста по CSV."""
    logging.info(f"Executing run_full_backtest_async for chat_id {chat_id} with threshold {threshold}.")
    try:
        await bot.send_message(chat_id, f"✅ Запускаю полный бэктест по историческим данным с фильтром {threshold}. Это может занять несколько минут...")
        stats, plot_file = await asyncio.to_thread(run_full_backtest, threshold)
        if plot_file:
            await bot.send_message(chat_id, f"📊 Результаты полного бэктеста:\n\n<pre>{stats}</pre>", parse_mode='HTML')
            with open(plot_file, 'r', encoding='utf-8') as f:
                html = f.read()
            token = str(uuid.uuid4())
            expire_time = time.time() + 1800
            reports[token] = (html, expire_time)
            await bot.send_game(chat_id, game_short_name='backtest_report', start_parameter=token)
            os.remove(plot_file)
        else:
            await bot.send_message(chat_id, f"❌ Ошибка во время полного бэктеста: {stats}")
    except Exception as e:
        await bot.send_message(chat_id, f"❌ Критическая ошибка в задаче полного бэктеста: {e}")

async def run_backtest_m5_async(chat_id):
    """Асинхронная функция для запуска бэктеста 5-минутной стратегии."""
    logging.info(f"Executing run_backtest_m5_async for chat_id {chat_id}.")
    try:
        await bot.send_message(chat_id, f"✅ Запускаю бэктест 5-минутной стратегии за 59 дней. Это может занять несколько минут...")
        stats, plot_file = await asyncio.to_thread(run_backtest_m5)
        if plot_file:
            await bot.send_message(chat_id, f"📊 Результаты 5-минутного бэктеста:\n\n<pre>{stats}</pre>", parse_mode='HTML')
            with open(plot_file, 'r', encoding='utf-8') as f:
                html = f.read()
            token = str(uuid.uuid4())
            expire_time = time.time() + 1800
            reports[token] = (html, expire_time)
            await bot.send_game(chat_id, game_short_name='backtest_report', start_parameter=token)
            os.remove(plot_file)
        else:
            await bot.send_message(chat_id, f"❌ Ошибка во время 5-минутного бэктеста: {stats}")
    except Exception as e:
        await bot.send_message(chat_id, f"❌ Критическая ошибка в задаче 5-минутного бэктеста: {e}")

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
                await bot.send_message(chat_id, "Вы уже подписаны.")
        elif text == '/check':
            logging.info(f"Ручная проверка по команде /check от chat_id {chat_id}")
            task = asyncio.create_task(run_check_and_report(chat_id))
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)
        elif text.startswith('/backtest_m5'):
            logging.info(f"'/backtest_m5' command recognized for chat_id {chat_id}.")
            task = asyncio.create_task(run_backtest_m5_async(chat_id))
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)
            logging.info(f"Backtest_m5 task for chat_id {chat_id} has been created and stored.")
        elif text.startswith('/backtest'):
            logging.info(f"'/backtest' command recognized for chat_id {chat_id}.")
            try:
                threshold = 0.55
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
                await bot.send_message(chat_id, "Неверный формат. Используйте: /backtest [уровень_фильтра], например: /backtest 0.55")
        elif text.startswith('/fullbacktest'):
            logging.info(f"'/fullbacktest' command recognized for chat_id {chat_id}.")
            try:
                threshold = 0.55
                parts = text.split()
                if len(parts) > 1:
                    threshold = float(parts[1])
                
                task = asyncio.create_task(run_full_backtest_async(chat_id, threshold))
                background_tasks.add(task)
                task.add_done_callback(background_tasks.discard)
                logging.info(f"Full backtest task for chat_id {chat_id} has been created.")
            except (ValueError, IndexError):
                logging.error("Failed to parse /fullbacktest command.", exc_info=True)
                await bot.send_message(chat_id, "Неверный формат. Используйте: /fullbacktest [уровень_фильтра], например: /fullbacktest 0.55")
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
    """
    Эндпоинт для внешних сервисов (например, UptimeRobot), 
    чтобы запускать проверку сигналов каждые 5 минут.
    """
    print("Получен запрос на /check от планировщика.")
    try:
        # Запускаем генерацию и рассылку сигналов в фоновом потоке
        loop = get_background_loop()
        task = asyncio.run_coroutine_threadsafe(
            generate_and_send_signals(),
            loop
        )
        return "Check initiated", 200

    except Exception as e:
        logging.error(f"Критическая ошибка в check_route: {e}", exc_info=True)
        return "Error", 500

async def run_check_and_report(chat_id):
    """Запускает проверку сигналов и отправляет отчет конкретному пользователю."""
    await bot.send_message(chat_id, "🔍 Начинаю ручную проверку сигналов...")
    
    try:
        # Проверяем оба ТФ
        _, _, _, _, _, _, _, status_5m = generate_signal_and_plot()
        _, _, _, _, _, _, _, status_30m = generate_signal_and_plot_30m()

        # Формируем и отправляем отчет
        report = (
            f"Отчет о проверке:\n\n"
            f"🔹 **5 минут:** {status_5m}\n"
            f"🔸 **30 минут:** {status_30m}"
        )
        await bot.send_message(chat_id, report, parse_mode='Markdown')

    except Exception as e:
        logging.error(f"Ошибка при выполнении ручной проверки для chat_id {chat_id}: {e}", exc_info=True)
        await bot.send_message(chat_id, f"❌ Произошла ошибка во время проверки: {e}")

async def generate_and_send_signals():
    """Генерирует, и если находит, рассылает все типы сигналов подписчикам."""
    try:
        # 5-минутный таймфрейм
        signal_5m, entry_5m, sl_5m, tp_5m, last_5m, image_path_5m, timeframe_5m, _ = generate_signal_and_plot()
        
        # 30-минутный таймфрейм
        signal_30m, entry_30m, sl_30m, tp_30m, last_30m, image_path_30m, timeframe_30m, _ = generate_signal_and_plot_30m()

        # Если есть хотя бы один сигнал, отправляем
        if signal_5m or signal_30m:
            await send_signals(
                signal_5m, entry_5m, sl_5m, tp_5m, last_5m, image_path_5m, timeframe_5m,
                signal_30m, entry_30m, sl_30m, tp_30m, last_30m, image_path_30m, timeframe_30m
            )
    except Exception as e:
        logging.error(f"Ошибка при генерации и отправке сигналов: {e}", exc_info=True)

async def send_signals(signal_5m, entry_5m, sl_5m, tp_5m, last_5m, image_path_5m, timeframe_5m,
                       signal_30m, entry_30m, sl_30m, tp_30m, last_30m, image_path_30m, timeframe_30m):
    """Асинхронно рассылает сигналы подписчикам."""
    subscribers = get_subscribers()
    if not subscribers:
        logging.info("Сигнал(ы) есть, но подписчиков нет.")
        return

    message_parts = []
    images_to_send = []

    if signal_5m:
        message_5m = (
            f"🚨 СИГНАЛ НА ПРОДАЖУ (SELL) EUR/USD ({timeframe_5m}) 🚨\n\n"
            f"Время сетапа (UTC): `{last_5m.name.strftime('%Y-%m-%d %H:%M')}`\n"
            f"Вход: {entry_5m:.5f}\n"
            f"Стоп: {sl_5m:.5f}\n"
            f"Тейк: {tp_5m:.5f}"
        )
        message_parts.append(message_5m)
        if image_path_5m and os.path.exists(image_path_5m):
            images_to_send.append(image_path_5m)

    if signal_30m:
        message_30m = (
            f"🚨 СИГНАЛ НА ПРОДАЖУ (SELL) EUR/USD ({timeframe_30m}) 🚨\n\n"
            f"Время сетапа (UTC): `{last_30m.name.strftime('%Y-%m-%d %H:%M')}`\n"
            f"Вход: {entry_30m:.5f}\n"
            f"Стоп: {sl_30m:.5f}\n"
            f"Тейк: {tp_30m:.5f}"
        )
        message_parts.append(message_30m)
        if image_path_30m and os.path.exists(image_path_30m):
            images_to_send.append(image_path_30m)

    if not message_parts:
        return

    final_message = "\n\n---\n\n".join(message_parts)

    for chat_id in subscribers:
        try:
            await bot.send_message(chat_id, final_message, parse_mode='Markdown')
            for img_path in images_to_send:
                with open(img_path, 'rb') as f:
                    await bot.send_photo(chat_id, photo=f)
                os.remove(img_path) # Удаляем после отправки
        except Exception as e:
            logging.error(f"Не удалось отправить сигнал подписчику {chat_id}: {e}")

@app.route('/')
def index():
    return "Trading Bot is running."

if __name__ == "__main__":
    # Локальный запуск для отладки. На Render будет использоваться gunicorn.
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 
