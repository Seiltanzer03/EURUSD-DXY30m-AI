import os
import json
import pandas as pd
import pandas_ta as ta
import joblib
import yfinance as yf
import telegram
from flask import Flask, request
import asyncio
from trading_strategy import run_backtest, run_backtest_local
import threading
import logging
import subprocess
import re
from signal_core import generate_signal_and_plot, generate_signal_and_plot_30m, TIMEFRAME
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

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

def check_signal_30m(model):
    """Проверяет сигнал на 30-минутном таймфрейме."""
    try:
        # Загрузка и подготовка данных
        data = get_live_data()
        if data is None: return "Рынок закрыт, проверка отменена."

        last_bar = data.iloc[-2]
        previous_bars = data.iloc[-22:-2]
        
        # Проверка условий
        is_trading_time = 13 <= last_bar.name.hour <= 17
        dxy_raid = last_bar['DXY_Low'] < previous_bars['DXY_Low'].min()
        eurusd_judas_swing = last_bar['High'] > previous_bars['High'].max()
        
        if is_trading_time and dxy_raid and eurusd_judas_swing:
            features = pd.DataFrame([{
                'RSI': last_bar['RSI'],
                'MACD': last_bar['MACD'],
                'MACD_hist': last_bar['MACD_hist'],
                'MACD_signal': last_bar['MACD_signal'],
                'ATR': last_bar['ATR']
            }])
            win_probability = model.predict_proba(features)[0][1]

            if win_probability >= 0.4:
                entry_price = data.iloc[-1]['Open']
                sl = entry_price * (1 + 0.004)
                tp = entry_price * (1 - 0.01)
                
                message = (
                    f"🚨 СИГНАЛ (M30) 🚨\n"
                    f"🔔 **Short EURUSD**\n"
                    f"📈 **Вероятность TP:** {win_probability:.2%}\n"
                    f"🔵 **Вход:** `{entry_price:.5f}`\n"
                    f"🔴 **Stop-Loss:** `{sl:.5f}`\n"
                    f"🟢 **Take-Profit:** `{tp:.5f}`\n"
                    f"🕗 **Время:** `{last_bar.name.strftime('%Y-%m-%d %H:%M:%S UTC')}`"
                )
                return message
            else:
                return f"Технический сетап есть, но ML-фильтр ({win_probability:.2%}) не пройден."
        else:
            return f"Нет сигнала на M30. Время: {data.iloc[-1].name.strftime('%Y-%m-%d %H:%M:%S UTC')}"

    except Exception as e:
        logging.error(f"Ошибка при проверке сигнала: {e}")
        return f"Ошибка получения данных или обработки сигнала: {e}"

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

async def handle_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Асинхронно обрабатывает входящие сообщения от пользователя."""
    if not update.message or not update.message.text:
        return

    chat_id = update.message.chat.id
    text = update.message.text
    logging.info(f"Получено сообщение от {chat_id}: {text}")

    command = text.split()[0]

    if command == '/start':
        await update.message.reply_text(
            "Добро пожаловать! Бот присылает сигналы по стратегиям M5 (SMC) и M30 (SMC+AI).\n\n"
            "Команды:\n"
            "/subscribe - Подписаться на сигналы\n"
            "/unsubscribe - Отписаться\n"
            "/check - Проверить сигналы прямо сейчас\n"
            "/backtest - Бэктест M30 на данных Yahoo\n"
            "/backtest_local 0.55 - Локальный бэктест M30\n"
            "/fullbacktest - Полный бэктест на локальных файлах за 3 года"
        )
    elif command == '/subscribe':
        if add_subscriber(chat_id):
            await update.message.reply_text("Вы успешно подписались на сигналы!")
        else:
            await update.message.reply_text("Вы уже подписаны.")
    elif command == '/unsubscribe':
        if remove_subscriber(chat_id):
            await update.message.reply_text("Вы успешно отписались.")
        else:
            await update.message.reply_text("Вы не были подписаны.")
    elif command == '/check':
        await update.message.reply_text("Проверяю сигналы на M5 и M30...")
        # Запускаем проверку и отправку в фоновом режиме
        loop = get_background_loop()
        task = asyncio.run_coroutine_threadsafe(check_and_send_signals_to_chat(chat_id), loop)
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
    elif command == '/backtest':
        await backtest(update, context) # Используем новую функцию-обработчик
    elif command == '/backtest_local':
        await backtest_local(update, context) # Используем новую функцию-обработчик
    elif command == '/fullbacktest':
        await fullbacktest(update, context)
    else:
        await update.message.reply_text(f"Команда '{text}' не распознана.")

async def send_signal_to_chat(chat_id, signal_data):
    """Отправляет один сформатированный сигнал в указанный чат."""
    signal, entry, sl, tp, last_bar, image_path, timeframe = signal_data

    if not signal:
        # await bot.send_message(chat_id, f"({timeframe}) Технического сетапа нет.")
        return

    message = (
        f"🚨 СИГНАЛ ({timeframe}) 🚨\n"
        f"🔔 **Short EURUSD**\n"
        f"🔵 **Вход:** `{entry:.5f}`\n"
        f"🔴 **Stop-Loss:** `{sl:.5f}`\n"
        f"🟢 **Take-Profit:** `{tp:.5f}`\n"
        f"🕗 **Время сетапа:** `{last_bar.name.strftime('%Y-%m-%d %H:%M:%S UTC')}`"
    )
    
    try:
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as img:
                await bot.send_photo(chat_id, photo=img, caption=message)
        else:
            await bot.send_message(chat_id, message)
    except Exception as e:
        logging.error(f"Не удалось отправить сигнал в чат {chat_id}: {e}")

async def check_and_send_signals_to_chat(chat_id):
    """Проверяет оба ТФ и отправляет результат в указанный чат."""
    try:
        # Проверка M5
        signal_data_5m = generate_signal_and_plot()
        if signal_data_5m[0]: # Если есть сигнал
            await send_signal_to_chat(chat_id, signal_data_5m)
        else:
            await bot.send_message(chat_id, "На M5 сетапа нет.")

        # Проверка M30
        signal_data_30m = generate_signal_and_plot_30m()
        if signal_data_30m[0]: # Если есть сигнал
            await send_signal_to_chat(chat_id, signal_data_30m)
        else:
            await bot.send_message(chat_id, "На M30 сетапа нет.")

    except Exception as e:
        logging.error(f"Ошибка при проверке и отправке сигналов: {e}")
        await bot.send_message(chat_id, f"Произошла ошибка: {e}")

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
    logging.info("Получен запрос на /check от планировщика.")
    try:
        loop = get_background_loop()
        task = asyncio.run_coroutine_threadsafe(check_and_send_to_subscribers(), loop)
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
        return "Проверка инициирована.", 200
    except Exception as e:
        logging.error(f"Ошибка в /check: {e}")
        return f"Ошибка: {e}", 500

async def check_and_send_to_subscribers():
    """Проверяет оба ТФ и рассылает сигналы подписчикам."""
    subscribers = get_subscribers()
    if not subscribers:
        logging.info("Нет подписчиков для рассылки.")
        return

    logging.info(f"Начинаю рассылку для {len(subscribers)} подписчиков.")
    
    # Проверяем сигналы один раз
    try:
        signal_data_5m = generate_signal_and_plot()
        signal_data_30m = generate_signal_and_plot_30m()
    except Exception as e:
        logging.error(f"Критическая ошибка при генерации сигналов: {e}")
        return

    # Рассылаем, если они есть
    for sub_id in subscribers:
        if signal_data_5m and signal_data_5m[0]:
            await send_signal_to_chat(sub_id, signal_data_5m)
        if signal_data_30m and signal_data_30m[0]:
            await send_signal_to_chat(sub_id, signal_data_30m)
            
    logging.info("Рассылка завершена.")

@app.route('/')
def index():
    """Стартовая страница для проверки, что сервис жив."""
    return "Telegram Bot is running!"

async def backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Запускает бэктест по команде с фиксированным порогом 0.55."""
    threshold = 0.55
    await update.message.reply_text('Запускаю бэктест с котировками Yahoo...')
    
    try:
        stats, plot_filename = run_backtest(threshold)
        if plot_filename:
            await update.message.reply_document(
                document=open(plot_filename, 'rb'),
                caption=f"📈 **Результаты бэктеста (Yahoo)**\n\n{format_stats_for_telegram(stats)}"
            )
        else:
            await update.message.reply_text("Не удалось сгенерировать отчет.")
    except Exception as e:
        logger.error(f"Ошибка при выполнении бэктеста: {e}")
        await update.message.reply_text(f"Произошла ошибка при выполнении бэктеста: {e}")

async def backtest_local(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Запускает локальный бэктест по команде."""
    try:
        threshold_str = context.args[0]
        threshold = float(threshold_str)
        if not (0 <= threshold <= 1):
            raise ValueError("Порог должен быть между 0 и 1.")
    except (IndexError, ValueError):
        await update.message.reply_text('Пожалуйста, укажите порог для бэктеста, например: /backtest_local 0.55')
        return

    chat_id = update.message.chat.id
    
    loop = get_background_loop()
    task = asyncio.run_coroutine_threadsafe(run_backtest_local_async(chat_id, threshold), loop)
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)

async def run_backtest_local_async(chat_id, threshold):
    """Асинхронно запускает локальный бэктест."""
    logging.info(f"Запуск локального бэктеста для {chat_id} с порогом {threshold}.")
    await bot.send_message(chat_id, f"✅ Запускаю локальный бэктест с фильтром {threshold}. Это может занять несколько минут...")
    
    try:
        # Запускаем в отдельном потоке, чтобы не блокировать бота
        stats, plot_file = await asyncio.to_thread(
            run_backtest_local, 
            'EURUSD_Candlestick_30_m_BID_18.06.2022-18.06.2025 (2).csv', 
            'DOLLAR.IDXUSD_Candlestick_30_m_BID_18.06.2022-18.06.2025 (2).csv',
            threshold
        )
        
        if plot_file and os.path.exists(plot_file):
            await bot.send_message(chat_id, f"📊 Результаты бэктеста:\n\n<pre>{stats}</pre>", parse_mode='HTML')
            with open(plot_file, 'rb') as f:
                await bot.send_document(chat_id, document=f, caption=f"Подробный отчет по локальному бэктесту с фильтром {threshold}")
            os.remove(plot_file)
        else:
            await bot.send_message(chat_id, f"❌ Ошибка во время локального бэктеста: {stats}")
            
    except Exception as e:
        logging.error(f"Критическая ошибка в задаче локального бэктеста: {e}")
        await bot.send_message(chat_id, f"❌ Критическая ошибка: {e}")

async def fullbacktest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Запускает полный бэктест на заданных файлах."""
    chat_id = update.message.chat.id
    loop = get_background_loop()
    task = asyncio.run_coroutine_threadsafe(run_fullbacktest_async(chat_id), loop)
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)

async def run_fullbacktest_async(chat_id):
    """Асинхронно запускает полный бэктест."""
    threshold = 0.55
    eurusd_file = 'EURUSD_Candlestick_30_m_BID_18.06.2022-18.06.2025 (2).csv'
    dxy_file = 'DOLLAR.IDXUSD_Candlestick_30_m_BID_18.06.2022-18.06.2025 (2).csv'

    logging.info(f"Запуск полного бэктеста для {chat_id} с порогом {threshold}.")
    await bot.send_message(chat_id, f"✅ Запускаю полный бэктест с фильтром {threshold}. Это может занять несколько минут...")
    
    try:
        # Проверка наличия файлов
        if not os.path.exists(eurusd_file) or not os.path.exists(dxy_file):
            await bot.send_message(chat_id, f"❌ Ошибка: Не найдены файлы для бэктеста. Убедитесь, что `{eurusd_file}` и `{dxy_file}` находятся в директории проекта.")
            return

        # Запускаем в отдельном потоке, чтобы не блокировать бота
        stats, plot_file = await asyncio.to_thread(
            run_backtest_local, 
            eurusd_file, 
            dxy_file,
            threshold
        )
        
        if plot_file and os.path.exists(plot_file):
            stats_text = format_stats_for_telegram(stats)
            await bot.send_message(chat_id, f"📊 Результаты полного бэктеста:\n\n{stats_text}", parse_mode='Markdown')
            
            with open(plot_file, 'rb') as f:
                await bot.send_document(chat_id, document=f, caption=f"Подробный отчет по полному бэктесту с фильтром {threshold}")
            os.remove(plot_file)
        else:
            await bot.send_message(chat_id, f"❌ Ошибка во время полного бэктеста: {stats}")
            
    except Exception as e:
        logging.error(f"Критическая ошибка в задаче полного бэктеста: {e}")
        await bot.send_message(chat_id, f"❌ Критическая ошибка: {e}")

if __name__ == "__main__":
    # Локальный запуск для отладки
    logging.info("Запуск бота в режиме опроса для локальной отладки...")
    
    # Загружаем модель один раз при старте
    try:
        model = joblib.load(MODEL_FILE)
        logging.info("ML модель успешно загружена.")
    except Exception as e:
        logging.error(f"Не удалось загрузить ML модель: {e}")
        model = None

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Добавляем обработчики команд
    application.add_handler(CommandHandler("start", handle_update))
    application.add_handler(CommandHandler("subscribe", handle_update))
    application.add_handler(CommandHandler("unsubscribe", handle_update))
    application.add_handler(CommandHandler("check", handle_update))
    application.add_handler(CommandHandler("backtest", backtest))
    application.add_handler(CommandHandler("backtest_local", backtest_local))
    application.add_handler(CommandHandler("fullbacktest", fullbacktest))
    
    # Запуск бота
    application.run_polling() 
