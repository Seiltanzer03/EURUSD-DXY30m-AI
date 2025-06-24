import os
import json
import pandas as pd
import pandas_ta as ta
import joblib
import yfinance as yf
import telegram
from flask import Flask, request, abort, Response
import asyncio
from trading_strategy import run_backtest, run_full_backtest
import threading
import logging
import subprocess
import re
from signal_core import generate_signal_and_plot_30m, find_signals_in_period, find_last_signal
import uuid
import requests
import time
from telegram_game_handler import handle_game_callback_query, send_report_to_game_server
import concurrent.futures

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
reports = {}  # {token: (html, expire_time)}

def cleanup_reports():
    while True:
        now = time.time()
        to_delete = [token for token, (_, exp) in reports.items() if exp < now]
        for token in to_delete:
            del reports[token]
        time.sleep(60)

# Запуск очистки отчётов в фоне
threading.Thread(target=cleanup_reports, daemon=True).start()

@app.route('/game_report')
def game_report():
    token = request.args.get('start') or request.args.get('token')
    logging.info(f"/game_report запрошен с токеном: {token}")
    
    if not token:
        logging.warning(f"/game_report: токен не предоставлен")
        return abort(404, 'Report not found')
        
    # Проверяем, если токен содержится в reports напрямую
    if token in reports:
        html, _ = reports[token]
        logging.info(f"/game_report: отдаю отчёт для токена: {token}")
        return Response(html, mimetype='text/html')
    
    # Если токен не найден напрямую, проверяем по user_id  
    # Предполагаем, что токен может быть просто user_id, и нам нужно найти последний отчет для этого пользователя
    try:
        user_id = int(token.split('_')[0]) if '_' in token else int(token)
        user_reports = sorted([
            (t, exp_time) 
            for t, (_, exp_time) in reports.items() 
            if t.startswith(f"{user_id}_")
        ], key=lambda x: x[1], reverse=True)
        
        if user_reports:
            # Берем самый свежий отчет для этого пользователя
            actual_token = user_reports[0][0]
            html, _ = reports[actual_token]
            logging.info(f"/game_report: отдаю отчёт для пользователя {user_id}, найден по токену: {actual_token}")
            return Response(html, mimetype='text/html')
    except (ValueError, IndexError, AttributeError) as e:
        logging.warning(f"/game_report: ошибка при поиске отчета по user_id из токена: {e}")
        
    # Если ничего не нашли
    logging.warning(f"/game_report: токен не найден: {token}. reports.keys(): {list(reports.keys())}")
    return abort(404, 'Report not found')

# Загрузка секретов из переменных окружения
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID') # ID администратора для отладки
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

# Настройки стратегии
MODEL_FILE = 'ml_model_final_fix.joblib'
PREDICTION_THRESHOLD = 0.55 # Оптимальный порог для live-сигналов
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
    logging.info(f"Executing run_backtest_async for chat_id {chat_id} with threshold {threshold}.")
    try:
        await bot.send_message(chat_id, f"✅ Запускаю бэктест с фильтром {threshold}. Это может занять несколько минут...")
        stats, plot_file = await asyncio.to_thread(run_backtest, threshold)
        if plot_file:
            msg = format_backtest_message(stats, '30m', '2024-01-01', '2024-06-01')
            await bot.send_message(chat_id, msg)
            with open(plot_file, 'r', encoding='utf-8') as f:
                html = f.read()
            # Формируем токен с указанием ID пользователя
            token = f"{chat_id}_{str(uuid.uuid4())}"
            expire_time = time.time() + 1800
            reports[token] = (html, expire_time)
            logging.info(f"Сохраняю отчёт для chat_id={chat_id}, token={token}, файл={plot_file}")
            
            # Отправляем отчет на сервер игры
            send_report_to_game_server(token, html)
            
            try:
                await bot.send_game(
                    chat_id=chat_id, 
                    game_short_name='backtest_report',
                    start_parameter=token
                )
            except (TypeError, ValueError):
                await bot.send_game(chat_id=chat_id, game_short_name='backtest_report')
            os.remove(plot_file)
        else:
            await bot.send_message(chat_id, f"❌ Ошибка во время бэктеста: {stats}")
    except Exception as e:
        logging.error(f"Ошибка в run_backtest_async: {e}", exc_info=True)
        await bot.send_message(chat_id, f"❌ Критическая ошибка в задаче бэктеста: {e}")

async def run_full_backtest_async(chat_id, threshold):
    logging.info(f"Executing run_full_backtest_async for chat_id {chat_id} with threshold {threshold}.")
    try:
        await bot.send_message(chat_id, f"✅ Запускаю полный бэктест по историческим данным с фильтром {threshold}. Это может занять несколько минут...")
        stats, plot_file = await asyncio.to_thread(run_full_backtest, threshold)
        if plot_file:
            msg = format_backtest_message(stats, '30m', '2024-01-01', '2024-06-01')
            await bot.send_message(chat_id, msg)
            with open(plot_file, 'r', encoding='utf-8') as f:
                html = f.read()
            # Формируем токен с указанием ID пользователя
            token = f"{chat_id}_{str(uuid.uuid4())}"
            expire_time = time.time() + 1800
            reports[token] = (html, expire_time)
            logging.info(f"Сохраняю полный отчёт для chat_id={chat_id}, token={token}, файл={plot_file}")
            
            # Отправляем отчет на сервер игры
            send_report_to_game_server(token, html)
            
            try:
                await bot.send_game(
                    chat_id=chat_id, 
                    game_short_name='backtest_report',
                    start_parameter=token
                )
            except (TypeError, ValueError):
                await bot.send_game(chat_id=chat_id, game_short_name='backtest_report')
            os.remove(plot_file)
        else:
            await bot.send_message(chat_id, f"❌ Ошибка во время полного бэктеста: {stats}")
    except Exception as e:
        logging.error(f"Ошибка в run_full_backtest_async: {e}", exc_info=True)
        await bot.send_message(chat_id, f"❌ Критическая ошибка в задаче полного бэктеста: {e}")

# Функция удалена, т.к. убран 5m сетап

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
        # Проверяем наличие callback_query (для обработки игровых запросов)
        if update.callback_query and update.callback_query.game_short_name == 'backtest_report':
            # Вызываем функцию обработки игровых callback_query
            return await handle_game_callback_query(bot, update, reports)
        
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
        # Команда /backtest_m5 удалена, т.к. убран 5m сетап
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
    await bot.send_message(chat_id, "🔍 Начинаю проверку последних сигналов...")
    
    try:
        # Устанавливаем таймаут для поиска сигналов
        timeout_seconds = 300
        
        # Ищем последний сигнал 30m с таймаутом
        last_signal_30m = None
        try:
            # Создаем задачу для поиска сигнала 30m
            task_30m = asyncio.create_task(asyncio.to_thread(find_last_signal, timeframe='30m'))
            # Ждем выполнения с таймаутом
            last_signal_30m = await asyncio.wait_for(task_30m, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logging.warning(f"Таймаут при поиске сигнала 30m (превышено {timeout_seconds} сек)")
        except Exception as e:
            logging.error(f"Ошибка при поиске сигнала 30m: {e}")
        
        # Формируем отчет для 30m таймфрейма
        if last_signal_30m:
            # Форматируем время и цены
            entry_time = last_signal_30m['time'].strftime('%Y-%m-%d %H:%M:%S')
            exit_time = last_signal_30m['exit_time'].strftime('%Y-%m-%d %H:%M:%S') if last_signal_30m['exit_time'] else "Не достигнут"
            
            report_30m = (
                f"🔸 **Последний сигнал 30m:**\n"
                f"   Время входа: {entry_time}\n"
                f"   Цена входа: {last_signal_30m['entry']:.5f}\n"
                f"   Стоп-лосс: {last_signal_30m['sl']:.5f}\n"
                f"   Тейк-профит: {last_signal_30m['tp']:.5f}\n"
                f"   Статус: {last_signal_30m['status']}\n"
            )
            
            # Добавляем информацию о выходе, если сделка закрыта
            if "закрыта" in last_signal_30m['status']:
                report_30m += f"   Время выхода: {exit_time}\n"
                report_30m += f"   Цена выхода: {last_signal_30m['exit_price']:.5f}\n"
        else:
            report_30m = "🔸 **30m:** Сигналов не найдено или превышен таймаут поиска."
        
        # Формируем и отправляем отчет
        report = (
            f"📊 Отчет о последних сигналах:\n\n"
            f"{report_30m}"
        )
        await bot.send_message(chat_id, report, parse_mode='Markdown')
        
        # Отправляем картинку для найденного сигнала
        if last_signal_30m and last_signal_30m['plot_path'] and os.path.exists(last_signal_30m['plot_path']):
            try:
                with open(last_signal_30m['plot_path'], 'rb') as f:
                    await bot.send_photo(
                        chat_id, 
                        photo=f, 
                        caption=f"Сигнал {last_signal_30m['timeframe']} от {last_signal_30m['time'].strftime('%Y-%m-%d %H:%M:%S')} - {last_signal_30m['status']}"
                    )
                # Удаляем файл после отправки
                os.remove(last_signal_30m['plot_path'])
            except Exception as e:
                logging.error(f"Ошибка при отправке изображения {last_signal_30m['plot_path']}: {e}", exc_info=True)
        elif last_signal_30m and last_signal_30m['plot_path']:
            logging.error(f"Файл изображения {last_signal_30m['plot_path']} не существует")
            await bot.send_message(
                chat_id,
                f"⚠️ Не удалось создать график для сигнала {last_signal_30m['timeframe']} от {last_signal_30m['time'].strftime('%Y-%m-%d %H:%M:%S')}"
            )

    except Exception as e:
        logging.error(f"Ошибка при выполнении проверки для chat_id {chat_id}: {e}", exc_info=True)
        await bot.send_message(chat_id, f"❌ Произошла ошибка во время проверки: {e}")

async def generate_and_send_signals():
    """Генерирует, и если находит, рассылает сигналы подписчикам."""
    try:
        # Ищем сигналы за последние 15 минут (интервал между проверками)
        signals_30m = find_signals_in_period(minutes=15, timeframe='30m')
        
        # Если есть сигналы, отправляем их
        if signals_30m:
            # Отправляем каждый сигнал отдельно
            for signal in signals_30m:
                if signal['time'] is not None and signal['entry'] is not None:
                    await send_signals(
                        True, signal['entry'], signal['sl'], signal['tp'], 
                        signal['time'], signal['plot_path'], signal['timeframe']
                    )
        
        # Если сигналов за последние 15 минут не найдено, проверяем текущий сигнал
        # (сохраняем текущее поведение для совместимости)
        if not signals_30m:
            # 30-минутный таймфрейм
            signal_30m, entry_30m, sl_30m, tp_30m, last_30m, image_path_30m, timeframe_30m, _ = generate_signal_and_plot_30m()

            # Если есть сигнал, отправляем
            if signal_30m:
                await send_signals(
                    signal_30m, entry_30m, sl_30m, tp_30m, last_30m, image_path_30m, timeframe_30m
                )
    except Exception as e:
        logging.error(f"Ошибка при генерации и отправке сигналов: {e}", exc_info=True)

async def send_signals(signal_30m, entry_30m, sl_30m, tp_30m, last_30m, image_path_30m, timeframe_30m):
    """Асинхронно рассылает сигналы подписчикам."""
    subscribers = get_subscribers()
    if not subscribers:
        logging.info("Сигнал есть, но подписчиков нет.")
        return

    message_parts = []
    images_to_send = []

    if signal_30m:
        # Корректно определяем время сетапа
        if last_30m is not None:
            setup_time_30m = last_30m.name if hasattr(last_30m, 'name') else last_30m
            try:
                setup_time_30m_str = setup_time_30m.strftime('%Y-%m-%d %H:%M')
            except Exception:
                setup_time_30m_str = str(setup_time_30m)
        else:
            setup_time_30m_str = "—"
        message_30m = (
            f"🚨 СИГНАЛ НА ПРОДАЖУ (SELL) EUR/USD ({timeframe_30m}) 🚨\n\n"
            f"Время сетапа (UTC): `{setup_time_30m_str}`\n"
            f"Вход: {entry_30m:.5f}\n"
            f"Стоп: {sl_30m:.5f}\n"
            f"Тейк: {tp_30m:.5f}"
        )
        message_parts.append(message_30m)
        if image_path_30m and os.path.exists(image_path_30m):
            images_to_send.append(image_path_30m)

    if not message_parts:
        return

    final_message = "\n\n".join(message_parts)

    for chat_id in subscribers:
        try:
            await bot.send_message(chat_id, final_message, parse_mode='Markdown')
            
            # Всегда отправляем картинки, если они есть
            for img_path in images_to_send:
                try:
                    with open(img_path, 'rb') as f:
                        await bot.send_photo(chat_id, photo=f)
                except Exception as img_error:
                    logging.error(f"Не удалось отправить изображение {img_path} подписчику {chat_id}: {img_error}")
            
            # Удаляем изображения после отправки всем подписчикам
            for img_path in images_to_send:
                try:
                    if os.path.exists(img_path):
                        os.remove(img_path)
                except Exception as del_error:
                    logging.error(f"Не удалось удалить файл {img_path}: {del_error}")
                    
        except Exception as e:
            logging.error(f"Не удалось отправить сигнал подписчику {chat_id}: {e}")

@app.route('/save_report', methods=['POST'])
def save_report():
    try:
        data = request.get_json()
        token = data['token']
        html = data['html']
        expire_time = time.time() + 1800  # 30 минут жизни
        reports[token] = (html, expire_time)
        logging.info(f"API: Сохранен отчет с токеном {token}")
        return {'status': 'ok'}
    except Exception as e:
        logging.error(f"Ошибка при обработке запроса /save_report: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}, 500

@app.route('/')
def index():
    return "Trading Bot is running."

def format_backtest_message(stats, timeframe, period_start, period_end):
    """
    Формирует красивое текстовое сообщение для пользователя по результатам бэктеста.
    """
    # Извлекаем нужные значения из stats
    total_return = stats.get('Return [%]', stats.get('Equity Final [$]', '—'))
    max_drawdown = stats.get('Max. Drawdown [%]', stats.get('Max. Drawdown', '—'))
    n_trades = stats.get('# Trades', stats.get('Trades', '—'))
    
    # Расчет количества прибыльных и убыточных сделок
    win_trades = stats.get('Win Rate [%]', None)
    if win_trades is not None:
        # Если у нас есть процент прибыльных сделок, рассчитаем количество
        win_count = int(round(float(n_trades) * float(win_trades) / 100))
        loss_count = int(n_trades) - win_count
    else:
        # Если нет процента, попробуем получить прямые значения
        win_count = stats.get('Win Trades', '—')
        loss_count = stats.get('Loss Trades', '—')
        
        # Если у нас нет прямых значений, но есть общее количество сделок
        if win_count == '—' and n_trades != '—':
            # Попробуем использовать Win Rate [%] для расчета
            win_rate = stats.get('Win Rate [%]', None)
            if win_rate is not None:
                win_count = int(round(float(n_trades) * float(win_rate) / 100))
                loss_count = int(n_trades) - win_count
    
    # Получаем проценты выигрышных и проигрышных сделок
    win_pct = stats.get('Win Rate [%]', None)
    if win_pct is not None:
        loss_pct = 100 - float(win_pct)
    else:
        loss_pct = stats.get('Loss Rate [%]', None)
    
    sharpe = stats.get('Sharpe Ratio', stats.get('Sharpe', '—'))

    # Форматируем проценты
    def fmt(val, is_pct=False):
        if val is None or val == '—':
            return '—'
        try:
            if is_pct:
                return f"{float(val):+.1f}%"
            return str(val)
        except Exception:
            return str(val)

    # Определяем реальный период бэктеста из данных
    start_date = stats.get('Start', period_start)
    end_date = stats.get('End', period_end)
    
    # Форматируем даты, если они есть
    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, pd.Timestamp):
        end_date = end_date.strftime('%Y-%m-%d')

    msg = f"""
📊 Результаты бэктеста ({timeframe}):

▫️ Итоговая доходность: {fmt(total_return, True)}
▫️ Максимальная просадка: {fmt(max_drawdown, True)}
▫️ Количество сделок: {n_trades}
▫️ Винрейт: {fmt(win_pct, True) if win_pct else '—'}
▫️ Коэффициент Шарпа: {fmt(sharpe)}

⏳ Период теста: {start_date} — {end_date}

Подробный отчёт ниже.
"""
    return msg

if __name__ == "__main__":
    # Локальный запуск для отладки. На Render будет использоваться gunicorn.
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 
