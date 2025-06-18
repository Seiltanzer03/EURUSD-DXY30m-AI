import os
import json
import pandas as pd
import pandas_ta as ta
import joblib
import yfinance as yf
import telegram
from flask import Flask, request
import asyncio
from trading_strategy import run_backtest, run_backtest_local, plot_backtest_results_to_pdf
import threading
import logging
import re
from signal_core import generate_signal_and_plot, generate_signal_and_plot_30m, load_data, create_signal_plot
from telegram import Update, InputMediaPhoto
from telegram.ext import Application, CommandHandler, ContextTypes
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- 1. Конфигурация и Инициализация ---

# Настройка логирования для отладки
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Глобальные переменные для "ленивой" инициализации.
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
                logging.info("Initializing PTB application in background thread...")
                # Инициализируем приложение. Это также инициализирует application.bot
                loop.run_until_complete(application.initialize())
                logging.info("PTB application initialized.")
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

# Создаем приложение и используем ЕГО экземпляр бота. Это гарантирует, что у нас только один объект бота.
application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
bot = application.bot

# Настройки стратегии
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

# --- 3. Вспомогательные функции ---

def format_stats_for_telegram(stats):
    """Форматирует статистику из backtesting.py для красивого вывода."""
    if isinstance(stats, str): # Если это сообщение об ошибке
        return f"`{stats}`"

    # Преобразуем Series в словарь для удобного доступа
    stats_dict = stats.to_dict()
    
    # Извлекаем нужные метрики
    start_date = stats_dict.get('Start')
    end_date = stats_dict.get('End')
    duration = stats_dict.get('Duration')
    equity_final = stats_dict.get('Equity Final [$]')
    equity_peak = stats_dict.get('Equity Peak [$]')
    return_pct = stats_dict.get('Return [%]')
    buy_hold_return_pct = stats_dict.get('Buy & Hold Return [%]')
    max_drawdown_pct = stats_dict.get('Max. Drawdown [%]')
    win_rate_pct = stats_dict.get('Win Rate [%]')
    profit_factor = stats_dict.get('Profit Factor')
    trades = stats_dict.get('# Trades')

    return (
        f"*Период:* `{start_date} - {end_date}`\n"
        f"*Длительность:* `{duration}`\n\n"
        f"*Итоговый капитал:* `${equity_final:,.2f}`\n"
        f"*Доходность:* `{return_pct:.2f}%`\n"
        f"*Max просадка:* `{max_drawdown_pct:.2f}%`\n\n"
        f"*Всего сделок:* `{int(trades)}`\n"
        f"*Процент побед:* `{win_rate_pct:.2f}%`\n"
        f"*Profit Factor:* `{profit_factor:.2f}`\n"
        f"*Доходность Buy & Hold:* `{buy_hold_return_pct:.2f}%`"
    )

# --- 4. Обработчики команд ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отправляет приветственное сообщение."""
    await update.message.reply_text(
        "Добро пожаловать! Бот присылает сигналы по стратегиям M5 (SMC) и M30 (SMC+AI).\n\n"
        "Команды:\n"
        "/subscribe - Подписаться на сигналы\n"
        "/unsubscribe - Отписаться\n"
        "/check - Проверить сигналы прямо сейчас\n"
        "/backtest - Бэктест M30 на данных Yahoo\n"
        "/backtest_local 0.55 - Локальный бэктест M30\n"
        "/fullbacktest - Полный бэктест на файлах проекта\n"
        "/backtest_m5 - Бэктест M5 на данных Yahoo за 30 дней"
    )

async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Подписывает пользователя на рассылку."""
    if add_subscriber(update.message.chat_id):
        await update.message.reply_text("Вы успешно подписались на сигналы!")
    else:
        await update.message.reply_text("Вы уже подписаны.")

async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отписывает пользователя от рассылки."""
    if remove_subscriber(update.message.chat_id):
        await update.message.reply_text("Вы успешно отписались.")
    else:
        await update.message.reply_text("Вы не были подписаны.")

async def check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Запускает проверку сигналов по запросу пользователя."""
    await update.message.reply_text("Проверяю сигналы на M5 и M30...")
    loop = get_background_loop()
    task = asyncio.run_coroutine_threadsafe(check_and_send_signals_to_chat(update.message.chat_id), loop)
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)

async def backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Запускает бэктест по команде с фиксированным порогом 0.55."""
    threshold = 0.55
    await update.message.reply_text('Запускаю бэктест с котировками Yahoo...')
    
    try:
        # Запускаем в отдельном потоке, чтобы не блокировать бота
        stats, plot_filename = await asyncio.to_thread(run_backtest, threshold)
        
        if plot_filename and os.path.exists(plot_filename):
            await update.message.reply_document(
                document=open(plot_filename, 'rb'),
                caption=f"📈 **Результаты бэктеста (Yahoo)**\n\n{format_stats_for_telegram(stats)}",
                parse_mode='Markdown'
            )
            os.remove(plot_filename)
        else:
            await update.message.reply_text(f"Не удалось сгенерировать отчет. {stats}")
    except Exception as e:
        logging.error(f"Ошибка при выполнении бэктеста: {e}")
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
    eurusd_file = 'EURUSD_Candlestick_30_m_BID_18.06.2022-18.06.2025 (2).csv'
    dxy_file = 'DOLLAR.IDXUSD_Candlestick_30_m_BID_18.06.2022-18.06.2025 (2).csv'
    
    logging.info(f"Запуск локального бэктеста для {chat_id} с порогом {threshold}.")
    await bot.send_message(chat_id, f"✅ Запускаю локальный бэктест с фильтром {threshold}. Это может занять несколько минут...")
    
    try:
        if not os.path.exists(eurusd_file) or not os.path.exists(dxy_file):
            await bot.send_message(chat_id, f"❌ Ошибка: Не найдены файлы для бэктеста. Убедитесь, что `{eurusd_file}` и `{dxy_file}` находятся в директории проекта.")
            return

        stats, plot_file = await asyncio.to_thread(
            run_backtest_local, 
            eurusd_file, 
            dxy_file,
            threshold
        )
        
        if plot_file and os.path.exists(plot_file):
            stats_text = format_stats_for_telegram(stats)
            await bot.send_message(chat_id, f"📊 Результаты бэктеста:\n\n{stats_text}", parse_mode='Markdown')
            
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
    chat_id = update.message.chat_id
    loop = get_background_loop()
    # Просто передаем управление в уже существующую функцию для локального бэктеста с фиксированным порогом
    task = asyncio.run_coroutine_threadsafe(run_backtest_local_async(chat_id, 0.55), loop)
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)

async def send_signal_to_chat(chat_id, signal_data):
    """Отправляет один сформатированный сигнал в указанный чат."""
    signal, entry, sl, tp, last_bar, image_path, timeframe = signal_data

    if not signal:
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
                await bot.send_photo(chat_id, photo=img, caption=message, parse_mode='Markdown')
            os.remove(image_path) # Удаляем временный файл
        else:
            await bot.send_message(chat_id, message, parse_mode='Markdown')
    except Exception as e:
        logging.error(f"Не удалось отправить сигнал в чат {chat_id}: {e}")

async def check_and_send_signals_to_chat(chat_id):
    """Проверяет оба ТФ и отправляет результат в указанный чат."""
    try:
        await bot.send_message(chat_id, "Проверяю M5...")
        signal_data_5m = generate_signal_and_plot()
        if signal_data_5m[0]: 
            await send_signal_to_chat(chat_id, signal_data_5m)
        else:
            await bot.send_message(chat_id, "На M5 сетапа нет.")

        await bot.send_message(chat_id, "Проверяю M30...")
        signal_data_30m = generate_signal_and_plot_30m()
        if signal_data_30m[0]:
            await send_signal_to_chat(chat_id, signal_data_30m)
        else:
            await bot.send_message(chat_id, "На M30 сетапа нет.")

    except Exception as e:
        logging.error(f"Ошибка при проверке и отправке сигналов: {e}")
        await bot.send_message(chat_id, f"Произошла ошибка: {e}")

def plot_m5_signals(data, signals, filename):
    """Рисует график цены и найденных сигналов M5."""
    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(15, 8), dpi=150)

        ax.plot(data.index, data.Close, label='EURUSD Close 5m', color='lightgray', alpha=0.7, linewidth=1)

        if signals:
            signal_times = [s['time'] for s in signals]
            signal_prices = [s['entry'] for s in signals]
            ax.plot(signal_times, signal_prices, 'v', color='#ff4d4d', markersize=8, label='Точки входа в шорт', linestyle='None')

        ax.set_title(f'Найденные M5 сигналы за последние {len(data.index.dayofyear.unique())} дней', fontsize=16)
        ax.set_ylabel('Цена')
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()
        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', format='pdf')
        plt.close(fig)
        return filename
    except Exception as e:
        logging.error(f"Ошибка при создании графика M5 сигналов: {e}")
        return None

async def backtest_m5(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Запускает бэктест пятиминутного SMC сигнала, отправляет PDF-отчет и PNG-изображения последних 5 сигналов."""
    chat_id = update.message.chat_id
    await bot.send_message(chat_id, '▶️ Запускаю поиск сигналов M5 за последние 30 дней. Это может занять до минуты...')
    
    try:
        LOOKBACK = 12
        START_HOUR = 13
        END_HOUR = 17
        SL_RATIO = 0.002
        TP_RATIO = 0.005
        
        # Запускаем загрузку данных в отдельном потоке
        data = await asyncio.to_thread(load_data, period='30d', interval='5m')
        
        signals = []
        # Проходим по данным и ищем сигналы
        for i in range(LOOKBACK + 2, len(data)):
            last_bar = data.iloc[i-2]
            entry_bar = data.iloc[i-1]
            previous_bars = data.iloc[i-(LOOKBACK+2):i-2]
            
            is_trading_time = START_HOUR <= last_bar.name.hour <= END_HOUR
            dxy_raid = last_bar['DXY_Low'] < previous_bars['DXY_Low'].min()
            eurusd_judas_swing = last_bar['High'] > previous_bars['High'].max()
            
            if is_trading_time and dxy_raid and eurusd_judas_swing:
                signals.append({
                    'time': entry_bar.name,
                    'entry': entry_bar['Open'],
                })

        # --- 1. Текстовый отчет ---
        msg = f'✅ *Отчет по M5 (SMC) Сигналам*\n\nНайдено сигналов за 30 дней: `{len(signals)}`'
        if signals:
            msg += f'\nДалее будут отправлены изображения последних {min(5, len(signals))} сигналов и общий PDF отчет.'
        else:
            msg += '\nСигналов для отображения нет.'
        await bot.send_message(chat_id, msg, parse_mode='Markdown')

        # --- 2. Изображения последних 5 сигналов ---
        if signals:
            last_signals = signals[-5:]
            media_group = []
            opened_files = []
            image_paths_to_delete = []

            await bot.send_message(chat_id, f"🖼️ Генерирую изображения для {len(last_signals)} последних сигналов...")

            for i, signal_info in enumerate(last_signals):
                try:
                    signal_time = signal_info['time']
                    entry_price = signal_info['entry']
                    sl_price = entry_price * (1 + SL_RATIO)
                    tp_price = entry_price * (1 - TP_RATIO)
                    
                    # Находим индекс сигнала, чтобы взять срез данных для графика
                    signal_index = data.index.get_loc(signal_time)
                    plot_data = data.iloc[max(0, signal_index - 59) : signal_index + 1]

                    plot_title = f"M5 Signal at {signal_time.strftime('%Y-%m-%d %H:%M UTC')}"
                    plot_filename = f"m5_signal_{i}_{chat_id}.png"
                    
                    # Генерируем график в потоке
                    await asyncio.to_thread(
                        create_signal_plot, 
                        plot_data, entry_price, sl_price, tp_price, plot_title, plot_filename
                    )

                    if os.path.exists(plot_filename):
                        image_paths_to_delete.append(plot_filename)
                        # Открываем файл и добавляем в список для отправки и последующего закрытия
                        f = open(plot_filename, 'rb')
                        opened_files.append(f)
                        # Добавляем в медиа-группу. Капшен только у первого фото.
                        caption = f"Последние {len(last_signals)} сигналов. Сигнал {i+1}/{len(last_signals)}" if i == 0 else None
                        media_group.append(InputMediaPhoto(media=f, caption=caption, parse_mode='Markdown'))

                except Exception as e:
                    logging.error(f"Ошибка при создании изображения для сигнала {i}: {e}")
                    await bot.send_message(chat_id, f"Не удалось создать изображение для сигнала {signal_time.strftime('%Y-%m-%d %H:%M')}.")
            
            # Отправляем медиа-группу если есть что отправлять
            if media_group:
                await bot.send_media_group(chat_id, media=media_group)
            
            # Закрываем все открытые файлы
            for f in opened_files:
                f.close()
            
            # Удаляем временные файлы изображений
            for path in image_paths_to_delete:
                if os.path.exists(path):
                    os.remove(path)

        # --- 3. Общий PDF-отчет ---
        if not data.empty:
            await bot.send_message(chat_id, "📄 Генерирую итоговый PDF-отчет со всеми сигналами...")
            pdf_plot_filename = f"m5_signals_report_{chat_id}.pdf"
            
            # Запускаем отрисовку в отдельном потоке
            await asyncio.to_thread(plot_m5_signals, data, signals, pdf_plot_filename)
            
            if os.path.exists(pdf_plot_filename):
                with open(pdf_plot_filename, 'rb') as doc:
                    await bot.send_document(chat_id, document=doc, caption="Итоговый PDF-отчет со всеми найденными сигналами M5.")
                os.remove(pdf_plot_filename)
            else:
                 await bot.send_message(chat_id, "Не удалось создать итоговый PDF-отчет.")

    except Exception as e:
        logging.error(f"Ошибка в /backtest_m5: {e}", exc_info=True)
        await bot.send_message(chat_id, f"❌ Критическая ошибка при выполнении M5 бэктеста: {e}")

# --- 5. Веб-сервер и Роуты ---

@app.route('/webhook', methods=['POST'])
def webhook():
    """Обрабатывает вебхуки от Telegram, используя PTB Application."""
    try:
        update_data = request.get_json(force=True)
        logging.info(f"Webhook received: {update_data}")
        # Используем application.bot, чтобы передать в update правильный, инициализированный экземпляр бота
        update = telegram.Update.de_json(update_data, application.bot)
        
        loop = get_background_loop()
        asyncio.run_coroutine_threadsafe(application.process_update(update), loop)
        logging.info("application.process_update task scheduled successfully.")
        
    except Exception as e:
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
    
    try:
        signal_data_5m = generate_signal_and_plot()
        signal_data_30m = generate_signal_and_plot_30m()
    except Exception as e:
        logging.error(f"Критическая ошибка при генерации сигналов: {e}")
        return

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

# --- 6. Запуск ---

# Регистрируем все обработчики команд
application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("subscribe", subscribe))
application.add_handler(CommandHandler("unsubscribe", unsubscribe))
application.add_handler(CommandHandler("check", check))
application.add_handler(CommandHandler("backtest", backtest))
application.add_handler(CommandHandler("backtest_local", backtest_local))
application.add_handler(CommandHandler("fullbacktest", fullbacktest))
application.add_handler(CommandHandler("backtest_m5", backtest_m5))
