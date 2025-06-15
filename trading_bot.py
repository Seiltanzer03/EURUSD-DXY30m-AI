import os
import json
import pandas as pd
import pandas_ta as ta
import joblib
import yfinance as yf
import telegram
from flask import Flask, request
import asyncio
from datetime import datetime, timedelta
import pytz

# Новые импорты для альтернативных источников
try:
    import FinanceDataReader as fdr
except ImportError:
    fdr = None
try:
    import efinance as ef
except ImportError:
    ef = None

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
SIGNALS_HISTORY_FILE = 'signals_history.json'

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

# --- 3. Управление историей сигналов ---
def get_signals_history():
    """Читает историю сигналов из файла."""
    if not os.path.exists(SIGNALS_HISTORY_FILE):
        return []
    with open(SIGNALS_HISTORY_FILE, 'r') as f:
        return json.load(f)

def save_signal(signal_data):
    """Сохраняет сигнал в историю."""
    history = get_signals_history()
    history.append(signal_data)
    with open(SIGNALS_HISTORY_FILE, 'w') as f:
        json.dump(history, f)

# --- 4. Универсальная загрузка данных ---
def try_yfinance(ticker, **kwargs):
    try:
        data = yf.download(ticker, **kwargs)
        if not data.empty:
            return data, 'yfinance'
    except Exception as e:
        print(f"yfinance error: {e}")
    return None, None

def try_fdr(ticker, start=None, end=None):
    if fdr is None:
        return None, None
    try:
        # FDR поддерживает только дневные данные для валют
        if ticker == 'EURUSD=X':
            data = fdr.DataReader('EUR/USD', start, end)
        elif ticker == 'DX-Y.NYB':
            data = fdr.DataReader('DXY', start, end)
        else:
            return None, None
        if not data.empty:
            return data, 'FinanceDataReader'
    except Exception as e:
        print(f"FDR error: {e}")
    return None, None

def try_efinance(ticker, start=None, end=None):
    if ef is None:
        return None, None
    try:
        # efinance поддерживает только дневные данные для валют
        if ticker == 'EURUSD=X':
            data = ef.currency.get_quote_history('EURUSD', beg=start, end=end, klt=24)
        elif ticker == 'DX-Y.NYB':
            data = ef.currency.get_quote_history('USDIDX', beg=start, end=end, klt=24)
        else:
            return None, None
        if not data.empty:
            return data, 'efinance'
    except Exception as e:
        print(f"efinance error: {e}")
    return None, None

def try_csv(ticker):
    try:
        if ticker == 'EURUSD=X':
            data = pd.read_csv('eurusd_data_2y.csv', parse_dates=[0], dayfirst=True)
        elif ticker == 'DX-Y.NYB':
            data = pd.read_csv('dxy_data_2y.csv', parse_dates=[0], dayfirst=True)
        else:
            return None, None
        if not data.empty:
            return data, 'csv'
    except Exception as e:
        print(f"csv error: {e}")
    return None, None

def get_data_universal(ticker, period=None, interval=None, start=None, end=None):
    # 1. yfinance (только если есть интервал)
    if interval:
        data, src = try_yfinance(ticker, period=period, interval=interval) if period else try_yfinance(ticker, start=start, end=end, interval=interval)
        if data is not None:
            return data, src
    # 2. FDR (только дневные)
    data, src = try_fdr(ticker, start, end)
    if data is not None:
        return data, src
    # 3. efinance (только дневные)
    data, src = try_efinance(ticker, start, end)
    if data is not None:
        return data, src
    # 4. CSV (локальный)
    data, src = try_csv(ticker)
    if data is not None:
        return data, src
    return None, None

# --- 5. Основная логика стратегии ---
def get_live_data():
    print("Загрузка свежих данных...")
    eurusd_data, src1 = get_data_universal('EURUSD=X', period='5d', interval='30m')
    if eurusd_data is None:
        print("Рынок EUR/USD закрыт или данные недоступны.")
        return None
    dxy_data, src2 = get_data_universal('DX-Y.NYB', period='5d', interval='30m')
    if dxy_data is None:
        print("Рынок DXY закрыт или данные недоступны.")
        return None
    # Приведение к нужному формату (универсально)
    eurusd_data = eurusd_data.copy()
    dxy_data = dxy_data.copy()
    if 'Datetime' not in eurusd_data.columns:
        eurusd_data.reset_index(inplace=True)
        date_col = next(col for col in ['Datetime', 'Date', 'index', 'Time'] if col in eurusd_data.columns)
        eurusd_data.rename(columns={date_col: 'Datetime'}, inplace=True)
    if 'Datetime' not in dxy_data.columns:
        dxy_data.reset_index(inplace=True)
        date_col = next(col for col in ['Datetime', 'Date', 'index', 'Time'] if col in dxy_data.columns)
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
    print(f"Данные успешно обработаны. Источник EURUSD: {src1}, DXY: {src2}")
    return data

def get_historical_data(start_date, end_date=None):
    print(f"Загрузка исторических данных с {start_date} по {end_date or 'сегодня'}...")
    eurusd_data, src1 = get_data_universal('EURUSD=X', start=start_date, end=end_date, interval='30m')
    if eurusd_data is None:
        print("Данные EUR/USD недоступны.")
        return None
    dxy_data, src2 = get_data_universal('DX-Y.NYB', start=start_date, end=end_date, interval='30m')
    if dxy_data is None:
        print("Данные DXY недоступны.")
        return None
    eurusd_data = eurusd_data.copy()
    dxy_data = dxy_data.copy()
    if 'Datetime' not in eurusd_data.columns:
        eurusd_data.reset_index(inplace=True)
        date_col = next(col for col in ['Datetime', 'Date', 'index', 'Time'] if col in eurusd_data.columns)
        eurusd_data.rename(columns={date_col: 'Datetime'}, inplace=True)
    if 'Datetime' not in dxy_data.columns:
        dxy_data.reset_index(inplace=True)
        date_col = next(col for col in ['Datetime', 'Date', 'index', 'Time'] if col in dxy_data.columns)
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
    print(f"Исторические данные успешно обработаны. Источник EURUSD: {src1}, DXY: {src2}")
    return data

def check_for_signal(data=None, candle_index=-2, save_to_history=True):
    """Проверяет сигнал и возвращает сообщение или None."""
    try:
        ml_model = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        return f"ОШИБКА: Файл модели {MODEL_FILE} не найден!"

    if data is None:
        data = get_live_data()
    if data is None: 
        return "Рынок закрыт, проверка отменена."

    if candle_index >= len(data) or candle_index < -len(data):
        return f"Ошибка: индекс свечи {candle_index} вне диапазона данных."

    last_candle = data.iloc[candle_index]
    last_candle_time = last_candle.name
    
    # Проверка на свежесть данных только для реальных данных (не бэктест)
    if candle_index == -2 and data is None:
        if (pd.Timestamp.now(tz='UTC') - last_candle_time.tz_convert('UTC')).total_seconds() > 3600 * 4:
            return f"Данные старые (последняя свеча: {last_candle_time}), рынок закрыт."

    # Проверка времени торговой сессии
    current_hour = last_candle_time.hour if isinstance(last_candle_time, pd.Timestamp) else last_candle_time.hour
    if not (13 <= current_hour <= 17):
        return f"Вне торгового времени (час UTC: {current_hour})."
    
    # Определение индексов для проверки паттерна
    if abs(candle_index) >= len(data):
        return "Недостаточно данных для анализа."
        
    end_index = candle_index
    start_index = end_index - LOOKBACK_PERIOD
    
    if start_index < -len(data):
        start_index = -len(data)
    
    # Проверка паттерна
    eurusd_judas_swing = last_candle['High'] > data['High'].iloc[start_index:end_index].max()
    dxy_raid = last_candle['DXY_Low'] < data['DXY_Low'].iloc[start_index:end_index].min()

    if eurusd_judas_swing and dxy_raid:
        features = [last_candle[col] for col in ['RSI', 'MACD', 'MACD_hist', 'MACD_signal', 'ATR']]
        win_prob = ml_model.predict_proba([features])[0][1]
        
        if win_prob >= PREDICTION_THRESHOLD:
            signal_message = (
                f"🚨 СИГНАЛ НА ПРОДАЖУ (SELL) EUR/USD 🚨\n\n"
                f"Вероятность успеха: *{win_prob:.2%}*\n"
                f"Время сетапа (UTC): `{last_candle_time}`"
            )
            
            # Сохраняем сигнал в историю
            if save_to_history:
                signal_data = {
                    "timestamp": last_candle_time.isoformat(),
                    "type": "SELL",
                    "probability": float(win_prob),
                    "price": float(last_candle['Close']),
                    "rsi": float(last_candle['RSI']),
                    "macd": float(last_candle['MACD']),
                    "atr": float(last_candle['ATR'])
                }
                save_signal(signal_data)
                
            return signal_message
    return "Активных сигналов нет."

def run_backtest(days=60):
    """Запускает бэктест на исторических данных."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = get_historical_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if data is None:
        return "Не удалось загрузить исторические данные для бэктеста."
    
    signals = []
    for i in range(LOOKBACK_PERIOD + 1, len(data)):
        # Проверяем каждую свечу в данных
        result = check_for_signal(data, i, save_to_history=False)
        if "СИГНАЛ НА ПРОДАЖУ" in result:
            candle_time = data.index[i]
            price = data.iloc[i]['Close']
            signals.append({
                "timestamp": candle_time.isoformat(),
                "price": float(price),
                "message": result
            })
    
    if not signals:
        return f"За последние {days} дней сигналов не найдено."
    
    # Формируем отчет
    report = f"Найдено {len(signals)} сигналов за последние {days} дней:\n\n"
    for i, signal in enumerate(signals, 1):
        date_str = pd.Timestamp(signal['timestamp']).strftime('%d.%m.%Y %H:%M')
        report += f"{i}. {date_str} - Цена: {signal['price']}\n"
    
    return report

def get_statistics():
    """Возвращает статистику по сигналам."""
    signals = get_signals_history()
    
    if not signals:
        return "История сигналов пуста."
    
    total_signals = len(signals)
    
    # Группируем по месяцам
    monthly_stats = {}
    for signal in signals:
        date = datetime.fromisoformat(signal['timestamp'])
        month_key = f"{date.year}-{date.month:02d}"
        
        if month_key not in monthly_stats:
            monthly_stats[month_key] = 0
        monthly_stats[month_key] += 1
    
    # Формируем отчет
    report = f"📊 *Статистика сигналов*\n\n"
    report += f"Всего сигналов: {total_signals}\n\n"
    
    report += "*Распределение по месяцам:*\n"
    for month, count in sorted(monthly_stats.items()):
        year, month = month.split('-')
        month_name = datetime(int(year), int(month), 1).strftime('%B %Y')
        report += f"- {month_name}: {count} сигналов\n"
    
    return report

# --- 7. Новые команды для теста источников и сигнала ---
@app.route('/source_test', methods=['GET'])
def source_test_route():
    results = {}
    for ticker in ['EURUSD=X', 'DX-Y.NYB']:
        for src in ['yfinance', 'FinanceDataReader', 'efinance', 'csv']:
            if src == 'yfinance':
                data, _ = try_yfinance(ticker, period='5d', interval='30m')
            elif src == 'FinanceDataReader':
                data, _ = try_fdr(ticker, start=(datetime.now()-timedelta(days=5)).strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))
            elif src == 'efinance':
                data, _ = try_efinance(ticker, start=(datetime.now()-timedelta(days=5)).strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))
            elif src == 'csv':
                data, _ = try_csv(ticker)
            else:
                data = None
            results[f'{ticker}_{src}'] = 'OK' if data is not None else 'FAIL'
    return json.dumps(results, ensure_ascii=False)

@app.route('/force_signal', methods=['GET'])
def force_signal_route():
    # Принудительно сгенерировать тестовый сигнал (если возможно)
    data = get_live_data()
    if data is None:
        return 'Нет данных для генерации сигнала.'
    result = check_for_signal(data, candle_index=-2, save_to_history=False)
    return result

# --- 5. Веб-сервер и Роуты ---
@app.route('/webhook', methods=['POST'])
def webhook():
    """Обрабатывает команды от пользователей Telegram."""
    update = telegram.Update.de_json(request.get_json(force=True), bot)
    chat_id = update.message.chat.id
    text = update.message.text

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
    elif text == '/test':
        bot.send_message(chat_id, "Запускаю бэктест за последние 60 дней...")
        result = run_backtest(60)
        bot.send_message(chat_id, result)
    elif text == '/stats':
        stats = get_statistics()
        bot.send_message(chat_id, stats, parse_mode='Markdown')
    elif text.startswith('/backtest'):
        try:
            # Формат: /backtest 30 (где 30 - количество дней)
            days = int(text.split()[1]) if len(text.split()) > 1 else 60
            bot.send_message(chat_id, f"Запускаю бэктест за последние {days} дней...")
            result = run_backtest(days)
            bot.send_message(chat_id, result)
        except Exception as e:
            bot.send_message(chat_id, f"Ошибка при запуске бэктеста: {e}")
    elif text == '/source_test':
        results = source_test_route()
        bot.send_message(chat_id, f'Результаты теста источников:\n{results}')
    elif text == '/force_signal':
        result = force_signal_route()
        bot.send_message(chat_id, f'Тестовый сигнал:\n{result}')
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

@app.route('/force_backtest', methods=['GET'])
def force_backtest_route():
    """Принудительно запускает бэктест (для тестирования)."""
    days = request.args.get('days', default=60, type=int)
    result = run_backtest(days)
    return result

@app.route('/')
def index():
    """Стартовая страница для проверки, что сервис жив."""
    return "Telegram Bot is running!"

if __name__ == "__main__":
    # Локальный запуск для отладки. На Render будет использоваться gunicorn.
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 
