import pandas as pd
import pandas_ta as ta
import joblib
import yfinance as yf
import schedule
import time
import telegram
import warnings
import asyncio
import os # <-- Добавляем импорт для работы с окружением

# --- 1. Конфигурация ---
# ВАЖНО: Секреты теперь читаются из окружения, а не из кода
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# Остальные настройки
MODEL_FILE = 'ml_model_final_fix.joblib'
# ML_MODEL = joblib.load(MODEL_FILE) # Убираем загрузку модели отсюда
PREDICTION_THRESHOLD = 0.55
LOOKBACK_PERIOD = 20

# Подавляем лишние предупреждения
warnings.filterwarnings("ignore", category=FutureWarning)


# --- 2. Функции ---

async def send_telegram_message(bot, chat_id, text):
    """Асинхронно отправляет сообщение в Telegram."""
    try:
        bot.send_message(chat_id=chat_id, text=text, parse_mode='Markdown')
        print(f"Сообщение успешно отправлено в Telegram: {text}")
    except Exception as e:
        print(f"Ошибка при отправке сообщения в Telegram: {e}")

def get_live_data():
    """Загружает последние 30-минутные данные для EUR/USD и DXY."""
    print("\nЗагрузка свежих данных...")
    try:
        eurusd_data = yf.download(tickers='EURUSD=X', period='5d', interval='30m')
        dxy_data = yf.download(tickers='DX-Y.NYB', period='5d', interval='30m')

        if eurusd_data.empty or dxy_data.empty:
            print("Ошибка: Не удалось загрузить данные. Проверьте тикеры или интернет-соединение.")
            return None

        # --- Надежная обработка индекса ---
        eurusd_data.reset_index(inplace=True)
        dxy_data.reset_index(inplace=True)
        
        # Находим колонку с датой (может называться 'Datetime', 'Date' или 'index')
        date_col_eur = next((col for col in ['Datetime', 'Date', 'index'] if col in eurusd_data.columns), None)
        date_col_dxy = next((col for col in ['Datetime', 'Date', 'index'] if col in dxy_data.columns), None)

        if not date_col_eur or not date_col_dxy:
            print("Критическая ошибка: не найдена колонка с датой в данных yfinance.")
            return None

        # Переименовываем колонку в 'Datetime' для унификации
        eurusd_data.rename(columns={date_col_eur: 'Datetime'}, inplace=True)
        dxy_data.rename(columns={date_col_dxy: 'Datetime'}, inplace=True)
        
        # Индикаторы рассчитываются, пока 'Datetime' является обычной колонкой
        eurusd_data.ta.rsi(length=14, append=True)
        eurusd_data.ta.macd(fast=12, slow=26, signal=9, append=True)
        eurusd_data.ta.atr(length=14, append=True)
        eurusd_data.rename(columns={'RSI_14':'RSI', 'MACD_12_26_9':'MACD', 'MACDh_12_26_9':'MACD_hist', 'MACDs_12_26_9':'MACD_signal', 'ATRr_14':'ATR'}, inplace=True)
        
        # Возвращаем 'Datetime' в качестве индекса для дальнейшей работы
        eurusd_data.set_index('Datetime', inplace=True)
        dxy_data.set_index('Datetime', inplace=True)
        
        dxy_data_renamed = dxy_data.rename(columns={'Low': 'DXY_Low'})
        
        data = pd.concat([eurusd_data, dxy_data_renamed['DXY_Low']], axis=1)
        data.dropna(inplace=True)
        
        print("Данные успешно загружены и обработаны.")
        return data
    except Exception as e:
        print(f"Критическая ошибка при загрузке данных: {e}")
        return None

async def check_for_signal(bot):
    """Основная функция проверки сигнала."""
    # Загружаем модель здесь, только когда она нужна
    try:
        ml_model = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Файл модели {MODEL_FILE} не найден! !!!")
        await send_telegram_message(bot, TELEGRAM_CHAT_ID, f"❌ *ОШИБКА: Файл модели `{MODEL_FILE}` не найден!* Бот не может работать.")
        return

    print("--- Запуск проверки сигнала ---")
    data = get_live_data()
    if data is None:
        return

    # Проверяем, не "старые" ли данные (признак выходного дня)
    last_candle_time = data.index[-1].tz_convert('UTC')
    time_now = pd.Timestamp.now(tz='UTC')
    time_diff = time_now - last_candle_time
    
    if time_diff.total_seconds() > 3600 * 4: # Если данные старше 4 часов
        print(f"Данные слишком старые (последняя свеча: {last_candle_time}). Вероятно, рынок закрыт. Пропускаю проверку.")
        return

    # Берем предпоследнюю ЗАВЕРШЕННУЮ свечу для анализа
    last_candle = data.iloc[-2]

    # Проверка торгового времени (UTC)
    current_hour = last_candle.name.hour
    if not (13 <= current_hour <= 17):
        print(f"Вне торгового времени (текущий час UTC: {current_hour}). Проверка отменена.")
        return
    
    print("Время торговое, начинаю анализ сетапа...")
    # Анализ сетапа
    start_index = len(data) - LOOKBACK_PERIOD - 2
    end_index = len(data) - 2
    
    recent_eurusd_high = data['High'].iloc[start_index:end_index].max()
    eurusd_judas_swing = last_candle['High'] > recent_eurusd_high

    recent_dxy_low = data['DXY_Low'].iloc[start_index:end_index].min()
    dxy_raid = last_candle['DXY_Low'] < recent_dxy_low

    if eurusd_judas_swing and dxy_raid:
        print("!!! НАЙДЕН СЕТАП !!! Проверяю с помощью ИИ...")
        
        features = [
            last_candle['RSI'],
            last_candle['MACD'],
            last_candle['MACD_hist'],
            last_candle['MACD_signal'],
            last_candle['ATR']
        ]
        
        win_probability = ml_model.predict_proba([features])[0][1]
        
        print(f"Вероятность успеха по мнению ИИ: {win_probability:.2%}")
        
        if win_probability >= PREDICTION_THRESHOLD:
            message = (
                f"🚨 *СИГНАЛ НА ПРОДАЖУ (SELL) EUR/USD* 🚨\n\n"
                f"Вероятность успеха: *{win_probability:.2%}*\n"
                f"Время сетапа (UTC): `{last_candle.name}`\n\n"
                f"Подготовьтесь к входу по рынку на открытии следующей свечи.\n\n"
                f"*Данные для проверки:*\n"
                f"- RSI: `{features[0]:.2f}`\n"
                f"- MACD: `{features[1]:.2f}`\n"
                f"- ATR: `{features[4]:.4f}`"
            )
            await send_telegram_message(bot, TELEGRAM_CHAT_ID, message)
        else:
            print("ИИ отфильтровал сигнал. Вероятность слишком низкая.")
    else:
        print("Сетап не найден. Жду следующей свечи.")


async def main():
    """Главная асинхронная функция, которая запускает планировщик."""
    
    # Проверка наличия секретов при старте
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("!!! КРИТИЧЕСКАЯ ОШИБКА: Не найдены переменные окружения TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID.")
        print("!!! Убедитесь, что вы добавили их в настройках хостинга (например, в Render).")
        return

    bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
    
    print("Бот запущен. Ожидаю запланированного времени для проверки...")
    await send_telegram_message(bot, TELEGRAM_CHAT_ID, "✅ *Торговый бот-сигнальщик запущен* ✅\n\nОжидаю торговую сессию (13:00 - 17:00 UTC)...")

    # --- Планировщик ---
    # Мы будем проверять сигнал на 2-й и 32-й минуте каждого часа,
    # чтобы точно успеть получить данные по закрытой 00-й или 30-й свече.
    schedule.every().hour.at(":02").do(lambda: asyncio.create_task(check_for_signal(bot)))
    schedule.every().hour.at(":32").do(lambda: asyncio.create_task(check_for_signal(bot)))

    while True:
        schedule.run_pending()
        await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nБот остановлен вручную.")
    except Exception as e:
        print(f"Произошла критическая ошибка: {e}")
        # Попытка отправить сообщение об ошибке в Telegram
        async def notify_shutdown():
            bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
            await send_telegram_message(bot, TELEGRAM_CHAT_ID, f"❌ *Критическая ошибка, бот остановлен!*\n\n`{e}`")
        asyncio.run(notify_shutdown()) 