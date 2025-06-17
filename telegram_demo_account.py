import os
import asyncio
import logging
from typing import Dict, List, Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
import matplotlib.pyplot as plt
import pandas as pd
import io
from datetime import datetime
import threading
import time
import signal
import sys

# Импортируем модули нашего проекта
from demo_account import demo_account, format_account_info, format_account_stats, format_trade_history
from integration import check_signals_and_process, get_current_price

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('demo_account_bot')

# Получаем токен из переменных окружения
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN не установлен в переменных окружения")
    raise ValueError("TELEGRAM_BOT_TOKEN не установлен в переменных окружения")

# URL для веб-приложения (если используется внешний хостинг)
WEBAPP_URL = os.environ.get('WEBAPP_URL', 'https://trading-bot-i36i.onrender.com')

# Префикс для команд демо-счета, чтобы избежать конфликтов с существующими командами
CMD_PREFIX = "demo_"

# Команды бота
async def demo_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет приветственное сообщение и показывает доступные команды."""
    user = update.effective_user
    
    # Создаем кнопку для веб-приложения
    keyboard = [
        [InlineKeyboardButton("📊 Открыть веб-интерфейс демо-счета", web_app=WebAppInfo(url=f"{WEBAPP_URL}/"))]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"👋 Привет, {user.first_name}! Я бот для управления демо-счетом.\n\n"
        "Доступные команды:\n"
        f"/{CMD_PREFIX}balance - Показать текущий баланс\n"
        f"/{CMD_PREFIX}stats - Показать статистику счета\n"
        f"/{CMD_PREFIX}trades - Показать историю сделок\n"
        f"/{CMD_PREFIX}chart - Показать график баланса\n"
        f"/{CMD_PREFIX}check - Проверить текущие сигналы\n"
        f"/{CMD_PREFIX}price - Получить текущую цену EURUSD\n"
        f"/{CMD_PREFIX}reset - Сбросить демо-счет\n"
        f"/{CMD_PREFIX}help - Показать эту справку\n\n"
        "Также вы можете использовать веб-интерфейс для более удобного просмотра:",
        reply_markup=reply_markup
    )

async def demo_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет справку по командам."""
    await update.message.reply_text(
        "📚 *Справка по командам демо-счета*\n\n"
        f"/{CMD_PREFIX}balance - Показать текущий баланс и информацию о счете\n"
        f"/{CMD_PREFIX}stats - Показать статистику счета (винрейт, профит, просадка и т.д.)\n"
        f"/{CMD_PREFIX}trades - Показать историю последних сделок\n"
        f"/{CMD_PREFIX}chart - Показать график изменения баланса\n"
        f"/{CMD_PREFIX}check - Проверить текущие торговые сигналы\n"
        f"/{CMD_PREFIX}price - Получить текущую цену EURUSD\n"
        f"/{CMD_PREFIX}reset - Сбросить демо-счет к начальному состоянию\n"
        f"/{CMD_PREFIX}help - Показать эту справку",
        parse_mode="Markdown"
    )

async def demo_balance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показывает текущий баланс и информацию о счете."""
    await update.message.reply_text("⏳ Получение информации о демо-счете...")
    
    try:
        account_info = await demo_account.get_account_info()
        formatted_info = format_account_info(account_info)
        
        await update.message.reply_text(
            formatted_info,
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Ошибка при получении информации о счете: {e}")
        await update.message.reply_text(f"❌ Произошла ошибка: {str(e)}")

async def demo_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показывает статистику счета."""
    await update.message.reply_text("⏳ Расчет статистики демо-счета...")
    
    try:
        stats = await demo_account.get_account_stats()
        formatted_stats = format_account_stats(stats)
        
        await update.message.reply_text(
            formatted_stats,
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Ошибка при получении статистики счета: {e}")
        await update.message.reply_text(f"❌ Произошла ошибка: {str(e)}")

async def demo_trades(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показывает историю сделок."""
    await update.message.reply_text("⏳ Получение истории сделок демо-счета...")
    
    try:
        trades = await demo_account.get_trade_history(limit=10)
        formatted_trades = format_trade_history(trades)
        
        await update.message.reply_text(
            formatted_trades,
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Ошибка при получении истории сделок: {e}")
        await update.message.reply_text(f"❌ Произошла ошибка: {str(e)}")

async def demo_chart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показывает график изменения баланса."""
    await update.message.reply_text("⏳ Создание графика баланса демо-счета...")
    
    try:
        # Получаем историю баланса
        balance_history = await demo_account.get_balance_history()
        
        if not balance_history:
            await update.message.reply_text("❌ Нет данных для построения графика.")
            return
        
        # Создаем DataFrame из истории баланса
        df = pd.DataFrame(balance_history)
        df['date'] = pd.to_datetime(df['date'])
        
        # Создаем график
        plt.figure(figsize=(10, 6))
        plt.plot(df['date'], df['balance'], marker='o', linestyle='-', color='blue')
        plt.title('История баланса демо-счета')
        plt.xlabel('Дата')
        plt.ylabel('Баланс ($)')
        plt.grid(True)
        plt.tight_layout()
        
        # Сохраняем график в буфер
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # Отправляем график пользователю
        await update.message.reply_photo(
            photo=buf,
            caption=f"📈 График баланса демо-счета\nТекущий баланс: ${df['balance'].iloc[-1]:.2f}"
        )
    except Exception as e:
        logger.error(f"Ошибка при создании графика баланса: {e}")
        await update.message.reply_text(f"❌ Произошла ошибка: {str(e)}")

async def demo_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Сбрасывает демо-счет к начальному состоянию."""
    # Создаем клавиатуру с кнопками подтверждения
    keyboard = [
        [
            InlineKeyboardButton("Да, сбросить", callback_data="demo_reset_confirm"),
            InlineKeyboardButton("Отмена", callback_data="demo_reset_cancel")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "⚠️ Вы уверены, что хотите сбросить демо-счет?\n"
        "Это действие удалит все сделки и вернет баланс к начальному значению.",
        reply_markup=reply_markup
    )

async def demo_check(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Проверяет текущие торговые сигналы."""
    await update.message.reply_text("⏳ Проверка торговых сигналов для демо-счета...")
    
    try:
        # Проверяем сигналы
        await check_signals_and_process()
        
        # Получаем открытые сделки
        trades = await demo_account.get_trade_history(limit=5)
        open_trades = [t for t in trades if t.get('status') == 'OPEN']
        
        if open_trades:
            message = "✅ Обнаружены новые сигналы! Открыты следующие сделки на демо-счете:\n\n"
            for trade in open_trades:
                message += (
                    f"🔹 *{trade['symbol']}* ({trade['timeframe']})\n"
                    f"Направление: {trade['direction']}\n"
                    f"Вход: {trade['entry_price']:.5f}\n"
                    f"SL: {trade['stop_loss']:.5f}\n"
                    f"TP: {trade['take_profit']:.5f}\n"
                    f"Размер лота: {trade['lot_size']:.2f}\n"
                    f"Открыта: {trade['opened_at'].split('T')[0]}\n\n"
                )
            
            await update.message.reply_text(message, parse_mode="Markdown")
        else:
            await update.message.reply_text("ℹ️ Нет активных сигналов на демо-счете на данный момент.")
    except Exception as e:
        logger.error(f"Ошибка при проверке сигналов: {e}")
        await update.message.reply_text(f"❌ Произошла ошибка при проверке сигналов: {str(e)}")

async def demo_price(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показывает текущую цену EURUSD."""
    await update.message.reply_text("⏳ Получение текущей цены EURUSD для демо-счета...")
    
    try:
        # Получаем текущую цену
        current_price = await get_current_price()
        
        if current_price is not None:
            await update.message.reply_text(
                f"💰 *Текущая цена EURUSD*: `{current_price:.5f}`",
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text("❌ Не удалось получить текущую цену.")
    except Exception as e:
        logger.error(f"Ошибка при получении текущей цены: {e}")
        await update.message.reply_text(f"❌ Произошла ошибка: {str(e)}")

async def demo_button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает нажатия на кнопки."""
    query = update.callback_query
    await query.answer()
    
    if query.data == "demo_reset_confirm":
        try:
            await query.edit_message_text("⏳ Сброс демо-счета...")
            result = await demo_account.reset_account()
            
            if 'error' in result:
                await query.edit_message_text(f"❌ Ошибка при сбросе демо-счета: {result['error']}")
            else:
                await query.edit_message_text(
                    "✅ Демо-счет успешно сброшен!\n"
                    f"Новый баланс: ${result['balance']:.2f}"
                )
        except Exception as e:
            logger.error(f"Ошибка при сбросе демо-счета: {e}")
            await query.edit_message_text(f"❌ Произошла ошибка: {str(e)}")
    
    elif query.data == "demo_reset_cancel":
        await query.edit_message_text("❌ Сброс демо-счета отменен.")

# Функция для периодической проверки сигналов в фоновом режиме
async def periodic_signal_check(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Периодически проверяет сигналы и обновляет открытые сделки."""
    try:
        logger.info("Выполняется периодическая проверка сигналов для демо-счета...")
        await check_signals_and_process()
        
        # Получаем открытые сделки
        trades = await demo_account.get_trade_history(limit=5)
        open_trades = [t for t in trades if t.get('status') == 'OPEN']
        
        # Если есть новые сделки, отправляем уведомления подписчикам
        # Эту часть можно реализовать по необходимости
        
    except Exception as e:
        logger.error(f"Ошибка при периодической проверке сигналов для демо-счета: {e}")

# Функция для добавления обработчиков команд в существующее приложение
def add_demo_account_handlers(application: Application) -> None:
    """Добавляет обработчики команд демо-счета в существующее приложение."""
    # Добавляем обработчики команд
    application.add_handler(CommandHandler(f"{CMD_PREFIX}start", demo_start))
    application.add_handler(CommandHandler(f"{CMD_PREFIX}help", demo_help))
    application.add_handler(CommandHandler(f"{CMD_PREFIX}balance", demo_balance))
    application.add_handler(CommandHandler(f"{CMD_PREFIX}stats", demo_stats))
    application.add_handler(CommandHandler(f"{CMD_PREFIX}trades", demo_trades))
    application.add_handler(CommandHandler(f"{CMD_PREFIX}chart", demo_chart))
    application.add_handler(CommandHandler(f"{CMD_PREFIX}reset", demo_reset))
    application.add_handler(CommandHandler(f"{CMD_PREFIX}check", demo_check))
    application.add_handler(CommandHandler(f"{CMD_PREFIX}price", demo_price))
    
    # Добавляем обработчик кнопок для демо-счета
    application.add_handler(CallbackQueryHandler(demo_button_callback, pattern=r"^demo_"))
    
    # Запускаем периодическую проверку сигналов в фоновом режиме
    application.job_queue.run_repeating(
        periodic_signal_check,
        interval=300,  # Каждые 5 минут
        first=10  # Первый запуск через 10 секунд после старта
    )
    
    logger.info("Обработчики команд демо-счета добавлены в приложение")

# Обработчик для обработки сигналов остановки (Ctrl+C)
def signal_handler(sig, frame):
    logger.info("Получен сигнал остановки, завершение работы...")
    sys.exit(0)

def main() -> None:
    """Запускает бота как отдельное приложение (только если запущен напрямую)."""
    # Регистрируем обработчик сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Создаем приложение
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Добавляем обработчики команд демо-счета
    add_demo_account_handlers(application)
    
    # Запускаем бота
    logger.info("Бот демо-счета запущен и готов к работе")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
