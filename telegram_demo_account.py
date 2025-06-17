import os
import asyncio
import logging
from typing import Dict, List, Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
import matplotlib.pyplot as plt
import pandas as pd
import io
from datetime import datetime

# Импортируем модули нашего проекта
from demo_account import demo_account, format_account_info, format_account_stats, format_trade_history

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('telegram_demo_account')

# Получаем токен из переменных окружения
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN не установлен в переменных окружения")
    raise ValueError("TELEGRAM_BOT_TOKEN не установлен в переменных окружения")

# Команды бота
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет приветственное сообщение и показывает доступные команды."""
    user = update.effective_user
    await update.message.reply_text(
        f"👋 Привет, {user.first_name}! Я бот для управления демо-счетом.\n\n"
        "Доступные команды:\n"
        "/balance - Показать текущий баланс\n"
        "/stats - Показать статистику счета\n"
        "/trades - Показать историю сделок\n"
        "/chart - Показать график баланса\n"
        "/reset - Сбросить демо-счет\n"
        "/help - Показать эту справку"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет справку по командам."""
    await update.message.reply_text(
        "📚 *Справка по командам*\n\n"
        "/balance - Показать текущий баланс и информацию о счете\n"
        "/stats - Показать статистику счета (винрейт, профит, просадка и т.д.)\n"
        "/trades - Показать историю последних сделок\n"
        "/chart - Показать график изменения баланса\n"
        "/reset - Сбросить демо-счет к начальному состоянию\n"
        "/help - Показать эту справку",
        parse_mode="Markdown"
    )

async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показывает текущий баланс и информацию о счете."""
    await update.message.reply_text("⏳ Получение информации о счете...")
    
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

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показывает статистику счета."""
    await update.message.reply_text("⏳ Расчет статистики счета...")
    
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

async def trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показывает историю сделок."""
    await update.message.reply_text("⏳ Получение истории сделок...")
    
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

async def chart_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показывает график изменения баланса."""
    await update.message.reply_text("⏳ Создание графика баланса...")
    
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

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Сбрасывает демо-счет к начальному состоянию."""
    # Создаем клавиатуру с кнопками подтверждения
    keyboard = [
        [
            InlineKeyboardButton("Да, сбросить", callback_data="reset_confirm"),
            InlineKeyboardButton("Отмена", callback_data="reset_cancel")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "⚠️ Вы уверены, что хотите сбросить демо-счет?\n"
        "Это действие удалит все сделки и вернет баланс к начальному значению.",
        reply_markup=reply_markup
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает нажатия на кнопки."""
    query = update.callback_query
    await query.answer()
    
    if query.data == "reset_confirm":
        try:
            await query.edit_message_text("⏳ Сброс демо-счета...")
            result = await demo_account.reset_account()
            
            if 'error' in result:
                await query.edit_message_text(f"❌ Ошибка при сбросе счета: {result['error']}")
            else:
                await query.edit_message_text(
                    "✅ Демо-счет успешно сброшен!\n"
                    f"Новый баланс: ${result['balance']:.2f}"
                )
        except Exception as e:
            logger.error(f"Ошибка при сбросе демо-счета: {e}")
            await query.edit_message_text(f"❌ Произошла ошибка: {str(e)}")
    
    elif query.data == "reset_cancel":
        await query.edit_message_text("❌ Сброс демо-счета отменен.")

def main() -> None:
    """Запускает бота."""
    # Создаем приложение
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Добавляем обработчики команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("balance", balance_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("trades", trades_command))
    application.add_handler(CommandHandler("chart", chart_command))
    application.add_handler(CommandHandler("reset", reset_command))
    
    # Добавляем обработчик кнопок
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Запускаем бота
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
