import os
import asyncio
import logging
from typing import Dict, Optional, Tuple
import yfinance as yf
import pandas as pd

# Импортируем модули нашего проекта
from signal_core import generate_signal_and_plot, generate_signal_and_plot_30m
from demo_account import process_signal, demo_account

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('integration')

async def check_signals_and_process():
    """
    Проверяет наличие сигналов и обрабатывает их для демо-счета.
    """
    try:
        logger.info("Проверка сигналов...")
        
        # Получаем сигналы для обоих таймфреймов
        signal_5m, entry_5m, sl_5m, tp_5m, last_5m, image_path_5m, timeframe_5m = generate_signal_and_plot()
        signal_30m, entry_30m, sl_30m, tp_30m, last_30m, image_path_30m, timeframe_30m = generate_signal_and_plot_30m()
        
        # Обрабатываем сигнал M5
        if signal_5m:
            logger.info(f"Получен сигнал M5: SELL EURUSD @ {entry_5m:.5f}")
            
            # Создаем сделку на демо-счете
            trade = await process_signal(
                symbol="EURUSD",
                direction="SELL",
                entry_price=entry_5m,
                stop_loss=sl_5m,
                take_profit=tp_5m,
                risk_percent=0.01,  # 1% риска на сделку
                timeframe="M5"
            )
            
            if 'error' in trade:
                logger.error(f"Ошибка при обработке сигнала M5: {trade['error']}")
            else:
                logger.info(f"Сигнал M5 успешно обработан, создана сделка: {trade['id']}")
        
        # Обрабатываем сигнал M30
        if signal_30m:
            logger.info(f"Получен сигнал M30: SELL EURUSD @ {entry_30m:.5f}")
            
            # Создаем сделку на демо-счете
            trade = await process_signal(
                symbol="EURUSD",
                direction="SELL",
                entry_price=entry_30m,
                stop_loss=sl_30m,
                take_profit=tp_30m,
                risk_percent=0.01,  # 1% риска на сделку
                timeframe="M30"
            )
            
            if 'error' in trade:
                logger.error(f"Ошибка при обработке сигнала M30: {trade['error']}")
            else:
                logger.info(f"Сигнал M30 успешно обработан, создана сделка: {trade['id']}")
        
        # Если нет сигналов, проверяем открытые сделки
        if not signal_5m and not signal_30m:
            logger.info("Нет новых сигналов. Проверка открытых сделок...")
            
            # Получаем текущую цену
            current_price = await get_current_price()
            
            if current_price is not None:
                # Проверяем открытые сделки на достижение TP/SL
                updated_trades = await demo_account.check_and_update_open_trades(current_price)
                
                if updated_trades:
                    for trade in updated_trades:
                        if 'error' not in trade:
                            logger.info(f"Сделка {trade['id']} закрыта с прибылью: ${trade.get('profit', 0):.2f}")
                        else:
                            logger.error(f"Ошибка при обновлении сделки: {trade['error']}")
            else:
                logger.warning("Не удалось получить текущую цену для проверки открытых сделок")
        
    except Exception as e:
        logger.error(f"Ошибка при проверке сигналов: {e}")

async def get_current_price() -> Optional[float]:
    """
    Получает текущую цену EURUSD.
    
    Returns:
        Optional[float]: Текущая цена или None в случае ошибки
    """
    try:
        data = yf.download(tickers='EURUSD=X', period='1d', interval='1m')
        if data.empty:
            return None
        return data['Close'].iloc[-1]
    except Exception as e:
        logger.error(f"Ошибка при получении текущей цены: {e}")
        return None

async def run_periodic_check(interval_seconds: int = 300):
    """
    Запускает периодическую проверку сигналов.
    
    Args:
        interval_seconds: Интервал между проверками в секундах (по умолчанию 5 минут)
    """
    logger.info(f"Запуск периодической проверки сигналов с интервалом {interval_seconds} секунд")
    
    while True:
        await check_signals_and_process()
        await asyncio.sleep(interval_seconds)

if __name__ == "__main__":
    # Проверяем наличие необходимых переменных окружения
    if not os.environ.get("SUPABASE_URL") or not os.environ.get("SUPABASE_KEY"):
        logger.error("Не установлены переменные окружения SUPABASE_URL и SUPABASE_KEY")
        exit(1)
    
    # Запускаем периодическую проверку сигналов
    asyncio.run(run_periodic_check()) 