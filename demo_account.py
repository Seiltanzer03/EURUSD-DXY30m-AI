import os
import json
import pandas as pd
import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Union
import asyncio
import logging
from supabase import create_client, Client

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('demo_account')

# Константы для демо-счета
INITIAL_BALANCE = 10000  # Начальный баланс в USD
DEFAULT_RISK_PERCENT = 0.01  # Риск на сделку (1% от баланса)

class DemoAccount:
    """
    Класс для управления демо-счетом, который отслеживает сделки и баланс
    с использованием Supabase для хранения данных.
    """
    
    def __init__(self):
        # Инициализация клиента Supabase
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        
        if not url or not key:
            raise ValueError("SUPABASE_URL и SUPABASE_KEY должны быть установлены в переменных окружения")
        
        self.supabase: Client = create_client(url, key)
        
        # Проверяем и создаем необходимые таблицы, если их нет
        self._init_database()
    
    def _init_database(self) -> None:
        """
        Инициализирует базу данных, создавая необходимые таблицы,
        если они еще не существуют.
        """
        try:
            # Проверяем существование таблицы accounts
            response = self.supabase.table('accounts').select('*').limit(1).execute()
            
            # Если таблица не существует, создаем ее
            if 'error' in response:
                logger.info("Создание таблиц в базе данных...")
                
                # SQL для создания таблицы accounts
                self.supabase.table('accounts').insert({
                    'id': 'demo',
                    'balance': INITIAL_BALANCE,
                    'currency': 'USD',
                    'created_at': datetime.datetime.now().isoformat(),
                    'updated_at': datetime.datetime.now().isoformat()
                }).execute()
                
                logger.info("Таблицы успешно созданы")
            else:
                logger.info("Таблицы уже существуют в базе данных")
        
        except Exception as e:
            logger.error(f"Ошибка при инициализации базы данных: {e}")
            raise
    
    async def get_account_info(self) -> Dict:
        """
        Получает информацию о демо-счете.
        
        Returns:
            Dict: Информация о счете (баланс, валюта, и т.д.)
        """
        try:
            response = self.supabase.table('accounts').select('*').eq('id', 'demo').execute()
            
            if not response.data:
                # Если счет не найден, создаем новый
                account_data = {
                    'id': 'demo',
                    'balance': INITIAL_BALANCE,
                    'currency': 'USD',
                    'created_at': datetime.datetime.now().isoformat(),
                    'updated_at': datetime.datetime.now().isoformat()
                }
                self.supabase.table('accounts').insert(account_data).execute()
                return account_data
            
            return response.data[0]
        
        except Exception as e:
            logger.error(f"Ошибка при получении информации о счете: {e}")
            return {
                'id': 'demo',
                'balance': INITIAL_BALANCE,
                'currency': 'USD',
                'error': str(e)
            }
    
    async def get_trade_history(self, limit: int = 20) -> List[Dict]:
        """
        Получает историю сделок.
        
        Args:
            limit: Максимальное количество сделок для возврата
            
        Returns:
            List[Dict]: Список сделок
        """
        try:
            response = self.supabase.table('trades').select('*').order('opened_at', desc=True).limit(limit).execute()
            return response.data
        
        except Exception as e:
            logger.error(f"Ошибка при получении истории сделок: {e}")
            return []
    
    async def get_account_stats(self) -> Dict:
        """
        Рассчитывает и возвращает статистику по счету.
        
        Returns:
            Dict: Статистика счета (прибыль, просадка, винрейт и т.д.)
        """
        try:
            # Получаем все сделки
            response = self.supabase.table('trades').select('*').order('opened_at').execute()
            trades = response.data
            
            if not trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'total_profit': 0,
                    'max_drawdown': 0,
                    'avg_profit': 0,
                    'avg_loss': 0
                }
            
            # Рассчитываем статистику
            total_trades = len(trades)
            winning_trades = [t for t in trades if t.get('profit', 0) > 0]
            losing_trades = [t for t in trades if t.get('profit', 0) <= 0]
            
            win_count = len(winning_trades)
            win_rate = win_count / total_trades if total_trades > 0 else 0
            
            total_profit = sum(t.get('profit', 0) for t in trades)
            total_win = sum(t.get('profit', 0) for t in winning_trades) if winning_trades else 0
            total_loss = abs(sum(t.get('profit', 0) for t in losing_trades)) if losing_trades else 0
            
            profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
            
            # Расчет максимальной просадки
            balance_curve = []
            current_balance = INITIAL_BALANCE
            for trade in trades:
                current_balance += trade.get('profit', 0)
                balance_curve.append(current_balance)
            
            max_balance = INITIAL_BALANCE
            max_drawdown = 0
            for balance in balance_curve:
                max_balance = max(max_balance, balance)
                drawdown = (max_balance - balance) / max_balance if max_balance > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            
            # Средняя прибыль/убыток
            avg_profit = total_win / win_count if win_count > 0 else 0
            avg_loss = total_loss / len(losing_trades) if losing_trades else 0
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_profit': total_profit,
                'max_drawdown': max_drawdown,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss
            }
        
        except Exception as e:
            logger.error(f"Ошибка при расчете статистики: {e}")
            return {
                'error': str(e),
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_profit': 0,
                'max_drawdown': 0,
                'avg_profit': 0,
                'avg_loss': 0
            }
    
    async def open_trade(self, 
                        symbol: str, 
                        direction: str, 
                        entry_price: float, 
                        stop_loss: float, 
                        take_profit: float, 
                        risk_percent: float = DEFAULT_RISK_PERCENT,
                        timeframe: str = "M30") -> Dict:
        """
        Открывает новую сделку на демо-счете.
        
        Args:
            symbol: Торговый инструмент (например, "EURUSD")
            direction: Направление сделки ("BUY" или "SELL")
            entry_price: Цена входа
            stop_loss: Уровень стоп-лосса
            take_profit: Уровень тейк-профита
            risk_percent: Процент риска от баланса (по умолчанию 1%)
            timeframe: Таймфрейм сигнала
            
        Returns:
            Dict: Информация о созданной сделке
        """
        try:
            # Получаем текущий баланс
            account = await self.get_account_info()
            balance = account.get('balance', INITIAL_BALANCE)
            
            # Рассчитываем размер позиции
            risk_amount = balance * risk_percent
            
            # Рассчитываем пипсы риска
            pip_value = 0.0001  # Для EURUSD 1 пипс = 0.0001
            pips_at_risk = abs(entry_price - stop_loss) / pip_value
            
            # Рассчитываем лот
            lot_size = risk_amount / pips_at_risk / 10  # 1 лот = 100,000 единиц базовой валюты
            
            # Создаем новую сделку
            trade_data = {
                'account_id': 'demo',
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'lot_size': lot_size,
                'risk_amount': risk_amount,
                'risk_percent': risk_percent,
                'opened_at': datetime.datetime.now().isoformat(),
                'status': 'OPEN',
                'timeframe': timeframe
            }
            
            response = self.supabase.table('trades').insert(trade_data).execute()
            
            if response.data:
                logger.info(f"Открыта новая сделка: {response.data[0]}")
                return response.data[0]
            else:
                logger.error("Не удалось открыть сделку")
                return {'error': 'Не удалось открыть сделку'}
        
        except Exception as e:
            logger.error(f"Ошибка при открытии сделки: {e}")
            return {'error': str(e)}
    
    async def close_trade(self, trade_id: str, close_price: float) -> Dict:
        """
        Закрывает существующую сделку.
        
        Args:
            trade_id: ID сделки для закрытия
            close_price: Цена закрытия
            
        Returns:
            Dict: Обновленная информация о сделке
        """
        try:
            # Получаем информацию о сделке
            response = self.supabase.table('trades').select('*').eq('id', trade_id).execute()
            
            if not response.data:
                return {'error': 'Сделка не найдена'}
            
            trade = response.data[0]
            
            if trade.get('status') != 'OPEN':
                return {'error': 'Сделка уже закрыта'}
            
            # Рассчитываем прибыль/убыток
            direction = trade.get('direction')
            entry_price = trade.get('entry_price')
            lot_size = trade.get('lot_size')
            
            pip_value = 0.0001  # Для EURUSD
            pip_difference = 0
            
            if direction == 'BUY':
                pip_difference = (close_price - entry_price) / pip_value
            else:  # SELL
                pip_difference = (entry_price - close_price) / pip_value
            
            # Рассчитываем прибыль (1 пипс на 1 лот = $10)
            profit = pip_difference * lot_size * 10
            
            # Округляем прибыль до 2 знаков после запятой
            profit = Decimal(str(profit)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            profit = float(profit)
            
            # Обновляем сделку
            trade_update = {
                'close_price': close_price,
                'closed_at': datetime.datetime.now().isoformat(),
                'status': 'CLOSED',
                'profit': profit,
                'pips': pip_difference
            }
            
            update_response = self.supabase.table('trades').update(trade_update).eq('id', trade_id).execute()
            
            if not update_response.data:
                return {'error': 'Не удалось обновить сделку'}
            
            # Обновляем баланс счета
            account = await self.get_account_info()
            new_balance = account.get('balance', INITIAL_BALANCE) + profit
            
            self.supabase.table('accounts').update({
                'balance': new_balance,
                'updated_at': datetime.datetime.now().isoformat()
            }).eq('id', 'demo').execute()
            
            logger.info(f"Закрыта сделка {trade_id} с прибылью {profit}")
            
            return update_response.data[0]
        
        except Exception as e:
            logger.error(f"Ошибка при закрытии сделки: {e}")
            return {'error': str(e)}
    
    async def check_and_update_open_trades(self, current_price: float) -> List[Dict]:
        """
        Проверяет открытые сделки и закрывает их, если цена достигла
        стоп-лосса или тейк-профита.
        
        Args:
            current_price: Текущая цена инструмента
            
        Returns:
            List[Dict]: Список обновленных сделок
        """
        try:
            # Получаем все открытые сделки
            response = self.supabase.table('trades').select('*').eq('status', 'OPEN').execute()
            
            if not response.data:
                return []
            
            updated_trades = []
            
            for trade in response.data:
                trade_id = trade.get('id')
                direction = trade.get('direction')
                stop_loss = trade.get('stop_loss')
                take_profit = trade.get('take_profit')
                
                # Проверяем, достигнут ли стоп-лосс или тейк-профит
                if direction == 'BUY':
                    if current_price <= stop_loss:
                        # Стоп-лосс достигнут
                        updated_trade = await self.close_trade(trade_id, stop_loss)
                        updated_trades.append(updated_trade)
                    elif current_price >= take_profit:
                        # Тейк-профит достигнут
                        updated_trade = await self.close_trade(trade_id, take_profit)
                        updated_trades.append(updated_trade)
                else:  # SELL
                    if current_price >= stop_loss:
                        # Стоп-лосс достигнут
                        updated_trade = await self.close_trade(trade_id, stop_loss)
                        updated_trades.append(updated_trade)
                    elif current_price <= take_profit:
                        # Тейк-профит достигнут
                        updated_trade = await self.close_trade(trade_id, take_profit)
                        updated_trades.append(updated_trade)
            
            return updated_trades
        
        except Exception as e:
            logger.error(f"Ошибка при проверке открытых сделок: {e}")
            return []
    
    async def reset_account(self) -> Dict:
        """
        Сбрасывает демо-счет к начальным значениям.
        
        Returns:
            Dict: Информация о сброшенном счете
        """
        try:
            # Удаляем все сделки
            self.supabase.table('trades').delete().neq('id', '0').execute()
            
            # Сбрасываем баланс
            response = self.supabase.table('accounts').update({
                'balance': INITIAL_BALANCE,
                'updated_at': datetime.datetime.now().isoformat()
            }).eq('id', 'demo').execute()
            
            logger.info("Демо-счет сброшен к начальным значениям")
            
            if response.data:
                return response.data[0]
            else:
                return {'error': 'Не удалось сбросить счет'}
        
        except Exception as e:
            logger.error(f"Ошибка при сбросе демо-счета: {e}")
            return {'error': str(e)}
    
    async def get_balance_history(self) -> List[Dict]:
        """
        Получает историю изменения баланса для построения графика.
        
        Returns:
            List[Dict]: История баланса по дням
        """
        try:
            # Получаем все сделки
            response = self.supabase.table('trades').select('*').order('closed_at').execute()
            trades = [t for t in response.data if t.get('status') == 'CLOSED' and t.get('closed_at')]
            
            if not trades:
                return [{'date': datetime.datetime.now().isoformat(), 'balance': INITIAL_BALANCE}]
            
            # Строим историю баланса
            balance_history = []
            current_balance = INITIAL_BALANCE
            
            # Группируем сделки по дням
            trades_by_day = {}
            for trade in trades:
                closed_at = trade.get('closed_at')
                if not closed_at:
                    continue
                
                date = closed_at.split('T')[0]  # Получаем только дату
                
                if date not in trades_by_day:
                    trades_by_day[date] = []
                
                trades_by_day[date].append(trade)
            
            # Сортируем дни
            sorted_days = sorted(trades_by_day.keys())
            
            # Добавляем начальный баланс
            if sorted_days:
                first_day = sorted_days[0]
                balance_history.append({
                    'date': first_day,
                    'balance': INITIAL_BALANCE
                })
            
            # Добавляем баланс для каждого дня
            for day in sorted_days:
                day_trades = trades_by_day[day]
                day_profit = sum(t.get('profit', 0) for t in day_trades)
                current_balance += day_profit
                
                balance_history.append({
                    'date': day,
                    'balance': current_balance
                })
            
            return balance_history
        
        except Exception as e:
            logger.error(f"Ошибка при получении истории баланса: {e}")
            return [{'date': datetime.datetime.now().isoformat(), 'balance': INITIAL_BALANCE, 'error': str(e)}]

# Создаем экземпляр класса для использования в других модулях
demo_account = DemoAccount()

# Функция для обработки сигнала и создания сделки
async def process_signal(symbol: str, direction: str, entry_price: float, 
                         stop_loss: float, take_profit: float, 
                         risk_percent: float = DEFAULT_RISK_PERCENT,
                         timeframe: str = "M30") -> Dict:
    """
    Обрабатывает торговый сигнал и открывает сделку на демо-счете.
    
    Args:
        symbol: Торговый инструмент
        direction: Направление сделки ("BUY" или "SELL")
        entry_price: Цена входа
        stop_loss: Уровень стоп-лосса
        take_profit: Уровень тейк-профита
        risk_percent: Процент риска от баланса
        timeframe: Таймфрейм сигнала
        
    Returns:
        Dict: Информация о созданной сделке
    """
    try:
        trade = await demo_account.open_trade(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_percent=risk_percent,
            timeframe=timeframe
        )
        
        logger.info(f"Сигнал обработан, создана сделка: {trade}")
        return trade
    
    except Exception as e:
        logger.error(f"Ошибка при обработке сигнала: {e}")
        return {'error': str(e)}

# Функция для форматирования статистики счета в текстовый формат для Telegram
def format_account_stats(stats: Dict) -> str:
    """
    Форматирует статистику счета для отображения в Telegram.
    
    Args:
        stats: Словарь со статистикой счета
        
    Returns:
        str: Отформатированная статистика
    """
    if 'error' in stats:
        return f"❌ Ошибка при получении статистики: {stats['error']}"
    
    return (
        f"📊 *Статистика демо-счета*\n\n"
        f"Всего сделок: {stats['total_trades']}\n"
        f"Винрейт: {stats['win_rate']:.2%}\n"
        f"Профит-фактор: {stats['profit_factor']:.2f}\n"
        f"Общая прибыль: ${stats['total_profit']:.2f}\n"
        f"Макс. просадка: {stats['max_drawdown']:.2%}\n"
        f"Средняя прибыль: ${stats['avg_profit']:.2f}\n"
        f"Средний убыток: ${stats['avg_loss']:.2f}\n"
    )

# Функция для форматирования информации о счете в текстовый формат для Telegram
def format_account_info(account: Dict) -> str:
    """
    Форматирует информацию о счете для отображения в Telegram.
    
    Args:
        account: Словарь с информацией о счете
        
    Returns:
        str: Отформатированная информация
    """
    if 'error' in account:
        return f"❌ Ошибка при получении информации о счете: {account['error']}"
    
    return (
        f"💰 *Информация о демо-счете*\n\n"
        f"Баланс: ${account['balance']:.2f}\n"
        f"Валюта: {account['currency']}\n"
        f"Создан: {account['created_at'].split('T')[0]}\n"
        f"Обновлен: {account['updated_at'].split('T')[0]}\n"
    )

# Функция для форматирования истории сделок в текстовый формат для Telegram
def format_trade_history(trades: List[Dict]) -> str:
    """
    Форматирует историю сделок для отображения в Telegram.
    
    Args:
        trades: Список сделок
        
    Returns:
        str: Отформатированная история сделок
    """
    if not trades:
        return "📝 *История сделок*\n\nНет сделок для отображения."
    
    result = "📝 *Последние сделки*\n\n"
    
    for trade in trades[:10]:  # Ограничиваем до 10 последних сделок
        status_emoji = "🟢" if trade.get('profit', 0) > 0 else "🔴"
        
        result += (
            f"{status_emoji} *{trade['symbol']}* ({trade['timeframe']})\n"
            f"Направление: {trade['direction']}\n"
            f"Статус: {trade['status']}\n"
        )
        
        if trade['status'] == 'CLOSED':
            result += (
                f"Прибыль: ${trade.get('profit', 0):.2f}\n"
                f"Пипсы: {trade.get('pips', 0):.1f}\n"
                f"Закрыт: {trade['closed_at'].split('T')[0]}\n\n"
            )
        else:
            result += (
                f"Вход: {trade['entry_price']:.5f}\n"
                f"SL: {trade['stop_loss']:.5f}\n"
                f"TP: {trade['take_profit']:.5f}\n"
                f"Открыт: {trade['opened_at'].split('T')[0]}\n\n"
            )
    
    return result

# Если этот файл запущен напрямую
if __name__ == "__main__":
    # Тестовый код для проверки функциональности
    async def test():
        try:
            # Инициализируем демо-счет
            account_info = await demo_account.get_account_info()
            print("Информация о счете:", account_info)
            
            # Открываем тестовую сделку
            trade = await demo_account.open_trade(
                symbol="EURUSD",
                direction="SELL",
                entry_price=1.0800,
                stop_loss=1.0850,
                take_profit=1.0700,
                risk_percent=0.01,
                timeframe="M30"
            )
            print("Открыта сделка:", trade)
            
            # Получаем статистику
            stats = await demo_account.get_account_stats()
            print("Статистика:", stats)
            
            # Закрываем сделку с прибылью
            if trade and 'id' in trade:
                closed_trade = await demo_account.close_trade(trade['id'], 1.0750)
                print("Закрыта сделка:", closed_trade)
            
            # Получаем историю баланса
            balance_history = await demo_account.get_balance_history()
            print("История баланса:", balance_history)
            
        except Exception as e:
            print(f"Ошибка при тестировании: {e}")
    
    # Запускаем тестовый код
    asyncio.run(test())
