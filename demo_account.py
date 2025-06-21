import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import time
import uuid
import logging
from flask import render_template_string, Response
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from threading import Lock
import joblib

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('demo_account')

# Константы
INITIAL_BALANCE = 100000  # Начальный баланс демо-счета в USD
DEMO_ACCOUNT_FILE = 'demo_account.json'  # Файл для хранения состояния счета
RISK_PER_TRADE = 0.01  # Риск 1% на сделку
ACCOUNT_LOCK = Lock()  # Для синхронизации доступа к счету

# Структура демо-счета
DEMO_ACCOUNT_TEMPLATE = {
    'balance': INITIAL_BALANCE,
    'equity': INITIAL_BALANCE,
    'trades': [],  # список сделок
    'created_at': None,
    'last_updated': None,
}

def load_demo_account():
    """Загружает состояние демо-счета из файла или создает новый."""
    if os.path.exists(DEMO_ACCOUNT_FILE):
        try:
            with open(DEMO_ACCOUNT_FILE, 'r') as f:
                account = json.load(f)
                logger.info("Демо-счет загружен: баланс ${:.2f}".format(account['balance']))
                return account
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Ошибка при загрузке демо-счета: {}. Создаю новый.".format(e))
    
    # Создаем новый счет
    account = DEMO_ACCOUNT_TEMPLATE.copy()
    account['created_at'] = datetime.now().isoformat()
    account['last_updated'] = account['created_at']
    save_demo_account(account)
    logger.info("Создан новый демо-счет с балансом ${:.2f}".format(INITIAL_BALANCE))
    return account

def save_demo_account(account):
    """Сохраняет состояние демо-счета в файл."""
    account['last_updated'] = datetime.now().isoformat()
    with open(DEMO_ACCOUNT_FILE, 'w') as f:
        json.dump(account, f, indent=4)
    logger.info("Демо-счет сохранен: баланс ${:.2f}".format(account['balance']))

def initialize_historical_signals():
    """
    Инициализирует демо-счет историческими сигналами на основе CSV файлов.
    Используется один раз при первом запуске или сбросе демо-счета.
    """
    with ACCOUNT_LOCK:
        # Загружаем текущий счет
        account = load_demo_account()
        
        # Проверяем, если счет уже содержит сделки, то не трогаем его
        if account['trades']:
            logger.info("Демо-счет уже содержит сделки. Инициализация историческими данными пропущена.")
            return False
        
        try:
            # 1. Загрузка исторических данных из CSV файлов
            eurusd_file = 'EURUSD_Candlestick_30_m_BID_18.06.2022-18.06.2025 (2).csv'
            dxy_file = 'DOLLAR.IDXUSD_Candlestick_30_m_BID_18.06.2022-18.06.2025 (2).csv'
            
            if not os.path.exists(eurusd_file) or not os.path.exists(dxy_file):
                logger.error(f"CSV файлы не найдены: {eurusd_file} или {dxy_file}")
                return False
                
            # Загрузка и парсинг дат с правильным именем колонки
            eurusd_data = pd.read_csv(eurusd_file, parse_dates=['Gmt time'], dayfirst=True)
            dxy_data = pd.read_csv(dxy_file, parse_dates=['Gmt time'], dayfirst=True)
            
            # Переименование колонок
            eurusd_data.rename(columns={'Gmt time': 'Datetime', 'Open': 'Open', 'High': 'High', 
                                      'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}, inplace=True)
            dxy_data.rename(columns={'Gmt time': 'Datetime', 'Low': 'DXY_Low'}, inplace=True)
            
            # Установка индекса
            eurusd_data.set_index('Datetime', inplace=True)
            dxy_data.set_index('Datetime', inplace=True)
            
            # 2. Подготовка данных для анализа
            # Добавляем индикаторы
            import pandas_ta as ta
            eurusd_data.ta.rsi(length=14, append=True)
            eurusd_data.ta.macd(fast=12, slow=26, signal=9, append=True)
            eurusd_data.ta.atr(length=14, append=True)
            eurusd_data.rename(columns={'RSI_14': 'RSI', 'MACD_12_26_9': 'MACD', 
                                     'MACDh_12_26_9': 'MACD_hist', 'MACDs_12_26_9': 'MACD_signal', 
                                     'ATRr_14': 'ATR'}, inplace=True)
            
            # Объединение данных
            data = pd.concat([eurusd_data, dxy_data['DXY_Low']], axis=1)
            data.dropna(inplace=True)
            
            # 3. Загружаем модель для фильтрации сигналов
            model_file = 'ml_model_final_fix.joblib'
            if not os.path.exists(model_file):
                logger.error(f"Файл модели не найден: {model_file}")
                return False
                
            ml_model = joblib.load(model_file)
            prediction_threshold = 0.55  # Порог для сигналов
            
            # 4. Генерация исторических сигналов
            logger.info("Начинаем генерацию исторических сигналов на основе CSV данных...")
            
            # Константы для стратегии
            lookback_period = 20
            sl_ratio = 0.004
            tp_ratio = 0.01
            start_hour = 13
            end_hour = 17
            
            # Проходимся по данным и генерируем сигналы
            signals_count = 0
            
            # Первые lookback_period свечей пропускаем
            for i in range(lookback_period + 1, len(data) - 1):
                current_candle = data.iloc[i]
                current_time = data.index[i]
                
                # Проверяем временные ограничения (13-17 UTC)
                current_hour = current_time.hour
                if not (start_hour <= current_hour <= end_hour):
                    continue
                
                # Проверяем паттерн (Judas Swing + DXY Raid)
                start_index = i - lookback_period
                end_index = i
                
                recent_highs = data['High'].iloc[start_index:end_index].max()
                recent_dxy_lows = data['DXY_Low'].iloc[start_index:end_index].min()
                
                eurusd_judas_swing = current_candle['High'] > recent_highs
                dxy_raid = current_candle['DXY_Low'] < recent_dxy_lows
                
                # Если паттерн найден, проверяем ML модель
                if eurusd_judas_swing and dxy_raid:
                    # Берем данные для ML предсказания
                    features = [current_candle['RSI'], current_candle['MACD'], 
                              current_candle['MACD_hist'], current_candle['MACD_signal'], 
                              current_candle['ATR']]
                    
                    # Проверяем на NaN
                    if not any(np.isnan(features)):
                        # Делаем предсказание
                        win_prob = ml_model.predict_proba([features])[0][1]
                        
                        # Если сигнал проходит порог, добавляем его на демо-счет
                        if win_prob >= prediction_threshold:
                            # Получаем данные для входа
                            next_candle = data.iloc[i+1]
                            entry_price = next_candle['Open']
                            sl_price = entry_price * (1 + sl_ratio)
                            tp_price = entry_price * (1 - tp_ratio)
                            
                            # Проверяем, что цены валидные
                            if np.isfinite(entry_price) and np.isfinite(sl_price) and np.isfinite(tp_price):
                                # Создаем сделку
                                trade = create_historical_trade(
                                    signal_type=-1,  # SELL сигнал
                                    entry_price=entry_price,
                                    sl_price=sl_price,
                                    tp_price=tp_price,
                                    open_time=data.index[i+1],  # Время открытия - следующая свеча
                                    timeframe='30m'
                                )
                                
                                # Определяем исход сделки (проверяем следующие свечи)
                                resolve_historical_trade(trade, data, i+1)
                                
                                # Добавляем сделку на счет
                                account['trades'].append(trade)
                                account['balance'] += trade['profit_loss']
                                signals_count += 1
            
            # 5. Обновляем equity и сохраняем счет
            account['equity'] = calculate_equity(account)
            save_demo_account(account)
            
            logger.info(f"Инициализация демо-счета завершена. Добавлено {signals_count} исторических сигналов.")
            logger.info(f"Текущий баланс: ${account['balance']:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при инициализации демо-счета историческими данными: {e}")
            return False

def create_historical_trade(signal_type, entry_price, sl_price, tp_price, open_time, timeframe):
    """Создает историческую сделку с заданными параметрами."""
    # Расчет объема позиции на основе риска
    risk_amount = INITIAL_BALANCE * RISK_PER_TRADE
    sl_distance = abs(sl_price - entry_price)
    position_size = risk_amount / sl_distance
    position_value = position_size * entry_price
    
    # Создаем объект сделки
    trade = {
        'id': str(uuid.uuid4()),
        'symbol': 'EURUSD',
        'type': 'SELL' if signal_type == -1 else 'BUY',
        'entry_price': entry_price,
        'sl_price': sl_price,
        'tp_price': tp_price,
        'position_size': position_size,
        'position_value': position_value,
        'open_time': open_time.isoformat(),
        'close_time': None,
        'status': 'OPEN',
        'profit_loss': 0,
        'timeframe': timeframe,
        'comment': f'Исторический сигнал {timeframe}'
    }
    
    return trade

def resolve_historical_trade(trade, data, start_index):
    """
    Определяет исход исторической сделки на основе исторических данных.
    """
    try:
        for i in range(start_index, len(data)):
            candle = data.iloc[i]
            
            if trade['type'] == 'SELL':
                # Для SELL сделок проверяем Low для TP и High для SL
                if candle['Low'] <= trade['tp_price']:
                    # Сработал Take Profit
                    trade['close_time'] = data.index[i].isoformat()
                    trade['close_price'] = trade['tp_price']
                    trade['status'] = 'CLOSED'
                    trade['close_reason'] = 'TP'
                    price_diff = trade['entry_price'] - trade['tp_price']
                    trade['profit_loss'] = price_diff * trade['position_size']
                    return True
                    
                elif candle['High'] >= trade['sl_price']:
                    # Сработал Stop Loss
                    trade['close_time'] = data.index[i].isoformat()
                    trade['close_price'] = trade['sl_price']
                    trade['status'] = 'CLOSED'
                    trade['close_reason'] = 'SL'
                    price_diff = trade['entry_price'] - trade['sl_price']
                    trade['profit_loss'] = price_diff * trade['position_size']
                    return True
            else:
                # Для BUY сделок (если будут добавлены)
                if candle['High'] >= trade['tp_price']:
                    # Сработал Take Profit
                    trade['close_time'] = data.index[i].isoformat()
                    trade['close_price'] = trade['tp_price']
                    trade['status'] = 'CLOSED'
                    trade['close_reason'] = 'TP'
                    price_diff = trade['tp_price'] - trade['entry_price']
                    trade['profit_loss'] = price_diff * trade['position_size']
                    return True
                    
                elif candle['Low'] <= trade['sl_price']:
                    # Сработал Stop Loss
                    trade['close_time'] = data.index[i].isoformat()
                    trade['close_price'] = trade['sl_price']
                    trade['status'] = 'CLOSED'
                    trade['close_reason'] = 'SL'
                    price_diff = trade['sl_price'] - trade['entry_price']
                    trade['profit_loss'] = price_diff * trade['position_size']
                    return True
        
        # Если дошли до конца данных и сделка не закрылась, 
        # закрываем по последней цене
        last_candle = data.iloc[-1]
        trade['close_time'] = data.index[-1].isoformat()
        trade['close_price'] = last_candle['Close']
        trade['status'] = 'CLOSED'
        trade['close_reason'] = 'END_OF_DATA'
        
        if trade['type'] == 'SELL':
            price_diff = trade['entry_price'] - trade['close_price']
        else:
            price_diff = trade['close_price'] - trade['entry_price']
            
        trade['profit_loss'] = price_diff * trade['position_size']
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при определении исхода исторической сделки: {e}")
        return False

def add_trade_to_demo_account(signal_type, entry_price, sl_price, tp_price, timeframe, symbol="EURUSD"):
    """Добавляет новую сделку на демо-счет."""
    with ACCOUNT_LOCK:
        account = load_demo_account()
        
        # Расчет объема позиции на основе риска
        risk_amount = account['balance'] * RISK_PER_TRADE
        sl_distance = abs(sl_price - entry_price)
        
        if sl_distance <= 0:
            logger.error(f"Некорректное расстояние SL: {sl_distance}")
            return False

        position_size = risk_amount / sl_distance
        position_value = position_size * entry_price  # значение позиции в USD
        
        # Создание новой сделки
        trade = {
            'id': str(uuid.uuid4()),
            'symbol': symbol,
            'type': 'SELL' if signal_type == -1 else 'BUY',  # Обычно у нас SELL сигналы
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'position_size': position_size,
            'position_value': position_value,
            'open_time': datetime.now().isoformat(),
            'close_time': None,
            'status': 'OPEN',
            'profit_loss': 0,
            'timeframe': timeframe,
            'comment': f'Сигнал {timeframe}'
        }
        
        # Добавление сделки в список
        account['trades'].append(trade)
        
        # Обновление equity (учитывает открытые позиции)
        account['equity'] = calculate_equity(account)
        
        save_demo_account(account)
        logger.info(f"Добавлена новая сделка {trade['type']} {symbol} на демо-счет, ID: {trade['id']}")
        return True

def update_open_trades():
    """Обновляет состояние открытых сделок на основе текущих цен."""
    with ACCOUNT_LOCK:
        account = load_demo_account()
        updated = False
        
        # Получаем текущие цены
        try:
            # Пытаемся получить цены через Yahoo Finance (онлайн)
            import yfinance as yf
            current_price = None
            
            # Приоритет 1: Онлайн цены через Yahoo Finance
            try:
                eurusd_data = yf.download(tickers='EURUSD=X', period='1d', interval='1m')
                if isinstance(eurusd_data.columns, pd.MultiIndex):
                    eurusd_data.columns = eurusd_data.columns.get_level_values(0)
                
                if not eurusd_data.empty:
                    current_price = eurusd_data['Close'].iloc[-1]
                    logger.info("Получены онлайн цены EURUSD: {}".format(current_price))
            except Exception as e:
                logger.warning("Ошибка при получении онлайн цен EURUSD: {}, пробуем CSV".format(e))
            
            # Приоритет 2: CSV файл с курсом EUR/USD
            if current_price is None:
                try:
                    csv_file_eurusd = 'EURUSD_Candlestick_30_m_BID_18.06.2022-18.06.2025 (2).csv'
                    
                    if os.path.exists(csv_file_eurusd):
                        eurusd_csv = pd.read_csv(csv_file_eurusd)
                        current_price = eurusd_csv['Close'].iloc[-1]
                        logger.info("Получены цены EURUSD из CSV: {}".format(current_price))
                    else:
                        logger.warning("CSV файл не найден: {}".format(csv_file_eurusd))
                except Exception as e:
                    logger.error("Ошибка при чтении CSV файла EURUSD: {}".format(e))
            
            # Если не удалось получить цену ни онлайн, ни из CSV
            if current_price is None:
                logger.error("Не удалось получить текущие цены EURUSD")
                return False
            
            for i, trade in enumerate(account['trades']):
                if trade['status'] == 'OPEN':
                    # Проверка условий закрытия
                    if trade['type'] == 'SELL':
                        # Для SELL сделок
                        if current_price <= trade['tp_price']:  # Достигнут Take Profit
                            close_trade(account, i, current_price, 'TP')
                            updated = True
                        elif current_price >= trade['sl_price']:  # Достигнут Stop Loss
                            close_trade(account, i, current_price, 'SL')
                            updated = True
                        else:
                            # Обновляем плавающую прибыль/убыток
                            price_diff = trade['entry_price'] - current_price
                            trade['profit_loss'] = price_diff * trade['position_size']
                            updated = True
                    else:
                        # Для BUY сделок
                        if current_price >= trade['tp_price']:  # Достигнут Take Profit
                            close_trade(account, i, current_price, 'TP')
                            updated = True
                        elif current_price <= trade['sl_price']:  # Достигнут Stop Loss
                            close_trade(account, i, current_price, 'SL')
                            updated = True
                        else:
                            # Обновляем плавающую прибыль/убыток
                            price_diff = current_price - trade['entry_price']
                            trade['profit_loss'] = price_diff * trade['position_size']
                            updated = True
            
            if updated:
                # Обновляем equity
                account['equity'] = calculate_equity(account)
                save_demo_account(account)
                logger.info("Обновлены открытые сделки, текущий equity: ${:.2f}".format(account['equity']))
            
            return updated
        
        except Exception as e:
            logger.error(f"Ошибка при обновлении открытых сделок: {e}")
            return False

def close_trade(account, trade_index, close_price, reason):
    """Закрывает сделку с указанным индексом."""
    trade = account['trades'][trade_index]
    trade['close_time'] = datetime.now().isoformat()
    trade['status'] = 'CLOSED'
    
    # Расчет прибыли/убытка
    if trade['type'] == 'SELL':
        price_diff = trade['entry_price'] - close_price
    else:
        price_diff = close_price - trade['entry_price']
    
    trade['profit_loss'] = price_diff * trade['position_size']
    trade['close_price'] = close_price
    trade['close_reason'] = reason
    
    # Обновляем баланс счета
    account['balance'] += trade['profit_loss']
    logger.info(f"Закрыта сделка ID: {trade['id']}, причина: {reason}, P/L: ${trade['profit_loss']:.2f}")

def calculate_equity(account):
    """Рассчитывает текущий equity счета (баланс + плавающая P/L)."""
    equity = account['balance']
    for trade in account['trades']:
        if trade['status'] == 'OPEN':
            equity += trade['profit_loss']
    return equity

def generate_account_html():
    """Генерирует HTML с отчетом о состоянии демо-счета."""
    account = load_demo_account()
    
    # Получаем статистику счета
    closed_trades = [t for t in account['trades'] if t['status'] == 'CLOSED']
    open_trades = [t for t in account['trades'] if t['status'] == 'OPEN']
    
    # Рассчитываем статистику
    total_trades = len(closed_trades)
    profitable_trades = sum(1 for t in closed_trades if t['profit_loss'] > 0)
    losing_trades = sum(1 for t in closed_trades if t['profit_loss'] <= 0)
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
    
    # Финансовая статистика
    total_profit = sum(t['profit_loss'] for t in closed_trades if t['profit_loss'] > 0)
    total_loss = sum(t['profit_loss'] for t in closed_trades if t['profit_loss'] <= 0)
    net_profit = total_profit + total_loss
    profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
    
    # Создание графиков
    fig = create_account_charts(account)
    chart_div = pio.to_html(fig, full_html=False)
    
    # Формируем таблицу открытых сделок
    open_trades_html = ""
    if open_trades:
        open_trades_html = """
        <h3>Открытые сделки</h3>
        <table class="trades-table">
            <tr>
                <th>ID</th>
                <th>Символ</th>
                <th>Тип</th>
                <th>Открытие</th>
                <th>Цена входа</th>
                <th>SL</th>
                <th>TP</th>
                <th>Размер</th>
                <th>P/L</th>
                <th>Таймфрейм</th>
            </tr>
        """
        
        for trade in open_trades:
            open_time = datetime.fromisoformat(trade['open_time']).strftime('%Y-%m-%d %H:%M')
            open_trades_html += f"""
            <tr class="{trade['type'].lower()}">
                <td>{trade['id'][:8]}...</td>
                <td>{trade['symbol']}</td>
                <td>{trade['type']}</td>
                <td>{open_time}</td>
                <td>{trade['entry_price']:.5f}</td>
                <td>{trade['sl_price']:.5f}</td>
                <td>{trade['tp_price']:.5f}</td>
                <td>{trade['position_size']:.2f}</td>
                <td class="{'profit' if trade['profit_loss'] > 0 else 'loss'}">${trade['profit_loss']:.2f}</td>
                <td>{trade['timeframe']}</td>
            </tr>
            """
        open_trades_html += "</table>"
    
    # Формируем таблицу закрытых сделок (последние 20)
    recent_closed_trades = sorted(closed_trades, key=lambda t: t['close_time'], reverse=True)[:20]
    closed_trades_html = ""
    if recent_closed_trades:
        closed_trades_html = """
        <h3>Последние закрытые сделки</h3>
        <table class="trades-table">
            <tr>
                <th>ID</th>
                <th>Символ</th>
                <th>Тип</th>
                <th>Открытие</th>
                <th>Закрытие</th>
                <th>Цена входа</th>
                <th>Цена выхода</th>
                <th>P/L</th>
                <th>Причина</th>
                <th>Таймфрейм</th>
            </tr>
        """
        
        for trade in recent_closed_trades:
            open_time = datetime.fromisoformat(trade['open_time']).strftime('%Y-%m-%d %H:%M')
            close_time = datetime.fromisoformat(trade['close_time']).strftime('%Y-%m-%d %H:%M')
            closed_trades_html += f"""
            <tr class="{trade['type'].lower()}">
                <td>{trade['id'][:8]}...</td>
                <td>{trade['symbol']}</td>
                <td>{trade['type']}</td>
                <td>{open_time}</td>
                <td>{close_time}</td>
                <td>{trade['entry_price']:.5f}</td>
                <td>{trade['close_price']:.5f}</td>
                <td class="{'profit' if trade['profit_loss'] > 0 else 'loss'}">${trade['profit_loss']:.2f}</td>
                <td>{trade.get('close_reason', 'N/A')}</td>
                <td>{trade['timeframe']}</td>
            </tr>
            """
        closed_trades_html += "</table>"
    
    # Формируем полный HTML
    created_date = datetime.fromisoformat(account['created_at']).strftime('%Y-%m-%d %H:%M')
    last_update = datetime.fromisoformat(account['last_updated']).strftime('%Y-%m-%d %H:%M:%S')
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Демо-счет торговли по сигналам</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .account-info {{
                display: flex;
                justify-content: space-around;
                margin-bottom: 30px;
                flex-wrap: wrap;
            }}
            .info-box {{
                background-color: #f8f9fa;
                border-left: 4px solid #4285F4;
                padding: 15px;
                margin: 10px;
                min-width: 200px;
                border-radius: 4px;
            }}
            .trades-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .trades-table th, .trades-table td {{
                padding: 10px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .trades-table th {{
                background-color: #f2f2f2;
            }}
            .profit {{
                color: green;
                font-weight: bold;
            }}
            .loss {{
                color: red;
                font-weight: bold;
            }}
            .buy {{
                background-color: #e6f7ff;
            }}
            .sell {{
                background-color: #fff2e6;
            }}
            .chart-container {{
                width: 100%;
                margin: 20px 0;
            }}
            .stats-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                margin-bottom: 30px;
            }}
            .stat-box {{
                background-color: #ffffff;
                border: 1px solid #e6e6e6;
                border-radius: 5px;
                padding: 15px;
                margin: 10px;
                flex: 1;
                min-width: 150px;
                text-align: center;
            }}
            .stat-box h4 {{
                margin: 0;
                color: #666;
                font-size: 14px;
            }}
            .stat-box p {{
                font-size: 18px;
                font-weight: bold;
                margin: 10px 0 0 0;
            }}
            .footer {{
                margin-top: 30px;
                text-align: center;
                font-size: 12px;
                color: #999;
            }}
            
            /* Добавим автоматическое обновление страницы каждые 5 минут */
            <meta http-equiv="refresh" content="300">
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Демо-счет торговли по сигналам</h1>
                <p>Автоматическая торговля на основе сигналов алгоритма</p>
            </div>
            
            <div class="account-info">
                <div class="info-box">
                    <h3>Баланс</h3>
                    <h2>${account['balance']:.2f}</h2>
                </div>
                <div class="info-box">
                    <h3>Средства</h3>
                    <h2>${account['equity']:.2f}</h2>
                </div>
                <div class="info-box">
                    <h3>Создан</h3>
                    <h2>{created_date}</h2>
                </div>
            </div>
            
            <div class="stats-container">
                <div class="stat-box">
                    <h4>Всего сделок</h4>
                    <p>{total_trades}</p>
                </div>
                <div class="stat-box">
                    <h4>Прибыльные</h4>
                    <p>{profitable_trades}</p>
                </div>
                <div class="stat-box">
                    <h4>Убыточные</h4>
                    <p>{losing_trades}</p>
                </div>
                <div class="stat-box">
                    <h4>Win Rate</h4>
                    <p>{win_rate:.2%}</p>
                </div>
                <div class="stat-box">
                    <h4>Чистая прибыль</h4>
                    <p class="{'profit' if net_profit > 0 else 'loss'}">{'$'}{net_profit:.2f}</p>
                </div>
                <div class="stat-box">
                    <h4>Profit Factor</h4>
                    <p>{profit_factor:.2f}</p>
                </div>
            </div>
            
            <div class="chart-container">
                {chart_div}
            </div>
            
            {open_trades_html}
            
            {closed_trades_html}
            
            <div class="footer">
                <p>Последнее обновление: {last_update}</p>
                <p>Демо-счет обновляется каждые 5 минут. Страница автоматически обновляется.</p>
                <p>Все сделки выполняются автоматически на основе сигналов алгоритма.</p>
            </div>
        </div>
        
        <!-- Добавим скрипт для автоматического обновления графиков и данных без перезагрузки страницы -->
        <script>
            // Функция для обновления данных через AJAX каждые 60 секунд
            function scheduleUpdate() {{
                setTimeout(function() {{
                    fetch(window.location.href)
                        .then(response => response.text())
                        .then(html => {{
                            const parser = new DOMParser();
                            const newDoc = parser.parseFromString(html, 'text/html');
                            
                            // Обновляем только нужные части страницы
                            document.querySelector('.account-info').innerHTML = newDoc.querySelector('.account-info').innerHTML;
                            document.querySelector('.stats-container').innerHTML = newDoc.querySelector('.stats-container').innerHTML;
                            document.querySelector('.chart-container').innerHTML = newDoc.querySelector('.chart-container').innerHTML;
                            
                            if (document.querySelector('.trades-table') && newDoc.querySelector('.trades-table')) {{
                                document.querySelectorAll('.trades-table').forEach((table, i) => {{
                                    if (newDoc.querySelectorAll('.trades-table')[i]) {{
                                        table.innerHTML = newDoc.querySelectorAll('.trades-table')[i].innerHTML;
                                    }}
                                }});
                            }}
                            
                            document.querySelector('.footer').innerHTML = newDoc.querySelector('.footer').innerHTML;
                            
                            scheduleUpdate();
                        }})
                        .catch(error => {{
                            console.error('Ошибка обновления данных:', error);
                            scheduleUpdate();
                        }});
                }}, 60000); // Обновление каждую минуту
            }}
            
            // Запускаем обновление после загрузки страницы
            window.addEventListener('load', scheduleUpdate);
        </script>
    </body>
    </html>
    """
    
    return html

def create_account_charts(account):
    """Создает графики для отчета о состоянии счета."""
    # Подготовка данных для графиков
    trades = account['trades']
    
    # Извлекаем даты и балансы для графика кривой баланса
    balance_data = []
    dates = []
    current_balance = INITIAL_BALANCE
    
    # Сортируем сделки по времени закрытия
    closed_trades = [t for t in trades if t['status'] == 'CLOSED']
    closed_trades.sort(key=lambda t: datetime.fromisoformat(t['close_time']))
    
    # Строим кривую баланса
    balance_data.append(current_balance)
    dates.append(datetime.fromisoformat(account['created_at']))
    
    for trade in closed_trades:
        current_balance += trade['profit_loss']
        balance_data.append(current_balance)
        dates.append(datetime.fromisoformat(trade['close_time']))
    
    # Создаем подграфики
    fig = make_subplots(rows=2, cols=1, 
                         shared_xaxes=True,
                         vertical_spacing=0.1,
                         subplot_titles=('Кривая баланса', 'Результаты сделок'),
                         row_heights=[0.7, 0.3])
    
    # График кривой баланса
    fig.add_trace(go.Scatter(
        x=dates,
        y=balance_data,
        mode='lines+markers',
        name='Баланс',
        line=dict(color='#2C82C9', width=2),
        marker=dict(size=6)
    ), row=1, col=1)
    
    # График отдельных сделок
    trade_results = [t['profit_loss'] for t in closed_trades]
    trade_dates = [datetime.fromisoformat(t['close_time']) for t in closed_trades]
    trade_colors = ['green' if r > 0 else 'red' for r in trade_results]
    
    fig.add_trace(go.Bar(
        x=trade_dates,
        y=trade_results,
        name='Результаты сделок',
        marker_color=trade_colors
    ), row=2, col=1)
    
    # Настройка макета
    fig.update_layout(
        title='История торговли демо-счета',
        showlegend=False,
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),
        template='plotly_white'
    )
    
    fig.update_xaxes(
        title_text='Дата',
        tickformat='%d %b %Y',
        tickangle=45,
        row=2, col=1
    )
    
    fig.update_yaxes(
        title_text='Баланс ($)',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text='Прибыль/Убыток ($)',
        row=2, col=1
    )
    
    return fig

def process_new_signal(signal_type, entry_price, sl_price, tp_price, timeframe, symbol="EURUSD"):
    """Обрабатывает новый сигнал от алгоритма."""
    result = add_trade_to_demo_account(signal_type, entry_price, sl_price, tp_price, timeframe, symbol)
    if result:
        logger.info(f"Добавлен новый сигнал {timeframe} на демо-счет")
    else:
        logger.error(f"Не удалось добавить сигнал {timeframe} на демо-счет")
    return result

def get_demo_account_summary():
    """Возвращает краткую статистику по демо-счету."""
    account = load_demo_account()
    
    closed_trades = [t for t in account['trades'] if t['status'] == 'CLOSED']
    open_trades = [t for t in account['trades'] if t['status'] == 'OPEN']
    
    total_trades = len(closed_trades)
    profitable_trades = sum(1 for t in closed_trades if t['profit_loss'] > 0)
    
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
    
    total_profit = sum(t['profit_loss'] for t in closed_trades if t['profit_loss'] > 0)
    total_loss = sum(t['profit_loss'] for t in closed_trades if t['profit_loss'] <= 0)
    net_profit = total_profit + total_loss
    
    return {
        'balance': account['balance'],
        'equity': account['equity'],
        'total_trades': total_trades,
        'win_rate': win_rate,
        'net_profit': net_profit,
        'open_positions': len(open_trades)
    }

