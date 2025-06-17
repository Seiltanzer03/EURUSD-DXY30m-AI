import os
import json
import asyncio
import logging
from flask import Flask, render_template, jsonify, request, send_from_directory, Blueprint
from threading import Thread
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

# Импортируем модули нашего проекта
from demo_account import demo_account

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('demo_webapp')

# Создаем Blueprint вместо приложения Flask для интеграции с существующим приложением
demo_blueprint = Blueprint('demo', __name__, url_prefix='/demo')

# Создаем директорию для статических файлов, если её нет
os.makedirs('static/demo', exist_ok=True)

# Функция для запуска асинхронных задач в фоновом режиме
def run_async(func):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(func(*args, **kwargs))
        loop.close()
        return result
    return wrapper

@demo_blueprint.route('/')
def index():
    """Главная страница с демо-счетом."""
    return render_template('demo_index.html')

@demo_blueprint.route('/api/account')
def get_account():
    """API для получения информации о счете."""
    try:
        account_info = run_async(demo_account.get_account_info)()
        return jsonify(account_info)
    except Exception as e:
        logger.error(f"Ошибка при получении информации о демо-счете: {e}")
        return jsonify({"error": str(e)}), 500

@demo_blueprint.route('/api/stats')
def get_stats():
    """API для получения статистики счета."""
    try:
        stats = run_async(demo_account.get_account_stats)()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Ошибка при получении статистики демо-счета: {e}")
        return jsonify({"error": str(e)}), 500

@demo_blueprint.route('/api/trades')
def get_trades():
    """API для получения истории сделок."""
    try:
        limit = request.args.get('limit', default=20, type=int)
        trades = run_async(demo_account.get_trade_history)(limit)
        return jsonify(trades)
    except Exception as e:
        logger.error(f"Ошибка при получении истории сделок демо-счета: {e}")
        return jsonify({"error": str(e)}), 500

@demo_blueprint.route('/api/balance-history')
def get_balance_history():
    """API для получения истории баланса."""
    try:
        balance_history = run_async(demo_account.get_balance_history)()
        return jsonify(balance_history)
    except Exception as e:
        logger.error(f"Ошибка при получении истории баланса демо-счета: {e}")
        return jsonify({"error": str(e)}), 500

@demo_blueprint.route('/api/balance-chart')
def get_balance_chart():
    """API для получения графика баланса в формате base64."""
    try:
        # Получаем историю баланса
        balance_history = run_async(demo_account.get_balance_history)()
        
        if not balance_history:
            return jsonify({"error": "Нет данных для построения графика демо-счета"}), 404
        
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
        
        # Кодируем в base64
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return jsonify({"chart": f"data:image/png;base64,{img_base64}"})
    except Exception as e:
        logger.error(f"Ошибка при создании графика баланса демо-счета: {e}")
        return jsonify({"error": str(e)}), 500

@demo_blueprint.route('/api/reset', methods=['POST'])
def reset_account():
    """API для сброса демо-счета."""
    try:
        result = run_async(demo_account.reset_account)()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Ошибка при сбросе демо-счета: {e}")
        return jsonify({"error": str(e)}), 500

@demo_blueprint.route('/static/<path:path>')
def serve_static(path):
    """Обслуживание статических файлов."""
    return send_from_directory('static/demo', path)

# Создаем шаблоны для веб-интерфейса
def create_templates():
    """Создает HTML шаблоны для веб-интерфейса демо-счета."""
    os.makedirs('templates', exist_ok=True)
    
    # Создаем основной шаблон demo_index.html
    index_html = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Демо-счет</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 20px;
            background-color: #f8f9fa;
        }
        .card {
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .card-header {
            font-weight: bold;
            background-color: #f1f8ff;
        }
        .trade-card {
            border-left: 5px solid #28a745;
            margin-bottom: 10px;
        }
        .trade-card.negative {
            border-left: 5px solid #dc3545;
        }
        .stats-value {
            font-weight: bold;
            font-size: 1.1rem;
        }
        #balance-chart {
            width: 100%;
            height: auto;
            max-height: 400px;
        }
        .loader {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 2s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row mb-4">
            <div class="col-12">
                <h1 class="text-center">Демо-счет</h1>
                <p class="text-center text-muted">Торговый бот с искусственным интеллектом</p>
            </div>
        </div>
        
        <div class="row">
            <!-- Информация о счете -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <i class="bi bi-wallet"></i> Информация о счете
                    </div>
                    <div class="card-body">
                        <div id="account-info-loader" class="loader"></div>
                        <div id="account-info" style="display: none;">
                            <h2 class="mb-3">Баланс: <span id="balance" class="text-primary"></span></h2>
                            <p>Валюта: <span id="currency"></span></p>
                            <p>Создан: <span id="created-at"></span></p>
                            <p>Обновлен: <span id="updated-at"></span></p>
                            <button id="reset-button" class="btn btn-danger">Сбросить счет</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Статистика счета -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <i class="bi bi-graph-up"></i> Статистика
                    </div>
                    <div class="card-body">
                        <div id="stats-loader" class="loader"></div>
                        <div id="stats-info" style="display: none;">
                            <div class="row">
                                <div class="col-6 mb-3">
                                    <div>Всего сделок:</div>
                                    <div id="total-trades" class="stats-value"></div>
                                </div>
                                <div class="col-6 mb-3">
                                    <div>Винрейт:</div>
                                    <div id="win-rate" class="stats-value"></div>
                                </div>
                                <div class="col-6 mb-3">
                                    <div>Профит-фактор:</div>
                                    <div id="profit-factor" class="stats-value"></div>
                                </div>
                                <div class="col-6 mb-3">
                                    <div>Общая прибыль:</div>
                                    <div id="total-profit" class="stats-value"></div>
                                </div>
                                <div class="col-6 mb-3">
                                    <div>Макс. просадка:</div>
                                    <div id="max-drawdown" class="stats-value"></div>
                                </div>
                                <div class="col-6 mb-3">
                                    <div>Средний профит:</div>
                                    <div id="avg-profit" class="stats-value"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <!-- График баланса -->
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <i class="bi bi-bar-chart-line"></i> График баланса
                    </div>
                    <div class="card-body text-center">
                        <div id="chart-loader" class="loader"></div>
                        <img id="balance-chart" style="display: none;" alt="График баланса">
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <!-- История сделок -->
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <i class="bi bi-list-ul"></i> История сделок
                    </div>
                    <div class="card-body">
                        <div id="trades-loader" class="loader"></div>
                        <div id="trades-container" style="display: none;"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Функция для форматирования даты
        function formatDate(dateString) {
            const date = new Date(dateString);
            return date.toLocaleString();
        }
        
        // Функция для форматирования денежных значений
        function formatMoney(amount) {
            return '$' + parseFloat(amount).toFixed(2);
        }
        
        // Функция для форматирования процентов
        function formatPercent(value) {
            return (parseFloat(value) * 100).toFixed(2) + '%';
        }
        
        // Загрузка информации о счете
        function loadAccountInfo() {
            document.getElementById('account-info-loader').style.display = 'block';
            document.getElementById('account-info').style.display = 'none';
            
            fetch('/demo/api/account')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('balance').textContent = formatMoney(data.balance);
                    document.getElementById('currency').textContent = data.currency;
                    document.getElementById('created-at').textContent = formatDate(data.created_at);
                    document.getElementById('updated-at').textContent = formatDate(data.updated_at);
                    
                    document.getElementById('account-info-loader').style.display = 'none';
                    document.getElementById('account-info').style.display = 'block';
                })
                .catch(error => {
                    console.error('Ошибка при загрузке информации о счете:', error);
                    document.getElementById('account-info-loader').style.display = 'none';
                    document.getElementById('account-info').style.display = 'block';
                    document.getElementById('account-info').innerHTML = '<div class="alert alert-danger">Ошибка при загрузке данных</div>';
                });
        }
        
        // Загрузка статистики счета
        function loadStats() {
            document.getElementById('stats-loader').style.display = 'block';
            document.getElementById('stats-info').style.display = 'none';
            
            fetch('/demo/api/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('total-trades').textContent = data.total_trades;
                    document.getElementById('win-rate').textContent = formatPercent(data.win_rate);
                    document.getElementById('profit-factor').textContent = data.profit_factor.toFixed(2);
                    document.getElementById('total-profit').textContent = formatMoney(data.total_profit);
                    document.getElementById('max-drawdown').textContent = formatPercent(data.max_drawdown);
                    document.getElementById('avg-profit').textContent = formatMoney(data.avg_profit);
                    
                    document.getElementById('stats-loader').style.display = 'none';
                    document.getElementById('stats-info').style.display = 'block';
                })
                .catch(error => {
                    console.error('Ошибка при загрузке статистики:', error);
                    document.getElementById('stats-loader').style.display = 'none';
                    document.getElementById('stats-info').style.display = 'block';
                    document.getElementById('stats-info').innerHTML = '<div class="alert alert-danger">Ошибка при загрузке данных</div>';
                });
        }
        
        // Загрузка графика баланса
        function loadBalanceChart() {
            document.getElementById('chart-loader').style.display = 'block';
            document.getElementById('balance-chart').style.display = 'none';
            
            fetch('/demo/api/balance-chart')
                .then(response => response.json())
                .then(data => {
                    if (data.chart) {
                        document.getElementById('balance-chart').src = data.chart;
                        document.getElementById('balance-chart').style.display = 'block';
                    } else {
                        document.getElementById('balance-chart').style.display = 'none';
                    }
                    document.getElementById('chart-loader').style.display = 'none';
                })
                .catch(error => {
                    console.error('Ошибка при загрузке графика:', error);
                    document.getElementById('chart-loader').style.display = 'none';
                    document.getElementById('balance-chart').style.display = 'none';
                    document.getElementById('chart-loader').insertAdjacentHTML('afterend', '<div class="alert alert-danger">Ошибка при загрузке графика</div>');
                });
        }
        
        // Загрузка истории сделок
        function loadTrades() {
            document.getElementById('trades-loader').style.display = 'block';
            document.getElementById('trades-container').style.display = 'none';
            
            fetch('/demo/api/trades')
                .then(response => response.json())
                .then(trades => {
                    const container = document.getElementById('trades-container');
                    container.innerHTML = '';
                    
                    if (trades.length === 0) {
                        container.innerHTML = '<div class="alert alert-info">Нет сделок для отображения</div>';
                    } else {
                        trades.forEach(trade => {
                            const isPositive = trade.profit > 0;
                            const cardClass = isPositive ? 'trade-card' : 'trade-card negative';
                            const profitClass = isPositive ? 'text-success' : 'text-danger';
                            
                            const tradeCard = `
                                <div class="card ${cardClass}">
                                    <div class="card-body">
                                        <div class="row">
                                            <div class="col-md-6">
                                                <h5>${trade.symbol} ${trade.direction}</h5>
                                                <p class="mb-1">Статус: <span class="badge ${trade.status === 'OPEN' ? 'bg-primary' : 'bg-secondary'}">${trade.status}</span></p>
                                                <p class="mb-1">Таймфрейм: ${trade.timeframe || 'Н/Д'}</p>
                                                <p class="mb-1">Открыта: ${formatDate(trade.opened_at)}</p>
                                                ${trade.closed_at ? `<p class="mb-1">Закрыта: ${formatDate(trade.closed_at)}</p>` : ''}
                                            </div>
                                            <div class="col-md-6">
                                                <p class="mb-1">Вход: ${trade.entry_price}</p>
                                                <p class="mb-1">SL: ${trade.stop_loss}</p>
                                                <p class="mb-1">TP: ${trade.take_profit}</p>
                                                <p class="mb-1">Лот: ${trade.lot_size.toFixed(2)}</p>
                                                ${trade.profit !== undefined ? `<p class="mb-1">Прибыль: <span class="${profitClass}">${formatMoney(trade.profit)}</span></p>` : ''}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            `;
                            container.innerHTML += tradeCard;
                        });
                    }
                    
                    document.getElementById('trades-loader').style.display = 'none';
                    document.getElementById('trades-container').style.display = 'block';
                })
                .catch(error => {
                    console.error('Ошибка при загрузке сделок:', error);
                    document.getElementById('trades-loader').style.display = 'none';
                    document.getElementById('trades-container').style.display = 'block';
                    document.getElementById('trades-container').innerHTML = '<div class="alert alert-danger">Ошибка при загрузке сделок</div>';
                });
        }
        
        // Обработка сброса счета
        document.getElementById('reset-button').addEventListener('click', function() {
            if (confirm('Вы уверены, что хотите сбросить демо-счет? Все сделки будут удалены, а баланс вернется к начальному значению.')) {
                fetch('/demo/api/reset', {
                    method: 'POST'
                })
                .then(response => response.json())
                .then(data => {
                    alert('Демо-счет успешно сброшен!');
                    // Перезагружаем все данные
                    loadAccountInfo();
                    loadStats();
                    loadBalanceChart();
                    loadTrades();
                })
                .catch(error => {
                    console.error('Ошибка при сбросе счета:', error);
                    alert('Произошла ошибка при сбросе счета');
                });
            }
        });
        
        // Загружаем все данные при загрузке страницы
        document.addEventListener('DOMContentLoaded', function() {
            loadAccountInfo();
            loadStats();
            loadBalanceChart();
            loadTrades();
        });
    </script>
</body>
</html>
    """
    
    with open('templates/demo_index.html', 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    logger.info("Шаблоны для веб-интерфейса демо-счета созданы")

# Функция для интеграции с существующим Flask приложением
def register_demo_blueprint(app):
    """
    Регистрирует Blueprint демо-счета в существующем Flask приложении.
    
    Args:
        app: Существующее Flask приложение
    """
    # Создаем шаблоны для веб-интерфейса
    create_templates()
    
    # Регистрируем Blueprint
    app.register_blueprint(demo_blueprint)
    
    logger.info("Blueprint демо-счета зарегистрирован в приложении Flask")

# Если файл запущен напрямую, создаем отдельное приложение Flask
if __name__ == "__main__":
    app = Flask(__name__)
    register_demo_blueprint(app)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
