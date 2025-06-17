import os
import json
import asyncio
import logging
from flask import Flask, render_template, jsonify, request, send_from_directory
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
logger = logging.getLogger('webapp')

app = Flask(__name__)

# Создаем директорию для статических файлов, если её нет
os.makedirs('static', exist_ok=True)

# Функция для запуска асинхронных задач в фоновом режиме
def run_async(func):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(func(*args, **kwargs))
        loop.close()
        return result
    return wrapper

@app.route('/')
def index():
    """Главная страница с демо-счетом."""
    return render_template('index.html')

@app.route('/api/account')
def get_account():
    """API для получения информации о счете."""
    try:
        account_info = run_async(demo_account.get_account_info)()
        return jsonify(account_info)
    except Exception as e:
        logger.error(f"Ошибка при получении информации о счете: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """API для получения статистики счета."""
    try:
        stats = run_async(demo_account.get_account_stats)()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Ошибка при получении статистики счета: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/trades')
def get_trades():
    """API для получения истории сделок."""
    try:
        limit = request.args.get('limit', default=20, type=int)
        trades = run_async(demo_account.get_trade_history)(limit)
        return jsonify(trades)
    except Exception as e:
        logger.error(f"Ошибка при получении истории сделок: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/balance-history')
def get_balance_history():
    """API для получения истории баланса."""
    try:
        balance_history = run_async(demo_account.get_balance_history)()
        return jsonify(balance_history)
    except Exception as e:
        logger.error(f"Ошибка при получении истории баланса: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/balance-chart')
def get_balance_chart():
    """API для получения графика баланса в формате base64."""
    try:
        # Получаем историю баланса
        balance_history = run_async(demo_account.get_balance_history)()
        
        if not balance_history:
            return jsonify({"error": "Нет данных для построения графика"}), 404
        
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
        logger.error(f"Ошибка при создании графика баланса: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_account():
    """API для сброса демо-счета."""
    try:
        result = run_async(demo_account.reset_account)()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Ошибка при сбросе демо-счета: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    """Обслуживание статических файлов."""
    return send_from_directory('static', path)

# Создаем шаблоны для веб-интерфейса
def create_templates():
    """Создает HTML шаблоны для веб-интерфейса."""
    os.makedirs('templates', exist_ok=True)
    
    # Создаем основной шаблон index.html
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
                                    <div>Средняя прибыль:</div>
                                    <div id="avg-profit" class="stats-value"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <!-- График баланса -->
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <i class="bi bi-bar-chart-line"></i> График баланса
                    </div>
                    <div class="card-body">
                        <div id="chart-loader" class="loader"></div>
                        <img id="balance-chart" src="" style="display: none;">
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
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
        
        <!-- Модальное окно подтверждения сброса -->
        <div class="modal fade" id="resetModal" tabindex="-1" aria-labelledby="resetModalLabel" aria-hidden="true">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="resetModalLabel">Подтверждение сброса</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <p>Вы уверены, что хотите сбросить демо-счет?</p>
                        <p>Это действие удалит все сделки и вернет баланс к начальному значению.</p>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Отмена</button>
                        <button type="button" class="btn btn-danger" id="confirm-reset">Да, сбросить</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Форматирование даты
        function formatDate(dateStr) {
            const date = new Date(dateStr);
            return date.toLocaleDateString();
        }

        // Загрузка информации о счете
        async function loadAccountInfo() {
            try {
                const response = await fetch('/api/account');
                const data = await response.json();
                
                document.getElementById('balance').textContent = '$' + data.balance.toFixed(2);
                document.getElementById('currency').textContent = data.currency;
                document.getElementById('created-at').textContent = formatDate(data.created_at);
                document.getElementById('updated-at').textContent = formatDate(data.updated_at);
                
                document.getElementById('account-info-loader').style.display = 'none';
                document.getElementById('account-info').style.display = 'block';
            } catch (error) {
                console.error('Ошибка при загрузке информации о счете:', error);
                document.getElementById('account-info-loader').style.display = 'none';
                document.getElementById('account-info').innerHTML = '<div class="alert alert-danger">Ошибка при загрузке данных</div>';
            }
        }

        // Загрузка статистики счета
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                
                document.getElementById('total-trades').textContent = data.total_trades;
                document.getElementById('win-rate').textContent = (data.win_rate * 100).toFixed(2) + '%';
                document.getElementById('profit-factor').textContent = data.profit_factor.toFixed(2);
                document.getElementById('total-profit').textContent = '$' + data.total_profit.toFixed(2);
                document.getElementById('max-drawdown').textContent = (data.max_drawdown * 100).toFixed(2) + '%';
                document.getElementById('avg-profit').textContent = '$' + data.avg_profit.toFixed(2);
                
                document.getElementById('stats-loader').style.display = 'none';
                document.getElementById('stats-info').style.display = 'block';
            } catch (error) {
                console.error('Ошибка при загрузке статистики:', error);
                document.getElementById('stats-loader').style.display = 'none';
                document.getElementById('stats-info').innerHTML = '<div class="alert alert-danger">Ошибка при загрузке данных</div>';
            }
        }

        // Загрузка графика баланса
        async function loadBalanceChart() {
            try {
                const response = await fetch('/api/balance-chart');
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                document.getElementById('balance-chart').src = data.chart;
                document.getElementById('balance-chart').style.display = 'block';
                document.getElementById('chart-loader').style.display = 'none';
            } catch (error) {
                console.error('Ошибка при загрузке графика:', error);
                document.getElementById('chart-loader').style.display = 'none';
                document.getElementById('balance-chart').style.display = 'none';
                document.getElementById('chart-loader').parentNode.innerHTML += '<div class="alert alert-danger">Ошибка при загрузке графика</div>';
            }
        }

        // Загрузка истории сделок
        async function loadTrades() {
            try {
                const response = await fetch('/api/trades');
                const trades = await response.json();
                
                const tradesContainer = document.getElementById('trades-container');
                tradesContainer.innerHTML = '';
                
                if (trades.length === 0) {
                    tradesContainer.innerHTML = '<div class="alert alert-info">Нет сделок для отображения</div>';
                } else {
                    trades.forEach(trade => {
                        const isProfit = trade.profit > 0;
                        const cardClass = isProfit ? 'trade-card' : 'trade-card negative';
                        const statusBadge = trade.status === 'OPEN' 
                            ? '<span class="badge bg-warning">Открыта</span>' 
                            : '<span class="badge bg-secondary">Закрыта</span>';
                        
                        let tradeHtml = `
                            <div class="card ${cardClass}">
                                <div class="card-body">
                                    <div class="d-flex justify-content-between align-items-center mb-2">
                                        <h5 class="card-title mb-0">${trade.symbol} (${trade.timeframe}) ${statusBadge}</h5>
                                        <span class="badge ${isProfit ? 'bg-success' : 'bg-danger'}">${trade.direction}</span>
                                    </div>
                                    <div class="row">
                                        <div class="col-md-6">
                                            <p class="card-text">Вход: ${trade.entry_price}</p>
                                            <p class="card-text">Стоп-лосс: ${trade.stop_loss}</p>
                                            <p class="card-text">Тейк-профит: ${trade.take_profit}</p>
                                        </div>
                                        <div class="col-md-6">`;
                        
                        if (trade.status === 'CLOSED') {
                            tradeHtml += `
                                            <p class="card-text">Закрытие: ${trade.close_price}</p>
                                            <p class="card-text">Прибыль: $${trade.profit.toFixed(2)}</p>
                                            <p class="card-text">Пипсы: ${trade.pips.toFixed(1)}</p>`;
                        } else {
                            tradeHtml += `
                                            <p class="card-text">Размер лота: ${trade.lot_size.toFixed(2)}</p>
                                            <p class="card-text">Риск: $${trade.risk_amount.toFixed(2)}</p>
                                            <p class="card-text">Риск %: ${(trade.risk_percent * 100).toFixed(2)}%</p>`;
                        }
                        
                        tradeHtml += `
                                        </div>
                                    </div>
                                    <div class="text-muted small mt-2">
                                        Открыта: ${formatDate(trade.opened_at)}
                                        ${trade.status === 'CLOSED' ? ' | Закрыта: ' + formatDate(trade.closed_at) : ''}
                                    </div>
                                </div>
                            </div>`;
                        
                        tradesContainer.innerHTML += tradeHtml;
                    });
                }
                
                document.getElementById('trades-loader').style.display = 'none';
                tradesContainer.style.display = 'block';
            } catch (error) {
                console.error('Ошибка при загрузке истории сделок:', error);
                document.getElementById('trades-loader').style.display = 'none';
                document.getElementById('trades-container').innerHTML = '<div class="alert alert-danger">Ошибка при загрузке истории сделок</div>';
                document.getElementById('trades-container').style.display = 'block';
            }
        }

        // Сброс демо-счета
        async function resetAccount() {
            try {
                const response = await fetch('/api/reset', {
                    method: 'POST'
                });
                const data = await response.json();
                
                if (data.error) {
                    alert('Ошибка при сбросе счета: ' + data.error);
                } else {
                    alert('Счет успешно сброшен!');
                    // Перезагружаем данные
                    loadAccountInfo();
                    loadStats();
                    loadBalanceChart();
                    loadTrades();
                }
            } catch (error) {
                console.error('Ошибка при сбросе счета:', error);
                alert('Ошибка при сбросе счета: ' + error.message);
            }
        }

        // Обработчики событий
        document.addEventListener('DOMContentLoaded', function() {
            // Загружаем данные при загрузке страницы
            loadAccountInfo();
            loadStats();
            loadBalanceChart();
            loadTrades();
            
            // Модальное окно сброса счета
            const resetModal = new bootstrap.Modal(document.getElementById('resetModal'));
            
            document.getElementById('reset-button').addEventListener('click', function() {
                resetModal.show();
            });
            
            document.getElementById('confirm-reset').addEventListener('click', function() {
                resetModal.hide();
                resetAccount();
            });
            
            // Обновление данных каждые 30 секунд
            setInterval(function() {
                loadAccountInfo();
                loadStats();
                loadTrades();
            }, 30000);
            
            // Обновление графика каждые 5 минут
            setInterval(function() {
                loadBalanceChart();
            }, 300000);
        });
    </script>
</body>
</html>
    """
    
    with open('templates/index.html', 'w') as f:
        f.write(index_html)
    
    logger.info("HTML шаблоны успешно созданы")

if __name__ == "__main__":
    # Создаем шаблоны перед запуском приложения
    create_templates()
    
    # Запускаем приложение
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
