import logging
import os
import requests
import uuid

async def handle_game_callback_query(bot, update, reports):
    """
    Обрабатывает callback_query от игровых запросов в Telegram
    
    Параметры:
    - bot: экземпляр бота Telegram
    - update: объект Update из python-telegram-bot
    - reports: словарь с отчетами {token: (html, expire_time)}
    
    Возвращает:
    - True если callback_query был для игры и обработан
    - False в противном случае
    """
    if not update.callback_query or update.callback_query.game_short_name != 'backtest_report':
        return False
        
    callback_query_id = update.callback_query.id
    user_id = update.callback_query.from_user.id
    
    # Получаем start_parameter из callback_query
    start_parameter = getattr(update.callback_query, 'start_parameter', None)
    
    # Находим подходящий токен отчета
    token = None
    
    # 1. Если есть start_parameter и он существует в reports - используем его
    if start_parameter and start_parameter in reports:
        token = start_parameter
        logging.info(f"Найден отчет по start_parameter: {start_parameter}")
    else:
        # 2. Ищем последний отчет для текущего чата
        chat_id = update.callback_query.message.chat.id if update.callback_query.message else user_id
        
        # Отфильтруем отчеты для текущего пользователя по времени создания (самый новый)
        recent_reports = sorted([
            (t, exp_time) 
            for t, (_, exp_time) in reports.items()
        ], key=lambda x: x[1], reverse=True)
        
        if recent_reports:
            # Берем самый свежий отчет
            token = recent_reports[0][0]
            logging.info(f"Найден свежий отчет: {token}")
        else:
            # 3. Если ничего не нашли, вернем ошибку
            logging.warning(f"Не найдено отчетов для пользователя {chat_id}!")
            await bot.answer_callback_query(
                callback_query_id=callback_query_id,
                text="Отчет не найден. Пожалуйста, запустите бэктест еще раз.",
                show_alert=True
            )
            return True
    
    if token in reports:
        # Получаем URL нашего сервера для игры
        server_url = os.environ.get('SERVER_URL', 'https://trading-bot-i36i.onrender.com')
        game_url = f"{server_url}/game_report?token={token}"
        
        # Отправляем отчет на сервер игры через API
        html, _ = reports[token]
        send_report_to_game_server(token, html)
        
        logging.info(f"Обрабатываю callback_query игры для пользователя {user_id}, токен: {token}, URL: {game_url}")
        
        # Отправляем URL игры в ответ на callback_query
        try:
            await bot.answer_callback_query(
                callback_query_id=callback_query_id,
                url=game_url
            )
            logging.info(f"URL игры успешно отправлен: {game_url}")
        except Exception as e:
            logging.error(f"Ошибка при ответе на callback_query: {e}", exc_info=True)
    else:
        logging.warning(f"Токен {token} не найден в словаре отчетов")
        await bot.answer_callback_query(
            callback_query_id=callback_query_id,
            text="Отчет не найден. Пожалуйста, запустите бэктест еще раз.",
            show_alert=True
        )
    return True

def send_report_to_game_server(token, html):
    """Отправляет отчет на сервер игры через API"""
    try:
        # Получаем URL сервера игры
        server_url = os.environ.get('GAME_SERVER_URL', 'https://trading-bot-i36i.onrender.com')
        save_url = f"{server_url}/save_report"
        
        # Отправляем JSON с токеном и HTML
        data = {"token": token, "html": html}
        response = requests.post(save_url, json=data)
        
        if response.status_code == 200:
            logging.info(f"Отчет успешно отправлен на сервер игры с токеном {token}")
            return True
        else:
            logging.error(f"Ошибка при отправке отчета на сервер игры: {response.status_code}")
            return False
    except Exception as e:
        logging.error(f"Ошибка при отправке отчета на сервер игры: {e}", exc_info=True)
        return False 