from flask import Flask, request, abort, Response
import threading
import time
import json

app = Flask(__name__)
reports = {}  # {token: (html, expire_time)}

# Фоновая очистка устаревших отчётов (старше 30 минут)
def cleanup_reports():
    while True:
        now = time.time()
        to_delete = [token for token, (_, exp) in reports.items() if exp < now]
        for token in to_delete:
            del reports[token]
        time.sleep(60)

threading.Thread(target=cleanup_reports, daemon=True).start()

@app.route('/save_report', methods=['POST'])
def save_report():
    data = request.get_json()
    token = data['token']
    html = data['html']
    expire_time = time.time() + 1800  # 30 минут жизни
    reports[token] = (html, expire_time)
    return {'status': 'ok'}

@app.route('/game_report')
def game_report():
    token = request.args.get('token')
    if not token or token not in reports:
        return abort(404, 'Report not found')
    html, _ = reports[token]
    return Response(html, mimetype='text/html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080) 
