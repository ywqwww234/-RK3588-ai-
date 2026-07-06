import matplotlib
matplotlib.use('Agg') # 强制使用无头渲染模式，防止 Flask 报错
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, jsonify, request
from recorder import Recorder
import logging
import socket
import pandas as pd

# 修复图表中文豆腐块
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
recorder = Recorder()

USER = "admin"
PASS = "123456"

def check_auth(username, password):
    return username == USER and password == PASS

def authenticate():
    return {'message': "MindRoom Guard: 请输入监护密码"}, 401, {'WWW-Authenticate': 'Basic realm="Login Required"'}

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

@app.route('/')
def index():
    auth = request.authorization
    if not auth or not check_auth(auth.username, auth.password):
        return authenticate()
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    """核心杀招：本地 Python 直接把图表画成图片传给前端，彻底抛弃外部依赖！"""
    try:
        df = recorder.get_decrypted_history()
        if df.empty:
            return jsonify({"status": "empty", "image": ""})

        # 将时间戳字符串转换为 datetime 对象
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # 1. 在内存中用 Matplotlib 画图
        fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
        ax.plot(df['timestamp'], df['risk'], marker='o', linestyle='-', color='#e74c3c', linewidth=2)
        ax.set_ylim(0, 1.0)
        ax.set_title("24小时综合风险趋势 (边缘端硬渲染)", pad=15)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 优化时间轴显示
        fig.autofmt_xdate()
        plt.tight_layout()

        # 2. 将画好的图转成 Base64 图片流
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', transparent=True)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        plt.close(fig) # 释放内存

        # 3. 发送给网页
        return jsonify({"status": "success", "image": img_base64})
    except Exception as e:
        print(f"API绘图错误: {e}")
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    local_ip = get_local_ip()
    print("\n" + "="*60)
    print("🚀 MindRoom Guard 局域网监护服务已启动 (原生图像渲染模式)！")
    print("==========================================================")
    print(f"1. 【本机测试】在浏览器输入: http://127.0.0.1:5000")
    print(f"2. 【手机测试】确保手机连上局域网，输入: http://{local_ip}:5000")
    print(f"\n🔐 安全登录账号: {USER}  |  密码: {PASS}")
    print("==========================================================\n")
    app.run(host='0.0.0.0', port=5000, debug=False)