import urllib.request
import os
import ssl

# 忽略本地 VPN 可能导致的 SSL 证书报错
ssl._create_default_https_context = ssl._create_unverified_context

# 1. 创建 Flask 专用的本地静态资源文件夹
if not os.path.exists('static'):
    os.makedirs('static')

print("正在为您拉取本地图表引擎，打造 100% 离线环境...")
# 使用国内镜像源下载
url = "https://registry.npmmirror.com/echarts/5.5.0/files/dist/echarts.min.js"

try:
    urllib.request.urlretrieve(url, "static/echarts.min.js")
    print("✅ 太棒了！本地引擎下载成功！文件已安全存放于 static/echarts.min.js")
    print("现在您的系统即使拔掉网线，也完全可以独立运行了！")
except Exception as e:
    print(f"❌ 下载失败，请检查网络: {e}")