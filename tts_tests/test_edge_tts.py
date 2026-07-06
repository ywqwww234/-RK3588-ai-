import subprocess, sys, os

# 安装
print("安装 edge-tts...")
subprocess.run([sys.executable, "-m", "pip", "install", "edge-tts", "-q"], check=True)

# 生成测试音频
text = "你好，这是智谱语音模型的替代方案测试。"
path = "D:/test_edge_tts.mp3"
print(f"生成音频: {text}")
subprocess.run([
    "edge-tts", "--voice", "zh-CN-XiaoxiaoNeural",
    "--text", text,
    "--write-media", path,
], check=True)

size = os.path.getsize(path)
print(f"OK! 文件: {path} 大小: {size} bytes")
print("可以用播放器打开 D:\\test_edge_tts.mp3 试听")
