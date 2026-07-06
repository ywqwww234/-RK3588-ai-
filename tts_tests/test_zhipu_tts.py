import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "zai-sdk", "-q"], check=True)

from zai import ZhipuAiClient
client = ZhipuAiClient(api_key="9d88f1c00bbe4ca5aff6071e83a75b13.nzxih6bumd1hgG1f")
response = client.audio.speech(
    model="glm-tts",
    input="你好呀，欢迎来到智谱开放平台，这是语音测试。",
    voice="female",
    response_format="wav",
    speed=1.0,
    volume=1.0,
)
response.stream_to_file("D:/test_zhipu_tts.wav")
print("OK, saved to D:/test_zhipu_tts.wav")
