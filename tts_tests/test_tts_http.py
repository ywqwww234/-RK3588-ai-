import requests
key = "9d88f1c00bbe4ca5aff6071e83a75b13.nzxih6bumd1hgG1f"
r = requests.post(
    "https://open.bigmodel.cn/api/paas/v4/audio/speech",
    headers={"Authorization": f"Bearer {key}"},
    json={"model": "glm-tts", "input": "你好呀欢迎测试语音", "voice": "female", "response_format": "wav"},
)
print("status:", r.status_code)
if r.status_code == 200:
    with open("D:/test_tts.wav", "wb") as f:
        f.write(r.content)
    print("OK, saved", len(r.content), "bytes")
else:
    print(r.text[:300])
