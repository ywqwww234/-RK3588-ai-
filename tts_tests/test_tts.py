import os
import requests

key = os.getenv("ZHIPU_API_KEY", "").strip()
if not key:
    raise SystemExit("Please set ZHIPU_API_KEY before running this script.")

r = requests.post(
    "https://open.bigmodel.cn/api/paas/v4/audio/speech",
    headers={"Authorization": f"Bearer {key}"},
    json={"model": "glm-4-voice", "input": "你好测试", "voice": "shunfenger", "response_format": "wav"},
)
print("status:", r.status_code)
if r.status_code == 200:
    with open("D:/test_tts.wav", "wb") as f:
        f.write(r.content)
    print("wav saved, size:", len(r.content))
else:
    print(r.text[:300])
