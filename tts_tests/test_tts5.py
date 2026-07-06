import requests
key = "9d88f1c00bbe4ca5aff6071e83a75b13.nzxih6bumd1hgG1f"

# 试1: chat API + glm-4-voice + modalities
print("=== 试1: glm-4-voice chat with audio modalities ===")
r = requests.post(
    "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    headers={"Authorization": f"Bearer {key}"},
    json={
        "model": "glm-4-voice",
        "messages": [{"role": "user", "content": "你好"}],
        "modalities": ["text", "audio"],
        "audio": {"voice": "alloy", "format": "wav"},
    },
)
print(f"status={r.status_code}")
print(r.text[:500])

# 试2: audio/speech + 不指定 voice
print("\n=== 试2: audio/speech no voice ===")
r2 = requests.post(
    "https://open.bigmodel.cn/api/paas/v4/audio/speech",
    headers={"Authorization": f"Bearer {key}"},
    json={"model": "glm-4-voice", "input": "你好", "response_format": "wav"},
)
print(f"status={r2.status_code}")
print(r2.text[:500])

# 试3: 查可用模型列表
print("\n=== 试3: list models ===")
r3 = requests.get(
    "https://open.bigmodel.cn/api/paas/v4/models",
    headers={"Authorization": f"Bearer {key}"},
)
print(f"status={r3.status_code}")
if r3.status_code == 200:
    for m in r3.json().get("data", []):
        mid = m.get("id", "")
        if "voice" in mid.lower() or "audio" in mid.lower() or "tts" in mid.lower() or "speech" in mid.lower():
            print(f"  AUDIO MODEL: {mid}")
