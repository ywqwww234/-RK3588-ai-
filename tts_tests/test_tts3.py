import requests
key = "9d88f1c00bbe4ca5aff6071e83a75b13.nzxih6bumd1hgG1f"
url = "https://open.bigmodel.cn/api/paas/v4/audio/speech"

tests = [
    {"model": "glm-4-voice", "input": "你好", "voice": "shunfenger"},
    {"model": "glm-4-voice", "input": "你好"},
    {"model": "glm-4-flash", "input": "你好"},
    {"model": "characterglm", "input": "你好"},
    # OpenAI-compatible format
    {"model": "tts-1", "input": "你好", "voice": "alloy"},
]

for t in tests:
    r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=t)
    print(f"model={t.get('model')} voice={t.get('voice','none')}: {r.status_code} {r.json().get('error',{}).get('message','')[:80]}")
