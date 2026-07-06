import os
import requests
key = os.getenv("ZHIPU_API_KEY", "").strip()
if not key:
    raise SystemExit("Please set ZHIPU_API_KEY before running this script.")
r = requests.post(
    "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    headers={"Authorization": f"Bearer {key}"},
    json={"model": "glm-4-flash", "messages": [{"role": "user", "content": "OK"}]},
)
print(r.status_code)
if r.status_code == 200:
    print("chat OK:", r.json()["choices"][0]["message"]["content"])
else:
    print(r.text[:200])
