import requests
key = "9d88f1c00bbe4ca5aff6071e83a75b13.nzxih6bumd1hgG1f"
url = "https://open.bigmodel.cn/api/paas/v4/audio/speech"

# Zhipu可能的音色: 数字编号 + 中文名
voices = [
    "101001", "101002", "101003", "101004",
    "shunfenger-1", "shunfeng",
    "B086", "A086",
    "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural",
    "xiaoling", "xiaoyan", "xiaomei",
]
for v in voices:
    r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json={"model":"glm-4-voice","input":"你好","voice":v,"response_format":"wav"})
    ok = r.status_code == 200
    print(f"voice={v}: {r.status_code} {'OK!' if ok else r.json().get('error',{}).get('message','')[:60]}")
    if ok:
        break

# Also try without model field
print("--- try without voice ---")
r2 = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json={"model":"glm-4-voice","input":"你好","response_format":"wav"})
print(f"no voice: {r2.status_code} {r2.json().get('error',{}).get('message','')[:80]}")
