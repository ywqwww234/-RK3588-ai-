import requests
key = "9d88f1c00bbe4ca5aff6071e83a75b13.nzxih6bumd1hgG1f"
url = "https://open.bigmodel.cn/api/paas/v4/audio/speech"

for voice in ["shunfenger", "zhiyu", "zhitian", "zhixiang", "xiaobei", "shunfeng", None]:
    body = {"model": "glm-4-voice", "input": "你好", "response_format": "wav"}
    if voice:
        body["voice"] = voice
    r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=body)
    print(f"voice={voice}: {r.status_code} {r.json().get('error',{}).get('message','')}")
    if r.status_code == 200:
        print("  SUCCESS!")
        break
