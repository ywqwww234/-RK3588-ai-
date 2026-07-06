from aip import AipFace
import base64
import cv2
import time

# 你的百度API配置
APP_ID = '你的APP_ID'
API_KEY = '你的API_KEY'
SECRET_KEY = '你的SECRET_KEY'

client = AipFace(APP_ID, API_KEY, SECRET_KEY)

# 打开摄像头，预览3秒后拍照
cap = cv2.VideoCapture(0)
print("摄像头已打开，请对准人脸，3秒后拍照...")
time.sleep(3)
ret, frame = cap.read()
cap.release()

if not ret:
    print("无法获取摄像头画面")
    exit()

# 显示拍摄的照片（可选）
cv2.imshow('Captured', frame)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# 保存图片用于检查
cv2.imwrite('debug_face.jpg', frame)

# 编码为base64
with open('debug_face.jpg', 'rb') as f:
    image = base64.b64encode(f.read()).decode()

# 调用百度人脸检测
result = client.detect(image, 'BASE64', options={'face_field': 'expression'})
print("API返回完整结果：")
print(result)

if result['error_code'] == 0:
    print(f"检测到人脸数：{result['result']['face_num']}")
    if result['result']['face_num'] > 0:
        face = result['result']['face_list'][0]
        expr = face['expression']
        print(f"表情类型: {expr['type']}, 概率: {expr['probability']}")
    else:
        print("未检测到人脸，请检查摄像头是否对准人脸，或光线是否充足。")
else:
    print(f"API调用失败，错误码：{result['error_code']}，错误信息：{result['error_msg']}")