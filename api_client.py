from aip import AipFace
import base64
import config

class FaceAPIClient:
    def __init__(self):
        self.client = AipFace(config.APP_ID, config.API_KEY, config.SECRET_KEY)

    def analyze_face(self, image_b64):
        """返回表情类型和概率 (type, probability)"""
        options = {'face_field': 'expression'}
        result = self.client.detect(image_b64, 'BASE64', options)
        if result['error_code'] == 0 and result['result']['face_num'] > 0:
            face = result['result']['face_list'][0]
            expr = face['expression']
            return expr['type'], expr['probability']
        else:
            return None, None