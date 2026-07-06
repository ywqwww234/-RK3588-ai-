"""nn/ 包入口（v2）。

对外稳定接口：
  - FeatureAligner    三模态 1Hz 对齐器
  - InferenceEngine   ONNX 推理（支持 strict / providers / scaler / reload）
  - InferenceMonitor  推理监控 + 数据漂移（生产可观测性）
"""
from .feature_aligner import FeatureAligner
from .inference_engine import InferenceEngine

try:
    from .monitor import InferenceMonitor, DriftDetector
except Exception:
    InferenceMonitor = None
    DriftDetector = None

__all__ = ['FeatureAligner', 'InferenceEngine',
           'InferenceMonitor', 'DriftDetector']
