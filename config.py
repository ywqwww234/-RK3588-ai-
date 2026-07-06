"""
全局配置中心。

集中维护 UI 尺寸、风险阈值、硬件链路、视觉后端和大模型等运行参数。
部署时建议优先修改本文件，而不是把环境差异散落到业务代码中。
"""

WINDOW_WIDTH = 1680
WINDOW_HEIGHT = 980
REAL_TIME_INTERVAL = 2
RECORD_INTERVAL = 10  
RISK_THRESHOLD_LOW = 0.3
RISK_THRESHOLD_MEDIUM = 0.6
RISK_THRESHOLD_HIGH = 0.8
ALERT_CONSECUTIVE_HIGH = 3

# 融合策略参数：视觉一票否决阈值
VISUAL_VETO_THRESHOLD = 0.6

# 社交电量动态参数（优化：高风险下降更快，恢复期回升更明显）
BATTERY_DRAIN_THRESHOLD = 0.45
BATTERY_DRAIN_STEP = 0.025
BATTERY_RECOVER_THRESHOLD = 0.34
BATTERY_RECOVER_STEP = 0.012

# 电量滞后保护：连续达到条件后才放电/回充，避免抖动误触
BATTERY_DRAIN_CONSECUTIVE = 2
BATTERY_RECOVER_CONSECUTIVE = 2

# 风险指数平滑参数：上升快、下降可回落
RISK_EMA_ALPHA_RISE = 0.55
RISK_EMA_ALPHA_FALL = 0.25

# ===== Vision 阶段A门控参数 =====
VISION_CONF_MISS_GOOD = 2
VISION_CONF_EAR_GOOD = 0.20
VISION_CONF_EXPR_HOLD_MIN = 0.45
VISION_GATE_DECAY_POSTURE = 0.94
VISION_GATE_DECAY_FORWARD = 0.95
VISION_GATE_DECAY_DOWN = 0.95
VISION_LOW_CONF_APPLY_AFTER = 20
VISION_LOW_CONF_WEIGHT_SCALE = 0.4

# 无串口时是否启用生理伪数据（用于本地联调）
PHYSIO_SIM_WHEN_NO_SERIAL = True
# 心率：本机直连 1.py 链路（不走 esp32_receiver / HTTP）
PHYSIO_LOCAL_SERIAL = True
PHYSIO_PORT = None          # None = 自动找 ESP32/USB Serial（排除蓝牙 TGAM 口）
PHYSIO_BAUD = 115200
PHYSIO_TICK_MS = 50
PHYSIO_FEATURE_INTERVAL_SEC = 0.5
ESP32_RECEIVER_ENABLE = False  # 单机板子+学生端同机时关闭 Flask 5001
# True：启动学生端时另开 1.py 完整窗口（推荐）；与嵌入式 PhysioThread 二选一，避免抢串口
PHYSIO_SHOW_1PY_WINDOW = False
PHYSIO_EMBED_THREAD = True    # 主系统启 PhysioThread，直喂 NN

# ===== 远端数据源配置 =====
# B电脑本机拉取地址（由 A电脑 推送到 B电脑 的 esp32_receiver 服务后，UI从本机读取）
ESP32_REMOTE_URL = "http://127.0.0.1:5001/esp32/latest"
ESP32_REMOTE_INTERVAL_MS = 1000
EEG_REMOTE_URL = "http://127.0.0.1:5001/eeg/latest"
EEG_REMOTE_INTERVAL_MS = 800
EEG_POOR_SIGNAL_MAX = 99

# ===== UI 刷新分层节奏 =====
UI_TICK_FAST_MS = 33    # ~30Hz 波形/动画
UI_TICK_MEDIUM_MS = 100 # 10Hz 指标
UI_TICK_SLOW_MS = 1000  # 1Hz 风险/状态

# ===== ELF2 本地部署配置 =====
CAMERA_SOURCE = "local"
CAMERA_STREAM_PORT = 9998
DATA_DIR = "/home/root/MindRoom/data"
RKNN_REMOTE_HOST = "192.168.137.100"
RKNN_REMOTE_PORT = 9999
RKNN_REMOTE_TIMEOUT = 10.0

# ===== 摄像头调试叠加层 =====
CAMERA_DEBUG_OVERLAY = True

# ===== Day1-2: YOLOv8 视觉后端开关 =====
VISION_BACKEND = "rknn_local"   # legacy | yolov8 | rknn_remote | rknn_local
YOLOV8_MODEL_PATH = "models/yolov8_classroom.pt"
YOLOV8_CONF = 0.55
YOLOV8_IMG_SIZE = 640

# ===== Day4.5: YOLO 稳定性参数（教室场景预设） =====
# 场景特点：多人干扰、走动频繁、光照变化较大，优先降低误报
YOLO_INFER_EVERY_N = 1          # 每帧推理（便于稳定 yolo_det+mesh）
VISION_EXPR_VOTE_WINDOW = 3     # 情绪投票帧数（越小越灵敏，3~4 推荐）
VISION_FACE_DETECTOR = "yunet"    # yunet | haar | person_crop（FER 用人脸框）
# FER+ 预处理：nchw + 减均值更接近官方；hwc 为旧 blobFromImage
VISION_FER_INPUT_LAYOUT = "nchw"  # nchw | hwc
VISION_FER_PREPROCESS = "mean128"  # mean128（FER+常用）| scale255
VISION_FER_DEBUG = False        # 详细 top3 调试（每 30 帧）
VISION_EXPR_LOG_TERMINAL = True # 终端打印最终表情（变化时或按间隔）
VISION_EXPR_LOG_INTERVAL_SEC = 0.35  # 同表情最短打印间隔（秒）
VISION_FER_NEUTRAL_MARGIN = 0.28  # neutral 与第二名差距小于此则改取第二名
VISION_FER_NEUTRAL_RELAX_GAP = 0.50  # neutral 仍第一但第二名情绪阈值且差距小于此 → 用第二名
VISION_FER_SECOND_MIN = 0.10
VISION_FER_OVERRIDE_NEUTRAL = True   # top1=neutral 时按 happy/负类概率抬 smile/negative
VISION_FER_HAPPY_OVERRIDE_MIN = 0.10
VISION_FER_NEG_OVERRIDE_MIN = 0.18   # sad+anger+fear+disgust 和
VISION_FER_ASSIST_WHEN_NEUTRAL_FER_TOP = True  # FER 第一是 neutral 时也允许几何 assist
# 几何 assist：配合 ASSIST_WHEN_UNCERTAIN 时，仅模型概率低才启用
VISION_FER_SMILE_ASSIST = True
VISION_FER_SMILE_BLEND_THRESH = 0.35
VISION_FER_SMILE_BLOCK_MOUTH_OPEN = 0.42
VISION_FER_SAD_DOMINANT = True
VISION_FER_SAD_DOMINANT_MIN = 0.18
VISION_FER_FROWN_ASSIST = True
VISION_FER_FROWN_THRESH = 0.32
VISION_FER_NEG_FER_MIN = 0.14
VISION_FER_SURPRISE_ASSIST = True
VISION_FER_MOUTH_OPEN_THRESH = 0.38
# 模型不确定时（最高类概率低于阈值）才启用上面 assist
VISION_FER_ASSIST_WHEN_UNCERTAIN = True
VISION_FER_UNCERTAIN_THRESH = 0.58  # 略放宽，assist 更易介入（更灵敏）
# 与采集一致：FER 用整幅画面 YuNet 找脸（不限制在人框内）
VISION_FER_USE_FULLFRAME_FACE = True
VISION_POSTURE_HIGH_GATE = 0.28 # 姿态触发阈值更保守
VISION_POSTURE_HIGH_NEED = 8    # 连续8帧异常才确认触发
VISION_POSTURE_LOW_RELEASE = 5  # 连续5帧低值后释放

# ===== Week3: YOLO 路径 + 人脸 ROI + MediaPipe =====
VISION_YOLO_USE_FACE_ROI = True       # 情绪 FER 用人脸上半区/ Haar 脸框，不用整身框
VISION_YOLO_USE_MEDIAPIPE = True      # 在脸 ROI 上跑 Face Mesh（EAR / 低头 / 探头）
VISION_MEDIAPIPE_STRIDE = 1           # 每 N 帧跑一次 mesh（1=每帧，2=隔帧）
VISION_MESH_HOLD_FRAMES = 15          # mesh 成功后 MODE 保持 +mesh 的帧数（防一闪而过）
VISION_MESH_TRY_FULL_FRAME = True     # 脸 ROI 失败时在整帧上再试一次 Face Landmarker

# ===== 脑机接口 (BCI / TGAM) =====
# 本地优先：尝试自动识别 TGAM 蓝牙串口；失败回退模拟；同时仍允许远端兜底
# 若 BCI_FORCE_REMOTE=True，则 B 电脑仅使用 Wi-Fi 远端脑电，不启动本地串口 BCI。
BCI_ENABLE = True
# 同机板子：本地 TGAM 串口直连；USB 蓝牙未配对时走模拟波形
BCI_FORCE_REMOTE = False
BCI_SIM_WHEN_FORCE_REMOTE = True   # ELF2 未配对 TGAM 时模拟脑电
BCI_PREFER_LOCAL_SERIAL = True
BCI_PORT = None           # TGAM 蓝牙未识别，走模拟。配好后改为 "/dev/rfcomm0"
BCI_BAUD = 57600
BCI_PUSH_INTERVAL_MS = 800
BCI_FALLBACK_TO_REMOTE = False
# 双击脑电卡放大 = 11.py / eeg_node_real.py 全屏；False 时主界面嵌 BCIThread 模拟
BCI_EMBED_THREAD = True

# ===== 智谱 AI 干预 / 微调模型 =====
# 微调完成后将 model id 填入；留空则使用下方默认 glm-4-flash / glm-4
ZHIPU_FT_MODEL_REALTIME = ""      # 例: "your-ft-flash-xxxx"
ZHIPU_FT_MODEL_DEEP = ""          # 深度分析仍建议用全量 glm-4，可不填
ZHIPU_LLM_SAVE_LAST_CONTEXT = True
ZHIPU_LLM_ONLINE_SAMPLES = True   # 脱敏样本追加到 data/zhipu_online_samples.jsonl
ZHIPU_LLM_REWRITE_ON_LOW_QUALITY = True
