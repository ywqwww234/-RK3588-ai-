# 基于 RK3588 的边缘 AI 抑郁风险预警系统

本项目面向校园/教室场景，基于 RK3588 边缘计算平台，融合视觉、生理、脑电三类信号，对学生状态进行实时风险评估、告警与可视化展示。仓库结构与模块说明参考了《源程序说明》，并补充了开发和部署时更常用的使用说明。

## 项目概览

- 学生端：实时采集摄像头、心率/HRV、脑电等数据，展示融合风险与 AI 干预结果。
- 家长端：读取历史风险记录，生成趋势分析和阶段性建议。
- 嵌入式端：包含 ESP32 心率/HRV 固件，以及 RK3588 上使用的 RKNN 推理链路。

## 核心目录

```text
.
├── main.py
├── ui_main.py
├── ui_parent.py
├── config.py
├── camera_thread.py
├── local_vision.py
├── vision_yolov8.py
├── physio_thread.py
├── bci_thread.py
├── risk_calculator.py
├── recorder.py
├── models/
├── nn/
├── esp32_c3_hr_fixed/
└── esp32_hrv_edge/
```

## 模块说明

- `main.py`：项目主入口，负责登录、角色分流和线程启动。
- `ui_main.py`：学生端主界面，负责实时监测、图表和 AI 干预展示。
- `ui_parent.py`：家长端界面，负责历史记录与阶段性分析。
- `camera_thread.py`：多模态实时总线，汇总视觉、生理、脑电链路。
- `local_vision.py`：本地视觉分析，输出表情、疲劳和姿态风险。
- `vision_yolov8.py`：统一适配 YOLOv8、RKNN local、RKNN remote 视觉后端。
- `physio_thread.py`：串口读取 ESP32 PPG 数据，提取 BPM 与 HRV 特征。
- `bci_thread.py`：解析 TGAM 脑电协议，输出注意力、冥想度和原始脑波。
- `risk_calculator.py`：多模态风险打分与融合规则。
- `recorder.py`：风险日志加密存储与历史数据读取。

## 运行说明

### 1. 本地运行

```powershell
python main.py
```

也可以直接双击：

- `start_project.bat`：显示控制台，适合调试
- `start_project_silent.bat`：静默启动，适合演示

### 2. 运行前建议检查

- `config.py` 中的 `VISION_BACKEND`
- `CAMERA_SOURCE`
- `PHYSIO_LOCAL_SERIAL` / `PHYSIO_SIM_WHEN_NO_SERIAL`
- `BCI_ENABLE` / `BCI_PREFER_LOCAL_SERIAL`
- `RKNN_REMOTE_HOST` / `RKNN_REMOTE_PORT`

### 3. 模型与依赖

项目依赖 PyQt5、OpenCV、NumPy、SciPy、pandas、pyserial、cryptography，视觉链路还可能需要 `mediapipe`、`ultralytics` 和 RKNN 运行时。

请确认模型文件可用，例如：

- `models/emotion-ferplus-8.onnx`
- `models/yolov8_classroom.pt`
- 对应的 `.rknn`、人脸检测模型和 FaceMesh 资源

### 4. 智谱 API Key

仓库中的智谱调用已经改成从环境变量读取：

```powershell
$env:ZHIPU_API_KEY="你的智谱 API Key"
python main.py
```

未设置时，相关 AI 建议功能会回退到本地逻辑或不可用状态。

## 阅读顺序建议

1. `README.md`
2. `config.py`
3. `main.py`
4. `camera_thread.py`
5. `local_vision.py` / `vision_yolov8.py`
6. `physio_thread.py` / `bci_thread.py`
7. `risk_calculator.py`
8. `ui_main.py` / `ui_parent.py`

## GitHub 发布建议

为避免泄露敏感信息，建议不要提交运行期生成的密钥、日志和临时文件。大于 100 MB 的演示视频建议单独通过 Git LFS 或 GitHub Release 附件发布。

## 附件位置

- 设计报告：`docs/嵌入式芯片与系统设计大赛报告_基于RK3588的边缘ai抑郁风险预警系统.pdf`
- 演示视频：`media/基于RK3588的边缘ai抑郁风险预警系统演示视频.mp4`

其中演示视频已通过 Git LFS 管理，首次克隆仓库时请确保本地安装并启用 Git LFS。

## GitHub 打开说明

GitHub 仓库页对较大的 `pdf` 和 `mp4` 文件通常不会直接在线预览，页面里出现
“Sorry about that, but we can't show files that are this big right now.”
属于正常现象，并不代表文件上传失败或损坏。

### 设计报告怎么打开

1. 进入 `docs/` 目录中的报告文件页面
2. 点击 `View raw` 或右上角下载按钮
3. 浏览器会下载 `pdf` 文件
4. 下载完成后使用浏览器、Adobe Acrobat、Edge 或 WPS Office 打开

### 演示视频怎么打开

1. 进入 `media/` 目录中的视频文件页面
2. 点击 `Raw` 或右上角下载按钮
3. 等待视频完整下载到本地
4. 使用本地播放器打开，例如系统自带播放器、PotPlayer、VLC 等

### 为什么视频不能在 GitHub 页面直接播放

- 本仓库中的视频文件较大
- 视频通过 Git LFS 存储
- GitHub 对这类大文件通常只提供下载，不提供网页内嵌播放

因此，正确使用方式是“下载后本地打开”，而不是在仓库网页里直接预览。
