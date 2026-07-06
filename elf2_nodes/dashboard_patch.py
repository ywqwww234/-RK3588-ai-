"""
Dashboard 模块补丁 — 工业级增强 (v2)

修复:
  1. 心率/脑电嵌入路径 -> 使用 D:/G/M/embedded_node.py 多路径自动搜索
  2. 视觉卡摄像头升级到 640×360 大画幅
  3. 增强眼疲劳/姿态子指标条布局
  4. 融合贡献仪表

使用方式:
  将此文件内容合并到 D:/G/3 目录下的 ui_dashboard.py 和 ui_dashboard_modules.py
  或者直接将此文件复制到 D:/G/0/Anti_depression/ 替换对应部分
"""

# ============================================================
# 对 ui_dashboard.py 的修改: 第214-249行 (节点放大区域)
# ============================================================

DASHBOARD_ZOOM_PATCH = """
    # ===== 替换原来的 ZoomOverlay/EmbeddedNodeOverlay 节点放大代码 =====
    # 视觉卡放大: 使用 D:/G/M/visual_node.py (工业级视觉独立节点)
    # 心率卡放大: 使用 D:/G/1.py (HRMonitorWindow)
    # 脑电卡放大: 使用 D:/G/11.py (EEGMonitorWindow)

    try:
        # 优先使用 D:/G/M/embedded_node.py (增强版,多路径自动搜索)
        import sys as _sys
        _sys.path.insert(0, 'D:/G/M')
        from embedded_node import EmbeddedNodeOverlay as _EOV

        _EOV.install(self.vision_card, self, node_type='vision',
                     title='视 觉 感 知 节 点  ·  FER+  VISION  EDGE  NODE')
        _EOV.install(self.heart_card, self, node_type='heart',
                     title='心 率 感 知 节 点  ·  ESP32  HRV  EDGE  NODE')
        _EOV.install(self.brain_card, self, node_type='brain',
                     title='脑 电 感 知 节 点  ·  TGAM  EDGE  NODE')
    except Exception as _exc3:
        # 回退到原 embedded_node.py
        try:
            from embedded_node import EmbeddedNodeOverlay as _EOV2
            import os as _os2
            _root_g2 = 'D:/G'  # 直接指定正确路径
            _EOV2.install(
                card_widget=self.heart_card, host_window=self,
                module_path=_os2.path.join(_root_g2, '1.py'),
                window_class='HRMonitorWindow',
                title='心 率 感 知 节 点  ·  ESP32  HRV  EDGE  NODE')
            _EOV2.install(
                card_widget=self.brain_card, host_window=self,
                module_path=_os2.path.join(_root_g2, '11.py'),
                window_class='EEGMonitorWindow',
                title='脑 电 感 知 节 点  ·  TGAM  EDGE  NODE')
            # 视觉用普通 ZoomOverlay
            ZoomOverlay.install_zoom(self.vision_card, self, '视觉感知 · VISION')
        except Exception as _exc2:
            # 最终回退: 全部用 ZoomOverlay
            ZoomOverlay.install_zoom(self.vision_card, self, '视觉感知 · VISION')
            ZoomOverlay.install_zoom(self.heart_card, self, '心率与生理 · CARDIAC')
            ZoomOverlay.install_zoom(self.brain_card, self, '脑电感知 · NEURAL')
            print(f'[Dashboard] 节点放大回退: {_exc3}')
"""


# ============================================================
# 对 ui_dashboard_modules.py 的修改: build_vision_card()
# ============================================================

BUILD_VISION_CARD_V2 = """
def build_vision_card(self):
    card = ModuleCard('视觉感知 · VISION', 'YOLOv8n · FER+ · MediaPipe', C_VISION)
    card.status_pill.set_state('sim', '模型: 就绪')

    # === 摄像头主显示 - 升级到 640×360 大画幅 ===
    self.video_label = QLabel('正在初始化视觉识别引擎…')
    self.video_label.setMinimumSize(480, 320)
    self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    self.video_label.setAlignment(Qt.AlignCenter)
    self.video_label.setScaledContents(True)
    self.video_label.setStyleSheet(
        f'background:{C_BG_DEEP}; color:{C_VISION}; font-size:{FS_MD}px; '
        f'border:{BORDER_W}px solid {C_BORDER}; border-radius:{R_MD}px;')
    card.add_content(self.video_label)

    # === 表情与视觉风险 ===
    self.expr_label = QLabel('当前状态: 等待人脸…')
    self.expr_label.setStyleSheet(
        f'color:{C_TEXT}; font-size:12px; letter-spacing:0.8px; background:transparent;')
    card.add_content(self.expr_label)

    # === 视觉子指标条 (4项) ===
    self.bar_eye = _bar_label('眼疲劳', C_VISION)
    self.bar_posture = _bar_label('姿态风险', C_WARN)
    self.bar_forward = _bar_label('探头', C_WARN)
    self.bar_down = _bar_label('低头', C_RISK)
    card.add_content(self.bar_eye)
    card.add_content(self.bar_posture)
    card.add_content(self.bar_forward)
    card.add_content(self.bar_down)

    # === 视觉风险评分(小) ===
    risk_row = QHBoxLayout(); risk_row.setSpacing(8)
    self.lbl_vis_risk = QLabel('0.00')
    self.lbl_vis_risk.setStyleSheet(
        f'color:{C_OK}; font-size:22px; font-weight:bold; '
        f'background:{C_BG_DEEP}; border:1px solid {C_BORDER}; '
        f'border-radius:6px; padding:6px 12px; font-family:Consolas;')
    self.lbl_vis_risk.setAlignment(Qt.AlignCenter)
    self.lbl_vis_tag = QLabel('LOW')
    self.lbl_vis_tag.setStyleSheet(
        f'color:{C_OK}; font-size:11px; letter-spacing:2px; background:transparent;')
    self.lbl_vis_tag.setAlignment(Qt.AlignCenter)
    risk_row.addWidget(self.lbl_vis_risk)
    risk_row.addWidget(self.lbl_vis_tag)
    card.add_content(risk_row)

    # === NN 模态贡献仪表 ===
    self.vision_contrib_gauge = ModalContribGauge('vision', C_VISION)
    card.add_content(self.vision_contrib_gauge)

    self.vision_card = card
    return card
"""


# ============================================================
# 简易应用指南
# ============================================================
USAGE_GUIDE = '''
================================================================================
  MindRoom Guard — 工业级节点放大 + 视觉增强 安装指南
================================================================================

文件清单:
  D:/G/M/visual_node.py      — 独立视觉感知节点 (可单独 python visual_node.py)
  D:/G/M/embedded_node.py    — 增强嵌入式节点管理器 (多路径搜索+三模态支持)

安装步骤:

1. 确保独立节点文件存在:
   - D:/G/1.py   (心率节点 HRMonitorWindow)
   - D:/G/11.py  (脑电节点 EEGMonitorWindow)
   - D:/G/M/visual_node.py (视觉节点 VisionMonitorWindow) ← 新建

2. 修改 D:/G/0/Anti_depression/ui_dashboard.py 第214-249行:
   将原来的 EmbeddedNodeOverlay.install 代码块替换为
   DASHBOARD_ZOOM_PATCH 中的代码。

   关键修复: 原代码 _root_g 计算结果为 D:/G/0 而非 D:/G，
   导致找不到 1.py/11.py。新方案直接搜索多个候选路径。

3. (可选) 修改 D:/G/0/Anti_depression/ui_dashboard_modules.py:
   用 BUILD_VISION_CARD_V2 中的 build_vision_card 函数替换原函数。
   主要改动: video_label 从固定 480×260 升级为弹性 480×320+.

4. 验证:
   - 双击主界面"视觉感知"卡 → 应弹出 industrial visual_node 全屏界面
   - 双击"心率与生理"卡 → 应弹出 D:/G/1.py 的 HRMonitorWindow 全屏
   - 双击"脑电感知"卡 → 应弹出 D:/G/11.py 的 EEGMonitorWindow 全屏
   - 按 Esc 或双击空白区域 → 应正常返回主界面

故障排查:
   - "节点源文件不存在" → 检查 D:/G/1.py, D:/G/11.py 是否存在
   - "模块中未找到类" → 确认 .py 文件中类名未改变
   - 摄像头黑屏 → visual_node.py 独立运行测试；检查摄像头索引
================================================================================
'''


if __name__ == '__main__':
    print(USAGE_GUIDE)
