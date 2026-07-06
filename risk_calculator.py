"""
多模态风险评分规则。

负责将视觉、生理、脑电三条链路的特征统一映射为 0~1 风险值，并输出融合结果。
"""

import config

def calculate_visual_risk(expr_type, expr_prob):
    """
    单模态1：视觉表情风险 (0-1)
    兼容多种标签体系，并对负向表情给出更敏感的风险抬升。
    """
    if expr_type is None:
        return 0.2

    expr = str(expr_type).lower()
    prob = max(0.0, min(1.0, float(expr_prob) if expr_prob is not None else 0.0))
    base_risk = 0.2

    # 明确正向类别
    if expr in ("laugh", "happy"):
        risk = 0.08 + 0.08 * (1.0 - prob)
    elif expr in ("smile", "neutral"):
        risk = base_risk + 0.15 * (1.0 - prob)

    # 负向类别：显著提高敏感度
    elif expr in ("negative", "sad", "angry", "anger", "fear", "disgust", "none", "contempt"):
        # 注意：你当前模型频繁输出 none，这里将其按负向高敏处理
        risk = base_risk + prob * 0.85

    else:
        risk = base_risk + prob * 0.40

    return max(0.0, min(1.0, risk))

def calculate_hrv_risk(rmssd, sdnn, hf, lf_hf, hr):
    """
    单模态2：心率变异性(HRV)风险 (0-1)
    严格按照论文中的 HRV 评分表进行计算
    """
    score = 0
    # RMSSD
    if rmssd <= 34: score += 2
    elif 35 <= rmssd <= 49: score += 1
    
    # SDNN
    if sdnn <= 69: score += 2
    elif 70 <= sdnn <= 99: score += 1
    
    # HF
    if hf <= 299: score += 2
    elif 300 <= hf <= 599: score += 1
    
    # LF/HF
    if lf_hf >= 4.1: score += 2
    elif 2.6 <= lf_hf <= 4.0: score += 1
    
    # 平均心率
    if hr >= 86: score += 2
    elif 76 <= hr <= 85: score += 1

    # 总分 0-10，直接映射为 0.0 - 1.0 的风险系数
    hrv_risk = score / 10.0
    return max(0.0, min(1.0, hrv_risk))

def calculate_eeg_risk(attention, meditation):
    """
    单模态3：脑电(EEG)风险 (0-1)
    严格按照论文中的脑电专注度/放松度阈值计算
    """
    # 专注度风险计算
    if attention <= 39: 
        att_risk = 0.8       # 显著低下
    elif 40 <= attention <= 59: 
        att_risk = 0.5       # 明显下降
    else: 
        att_risk = 0.2       # 稳定

    # 放松度风险计算
    if meditation <= 29 or meditation > 85: 
        med_risk = 0.8       # 两极化
    elif 30 <= meditation <= 39 or 71 <= meditation <= 85: 
        med_risk = 0.5       # 轻度失衡
    else: 
        med_risk = 0.2       # 平衡

    # 综合脑电风险（取均值）
    eeg_risk = (att_risk + med_risk) / 2.0
    return max(0.0, min(1.0, eeg_risk))

def calculate_total_risk(visual_risk, hrv_risk, eeg_risk):
    """
    【核心融合算法】
    三模态加权融合 + 木桶短板/一票否决：
    1) 先算加权均值风险 avg_risk
    2) 若视觉风险 >= 0.6，则总风险至少不低于视觉风险
       （恶劣表情/姿态被捕捉到时，整体风险立即拉高）
    """
    avg_risk = (visual_risk * 0.35) + (hrv_risk * 0.40) + (eeg_risk * 0.25)

    veto_th = float(getattr(config, 'VISUAL_VETO_THRESHOLD', 0.6))
    if visual_risk >= veto_th:
        total_risk = max(avg_risk, visual_risk)
    else:
        total_risk = avg_risk

    return max(0.0, min(1.0, total_risk))

def get_risk_level(risk):
    if risk is None:
        return "未知"
    
    if risk >= config.RISK_THRESHOLD_HIGH:
        return "极高风险"
    elif risk >= config.RISK_THRESHOLD_MEDIUM:
        return "高风险"
    elif risk >= config.RISK_THRESHOLD_LOW:
        return "中等风险"
    else:
        return "低风险"
