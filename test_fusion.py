import time
from mock_sensors import MockHardwareSensors
from risk_calculator import calculate_visual_risk, calculate_hrv_risk, calculate_eeg_risk, calculate_total_risk, get_risk_level

def run_test():
    # 初始化模拟传感器
    sensor = MockHardwareSensors()
    
    # 我们要测试的三种极端/典型场景
    test_scenarios = ["normal", "stressed", "depressed"]
    
    # 既然没有接摄像头，我们为三种场景手动提供三个假表情
    # normal: 微笑; stressed: 没表情; depressed: 负面表情
    visual_mocks = [
        ("smile", 0.90),      
        ("none", 0.85),       
        ("negative", 0.88)    
    ]

    print("="*40)
    print("🚀 MindRoom Guard 多模态融合算法测试启动")
    print("="*40)

    for i, state in enumerate(test_scenarios):
        print(f"\n▶ 正在模拟用户状态: 【{state.upper()}】")
        sensor.set_simulation_state(state)

        # 1. 采集“假”数据
        hrv_data = sensor.get_hrv_data()
        eeg_data = sensor.get_eeg_data()
        vis_expr, vis_prob = visual_mocks[i]

        # 2. 分别计算各单模态风险分
        vis_risk = calculate_visual_risk(vis_expr, vis_prob)
        # 字典解包直接传参，对应 rmssd, sdnn, hf, lf_hf, hr
        hrv_risk = calculate_hrv_risk(**hrv_data) 
        # 对应 attention, meditation
        eeg_risk = calculate_eeg_risk(**eeg_data)

        # 3. 核心大融合！
        total_risk = calculate_total_risk(vis_risk, hrv_risk, eeg_risk)
        level = get_risk_level(total_risk)

        # 4. 打印极其硬核的中间过程
        print(f"  👁️ 视觉模态 (权重 0.35) -> 识别: {vis_expr}, 风险分: {vis_risk:.2f}")
        print(f"  🫀 生理模态 (权重 0.40) -> HRV(RMSSD): {hrv_data['rmssd']:.1f}, 风险分: {hrv_risk:.2f}")
        print(f"  🧠 脑电模态 (权重 0.25) -> 专注度: {eeg_data['attention']}, 放松度: {eeg_data['meditation']}, 风险分: {eeg_risk:.2f}")
        print("-" * 30)
        print(f"  🔥 综合融合风险分: {total_risk:.2f}")
        print(f"  🚩 系统预警等级: 【{level}】")
        
        time.sleep(2) # 停顿2秒，让你看清每次的变化

    print("\n✅ 测试完毕！算法逻辑完美闭环。")

if __name__ == "__main__":
    run_test()