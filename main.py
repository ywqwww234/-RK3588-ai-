"""
项目主入口。

启动登录窗口后，根据角色进入学生端或家长端界面；学生端会进一步启动
`CameraThread` 作为实时多模态采集主线程。
"""

import sys
from PyQt5.QtWidgets import QApplication
from ui_login import LoginWindow
from ui_main import MainWindow
from ui_parent import ParentWindow
from camera_thread import CameraThread
from recorder import Recorder
import subprocess
import config

if __name__ == "__main__":
    print(f">>> Python executable: {sys.executable}")
    # 仅双机/远端心率+脑电转发时启动 5001；本机直连串口时关闭
    if bool(getattr(config, "ESP32_RECEIVER_ENABLE", False)):
        try:
            subprocess.Popen(
                [sys.executable, "esp32_receiver.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
    else:
        print(">>> esp32_receiver 未启动（PHYSIO_LOCAL_SERIAL / ESP32_RECEIVER_ENABLE=False）")

    app = QApplication(sys.argv)
    
    login = LoginWindow()
    if login.exec_() == LoginWindow.Accepted:
        recorder = Recorder(data_dir=r'D:\Anti_depression\data')
        if login.role == 'student':
            print(">>> 启动学生端...")
            thread = CameraThread(recorder)
            window = MainWindow(thread, recorder)
            thread.start()
            app.aboutToQuit.connect(thread.stop)
            window.showFullScreen()
        else:
            print(">>> 启动家长端...")
            window = ParentWindow(recorder)
            window.showFullScreen()
        sys.exit(app.exec_())
    else:
        sys.exit(0)
