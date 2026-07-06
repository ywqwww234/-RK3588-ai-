@echo off
chcp 65001 >nul
setlocal

REM 切换到脚本所在目录
cd /d "%~dp0"

echo >>> 启动学生端...
python main.py

if errorlevel 1 (
  echo.
  echo [ERROR] 启动失败，请检查 Python 环境与依赖是否安装完整。
)

echo.
echo 按任意键退出...
pause >nul
