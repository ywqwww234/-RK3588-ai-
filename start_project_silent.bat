@echo off
setlocal
cd /d "%~dp0"

REM 静默启动（无终端窗口）
start "" /min pythonw main.py

exit
