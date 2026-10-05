@echo off
setlocal
if not defined SOL_PYTHON set "SOL_PYTHON=C:\Users\Subhajeet Bose\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not exist "%SOL_PYTHON%" (
  echo Python runtime not found. See README.md for setup on another computer.
  pause
  exit /b 1
)
cd /d "%~dp0"
echo SOL chat: http://127.0.0.1:8765/chat by default. Use Ctrl+C to stop.
"%SOL_PYTHON%" -X utf8 -u server.py --config backend_config.json
pause
