@echo off
setlocal
if defined SOL_PYTHON goto run
if exist "%~dp0..\.venv\Scripts\python.exe" (
  set "SOL_PYTHON=%~dp0..\.venv\Scripts\python.exe"
  goto run
)
if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" (
  set "SOL_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
  goto run
)
set "SOL_PYTHON=python"
:run
echo SOL Chat: http://127.0.0.1:8765/chat
echo Keep this window open. Ctrl+C stops the application.
"%SOL_PYTHON%" -X utf8 -u "%~dp0run_sol.py" %*
if errorlevel 1 pause
