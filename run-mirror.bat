@echo off
title Smart mirror 8002 (ws 8765)
cd /d "%~dp0"
REM Stop any earlier smart-mirror backend / UI server started by START-ALL.
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'backend\.server' -or $_.CommandLine -match 'http\.server 8002' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
set MIRROR_HTTP_PORT=8002
set PYTHONUNBUFFERED=1
echo Smart mirror UI: http://localhost:8002
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '.venv311\Scripts\python.exe' -m backend.server %* 2>&1 | Tee-Object -FilePath 'mirror-run.log'"
pause
