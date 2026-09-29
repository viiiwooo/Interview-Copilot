@echo off
REM Launch the Interview Copilot web interface
REM Usage: launch_web.bat

cd /d "%~dp0"
echo Starting Interview Copilot web interface...
echo Open http://localhost:9090 in your browser.
echo Press Ctrl+C to stop.
.venv\Scripts\python.exe -m uvicorn app.web:app --host 0.0.0.0 --port 9090
pause
