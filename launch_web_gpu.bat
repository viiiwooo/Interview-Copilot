@echo off
REM Launch the Interview Copilot web interface with GigaAM on the GPU
REM Usage: launch_web_gpu.bat

cd /d "%~dp0"
set GIGAAM_DEVICE=cuda
echo Starting Interview Copilot web interface (GigaAM on GPU)...
echo Open http://localhost:9090 in your browser.
echo Press Ctrl+C to stop.
.venv\Scripts\python.exe -m uvicorn app.web:app --host 0.0.0.0 --port 9090
pause
