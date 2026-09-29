@echo off
REM Launch your normal Chrome with remote debugging enabled, so the Interview
REM Copilot can drive the chat.deepseek.com tab via CDP.
REM Close all existing Chrome windows first (Chrome only opens one debug port).
set PORT=9222
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" ^
  --remote-debugging-port=%PORT% ^
  --user-data-dir="C:\Users\msh\AppData\Local\Google\Chrome\User Data"

echo Chrome launched with CDP on port %PORT%.
echo Now open https://chat.deepseek.com in that window and log in.
pause
