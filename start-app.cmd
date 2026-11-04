@echo off
rem Double-click to start Lost and Frame on this computer; it opens in your browser.
cd /d "%~dp0"
start "" cmd /c "timeout /t 2 >nul & start http://localhost:8787"
node --env-file-if-exists=.env --env-file-if-exists=ffmpeg.env app.mjs
pause
