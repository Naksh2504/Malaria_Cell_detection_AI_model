@echo off
cd /d "%~dp0"
echo ========================================================
echo        Starting Malaria Detection AI Server
echo ========================================================
echo.
echo Opening server at http://127.0.0.1:5000 ...
echo Press Ctrl+C in this window to stop the server anytime.
echo.
python app.py
pause
