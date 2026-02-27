@echo off
echo Starting Phyto-Research Assistant...
cd /d "%~dp0"
call .\venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Could not activate virtual environment! Make sure you are running this from the project folder.
    pause
    exit /b
)
streamlit run app.py
pause
