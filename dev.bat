@echo off
REM Backend Development Script for Windows
REM Runs FastAPI with hot reload

cd /d "%~dp0"

echo Starting FlowScope AI Backend (Development Mode)
echo ==================================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -q -r requirements.txt

REM Create .env if it doesn't exist
if not exist ".env" (
    echo Creating .env file...
    copy .env.example .env
)

REM Create data directory
if not exist "data" mkdir data

echo.
echo Backend ready!
echo    API: http://localhost:8000
echo    Docs: http://localhost:8000/docs
echo.

REM Run with hot reload
python main.py
