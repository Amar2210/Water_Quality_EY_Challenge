@echo off
echo ============================================
echo EY Water Quality Challenge - Setup
echo ============================================

echo.
echo [1/4] Creating virtual environment...
python -m venv venv

echo.
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [3/4] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [4/4] Installing requirements...
pip install -r requirements.txt

echo.
echo ============================================
echo Setup complete! 
echo.
echo To activate the venv in future:
echo   venv\Scripts\activate
echo.
echo To run data analysis:
echo   python scripts\01_analyze_data.py
echo ============================================
pause
