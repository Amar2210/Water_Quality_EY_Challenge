@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1

echo ============================================
echo EY Water Quality Challenge - Setup
echo Use Python 3.11 or 3.12 for the pinned requirements.
echo ============================================

echo.
echo [1/3] Creating virtual environment if needed...
if not exist "venv\Scripts\python.exe" (
    python -m venv venv
    if errorlevel 1 goto :error
)

echo.
echo [2/3] Upgrading pip...
"venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error

echo.
echo [3/3] Installing requirements...
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Setup complete.
echo To activate: venv\Scripts\activate
echo To check the saved submission: python scripts\validate_submission.py
popd
exit /b 0

:error
echo.
echo Setup failed. Check the error above; installation did not complete.
popd
exit /b 1
