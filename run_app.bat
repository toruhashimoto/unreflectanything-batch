@echo off
REM ===========================================================================
REM  UnReflect Batch — launch the Streamlit GUI.
REM  Double-click to start. On first run it sets up the environment and
REM  downloads the model weights (~5.9 GB); later runs start immediately.
REM ===========================================================================
setlocal
set "ROOT=%~dp0"
set "PYTHONUTF8=1"

if not exist "%ROOT%.venv\Scripts\python.exe" (
  echo [*] First run: setting up environment + GUI ^(downloads PyTorch and ~5.9GB weights^)...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\setup_env.ps1" -Gui
  if errorlevel 1 ( echo Setup failed. & pause & exit /b 1 )
)

REM Ensure Streamlit is present even if the venv was created CLI-only.
"%ROOT%.venv\Scripts\python.exe" -c "import streamlit" 1>nul 2>nul
if errorlevel 1 (
  echo [*] Installing Streamlit...
  "%ROOT%.venv\Scripts\python.exe" -m pip install "streamlit>=1.40"
)

echo [*] Starting UnReflect Batch GUI — your browser will open shortly.
"%ROOT%.venv\Scripts\python.exe" -m streamlit run "%ROOT%app.py"
pause
