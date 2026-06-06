@echo off
REM ===========================================================================
REM  UnReflect Batch — example CLI batch run.
REM  Edit INPUT and OUTPUT below, then double-click this file.
REM  OUTPUT must be a SEPARATE folder (never inside INPUT). Originals are kept.
REM ===========================================================================
setlocal
set "ROOT=%~dp0"
set "PYTHONUTF8=1"

REM --- EDIT THESE TWO PATHS ---------------------------------------------------
set "INPUT=D:\photo_input"
set "OUTPUT=D:\photo_unreflect"
REM ---------------------------------------------------------------------------

if not exist "%ROOT%.venv\Scripts\python.exe" (
  echo [!] Environment not set up yet. Running first-time setup...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\setup_env.ps1"
  if errorlevel 1 ( echo Setup failed. & pause & exit /b 1 )
)

"%ROOT%.venv\Scripts\python.exe" "%ROOT%main.py" ^
  --input "%INPUT%" ^
  --output "%OUTPUT%" ^
  --recursive ^
  --make-preview ^
  --heatmap ^
  --device auto

echo.
echo Done. See "%OUTPUT%\logs" for process_log.csv / errors.csv / run_summary.json
pause
