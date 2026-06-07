@echo off
REM ===========================================================================
REM  ReflectMask for RealityScan - example CLI run (reflectmask mode).
REM  Edit INPUT and OUTPUT below, then double-click this file.
REM  Produces "<OUTPUT>\realityscan\": a byte-exact copy of each original image plus a
REM  "<name>.mask.png" alignment mask (white = kept, black = excluded reflection).
REM  Import that folder into RealityScan and enable "masks for alignment".
REM  OUTPUT must be a SEPARATE folder (never inside INPUT). Originals are never modified.
REM ===========================================================================
setlocal
set "ROOT=%~dp0"
set "PYTHONUTF8=1"

REM --- EDIT THESE TWO PATHS ---------------------------------------------------
set "INPUT=D:\photo_input"
set "OUTPUT=D:\rs_reflectmask"
REM ---------------------------------------------------------------------------

if not exist "%ROOT%.venv\Scripts\python.exe" (
  echo [!] Environment not set up yet. Running first-time setup...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\setup_env.ps1"
  if errorlevel 1 ( echo Setup failed. & pause & exit /b 1 )
)

"%ROOT%.venv\Scripts\python.exe" "%ROOT%main.py" reflectmask ^
  --input "%INPUT%" ^
  --output "%OUTPUT%" ^
  --recursive ^
  --device auto

echo.
echo Done. Import "%OUTPUT%\realityscan" into RealityScan (photos + masks together),
echo then enable "masks for alignment". See "%OUTPUT%\logs" for the CSV / JSON logs.
pause
