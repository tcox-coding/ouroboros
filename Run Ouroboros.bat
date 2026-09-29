@echo off
rem Starts the Ouroboros web UI and opens it in your browser.
rem Keep this window open while you use it; close it (or Ctrl+C) to quit.
cd /d "%~dp0"
set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
%PY% -m ouroboros serve
if errorlevel 1 pause
