@echo off
cd /d "%~dp0"
if not exist input mkdir input
if not exist output mkdir output
if not exist temp mkdir temp
python "%~dp0episode_clipper.py"
echo.
pause
