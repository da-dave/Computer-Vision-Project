@echo off
rem Dataset and evaluation tools (Windows). First run creates the virtualenv and installs dependencies.
rem   run.bat capture --person P01 --site forearm --light dim
rem   run.bat label --labeler A
rem   run.bat evaluate
rem   run.bat stream-fps
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto :run
call :find_python || goto :fail
echo First run: installing Python packages, this can take a few minutes...
%PY% -m venv .venv || goto :fail_setup
".venv\Scripts\python.exe" -m pip install -q -r requirements.txt || goto :fail_setup

:run
set "COMMAND=%~1"
shift
set "ARGS="
:collect
if "%~1"=="" goto :dispatch
set ARGS=%ARGS% %1
shift
goto :collect

:dispatch
if /i "%COMMAND%"=="capture" set "SCRIPT=capture.py"
if /i "%COMMAND%"=="label" set "SCRIPT=label.py"
if /i "%COMMAND%"=="evaluate" set "SCRIPT=evaluate.py"
if /i "%COMMAND%"=="stream-fps" set "SCRIPT=measure_stream.py"
if /i "%COMMAND%"=="test" (
  ".venv\Scripts\python.exe" -m unittest discover -s tests -v
  exit /b
)
if not defined SCRIPT (
  echo Usage: run.bat capture^|label^|evaluate^|stream-fps^|test [options]
  exit /b 1
)
".venv\Scripts\python.exe" %SCRIPT%%ARGS%
if errorlevel 1 pause
exit /b

:find_python
rem Prefer the "py" launcher; plain "python" can be a Microsoft Store placeholder that doesn't run.
py -3 --version >nul 2>&1 && (set "PY=py -3" & exit /b 0)
python --version >nul 2>&1 && (set "PY=python" & exit /b 0)
echo Python 3 was not found. Install it from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during setup, then run this again.
exit /b 1

:fail_setup
echo Setup failed. Check the internet connection, then run this again.
if exist ".venv" rmdir /s /q ".venv"
:fail
pause
exit /b 1
