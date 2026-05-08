@echo off
setlocal
set "REPO=%~dp0"
for /f "usebackq delims=" %%I in (`python "%REPO%tar_storage.py" workspace`) do set "TAR_WORKSPACE=%%I"
for /f "usebackq delims=" %%I in (`python "%REPO%tar_storage.py" cmd-env`) do call %%I
set "LOGDIR=%TAR_WORKSPACE%\tar_state\logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

echo ============================================================
echo  TAR Watchdog
echo  %date% %time%
echo  REPO=%REPO%
echo  TAR_WORKSPACE=%TAR_WORKSPACE%
echo ============================================================
echo.
echo Starting watchdog...
echo Log: %LOGDIR%\watchdog.log

python "%REPO%tar_watchdog.py" --poll-interval-s 15 >> "%LOGDIR%\watchdog.log" 2>&1

endlocal
