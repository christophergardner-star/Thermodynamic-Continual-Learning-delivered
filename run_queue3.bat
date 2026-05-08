@echo off
setlocal
set "REPO=%~dp0"
for /f "usebackq delims=" %%I in (`python "%REPO%tar_storage.py" workspace`) do set "TAR_WORKSPACE=%%I"
for /f "usebackq delims=" %%I in (`python "%REPO%tar_storage.py" cmd-env`) do call %%I
set "LOGDIR=%TAR_WORKSPACE%\tar_state\logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

echo ============================================================
echo  TAR Queue 3 - Living Research Ecosystem
echo  %date% %time%
echo  REPO=%REPO%
echo  TAR_WORKSPACE=%TAR_WORKSPACE%
echo ============================================================

:: Living research portfolio: scale-up + autonomous queue
echo.
echo [1/3] TAR Living Research Ecosystem
echo       Log: %LOGDIR%\living_research.log
python "%REPO%tar_living_research.py" > "%LOGDIR%\living_research.log" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Living research failed - check %LOGDIR%\living_research.log
    echo         Continuing to TAR-Author...
) else (
    echo [OK] Living research complete.
)

:: TAR-Author: compile paper with all results
echo.
echo [2/3] TAR-Author - compiling paper
echo       Log: %LOGDIR%\tar_author.log
python "%REPO%tar_author.py" > "%LOGDIR%\tar_author.log" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] TAR-Author failed - check %LOGDIR%\tar_author.log
    echo         Continuing to Autonomous Research...
) else (
    echo [OK] TAR-Author complete.
)

echo.
echo ============================================================
echo  TAR Queue 3 complete: %date% %time%
echo ============================================================
endlocal
