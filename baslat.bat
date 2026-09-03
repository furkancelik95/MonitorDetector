@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo  MonitorDetector
echo  ----------------

set "PY=%LocalAppData%\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY=%LocalAppData%\Programs\Python\Python311\python.exe"
if not exist "%PY%" (
    echo  Resmi Python 3.12 bulunamadi.
    echo  https://www.python.org/downloads/ adresinden 3.12 kurun.
    echo.
    pause
    exit /b 1
)

echo  Python:
"%PY%" --version

set "NEED_VENV=0"
if not exist ".venv\Scripts\python.exe" set "NEED_VENV=1"
if exist ".venv\pyvenv.cfg" (
    findstr /i /c:"anaconda3" ".venv\pyvenv.cfg" >nul 2>&1
    if not errorlevel 1 set "NEED_VENV=1"
)

if "%NEED_VENV%"=="1" goto MAKE_VENV
goto VENV_READY

:MAKE_VENV
echo  [1/3] Sanal ortam olusturuluyor...
if exist ".venv" rmdir /s /q ".venv"
"%PY%" -m venv .venv
if errorlevel 1 (
    echo  Sanal ortam olusturulamadi.
    echo.
    pause
    exit /b 1
)

:VENV_READY
set "VENV_PY=.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo  Sanal ortam python.exe bulunamadi.
    echo.
    pause
    exit /b 1
)

set "REQ_STAMP=.venv\.requirements.ok"
set "NEED_PIP=1"
if exist "%REQ_STAMP%" (
    fc /b "requirements.txt" "%REQ_STAMP%" >nul 2>&1
    if not errorlevel 1 set "NEED_PIP=0"
)

if "%NEED_PIP%"=="0" goto PIP_DONE

echo  [2/3] Gereksinimler kuruluyor / guncelleniyor...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
    echo  pip guncellenemedi.
    echo.
    pause
    exit /b 1
)
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo  Paket kurulumu basarisiz oldu.
    echo.
    pause
    exit /b 1
)
copy /y "requirements.txt" "%REQ_STAMP%" >nul
goto RUN_APP

:PIP_DONE
echo  [2/3] Gereksinimler zaten kurulu.

:RUN_APP
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo.
    echo  Uyari: ffmpeg PATH'te yok. Kamera ve ses klibi icin gerekir.
    echo  https://www.gyan.dev/ffmpeg/builds/
    echo.
)

echo  [3/3] Uygulama aciliyor...
echo.
"%VENV_PY%" main.py
set "APP_EXIT=%errorlevel%"
if not "%APP_EXIT%"=="0" (
    echo.
    echo  Uygulama hata ile kapandi (kod: %APP_EXIT%).
)
echo.
pause
exit /b %APP_EXIT%
