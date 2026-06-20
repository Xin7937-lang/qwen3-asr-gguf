@echo off
chcp 65001 >nul
title Qwen3-ASR GGUF Server (AMD GPU)

echo ========================================
echo    Qwen3-ASR GGUF Server
echo    AMD GPU + Vulkan 加速
echo ========================================
echo.

REM ─── Check Python ────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+
    echo         Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM ─── Create venv if needed ────────────────────────────────────────────
if not exist "venv" (
    echo [1/6] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
) else (
    echo [1/6] Virtual environment exists
)

REM ─── Activate venv ────────────────────────────────────────────────────
echo [2/6] Activating environment...
call venv\Scripts\activate.bat

REM ─── Install dependencies ────────────────────────────────────────────
echo [3/6] Installing dependencies...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo [WARNING] Some dependencies may have failed to install
    echo.
)

REM ─── Check Vulkan environment ───────────────────────────────────────────
echo [4/6] Checking Vulkan support...

set ASR_ENABLE_VULKAN=false

REM Check via vulkaninfo (most reliable)
vulkaninfo --version >nul 2>&1
if not errorlevel 1 (
    echo [OK] Vulkan SDK detected (vulkaninfo)
    set ASR_ENABLE_VULKAN=true
    goto :vulkan_done
)

REM Fallback: check VULKAN_SDK env var (set by Vulkan SDK installer)
if not "%VULKAN_SDK%"=="" (
    echo [OK] Vulkan SDK detected (VULKAN_SDK=%VULKAN_SDK%)
    set ASR_ENABLE_VULKAN=true
    goto :vulkan_done
)

REM Fallback: check for vulkan-1.dll in system path
if exist "%SystemRoot%\System32\vulkan-1.dll" (
    echo [OK] Vulkan detected (vulkan-1.dll)
    set ASR_ENABLE_VULKAN=true
    goto :vulkan_done
)

echo [WARNING] Vulkan SDK not detected
echo.
echo          Without Vulkan SDK, server will run in CPU-only mode.
echo          For AMD GPU acceleration, install Vulkan SDK:
echo          https://vulkan.lunarg.com/
echo.

:vulkan_done

REM ─── Check model ──────────────────────────────────────────────────────
echo [5/6] Checking model...
python -c "import config; print('Model exists:', config.MODEL_PATH.exists()); print('mmproj exists:', config.MMPROJ_PATH.exists())" 2>nul
if errorlevel 1 (
    echo [INFO] Model not found
    echo [TIP] Download the model using: python download_model.py
    echo.
) else (
    python -c "import config; print('  Model:', config.MODEL_PATH.name); print('  mmproj:', config.MMPROJ_PATH.name)"
    python -c "import config; print('  Size:', round(config.MODEL_PATH.stat().st_size/1024/1024, 1) if config.MODEL_PATH.exists() else '0', 'MB')"
)

REM ─── Start server ──────────────────────────────────────────────────────
echo.
echo ========================================
echo    Starting Server
echo ========================================
echo.
echo Server addresses:
echo   Local:  http://localhost:8001
echo   API docs: http://localhost:8001/docs
echo.
echo Internal llama-server: http://localhost:8080
echo.
echo Environment:
echo   Vulkan: %ASR_ENABLE_VULKAN%
echo.
echo Press Ctrl+C to stop
echo ========================================
echo.

python main.py

pause