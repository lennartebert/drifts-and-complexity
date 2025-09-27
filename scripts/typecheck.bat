@echo off
REM Type checking script for Windows

if "%1"=="--help" (
    python scripts/typecheck.py --help
    exit /b 0
)

if "%1"=="--mode" (
    python scripts/typecheck.py %*
) else (
    python scripts/typecheck.py --mode basic %*
)
