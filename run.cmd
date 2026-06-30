@echo off
REM Doble-click o ".\run.cmd" para lanzar la app. Evita el bloqueo de scripts de
REM PowerShell ejecutando run.ps1 con -ExecutionPolicy Bypass (solo este proceso).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
