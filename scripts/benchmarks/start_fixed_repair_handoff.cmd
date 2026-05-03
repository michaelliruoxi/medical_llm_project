@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_fixed_repair_handoff.ps1"
