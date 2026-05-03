@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_self_repair_pipeline.ps1"
