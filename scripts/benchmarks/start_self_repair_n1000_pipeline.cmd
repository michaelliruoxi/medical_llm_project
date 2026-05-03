@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_self_repair_n1000_pipeline.ps1"
