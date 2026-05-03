@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_fixed_question_sets_n1000_pipeline.ps1"
