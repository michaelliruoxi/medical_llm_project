$projectRoot = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\Owner\AppData\Local\Programs\Python\Python311\python.exe"
$outputDir = Join-Path $projectRoot "data\outputs\comparison"

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $outputDir ("run_" + $timestamp + ".log")
$stderr = Join-Path $outputDir ("run_" + $timestamp + "_stderr.log")

$cmd = 'cd /d "{0}" && "{1}" scripts\run_model_comparison.py 1>>"{2}" 2>>"{3}"' -f `
    $projectRoot, $python, $stdout, $stderr

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "cmd.exe"
$psi.Arguments = '/c ' + $cmd
$psi.WorkingDirectory = $projectRoot
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true

$proc = [System.Diagnostics.Process]::Start($psi)

Write-Output ("PID=" + $proc.Id)
Write-Output ("STDOUT=" + $stdout)
Write-Output ("STDERR=" + $stderr)
