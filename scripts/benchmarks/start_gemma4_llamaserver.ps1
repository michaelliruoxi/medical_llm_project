$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$serverExe = Join-Path $projectRoot "Models\runtime\llama.cpp\llama-server.exe"
$modelPath = Join-Path $projectRoot "Models\models\gemma-4-31b-it\google.gemma-4-31B-it.Q6_K.gguf"
$mmprojPath = Join-Path $projectRoot "Models\models\gemma-4-31b-it\mmproj-google.gemma-4-31B-it.f16.gguf"
$port = 8080
$alias = "gemma-4-31B-it"
$logDir = Join-Path $projectRoot "data\outputs\benchmarks\gemma4_llamaserver\logs"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

if (-not (Test-Path $serverExe)) {
    throw "llama-server executable not found at $serverExe"
}
if (-not (Test-Path $modelPath)) {
    throw "Gemma 4 GGUF not found at $modelPath"
}
if (-not (Test-Path $mmprojPath)) {
    throw "Gemma 4 mmproj not found at $mmprojPath"
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Quote-ProcessArg {
    param([string]$Arg)

    if ($null -eq $Arg) {
        return '""'
    }
    if ($Arg -match '[\s"]') {
        return '"' + ($Arg -replace '"', '\"') + '"'
    }
    return $Arg
}

$stdout = Join-Path $logDir ("{0}_stdout.log" -f $timestamp)
$stderr = Join-Path $logDir ("{0}_stderr.log" -f $timestamp)
$healthUrl = "http://127.0.0.1:$port/health"

$otherServers = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq "llama-server.exe" -and $_.CommandLine -notlike "*--port $port*"
})
foreach ($proc in $otherServers) {
    Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
}

$existing = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq "llama-server.exe" -and $_.CommandLine -like "*--port $port*"
})
if (@($existing).Count -gt 0) {
    try {
        $resp = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 5
        if ($resp.status -eq "ok") {
            Write-Output ("PID=" + $existing[0].ProcessId)
            Write-Output ("HEALTH=" + $healthUrl)
            Write-Output ("STATUS=already_running")
            exit 0
        }
    } catch {
        # Existing process is unhealthy; stop it below and start a fresh server.
    }
}

foreach ($proc in $existing) {
    Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
}

$arguments = @(
    "-m", $modelPath,
    "--mmproj", $mmprojPath,
    "--host", "127.0.0.1",
    "--port", "$port",
    "--alias", $alias,
    "--ctx-size", "2048",
    "--n-gpu-layers", "999",
    "--no-webui"
)

$argumentString = ($arguments | ForEach-Object { Quote-ProcessArg -Arg $_ }) -join " "

$proc = Start-Process `
    -FilePath $serverExe `
    -ArgumentList $argumentString `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

$deadline = (Get-Date).AddMinutes(30)
$ready = $false

while ((Get-Date) -lt $deadline) {
    if ($proc.HasExited) {
        throw "llama-server exited early with code $($proc.ExitCode). See $stderr"
    }

    try {
        $resp = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 5
        if ($resp.status -eq "ok") {
            $ready = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 10
    }
}

if (-not $ready) {
    throw "Timed out waiting for llama-server health check at $healthUrl. See $stderr"
}

Write-Output ("PID=" + $proc.Id)
Write-Output ("HEALTH=" + $healthUrl)
Write-Output ("STDOUT=" + $stdout)
Write-Output ("STDERR=" + $stderr)
exit 0
