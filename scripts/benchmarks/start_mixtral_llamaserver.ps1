$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$serverExe = Join-Path $projectRoot "Models\runtime\llama.cpp\llama-server.exe"
$port = 8081
$alias = "Mixtral-8x7B-Instruct-v0.1"
$repo = "worthdoing/Mixtral-8x7B-Instruct-v0.1-GGUF:Q4_K_M"
$file = "mixtral-8x7b-instruct-v0.1-Q4_K_M-worthdoing.gguf"
$logDir = Join-Path $projectRoot "data\outputs\benchmarks\mixtral_llamaserver\logs"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

if (-not (Test-Path $serverExe)) {
    throw "llama-server executable not found at $serverExe"
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$stdout = Join-Path $logDir ("{0}_stdout.log" -f $timestamp)
$stderr = Join-Path $logDir ("{0}_stderr.log" -f $timestamp)
$healthUrl = "http://127.0.0.1:$port/health"

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
    "--hf-repo", $repo,
    "--hf-file", $file,
    "--no-mmproj",
    "--host", "127.0.0.1",
    "--port", "$port",
    "--alias", $alias,
    "--ctx-size", "8192",
    "--gpu-layers", "auto",
    "--fit-target", "1024",
    "--reasoning", "off",
    "--no-webui"
)

$proc = Start-Process `
    -FilePath $serverExe `
    -ArgumentList $arguments `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

$deadline = (Get-Date).AddMinutes(45)
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
