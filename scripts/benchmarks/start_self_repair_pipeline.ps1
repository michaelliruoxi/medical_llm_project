$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = "C:\Users\Owner\AppData\Local\Programs\Python\Python311\python.exe"
$logDir = Join-Path $projectRoot "data\outputs\benchmarks\self_repair\logs"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

if (-not (Test-Path $python)) {
    throw "Python executable not found at $python"
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$StepArgs
    )

    $stdout = Join-Path $logDir ("{0}_{1}.log" -f $timestamp, $Name)
    $stderr = Join-Path $logDir ("{0}_{1}_stderr.log" -f $timestamp, $Name)

    Write-Output ("Running " + $Name + "...")
    Write-Output ("STDOUT=" + $stdout)
    Write-Output ("STDERR=" + $stderr)

    $proc = Start-Process `
        -FilePath $python `
        -ArgumentList $StepArgs `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -NoNewWindow `
        -Wait `
        -PassThru

    if ($proc.ExitCode -ne 0) {
        throw ("Step '{0}' failed with exit code {1}. See {2} and {3}." -f `
            $Name, $proc.ExitCode, $stdout, $stderr)
    }
}

$steps = @(
    @{
        Name = "run_self_repair"
        Args = @(
            "scripts/benchmarks/run_comparison.py",
            "--benchmark-mode",
            "self_repair"
        )
    },
    @{
        Name = "backfill_metrics"
        Args = @(
            "scripts/benchmarks/backfill_metrics.py",
            "--mode",
            "self_repair"
        )
    },
    @{
        Name = "aggregate_stats"
        Args = @(
            "-m",
            "src.aggregate",
            "--mode",
            "self_repair"
        )
    }
)

foreach ($step in $steps) {
    Invoke-Step -Name $step.Name -StepArgs $step.Args
}

Write-Output "Self-repair pipeline completed."
Write-Output ("Logs=" + $logDir)
