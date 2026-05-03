$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = "C:\Users\Owner\AppData\Local\Programs\Python\Python311\python.exe"
$logDir = Join-Path $projectRoot "data\outputs\benchmarks_n1000\self_repair\logs"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

if (-not (Test-Path $python)) {
    throw "Python executable not found at $python"
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$mixtralServerScript = Join-Path $PSScriptRoot "start_mixtral_llamaserver.ps1"
if (Test-Path $mixtralServerScript) {
    Write-Output "Ensuring Mixtral llama.cpp server is running..."
    & $mixtralServerScript
}

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

$questionSetDir = "data/processed/benchmarks/fixed_question_sets_gpt54_n1000"
$outputRoot = "data/outputs/benchmarks_n1000"
$outputDir = "$outputRoot/self_repair"

$steps = @(
    @{
        Name = "run_self_repair_n1000"
        Args = @(
            "scripts/benchmarks/run_comparison.py",
            "--benchmark-mode",
            "self_repair",
            "--n-examples",
            "1000",
            "--question-set-dir",
            $questionSetDir,
            "--output-dir",
            $outputDir
        )
    },
    @{
        Name = "backfill_metrics_n1000"
        Args = @(
            "scripts/benchmarks/backfill_metrics.py",
            "--mode",
            "self_repair",
            "--output-root",
            $outputRoot
        )
    },
    @{
        Name = "aggregate_stats_n1000"
        Args = @(
            "-m",
            "src.aggregate",
            "--mode",
            "self_repair",
            "--output-root",
            $outputRoot
        )
    }
)

foreach ($step in $steps) {
    Invoke-Step -Name $step.Name -StepArgs $step.Args
}

Write-Output "Self-repair n=1000 pipeline completed."
Write-Output ("Logs=" + $logDir)
