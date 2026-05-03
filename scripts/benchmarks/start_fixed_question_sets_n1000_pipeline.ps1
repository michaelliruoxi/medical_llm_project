$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = "C:\Users\Owner\AppData\Local\Programs\Python\Python311\python.exe"
$logDir = Join-Path $projectRoot "data\outputs\benchmarks\fixed_question_sets_gpt54_n1000\logs"
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
        Name = "build_fixed_question_sets_n1000"
        Args = @(
            "scripts/benchmarks/build_fixed_question_sets.py",
            "--config",
            "configs/experiments/fixed_question_sets_gpt54_n1000.yaml"
        )
    },
    @{
        Name = "validate_fixed_question_sets_n1000"
        Args = @(
            "scripts/benchmarks/validate_fixed_question_sets.py",
            "--question-set-dir",
            "data/processed/benchmarks/fixed_question_sets_gpt54_n1000",
            "--output-dir",
            "data/outputs/benchmarks/fixed_question_set_validation/fixed_question_sets_gpt54_n1000"
        )
    }
)

foreach ($step in $steps) {
    Invoke-Step -Name $step.Name -StepArgs $step.Args
}

Write-Output "Fixed-question-set n=1000 pipeline completed."
Write-Output ("Logs=" + $logDir)
