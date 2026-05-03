$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = "C:\Users\Owner\AppData\Local\Programs\Python\Python311\python.exe"
$logDir = Join-Path $projectRoot "data\outputs\benchmarks\fixed_repair\logs"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

if (-not (Test-Path $python)) {
    throw "Python executable not found at $python"
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Format-CmdArg {
    param([string]$Value)
    return '"' + ($Value -replace '"', '\"') + '"'
}

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Args
    )

    $stdout = Join-Path $logDir ("{0}_{1}.log" -f $timestamp, $Name)
    $stderr = Join-Path $logDir ("{0}_{1}_stderr.log" -f $timestamp, $Name)
    $quotedArgs = $Args | ForEach-Object { Format-CmdArg $_ }
    $cmd = 'cd /d "{0}" && "{1}" {2} 1>>"{3}" 2>>"{4}"' -f `
        $projectRoot, $python, ($quotedArgs -join " "), $stdout, $stderr

    Write-Output ("Running " + $Name + "...")
    Write-Output ("STDOUT=" + $stdout)
    Write-Output ("STDERR=" + $stderr)

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "cmd.exe"
    $psi.Arguments = "/c " + $cmd
    $psi.WorkingDirectory = $projectRoot
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $proc = [System.Diagnostics.Process]::Start($psi)
    $proc.WaitForExit()

    if ($proc.ExitCode -ne 0) {
        throw ("Step '{0}' failed with exit code {1}. See {2} and {3}." -f `
            $Name, $proc.ExitCode, $stdout, $stderr)
    }
}

$steps = @(
    @{
        Name = "preflight"
        Args = @(
            "scripts/benchmarks/preflight.py",
            "--configs",
            "configs/models/gemma4_31b_q6k_ref.yaml",
            "configs/models/qwen3_32b_4bit.yaml"
        )
    },
    @{
        Name = "run_fixed_repair"
        Args = @(
            "scripts/benchmarks/run_comparison.py",
            "--configs",
            "configs/models/gemma4_31b_q6k_ref.yaml",
            "configs/models/qwen3_32b_4bit.yaml",
            "--benchmark-mode",
            "fixed_repair"
        )
    },
    @{
        Name = "backfill_metrics"
        Args = @(
            "scripts/benchmarks/backfill_metrics.py",
            "--mode",
            "fixed_repair"
        )
    },
    @{
        Name = "aggregate_stats"
        Args = @(
            "-m",
            "src.aggregate",
            "--mode",
            "fixed_repair"
        )
    }
)

foreach ($step in $steps) {
    Invoke-Step -Name $step.Name -Args $step.Args
}

Write-Output "Fixed-repair handoff sequence completed."
Write-Output ("Logs=" + $logDir)
