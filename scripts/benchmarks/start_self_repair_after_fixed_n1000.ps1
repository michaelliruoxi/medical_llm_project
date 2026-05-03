param(
    [int]$FixedWrapperPid = 0,
    [int]$PollSeconds = 60
)

$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$selfScript = Join-Path $PSScriptRoot "start_self_repair_n1000_pipeline.ps1"
$comparisonPath = Join-Path $projectRoot "data\outputs\benchmarks_n1000\fixed_repair\model_comparison.csv"

if (-not (Test-Path $selfScript)) {
    throw "Self-repair n=1000 pipeline script not found at $selfScript"
}

function Test-FixedRepairCompleted {
    if (-not (Test-Path $comparisonPath)) {
        return $false
    }

    $rows = Import-Csv $comparisonPath
    if (-not $rows) {
        return $false
    }

    if ($rows.Count -lt 11) {
        return $false
    }

    $completed = @($rows | Where-Object { $_.Status -like "COMPLETED*" })
    return $completed.Count -ge 11
}

Write-Output "Waiting for fixed_repair n=1000 to complete before launching self_repair n=1000..."

while ($true) {
    if (Test-FixedRepairCompleted) {
        break
    }

    if ($FixedWrapperPid -gt 0) {
        $fixedProc = Get-Process -Id $FixedWrapperPid -ErrorAction SilentlyContinue
        if (-not $fixedProc -and -not (Test-FixedRepairCompleted)) {
            throw "Fixed-repair wrapper PID $FixedWrapperPid exited before all 11 models were marked completed."
        }
    }

    Start-Sleep -Seconds $PollSeconds
}

Write-Output "fixed_repair n=1000 is complete. Launching self_repair n=1000..."
& $selfScript
