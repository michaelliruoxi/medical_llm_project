param(
    [int]$ExpectedModels = 11,
    [int]$ExpectedRows = 1000,
    [int]$MaxAttemptsPerMode = 100,
    [int]$RestartDelaySeconds = 30
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = "C:\Users\Owner\AppData\Local\Programs\Python\Python311\python.exe"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputRootRel = "data/outputs/benchmarks_n1000"
$questionSetDirRel = "data/processed/benchmarks/fixed_question_sets_gpt54_n1000"
$orchestrationDir = Join-Path $projectRoot "$outputRootRel\orchestration"
$logDir = Join-Path $orchestrationDir "logs"
$statusDir = Join-Path $orchestrationDir "status"
$statusPath = Join-Path $statusDir ("all_models_n1000_status_{0}.json" -f $timestamp)
$mainLog = Join-Path $logDir ("all_models_n1000_supervisor_{0}.log" -f $timestamp)
$mixtralServerScript = Join-Path $PSScriptRoot "start_mixtral_llamaserver.ps1"

if (-not (Test-Path $python)) {
    throw "Python executable not found at $python"
}

New-Item -ItemType Directory -Force -Path $orchestrationDir | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $statusDir | Out-Null

function Write-Status {
    param(
        [string]$Step,
        [string]$State,
        [string]$Message,
        [hashtable]$Extra = @{}
    )

    $line = "[{0}] [{1}] [{2}] {3}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Step, $State.ToUpperInvariant(), $Message
    Add-Content -Path $mainLog -Value $line

    $payload = [ordered]@{
        updated_at = (Get-Date).ToString("s")
        step = $Step
        state = $State
        message = $Message
        extra = $Extra
    }
    $payload | ConvertTo-Json -Depth 8 | Set-Content -Path $statusPath
}

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

function Quote-PowerShellLiteral {
    param([string]$Arg)

    if ($null -eq $Arg) {
        return "''"
    }

    return "'" + ($Arg -replace "'", "''") + "'"
}

function Invoke-LoggedProcess {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$ArgumentList
    )

    $stdout = Join-Path $logDir ("{0}_{1}.log" -f $timestamp, $Name)
    $stderr = Join-Path $logDir ("{0}_{1}_stderr.log" -f $timestamp, $Name)

    Write-Status -Step $Name -State "running" -Message ("Launching {0} {1}" -f $FilePath, ($ArgumentList -join " ")) -Extra @{
        stdout = $stdout
        stderr = $stderr
    }

    $argumentString = (
        $ArgumentList |
        ForEach-Object { Quote-ProcessArg -Arg $_ }
    ) -join " "

    $invokeParts = @("&", (Quote-PowerShellLiteral -Arg $FilePath))
    $invokeParts += @($ArgumentList | ForEach-Object { Quote-PowerShellLiteral -Arg $_ })
    $wrapperCommand = (
        ($invokeParts -join " ") +
        " > " + (Quote-PowerShellLiteral -Arg $stdout) +
        " 2> " + (Quote-PowerShellLiteral -Arg $stderr) +
        "; exit `$LASTEXITCODE"
    )

    $proc = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $wrapperCommand) `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -PassThru

    $proc.WaitForExit()
    $proc.Refresh()
    $exitCode = $proc.ExitCode

    return [pscustomobject]@{
        Name = $Name
        ExitCode = $exitCode
        Stdout = $stdout
        Stderr = $stderr
    }
}

function Get-ModeProgress {
    param([string]$ModeDir)

    $comparisonPath = Join-Path $ModeDir "model_comparison.csv"
    $targetRows = "{0}/{0}" -f $ExpectedRows
    $result = [ordered]@{
        comparison_path = $comparisonPath
        exists = $false
        total_models = 0
        completed_models = 0
        complete = $false
        outstanding = @()
    }

    $progressFiles = @(Get-ChildItem $ModeDir -Filter "progress_*.json" -File -ErrorAction SilentlyContinue)
    if ($progressFiles.Count -gt 0) {
        $rows = @()
        foreach ($file in $progressFiles) {
            try {
                $payload = Get-Content -Path $file.FullName -Raw | ConvertFrom-Json
            } catch {
                continue
            }

            $model = $payload.model
            if ([string]::IsNullOrWhiteSpace($model)) {
                $model = $file.BaseName -replace '^progress_', ''
            }

            $status = [string]$payload.status
            $rowsCompleted = 0
            $rowsExpected = $ExpectedRows
            if ($null -ne $payload.rows_completed) {
                $rowsCompleted = [int]$payload.rows_completed
            }
            if ($null -ne $payload.rows_expected) {
                $rowsExpected = [int]$payload.rows_expected
            }

            $rows += [pscustomobject]@{
                Model = $model
                Status = $status
                Rows = ("{0}/{1}" -f $rowsCompleted, $rowsExpected)
                RowsCompleted = $rowsCompleted
                RowsExpected = $rowsExpected
            }
        }

        if ($rows.Count -gt 0) {
            $completed = @(
                $rows | Where-Object {
                    $_.Status -like "completed*" -and
                    $_.RowsCompleted -ge $ExpectedRows -and
                    $_.RowsExpected -eq $ExpectedRows
                }
            )
            $outstanding = @(
                $rows | Where-Object {
                    $_.Status -notlike "completed*" -or
                    $_.RowsCompleted -lt $ExpectedRows -or
                    $_.RowsExpected -ne $ExpectedRows
                } | Select-Object Model, Status, Rows
            )

            $result.exists = $true
            $result.total_models = $rows.Count
            $result.completed_models = $completed.Count
            $result.complete = ($rows.Count -ge $ExpectedModels -and $completed.Count -ge $ExpectedModels)
            $result.outstanding = @($outstanding | Select-Object -First 8)
            return [pscustomobject]$result
        }
    }

    if (-not (Test-Path $comparisonPath)) {
        return [pscustomobject]$result
    }

    try {
        $rows = @(Import-Csv $comparisonPath)
    } catch {
        return [pscustomobject]$result
    }

    if (-not $rows) {
        return [pscustomobject]$result
    }

    $completed = @(
        $rows | Where-Object {
            $_.Status -like "COMPLETED*" -and $_.Rows -eq $targetRows
        }
    )
    $outstanding = @(
        $rows | Where-Object {
            $_.Status -notlike "COMPLETED*" -or $_.Rows -ne $targetRows
        } | Select-Object Model, Status, Rows
    )

    $result.exists = $true
    $result.total_models = $rows.Count
    $result.completed_models = $completed.Count
    $result.complete = ($rows.Count -ge $ExpectedModels -and $completed.Count -ge $ExpectedModels)
    $result.outstanding = @($outstanding | Select-Object -First 8)
    return [pscustomobject]$result
}

function Get-ActiveModeRuns {
    param([string]$Mode)

    $outputDirRel = "$outputRootRel/$Mode"
    $procs = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -eq "python.exe" -and
        $_.CommandLine -like "*run_comparison.py*" -and
        $_.CommandLine -like "*--benchmark-mode $Mode*" -and
        $_.CommandLine -like "*--output-dir $outputDirRel*"
    } | Select-Object ProcessId, CommandLine)

    return @($procs)
}

function Wait-ForActiveModeRuns {
    param(
        [string]$Mode,
        [int]$PollSeconds = 120
    )

    while ($true) {
        $active = @(Get-ActiveModeRuns -Mode $Mode)
        if ($active.Count -eq 0) {
            return
        }

        $ids = @($active | Select-Object -ExpandProperty ProcessId)
        Write-Status -Step $Mode -State "waiting" -Message (
            "Detected active {0} run(s); waiting on PID(s): {1}" -f $Mode, ($ids -join ", ")
        ) -Extra @{
            process_ids = $ids
        }
        Start-Sleep -Seconds $PollSeconds
    }
}

function Test-StatsSetComplete {
    param([string]$ModeDir)

    $statsDir = Join-Path $ModeDir "stats"
    if (-not (Test-Path $statsDir)) {
        return $false
    }

    $files = @(Get-ChildItem $statsDir -Filter "*.csv" -ErrorAction SilentlyContinue)
    $paired = @($files | Where-Object { $_.Name -like "paired_tests_*" }).Count
    $bootstrap = @($files | Where-Object { $_.Name -like "bootstrap_cis_*" }).Count
    $robustness = @($files | Where-Object { $_.Name -like "robustness_*" }).Count
    $summary = @($files | Where-Object { $_.Name -like "summary_*" -and $_.Name -notlike "summary_noise_*" }).Count
    $summaryNoise = @($files | Where-Object { $_.Name -like "summary_noise_*" }).Count

    return (
        $paired -ge $ExpectedModels -and
        $bootstrap -ge $ExpectedModels -and
        $robustness -ge $ExpectedModels -and
        $summary -ge $ExpectedModels -and
        $summaryNoise -ge $ExpectedModels
    )
}

function Ensure-MixtralServer {
    if (-not (Test-Path $mixtralServerScript)) {
        throw "Mixtral server script not found at $mixtralServerScript"
    }

    $name = "ensure_mixtral_server"
    $stdout = Join-Path $logDir ("{0}_{1}.log" -f $timestamp, $name)
    $stderr = Join-Path $logDir ("{0}_{1}_stderr.log" -f $timestamp, $name)

    Write-Status -Step $name -State "running" -Message ("Launching " + $mixtralServerScript) -Extra @{
        stdout = $stdout
        stderr = $stderr
    }

    Push-Location $projectRoot
    try {
        & $mixtralServerScript 1> $stdout 2> $stderr
    } finally {
        Pop-Location
    }
}

function Stop-MixtralServer {
    $existing = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -eq "llama-server.exe"
    })
    if ($existing.Count -eq 0) {
        return
    }

    $ids = ($existing | ForEach-Object { $_.ProcessId }) -join ", "
    Write-Status -Step "stop_llama_servers" -State "running" -Message ("Stopping llama-server PID(s): {0}" -f $ids)
    foreach ($proc in $existing) {
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 5
}

function Invoke-PythonStepOrThrow {
    param(
        [string]$Name,
        [string[]]$StepArgs
    )

    $result = Invoke-LoggedProcess -Name $Name -FilePath $python -ArgumentList $StepArgs
    if ($result.ExitCode -ne 0) {
        throw ("Python step '{0}' failed with exit code {1}. See {2} and {3}." -f $Name, $result.ExitCode, $result.Stdout, $result.Stderr)
    }
    return $result
}

function Finalize-Mode {
    param([string]$Mode)

    $modeDir = Join-Path $projectRoot "$outputRootRel\$Mode"

    Invoke-PythonStepOrThrow `
        -Name ("{0}_backfill_metrics" -f $Mode) `
        -StepArgs @(
            "scripts/benchmarks/backfill_metrics.py",
            "--mode",
            $Mode,
            "--output-root",
            $outputRootRel
        )

    Invoke-PythonStepOrThrow `
        -Name ("{0}_refresh_geval" -f $Mode) `
        -StepArgs @(
            "scripts/benchmarks/refresh_geval_from_cache.py",
            "--mode",
            $Mode,
            "--output-root",
            $outputRootRel
        )

    Invoke-PythonStepOrThrow `
        -Name ("{0}_aggregate_stats" -f $Mode) `
        -StepArgs @(
            "-m",
            "src.aggregate",
            "--mode",
            $Mode,
            "--output-root",
            $outputRootRel
        )

    if (-not (Test-StatsSetComplete -ModeDir $modeDir)) {
        throw ("Stats set for {0} is still incomplete under {1}" -f $Mode, $modeDir)
    }
}

function Invoke-ModeUntilComplete {
    param([string]$Mode)

    $modeDir = Join-Path $projectRoot "$outputRootRel\$Mode"

    for ($attempt = 1; $attempt -le $MaxAttemptsPerMode; $attempt++) {
        $outputDirRel = "$outputRootRel/$Mode"
        $progress = Get-ModeProgress -ModeDir $modeDir
        if ($progress.complete) {
            Write-Status -Step $Mode -State "completed" -Message ("All {0} model rows already complete." -f $ExpectedModels)
            Finalize-Mode -Mode $Mode
            return
        }

        $active = @(Get-ActiveModeRuns -Mode $Mode)
        if ($active.Count -gt 0) {
            Wait-ForActiveModeRuns -Mode $Mode
            $progress = Get-ModeProgress -ModeDir $modeDir
            if ($progress.complete) {
                Write-Status -Step $Mode -State "completed" -Message ("All {0} model rows completed while attached to an existing run." -f $ExpectedModels)
                Finalize-Mode -Mode $Mode
                return
            }
        }

        Stop-MixtralServer

        $result = Invoke-LoggedProcess `
            -Name ("{0}_run_attempt{1}" -f $Mode, $attempt) `
            -FilePath $python `
            -ArgumentList @(
                "scripts/benchmarks/run_comparison.py",
                "--benchmark-mode",
                $Mode,
                "--n-examples",
                "$ExpectedRows",
                "--question-set-dir",
                $questionSetDirRel,
                "--output-dir",
                $outputDirRel
            )

        $progress = Get-ModeProgress -ModeDir $modeDir
        if ($progress.complete) {
            Write-Status -Step $Mode -State "completed" -Message ("Run attempt {0} completed all {1} models." -f $attempt, $ExpectedModels) -Extra @{
                stdout = $result.Stdout
                stderr = $result.Stderr
            }
            Finalize-Mode -Mode $Mode
            return
        }

        $outstandingText = ""
        if ($progress.outstanding.Count -gt 0) {
            $outstandingText = ($progress.outstanding | ForEach-Object {
                "{0} [{1}; {2}]" -f $_.Model, $_.Status, $_.Rows
            }) -join "; "
        }

        Write-Status -Step $Mode -State "retrying" -Message (
            "Attempt {0}/{1} finished with exit code {2}. Completed models: {3}/{4}. {5}" -f `
                $attempt, $MaxAttemptsPerMode, $result.ExitCode, $progress.completed_models, $ExpectedModels, $outstandingText
        ) -Extra @{
            stdout = $result.Stdout
            stderr = $result.Stderr
        }

        if ($attempt -lt $MaxAttemptsPerMode) {
            Start-Sleep -Seconds $RestartDelaySeconds
        }
    }

    throw ("{0} did not reach {1} completed model rows after {2} attempt(s)." -f $Mode, $ExpectedModels, $MaxAttemptsPerMode)
}

$reportRel = "reports/fixed_vs_self_repair_n1000_{0}.md" -f (Get-Date -Format "yyyy-MM-dd")

try {
    Write-Status -Step "supervisor" -State "running" -Message "Starting full n=1000 benchmark supervisor."
    Invoke-ModeUntilComplete -Mode "fixed_repair"
    Invoke-ModeUntilComplete -Mode "self_repair"
    Stop-MixtralServer

    Invoke-PythonStepOrThrow `
        -Name "write_n1000_comparison_report" `
        -StepArgs @(
            "scripts/benchmarks/write_repair_mode_report.py",
            "--fixed-dir",
            "data/outputs/benchmarks_n1000/fixed_repair",
            "--self-dir",
            "data/outputs/benchmarks_n1000/self_repair",
            "--report-path",
            $reportRel
        )

    Write-Status -Step "supervisor" -State "completed" -Message "Fixed-repair and self-repair n=1000 are complete." -Extra @{
        report = $reportRel
    }
} catch {
    Write-Status -Step "supervisor" -State "failed" -Message $_.Exception.Message
    throw
}
