$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = "C:\Users\Owner\AppData\Local\Programs\Python\Python311\python.exe"
$statusDir = Join-Path $projectRoot "data\outputs\benchmarks\orchestration"
$logDir = Join-Path $statusDir "logs"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$statusPath = Join-Path $statusDir ("overnight_pipeline_status_{0}.json" -f $timestamp)

New-Item -ItemType Directory -Force -Path $statusDir | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Write-Status {
    param(
        [string]$Step,
        [string]$State,
        [string]$Message,
        [hashtable]$Extra = @{}
    )

    $payload = @{
        updated_at = (Get-Date).ToString("s")
        step = $Step
        state = $State
        message = $Message
        extra = $Extra
    } | ConvertTo-Json -Depth 6

    Set-Content -Path $statusPath -Value $payload
    Write-Output ("[{0}] {1}: {2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Step, $Message)
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

function Get-PythonProcessesByPattern {
    param([string]$Pattern)

    $running = @(Get-Process -Name python -ErrorAction SilentlyContinue)
    if ($running.Count -eq 0) {
        return @()
    }

    $matches = foreach ($proc in $running) {
        try {
            $cim = Get-CimInstance Win32_Process -Filter ("ProcessId = {0}" -f $proc.Id) -ErrorAction Stop
            if ($cim -and $cim.CommandLine -like "*$Pattern*") {
                [pscustomobject]@{
                    ProcessId = $proc.Id
                    CommandLine = $cim.CommandLine
                }
            }
        } catch {
            continue
        }
    }

    return @($matches)
}

function Wait-WhilePythonRunning {
    param(
        [string]$Pattern,
        [string]$Step,
        [int]$PollSeconds = 120
    )

    while ($true) {
        $procs = @(Get-PythonProcessesByPattern -Pattern $Pattern)
        if ($procs.Count -eq 0) {
            return
        }

        $processIds = @($procs | Select-Object -ExpandProperty ProcessId)
        Write-Status -Step $Step -State "waiting" -Message ("Waiting for active python process(es): " + ($processIds -join ", ")) -Extra @{
            pattern = $Pattern
            process_ids = $processIds
        }
        Start-Sleep -Seconds $PollSeconds
    }
}

function Invoke-PowerShellFile {
    param(
        [string]$Name,
        [string]$ScriptPath
    )

    $stdout = Join-Path $logDir ("{0}_{1}.log" -f $timestamp, $Name)
    $stderr = Join-Path $logDir ("{0}_{1}_stderr.log" -f $timestamp, $Name)

    Write-Status -Step $Name -State "running" -Message ("Launching " + $ScriptPath) -Extra @{
        stdout = $stdout
        stderr = $stderr
    }

    $argList = (
        @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $ScriptPath) |
        ForEach-Object { Quote-ProcessArg -Arg $_ }
    ) -join " "
    $proc = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList $argList `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -Wait `
        -PassThru

    if ($proc.ExitCode -ne 0) {
        throw ("Step '{0}' failed with exit code {1}. See {2} and {3}." -f $Name, $proc.ExitCode, $stdout, $stderr)
    }
}

function Invoke-PythonStep {
    param(
        [string]$Name,
        [string[]]$Args
    )

    $stdout = Join-Path $logDir ("{0}_{1}.log" -f $timestamp, $Name)
    $stderr = Join-Path $logDir ("{0}_{1}_stderr.log" -f $timestamp, $Name)

    Write-Status -Step $Name -State "running" -Message ("Launching python " + ($Args -join " ")) -Extra @{
        stdout = $stdout
        stderr = $stderr
    }

    $argList = (
        $Args |
        ForEach-Object { Quote-ProcessArg -Arg $_ }
    ) -join " "
    $proc = Start-Process `
        -FilePath $python `
        -ArgumentList $argList `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -Wait `
        -PassThru

    if ($proc.ExitCode -ne 0) {
        throw ("Python step '{0}' failed with exit code {1}. See {2} and {3}." -f $Name, $proc.ExitCode, $stdout, $stderr)
    }
}

function Test-StatsSetComplete {
    param(
        [string]$ModeDir,
        [int]$ExpectedModels = 11
    )

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
        $paired -eq $ExpectedModels -and
        $bootstrap -eq $ExpectedModels -and
        $robustness -eq $ExpectedModels -and
        $summary -eq $ExpectedModels -and
        $summaryNoise -eq $ExpectedModels
    )
}

function Test-ModeComplete {
    param(
        [string]$ModeDir,
        [int]$ExpectedRows,
        [int]$ExpectedModels = 11
    )

    $comparisonPath = Join-Path $ModeDir "model_comparison.csv"
    if (-not (Test-Path $comparisonPath)) {
        return $false
    }

    $rows = @(Import-Csv $comparisonPath)
    if ($rows.Count -lt $ExpectedModels) {
        return $false
    }

    $completed = @(
        $rows | Where-Object {
            $_.Status -like "COMPLETED*" -and $_.Rows -eq ("{0}/{0}" -f $ExpectedRows)
        }
    )

    return ($completed.Count -eq $ExpectedModels) -and (Test-StatsSetComplete -ModeDir $ModeDir -ExpectedModels $ExpectedModels)
}

function Invoke-ModeUntilComplete {
    param(
        [string]$StepName,
        [string]$ScriptRelativePath,
        [string]$ModeDir,
        [int]$ExpectedRows,
        [int]$MaxAttempts = 4
    )

    $scriptPath = Join-Path $projectRoot $ScriptRelativePath

    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        if (Test-ModeComplete -ModeDir $ModeDir -ExpectedRows $ExpectedRows) {
            Write-Status -Step $StepName -State "completed" -Message ("Verified complete outputs under " + $ModeDir)
            return
        }

        Write-Status -Step $StepName -State "running" -Message ("Attempt {0}/{1}" -f $attempt, $MaxAttempts)
        Invoke-PowerShellFile -Name ("{0}_attempt{1}" -f $StepName, $attempt) -ScriptPath $scriptPath

        if (Test-ModeComplete -ModeDir $ModeDir -ExpectedRows $ExpectedRows) {
            Write-Status -Step $StepName -State "completed" -Message ("Verified complete outputs under " + $ModeDir)
            return
        }
    }

    throw ("{0} did not reach a complete 11-model / {1}-row state after {2} attempt(s)." -f $StepName, $ExpectedRows, $MaxAttempts)
}

function Test-QuestionSetReady {
    param(
        [string]$QuestionSetDir,
        [string]$ValidationDir,
        [int]$ExpectedRows = 1000
    )

    $cleanPath = Join-Path $QuestionSetDir "clean_fixed.csv"
    $noisyPath = Join-Path $QuestionSetDir "noisy_fixed_gpt54.csv"
    $repairedPath = Join-Path $QuestionSetDir "repaired_fixed_gpt54.csv"
    $summaryPath = Join-Path $ValidationDir "summary.json"

    if (-not (Test-Path $cleanPath) -or -not (Test-Path $noisyPath) -or -not (Test-Path $repairedPath) -or -not (Test-Path $summaryPath)) {
        return $false
    }

    $cleanCount = @(Import-Csv $cleanPath).Count
    $noisyCount = @(Import-Csv $noisyPath).Count
    $repairedCount = @(Import-Csv $repairedPath).Count
    if ($cleanCount -ne $ExpectedRows -or $noisyCount -ne $ExpectedRows -or $repairedCount -ne $ExpectedRows) {
        return $false
    }

    $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
    return (
        $summary.status -eq "ok" -and
        [int]$summary.noisy_rows -eq $ExpectedRows -and
        [int]$summary.repaired_rows -eq $ExpectedRows -and
        [int]$summary.noisy_invalid -eq 0 -and
        [int]$summary.repaired_invalid -eq 0
    )
}

function Invoke-QuestionSetUntilReady {
    param(
        [string]$ScriptRelativePath,
        [string]$QuestionSetDir,
        [string]$ValidationDir,
        [int]$ExpectedRows = 1000,
        [int]$MaxAttempts = 4
    )

    $scriptPath = Join-Path $projectRoot $ScriptRelativePath

    Wait-WhilePythonRunning -Pattern "build_fixed_question_sets.py --config configs/experiments/fixed_question_sets_gpt54_n1000.yaml" -Step "question_set_wait"

    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        if (Test-QuestionSetReady -QuestionSetDir $QuestionSetDir -ValidationDir $ValidationDir -ExpectedRows $ExpectedRows) {
            Write-Status -Step "question_set_n1000" -State "completed" -Message ("Validated 1000-row frozen question set under " + $QuestionSetDir)
            return
        }

        Write-Status -Step "question_set_n1000" -State "running" -Message ("Attempt {0}/{1}" -f $attempt, $MaxAttempts)
        Invoke-PowerShellFile -Name ("question_set_n1000_attempt{0}" -f $attempt) -ScriptPath $scriptPath

        if (Test-QuestionSetReady -QuestionSetDir $QuestionSetDir -ValidationDir $ValidationDir -ExpectedRows $ExpectedRows) {
            Write-Status -Step "question_set_n1000" -State "completed" -Message ("Validated 1000-row frozen question set under " + $QuestionSetDir)
            return
        }
    }

    throw "n=1000 frozen question set did not validate successfully after repeated attempts."
}

$selfRepairDir = Join-Path $projectRoot "data\outputs\benchmarks\self_repair"
$fixedRepairDir = Join-Path $projectRoot "data\outputs\benchmarks\fixed_repair"
$questionSetDir = Join-Path $projectRoot "data\processed\benchmarks\fixed_question_sets_gpt54_n1000"
$validationDir = Join-Path $projectRoot "data\outputs\benchmarks\fixed_question_set_validation\fixed_question_sets_gpt54_n1000"
$benchmarksN1000Root = Join-Path $projectRoot "data\outputs\benchmarks_n1000"
$fixedRepairN1000Dir = Join-Path $benchmarksN1000Root "fixed_repair"
$selfRepairN1000Dir = Join-Path $benchmarksN1000Root "self_repair"
$reportPath = Join-Path $projectRoot ("reports\fixed_vs_self_repair_{0}.md" -f (Get-Date -Format "yyyy-MM-dd"))

Write-Status -Step "overnight_pipeline" -State "running" -Message "Starting overnight repair benchmark supervisor."

Wait-WhilePythonRunning -Pattern "run_comparison.py --benchmark-mode self_repair" -Step "self_repair_n50_wait"
Invoke-ModeUntilComplete `
    -StepName "self_repair_n50" `
    -ScriptRelativePath "scripts\benchmarks\start_self_repair_pipeline.ps1" `
    -ModeDir $selfRepairDir `
    -ExpectedRows 50 `
    -MaxAttempts 4

Invoke-PythonStep `
    -Name "write_repair_mode_report" `
    -Args @(
        "scripts/benchmarks/write_repair_mode_report.py",
        "--fixed-dir",
        $fixedRepairDir,
        "--self-dir",
        $selfRepairDir,
        "--report-path",
        $reportPath
    )

Invoke-QuestionSetUntilReady `
    -ScriptRelativePath "scripts\benchmarks\start_fixed_question_sets_n1000_pipeline.ps1" `
    -QuestionSetDir $questionSetDir `
    -ValidationDir $validationDir `
    -ExpectedRows 1000 `
    -MaxAttempts 4

Invoke-ModeUntilComplete `
    -StepName "fixed_repair_n1000" `
    -ScriptRelativePath "scripts\benchmarks\start_fixed_repair_n1000_pipeline.ps1" `
    -ModeDir $fixedRepairN1000Dir `
    -ExpectedRows 1000 `
    -MaxAttempts 3

Invoke-ModeUntilComplete `
    -StepName "self_repair_n1000" `
    -ScriptRelativePath "scripts\benchmarks\start_self_repair_n1000_pipeline.ps1" `
    -ModeDir $selfRepairN1000Dir `
    -ExpectedRows 1000 `
    -MaxAttempts 3

Write-Status -Step "overnight_pipeline" -State "completed" -Message "Overnight repair benchmark pipeline finished successfully." -Extra @{
    report = $reportPath
    status = $statusPath
}

Write-Output ("STATUS=" + $statusPath)
Write-Output ("REPORT=" + $reportPath)
