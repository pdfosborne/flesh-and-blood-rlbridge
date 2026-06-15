<#
.SYNOPSIS
    Run randomized full-episode parity sweeps across local Talishar deck matchups.

.DESCRIPTION
    Discovers Talishar local deck files from Talishar/Assets, creates the complete
    ordered matchup range, shuffles it, and invokes scripts/run_parity_check.ps1
    for each matchup using random actions (Mode=stress-test).

    This script intentionally delegates each actual comparison to
    run_parity_check.ps1 so all detailed JSON/summary/HTML discrepancy reports
    keep the same format and location.

.PARAMETER TalisharUrl
    Talishar HTTP base URL. Docker Compose in this repo usually exposes the app
    at http://localhost:8080/game.

.PARAMETER DeckNamePattern
    Asset filename glob(s) under Talishar/Assets to include. Defaults to the
    curated local decks: Ira.txt and *SAGEPrecon.txt.

.PARAMETER EpisodesPerMatchup
    Number of randomized episodes to run for each matchup.

.PARAMETER MaxStepsPerEpisode
    Maximum random actions per episode. In stress-test mode, this is passed as
    run_parity_check.ps1 -StepsPerEpisode. Use a large value for full episodes.

.PARAMETER MaxMatchups
    Optional cap after shuffling the complete matchup range. 0 means run all.

.PARAMETER Seed
    Random seed for matchup order. Use the same seed to reproduce a sweep order.

.PARAMETER BuildMissingEngines
    Build a C++ engine for each matchup before running parity. Useful for a first
    sweep, but can be slow because it may compile many matchups.

.PARAMETER ContinueAfterFailure
    Keep running requested episodes for a matchup after the first discrepancy.
    Default behavior is fail-fast per matchup because later random actions are
    not comparable once reset or step observations diverge.

.EXAMPLE
    .\scripts\run_random_parity_sweep.ps1 -TalisharUrl http://localhost:8080/game -MaxMatchups 10

.EXAMPLE
    .\scripts\run_random_parity_sweep.ps1 -BuildMissingEngines -EpisodesPerMatchup 2 -MaxStepsPerEpisode 2000
#>

[CmdletBinding()]
param(
    [string]$TalisharUrl = "http://localhost:8080/game",
    [string]$Format = "silver_age",
    [string[]]$DeckNamePattern = @("Ira.txt", "*SAGEPrecon.txt"),
    [int]$EpisodesPerMatchup = 1,
    [int]$MaxStepsPerEpisode = 2000,
    [int]$MaxMatchups = 0,
    [int]$Seed = 8675309,
    [switch]$BuildMissingEngines,
    [switch]$ContinueAfterFailure,
    [string]$OutputDir = "results\parity_sweeps"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$ScriptDir = $PSScriptRoot
$RepoRoot = Split-Path -Parent $ScriptDir
$AssetsDir = Join-Path $RepoRoot "Talishar\Assets"
$ParityRunner = Join-Path $ScriptDir "run_parity_check.ps1"
$EngineBuilder = Join-Path $RepoRoot "build_cpp_engine_for_matchup.ps1"
$SweepRoot = Join-Path $RepoRoot $OutputDir
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SweepDir = Join-Path $SweepRoot "sweep_$Timestamp"

function Write-Section {
    param([string]$Text)
    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host ("=" * 72) -ForegroundColor Cyan
}

function Get-SafeMatchupDir {
    param([string]$Deck1, [string]$Deck2)
    return "$(($Deck1).ToLower())_vs_$(($Deck2).ToLower())"
}

function Get-ReportPath {
    param([string]$Deck1, [string]$Deck2)
    return Join-Path $RepoRoot (Join-Path "results\parity_checks" (Join-Path (Get-SafeMatchupDir $Deck1 $Deck2) "parity_report.json"))
}

if (-not (Test-Path $ParityRunner)) {
    throw "Parity runner not found: $ParityRunner"
}
if (-not (Test-Path $AssetsDir)) {
    throw "Talishar Assets directory not found: $AssetsDir"
}
if ($BuildMissingEngines -and -not (Test-Path $EngineBuilder)) {
    throw "Engine builder not found: $EngineBuilder"
}

New-Item -ItemType Directory -Force -Path $SweepDir | Out-Null

Write-Section "Discovering Decks"
$deckFiles = New-Object 'System.Collections.Generic.List[System.IO.FileInfo]'
foreach ($pattern in $DeckNamePattern) {
    Get-ChildItem -Path $AssetsDir -Filter $pattern -File |
        Where-Object { $_.BaseName -ne "Dummy" -and $_.Name -notmatch '^(eval_|rl_|MetafyDictionary)' } |
        ForEach-Object { $deckFiles.Add($_) }
}

$decks = $deckFiles |
    Sort-Object FullName -Unique |
    ForEach-Object { $_.BaseName } |
    Sort-Object

if (-not $decks -or $decks.Count -lt 2) {
    throw "Need at least two deck assets. Patterns: $($DeckNamePattern -join ', ')"
}

Write-Host "  Decks discovered: $($decks.Count)"
Write-Host "  Patterns        : $($DeckNamePattern -join ', ')"
Write-Host "  Deck list       : $($decks -join ', ')"

Write-Section "Preparing Randomized Matchups"
$matchups = New-Object 'System.Collections.Generic.List[object]'
foreach ($deck1 in $decks) {
    foreach ($deck2 in $decks) {
        $matchups.Add([pscustomobject]@{ Deck1 = $deck1; Deck2 = $deck2 })
    }
}

$random = [System.Random]::new($Seed)
$shuffled = $matchups | Sort-Object { $random.Next() }
if ($MaxMatchups -gt 0) {
    $shuffled = $shuffled | Select-Object -First $MaxMatchups
}

Write-Host "  Seed           : $Seed"
Write-Host "  Matchups queued: $(($shuffled | Measure-Object).Count)"
Write-Host "  Mode           : stress-test (random actions)"
Write-Host "  Episodes/match : $EpisodesPerMatchup"
Write-Host "  Max steps/ep   : $MaxStepsPerEpisode"
Write-Host "  Fail-fast      : $(-not $ContinueAfterFailure)"
Write-Host "  Talishar URL   : $TalisharUrl"
Write-Host "  Sweep dir      : $SweepDir"

$summary = New-Object 'System.Collections.Generic.List[object]'
$index = 0
$total = ($shuffled | Measure-Object).Count

foreach ($matchup in $shuffled) {
    $index += 1
    $deck1 = [string]$matchup.Deck1
    $deck2 = [string]$matchup.Deck2
    $label = "$deck1 vs $deck2"

    Write-Section "[$index / $total] $label"

    $buildExit = $null
    if ($BuildMissingEngines) {
        Write-Host "  Building/checking C++ engine cache..."
        & $EngineBuilder `
            -Deck1 $deck1 `
            -Deck2 $deck2 `
            -TalisharSrc "Talishar" `
            -TalisharUrl $TalisharUrl
        $buildExit = $LASTEXITCODE
        if ($buildExit -ne 0) {
            Write-Host "  Build failed for $label (exit $buildExit); parity run will still be recorded as skipped." -ForegroundColor Red
            $summary.Add([pscustomobject]@{
                index = $index
                deck1 = $deck1
                deck2 = $deck2
                status = "build_failed"
                exit_code = $buildExit
                discrepancies_found = $null
                setup_failures = $null
                total_steps = 0
                first_failure = "C++ engine build failed"
                report = $null
            })
            continue
        }
    }

    $parityArgs = @{
        Deck1Source = $deck1
        Deck2Source = $deck2
        Format = $Format
        Mode = "stress-test"
        Episodes = $EpisodesPerMatchup
        StepsPerEpisode = $MaxStepsPerEpisode
        TalisharUrl = $TalisharUrl
    }
    if ($ContinueAfterFailure) {
        $parityArgs["ContinueAfterFailure"] = $true
    }

    & $ParityRunner @parityArgs

    $exitCode = $LASTEXITCODE
    $reportPath = Get-ReportPath $deck1 $deck2
    $report = $null
    $firstFailure = ""
    $discrepancies = $null
    $setupFailures = $null
    $totalSteps = $null

    if (Test-Path $reportPath) {
        try {
            $report = Get-Content -Raw -Path $reportPath | ConvertFrom-Json
            $discrepancies = $report.discrepancies_found
            $setupFailures = $report.setup_failures
            $totalSteps = $report.total_steps
            if ($report.discrepancies -and $report.discrepancies.Count -gt 0) {
                $firstFailure = [string]$report.discrepancies[0].description
            }
        } catch {
            $firstFailure = "Could not parse report: $($_.Exception.Message)"
        }
    } else {
        $firstFailure = "Report not found: $reportPath"
    }

    $status = switch ($exitCode) {
        0 { "passed" }
        1 { "discrepancy" }
        2 { "setup_failed" }
        default { "error" }
    }

    $summary.Add([pscustomobject]@{
        index = $index
        deck1 = $deck1
        deck2 = $deck2
        status = $status
        exit_code = $exitCode
        discrepancies_found = $discrepancies
        setup_failures = $setupFailures
        total_steps = $totalSteps
        first_failure = $firstFailure
        report = $reportPath
    })

    $summaryPath = Join-Path $SweepDir "summary.json"
    $summary | ConvertTo-Json -Depth 6 | Set-Content -Path $summaryPath -Encoding UTF8
    $summary | Export-Csv -Path (Join-Path $SweepDir "summary.csv") -NoTypeInformation -Encoding UTF8
}

Write-Section "Sweep Complete"
$finalJson = Join-Path $SweepDir "summary.json"
$finalCsv = Join-Path $SweepDir "summary.csv"
$summary | ConvertTo-Json -Depth 6 | Set-Content -Path $finalJson -Encoding UTF8
$summary | Export-Csv -Path $finalCsv -NoTypeInformation -Encoding UTF8

$passed = ($summary | Where-Object { $_.status -eq "passed" } | Measure-Object).Count
$discrepancy = ($summary | Where-Object { $_.status -eq "discrepancy" } | Measure-Object).Count
$setupFailed = ($summary | Where-Object { $_.status -eq "setup_failed" } | Measure-Object).Count
$buildFailed = ($summary | Where-Object { $_.status -eq "build_failed" } | Measure-Object).Count
$errors = ($summary | Where-Object { $_.status -eq "error" } | Measure-Object).Count

Write-Host "  Passed       : $passed" -ForegroundColor Green
Write-Host "  Discrepancies: $discrepancy" -ForegroundColor Yellow
Write-Host "  Setup failed : $setupFailed" -ForegroundColor Red
Write-Host "  Build failed : $buildFailed" -ForegroundColor Red
Write-Host "  Errors       : $errors" -ForegroundColor Red
Write-Host ""
Write-Host "  JSON summary : $finalJson"
Write-Host "  CSV summary  : $finalCsv"

if (($discrepancy + $setupFailed + $buildFailed + $errors) -gt 0) {
    exit 1
}
exit 0
