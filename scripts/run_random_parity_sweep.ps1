<#
.SYNOPSIS
    Run randomized parity sweeps across local Talishar deck matchups.

.DESCRIPTION
    Discovers Talishar local deck asset files, builds the complete deck-vs-deck
    matchup range, shuffles it, and invokes scripts/run_parity_check.ps1 for each
    pair.

    Each matchup writes the usual per-matchup artifacts under
    results/parity_checks/<deck1>_vs_<deck2>/.
    This script also aggregates a sweep-level summary under
    results/parity_sweeps/sweep_<timestamp>/.

    By default only matchups with a compiled C++ engine in results/cpp_engines
    are run. Use -BuildMissingEngines to generate/build engines first, or
    -IncludeMissingEngines to attempt parity anyway (likely setup failures).

.PARAMETER TalisharUrl
    Talishar HTTP base URL. Docker Compose in this repo usually exposes the app
    at http://localhost:8080/game.

.PARAMETER DeckNamePattern
    Asset filename glob(s) under Talishar/Assets to include. Defaults to Ira.txt
    and all *SAGEPrecon.txt decks.

.PARAMETER Format
    Game format passed to run_parity_check.ps1. Default: silver_age.

.PARAMETER Mode
    Parity mode: single-step | multi-step | full-episode | stress-test.
    Default: full-episode (matches the passing single-matchup workflow).

.PARAMETER EpisodesPerMatchup
    Episodes per matchup. Default: 1.

.PARAMETER StepsPerEpisode
  Used for multi-step and stress-test modes. For full-episode, the checker uses
  the environment max-turn budget when this is 0. Default: 0.

.PARAMETER MaxMatchups
    Optional cap after shuffling the complete matchup range. 0 means run all
    eligible matchups.

.PARAMETER Seed
    Random seed for matchup order.

.PARAMETER UnorderedOnly
    When set, only run each unordered pair once (deck1 < deck2 lexicographically).
    Excludes mirror duplicates like A-vs-B and B-vs-A.

.PARAMETER ExcludeSelfMatchups
    Skip deck-vs-itself pairings (e.g. Ira vs Ira).

.PARAMETER BuildMissingEngines
    Build a C++ engine for each matchup before running parity. Slow on first sweep.

.PARAMETER SkipMissingEngines
    Skip matchups without a compiled C++ engine. Default: true.

.PARAMETER IncludeMissingEngines
    Attempt parity even when no compiled engine exists (usually setup failures).

.PARAMETER StopAfterFailure
    Stop each matchup at the first discrepancy instead of collecting all steps.

.PARAMETER CppEngineCacheDir
    Optional override for results/cpp_engines.

.EXAMPLE
    .\scripts\run_random_parity_sweep.ps1 -MaxMatchups 5

.EXAMPLE
    .\scripts\run_random_parity_sweep.ps1 -BuildMissingEngines -UnorderedOnly -EpisodesPerMatchup 1

.EXAMPLE
    .\scripts\run_random_parity_sweep.ps1 -Mode stress-test -EpisodesPerMatchup 2 -StepsPerEpisode 100
#>

[CmdletBinding()]
param(
    [string]$TalisharUrl = "",
    [string]$Format = "silver_age",
    [ValidateSet("single-step", "multi-step", "full-episode", "stress-test")]
    [string]$Mode = "full-episode",
    [string[]]$DeckNamePattern = @("Ira.txt", "*SAGEPrecon.txt"),
    [int]$EpisodesPerMatchup = 100,
    [int]$StepsPerEpisode = 500,
    [int]$MaxMatchups = 0,
    [int]$Seed = 8675309,
    [switch]$UnorderedOnly,
    [switch]$ExcludeSelfMatchups,
    [switch]$BuildMissingEngines,
    [switch]$SkipMissingEngines = $true,
    [switch]$IncludeMissingEngines,
    [switch]$StopAfterFailure,
    [string]$CppEngineCacheDir = "",
    [string]$OutputDir = "results\parity_sweeps"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

if ($IncludeMissingEngines) {
    $SkipMissingEngines = $false
}

if (-not $TalisharUrl) {
    $TalisharUrl = if ($env:TALISHAR_URL) { $env:TALISHAR_URL } else { "http://localhost:8080/game" }
}

$ScriptDir = $PSScriptRoot
$RepoRoot = Split-Path -Parent $ScriptDir
$AssetsDir = Join-Path $RepoRoot "Talishar\Assets"
$ParityRunner = Join-Path $ScriptDir "run_parity_check.ps1"
$EngineBuilder = Join-Path $RepoRoot "build_cpp_engine_for_matchup.ps1"
$Python = "python"
$SweepRoot = Join-Path $RepoRoot $OutputDir
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SweepDir = Join-Path $SweepRoot "sweep_$Timestamp"
$SrcDir = Join-Path $RepoRoot "src"
$env:PYTHONPATH = "$SrcDir;$env:PYTHONPATH"

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

function Test-CppEngineAvailable {
    param(
        [string]$Deck1,
        [string]$Deck2,
        [string]$CacheDir
    )

    $cacheArg = ""
    if ($CacheDir) {
        $escaped = $CacheDir.Replace("\", "\\")
        $cacheArg = ", r'$escaped'"
    }

    $py = @"
from flesh_and_blood_rlbridge.cpp_engine_environment import get_engine_dir, is_cpp_engine_available
import sys
engine_dir = get_engine_dir('$Deck1', '$Deck2'$cacheArg)
sys.exit(0 if is_cpp_engine_available(engine_dir) else 1)
"@

    & $Python -c $py
    return $LASTEXITCODE -eq 0
}

function Save-SweepSummary {
    param(
        [System.Collections.Generic.List[object]]$Rows,
        [string]$Directory
    )

    $jsonPath = Join-Path $Directory "summary.json"
    $csvPath = Join-Path $Directory "summary.csv"
    $txtPath = Join-Path $Directory "summary.txt"

    $Rows | ConvertTo-Json -Depth 6 | Set-Content -Path $jsonPath -Encoding UTF8
    $Rows | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8

    $passed = ($Rows | Where-Object { $_.status -eq "passed" } | Measure-Object).Count
    $discrepancy = ($Rows | Where-Object { $_.status -eq "discrepancy" } | Measure-Object).Count
    $setupFailed = ($Rows | Where-Object { $_.status -eq "setup_failed" } | Measure-Object).Count
    $buildFailed = ($Rows | Where-Object { $_.status -eq "build_failed" } | Measure-Object).Count
    $skipped = ($Rows | Where-Object { $_.status -eq "skipped_no_engine" } | Measure-Object).Count
    $errors = ($Rows | Where-Object { $_.status -eq "error" } | Measure-Object).Count

    $lines = New-Object 'System.Collections.Generic.List[string]'
    $lines.Add("RANDOM PARITY SWEEP SUMMARY")
    $lines.Add("===========================")
    $lines.Add("")
    $lines.Add("Timestamp       : $Timestamp")
    $lines.Add("Mode            : $Mode")
    $lines.Add("Format          : $Format")
    $lines.Add("Episodes/matchup: $EpisodesPerMatchup")
    $lines.Add("Seed            : $Seed")
    $lines.Add("Talishar URL    : $TalisharUrl")
    $lines.Add("")
    $lines.Add("Matchups run    : $(($Rows | Measure-Object).Count)")
    $lines.Add("Passed          : $passed")
    $lines.Add("Discrepancies   : $discrepancy")
    $lines.Add("Setup failed    : $setupFailed")
    $lines.Add("Build failed    : $buildFailed")
    $lines.Add("Skipped (no C++): $skipped")
    $lines.Add("Errors          : $errors")
    $lines.Add("")
    $lines.Add("DETAILS")
    $lines.Add("-------")

    foreach ($row in $Rows) {
        $detail = "[$($row.status)] $($row.deck1) vs $($row.deck2)"
        if ($null -ne $row.discrepancies_found) {
            $detail += " | discrepancies=$($row.discrepancies_found) steps=$($row.total_steps)"
        }
        if ($row.first_failure) {
            $detail += " | $($row.first_failure)"
        }
        $lines.Add($detail)
    }

    $lines | Set-Content -Path $txtPath -Encoding UTF8
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
        Where-Object {
            $_.BaseName -ne "Dummy" -and
            $_.Name -notmatch '^(eval_|rl_|MetafyDictionary)'
        } |
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
        if ($ExcludeSelfMatchups -and $deck1 -eq $deck2) {
            continue
        }
        if ($UnorderedOnly -and ([string]::Compare($deck1, $deck2, $true) -ge 0)) {
            continue
        }
        $matchups.Add([pscustomobject]@{ Deck1 = $deck1; Deck2 = $deck2 })
    }
}

$random = [System.Random]::new($Seed)
$shuffled = $matchups | Sort-Object { $random.Next() }
if ($MaxMatchups -gt 0) {
    $shuffled = $shuffled | Select-Object -First $MaxMatchups
}

Write-Host "  Seed            : $Seed"
Write-Host "  Matchups queued : $(($shuffled | Measure-Object).Count)"
Write-Host "  Mode            : $Mode"
Write-Host "  Episodes/match  : $EpisodesPerMatchup"
if ($StepsPerEpisode -gt 0) {
    Write-Host "  Steps/episode   : $StepsPerEpisode"
}
Write-Host "  Unordered only  : $UnorderedOnly"
Write-Host "  Exclude self    : $ExcludeSelfMatchups"
Write-Host "  Skip no engine  : $SkipMissingEngines"
Write-Host "  Build missing   : $BuildMissingEngines"
Write-Host "  Stop on failure : $StopAfterFailure"
Write-Host "  Talishar URL    : $TalisharUrl"
Write-Host "  Sweep dir       : $SweepDir"

$summary = New-Object 'System.Collections.Generic.List[object]'
$index = 0
$total = ($shuffled | Measure-Object).Count

foreach ($matchup in $shuffled) {
    $index += 1
    $deck1 = [string]$matchup.Deck1
    $deck2 = [string]$matchup.Deck2
    $label = "$deck1 vs $deck2"

    Write-Section "[$index / $total] $label"

    if ($SkipMissingEngines -and -not $BuildMissingEngines) {
        if (-not (Test-CppEngineAvailable -Deck1 $deck1 -Deck2 $deck2 -CacheDir $CppEngineCacheDir)) {
            Write-Host "  No compiled C++ engine found; skipping." -ForegroundColor DarkYellow
            $summary.Add([pscustomobject]@{
                index = $index
                deck1 = $deck1
                deck2 = $deck2
                status = "skipped_no_engine"
                exit_code = $null
                discrepancies_found = $null
                setup_failures = $null
                total_steps = 0
                first_failure = "No compiled fab_engine in results/cpp_engines"
                report = $null
            })
            Save-SweepSummary -Rows $summary -Directory $SweepDir
            continue
        }
    }

    if ($BuildMissingEngines) {
        Write-Host "  Building/checking C++ engine cache..."
        $buildArgs = @{
            Deck1 = $deck1
            Deck2 = $deck2
            TalisharSrc = "Talishar"
            TalisharUrl = $TalisharUrl
        }
        & $EngineBuilder @buildArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  Build failed for $label (exit $LASTEXITCODE)." -ForegroundColor Red
            $summary.Add([pscustomobject]@{
                index = $index
                deck1 = $deck1
                deck2 = $deck2
                status = "build_failed"
                exit_code = $LASTEXITCODE
                discrepancies_found = $null
                setup_failures = $null
                total_steps = 0
                first_failure = "C++ engine build failed"
                report = $null
            })
            Save-SweepSummary -Rows $summary -Directory $SweepDir
            continue
        }
    }

    $parityArgs = @{
        Deck1Source = $deck1
        Deck2Source = $deck2
        Format = $Format
        Mode = $Mode
        Episodes = $EpisodesPerMatchup
        TalisharUrl = $TalisharUrl
    }
    if ($StepsPerEpisode -gt 0) {
        $parityArgs["StepsPerEpisode"] = $StepsPerEpisode
    }
    if ($CppEngineCacheDir) {
        $parityArgs["CppEngineCacheDir"] = $CppEngineCacheDir
    }
    if ($StopAfterFailure) {
        $parityArgs["StopAfterFailure"] = $true
    }

    Push-Location $RepoRoot
    try {
        & $ParityRunner @parityArgs
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }

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

    Save-SweepSummary -Rows $summary -Directory $SweepDir
}

Write-Section "Sweep Complete"
Save-SweepSummary -Rows $summary -Directory $SweepDir

$passed = ($summary | Where-Object { $_.status -eq "passed" } | Measure-Object).Count
$discrepancy = ($summary | Where-Object { $_.status -eq "discrepancy" } | Measure-Object).Count
$setupFailed = ($summary | Where-Object { $_.status -eq "setup_failed" } | Measure-Object).Count
$buildFailed = ($summary | Where-Object { $_.status -eq "build_failed" } | Measure-Object).Count
$skipped = ($summary | Where-Object { $_.status -eq "skipped_no_engine" } | Measure-Object).Count
$errors = ($summary | Where-Object { $_.status -eq "error" } | Measure-Object).Count

Write-Host "  Passed          : $passed" -ForegroundColor Green
Write-Host "  Discrepancies   : $discrepancy" -ForegroundColor Yellow
Write-Host "  Setup failed    : $setupFailed" -ForegroundColor Red
Write-Host "  Build failed    : $buildFailed" -ForegroundColor Red
Write-Host "  Skipped (no C++): $skipped" -ForegroundColor DarkYellow
Write-Host "  Errors          : $errors" -ForegroundColor Red
Write-Host ""
Write-Host "  Sweep summary   : $(Join-Path $SweepDir 'summary.txt')"
Write-Host "  Sweep JSON      : $(Join-Path $SweepDir 'summary.json')"
Write-Host "  Sweep CSV       : $(Join-Path $SweepDir 'summary.csv')"

if (($discrepancy + $setupFailed + $buildFailed + $errors) -gt 0) {
    exit 1
}
exit 0
