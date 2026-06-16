
<#
.SYNOPSIS
    Run C++ Engine vs Talishar HTTP Parity Check.

.DESCRIPTION
    This script orchestrates parity verification tests between CppEngineEnvironment
    and TalisharEngineEnvironment (HTTP-backed) for the Ira vs Briar matchup.

    The parity check verifies that both environments produce identical:
    - Observation JSON format
    - Legal actions list
    - Reward values (within ±0.001 tolerance)
    - Termination flags (terminated/truncated)
    - Game outcomes (winner/draw/timeout)

.PARAMETER Deck1Source
    FaBrary URL, 26-char deck slug, or path to a local JSON file (deck 1 / P1).
    Defaults to "Ira" for the Ira vs Briar matchup.

.PARAMETER Deck2Source
    FaBrary URL, 26-char deck slug, or path to a local JSON file (deck 2 / P2).
    Defaults to "Briar" for the Ira vs Briar matchup.

.PARAMETER Format
    Game format: "silver_age" | "classic_constructed". Default: "silver_age".

.PARAMETER Episodes
    Number of episodes to test. Default: 10.

.PARAMETER Mode
    Test mode: "single-step" | "multi-step" | "full-episode" | "stress-test".
    Default: "full-episode".

.PARAMETER StepsPerEpisode
    Number of steps per episode (for multi-step mode). Only used in multi-step mode.
    Default: 50.

.PARAMETER TalisharUrl
    Talishar server URL. Defaults to $env:TALISHAR_URL or http://localhost.

.PARAMETER CppEngineDir
    Explicit directory containing a compiled fab_engine module for this matchup.

.PARAMETER CppEngineCacheDir
    Cache directory containing compiled C++ engine matchup subdirectories.

.PARAMETER CppEngineDeck1
    Override deck 1 name used only for C++ engine cache lookup.

.PARAMETER CppEngineDeck2
    Override deck 2 name used only for C++ engine cache lookup.

.PARAMETER StopAfterFailure
    Stop at the first discrepancy instead of recording it and continuing.
    By default the checker keeps running through all requested episodes and
    steps to collect as many findings as possible.

.PARAMETER ContinueAfterFailure
    Deprecated alias for the previous default behavior. Has no effect because
    continuing after failures is now the default.

.EXAMPLE
    # Full episode parity check for Ira vs Briar
    .\run_parity_check.ps1 -Deck1Source "Ira" -Deck2Source "Briar" -Format silver_age -Episodes 10

.EXAMPLE
    # Stress test with 100 episodes
    .\run_parity_check.ps1 -Deck1Source "Ira" -Deck2Source "Briar" -Mode stress-test -Episodes 100

.EXAMPLE
    # Single-step check for quick validation
    .\run_parity_check.ps1 -Deck1Source "Ira" -Deck2Source "Briar" -Mode single-step -Episodes 5

.NOTES
    Set $env:TALISHAR_URL before running if your Talishar is not on http://localhost.
    The C++ engine must be pre-built for the matchup.
#>

[CmdletBinding()]
param(
    [string]$Deck1Source = "Ira",
    [string]$Deck2Source = "Briar",
    [string]$Format = "silver_age",
    [int]$Episodes = 10,
    [string]$Mode = "full-episode",
    [int]$StepsPerEpisode = 50,
    [string]$TalisharUrl = "",
    [string]$CppEngineDir = "",
    [string]$CppEngineCacheDir = "",
    [string]$CppEngineDeck1 = "",
    [string]$CppEngineDeck2 = "",
    [switch]$StopAfterFailure,
    [switch]$ContinueAfterFailure
)

# =============================================================================
#  ── CONFIGURATION ────────────────────────────────────────────────────────────
# =============================================================================

if (-not $TalisharUrl) {
    $TalisharUrl = if ($env:TALISHAR_URL) { $env:TALISHAR_URL } else { "http://localhost:8080/game" }
}

# Script is already in scripts/, so PSScriptRoot is the scripts directory
$Python = "python"

# =============================================================================
#  ── STEP 1: validate arguments ────────────────────────────────────────────────
# =============================================================================

Write-Host ""
Write-Host "================================================================"
Write-Host "  C++ vs Talishar Parity Check"
Write-Host "================================================================"
Write-Host "  Deck 1   : $Deck1Source"
Write-Host "  Deck 2   : $Deck2Source"
Write-Host "  Format   : $Format"
Write-Host "  Mode     : $Mode"
Write-Host "  Episodes : $Episodes"
if ($StepsPerEpisode -gt 0) {
    Write-Host "  Steps/Ep : $StepsPerEpisode"
}
Write-Host "  Talishar : $TalisharUrl"
if ($CppEngineDir) {
    Write-Host "  C++ Dir  : $CppEngineDir"
}
if ($CppEngineCacheDir) {
    Write-Host "  C++ Cache: $CppEngineCacheDir"
}
if ($StopAfterFailure) {
    Write-Host "  Stop     : after first discrepancy"
}
Write-Host "================================================================"
Write-Host ""

# Validate mode
$ValidModes = @("single-step", "multi-step", "full-episode", "stress-test")
if ($Mode -notin $ValidModes) {
    Write-Host "ERROR: Invalid mode '$Mode'. Must be one of: $($ValidModes -join ', ')" -ForegroundColor Red
    exit 2
}

# =============================================================================
#  ── STEP 2: run parity check ──────────────────────────────────────────────────
# =============================================================================

Write-Host "  Starting parity check..."
Write-Host ""

# Add src directory to PYTHONPATH
$ProjectRoot = $PSScriptRoot.Replace("\scripts","")
$SrcDir = Join-Path $ProjectRoot "src"
$env:PYTHONPATH = "$SrcDir;$env:PYTHONPATH"

# Build the Python command arguments
$PyScript = "scripts\check_cpp_vs_talishar_parity.py"
$ArgsList = @(
    "`"$PyScript`"",
    "--deck1", "`"$Deck1Source`"",
    "--deck2", "`"$Deck2Source`"",
    "--format", "`"$Format`"",
    "--mode", "`"$Mode`"",
    "--episodes", "$Episodes"
)

if ($StepsPerEpisode -gt 0) {
    $ArgsList += "--steps-per-episode", "$StepsPerEpisode"
}

$ArgsList += "--talishar-url", "`"$TalisharUrl`""
$ArgsList += "--out-dir", "results/parity_checks"

if ($CppEngineDir) {
    $ArgsList += "--cpp-engine-dir", "`"$CppEngineDir`""
}
if ($CppEngineCacheDir) {
    $ArgsList += "--cpp-engine-cache-dir", "`"$CppEngineCacheDir`""
}
if ($CppEngineDeck1) {
    $ArgsList += "--cpp-engine-deck1", "`"$CppEngineDeck1`""
}
if ($CppEngineDeck2) {
    $ArgsList += "--cpp-engine-deck2", "`"$CppEngineDeck2`""
}
if ($StopAfterFailure) {
    $ArgsList += "--stop-after-failure"
}

Write-Host "  Command:"
Write-Host "  $Python $($ArgsList -join ' ')"
Write-Host ""

& $Python $ArgsList

$ExitCode = $LASTEXITCODE

# =============================================================================
#  ── STEP 3: results summary ───────────────────────────────────────────────────
# =============================================================================

Write-Host ""
Write-Host "================================================================"
Write-Host "  Parity Check Complete"
Write-Host "================================================================"

if ($ExitCode -eq 0) {
    Write-Host "  Status   : ALL CHECKS PASSED" -ForegroundColor Green
    Write-Host "  Verdict  : C++ and Talishar environments are PARITY-MATCHED" -ForegroundColor Green
} elseif ($ExitCode -eq 1) {
    Write-Host "  Status   : DISCREPANCIES DETECTED" -ForegroundColor Yellow
    Write-Host "  Verdict  : See parity_summary.txt for details" -ForegroundColor Yellow
} elseif ($ExitCode -eq 2) {
    Write-Host "  Status   : SETUP FAILED" -ForegroundColor Red
    Write-Host "  Verdict  : C++ engine or Talishar setup is not ready; see parity_summary.txt" -ForegroundColor Red
} else {
    Write-Host "  Status   : ERROR ($ExitCode)" -ForegroundColor Red
}

Write-Host ""
Write-Host "  Output files:"
$MatchupDir = "$($Deck1Source.ToLower())_vs_$($Deck2Source.ToLower())"
Write-Host "    JSON Report: results/parity_checks/$MatchupDir/parity_report.json"
Write-Host "    Summary     : results/parity_checks/$MatchupDir/parity_summary.txt"
if ($ExitCode -ne 0) {
    Write-Host "    HTML Diff   : results/parity_checks/$MatchupDir/discrepancies.html"
}

Write-Host ""
Write-Host "================================================================"

exit $ExitCode
