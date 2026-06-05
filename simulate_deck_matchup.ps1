#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Simulate a Flesh and Blood deck matchup and report win percentages.

.DESCRIPTION
    Given two decks (FaBrary URLs/slugs or local JSON files), this script:

      1. Fetches deck data from FaBrary (or uses local JSON files directly).
      2. Parses hero IDs, classes, and equipment headers from the deck JSON.
      3. Builds the C++ engine for fast simulation (optional but recommended).
      4. Handles sideboard automatically:
           - Deck already at game size (e.g. 40 cards Silver Age): used as-is.
           - Deck is a full pool (e.g. 55 cards Silver Age): greedy-cut to 40.
           - Deck is below minimum: RL sideboard agent selects the game deck.
      5. Trains play agents for both decks via Phase 3 (no deckbuilding).
      6. Runs a final evaluation to report win percentages.

    Both decks are FIXED — no deckbuilding phase runs.  Sideboard only runs
    if a deck is below the format minimum size.

.PARAMETER Deck1Source
    FaBrary URL, 26-char deck slug, or path to a local JSON file (deck 1 / P1).
    Defaults to the value of $Deck1Source set in the Configuration section below.

.PARAMETER Deck2Source
    FaBrary URL, 26-char deck slug, or path to a local JSON file (deck 2 / P2).
    Defaults to the value of $Deck2Source set in the Configuration section below.

.EXAMPLE
    # Two FaBrary URLs
    .\simulate_deck_matchup.ps1 `
        -Deck1Source "https://fabrary.net/decks/01KST88R7JVEQ73M82ZA0PJ9RN" `
        -Deck2Source "https://fabrary.net/decks/01KTBBVEZE0TPDAZ74Z4D787G4"

.EXAMPLE
    # One FaBrary slug, one already-fetched JSON file
    .\simulate_deck_matchup.ps1 `
        -Deck1Source "01KST88R7JVEQ73M82ZA0PJ9RN" `
        -Deck2Source "C:\Decks\briar_sa.json"

.NOTES
    Set $env:TALISHAR_URL before running if your Talishar is not on
    http://localhost:8080/game.
    Set $env:FABRARY_API_KEY for authenticated FaBrary fetches.
#>

[CmdletBinding()]
param(
    [string]$Deck1Source = "",
    [string]$Deck2Source = ""
)

# =============================================================================
#  ── CONFIGURATION ────────────────────────────────────────────────────────────
#
#  Edit these if you are not passing sources on the command line.
# =============================================================================

if (-not $Deck1Source) {
    $Deck1Source = "https://fabrary.net/decks/01KR0XXRF5MESBQWQH7FW5Y8MG"   # Riptide fixed
}
if (-not $Deck2Source) {
    $Deck2Source = "https://fabrary.net/decks/01KTBBVEZE0TPDAZ74Z4D787G4"   # Briar fixed
}

$Format = "silver_age"   # silver_age | classic_constructed | blitz | upf

# ── Simulation volumes ────────────────────────────────────────────────────────
# Play episodes train the agents enough to produce a meaningful win rate.
# Final-eval episodes are used for the reported win %.
$PlayEpisodes         = 200    # Phase 3 training games per iteration
$FinalEvalEpisodes    = 500    # Games for the final win % measurement
$FinalEvalMaxSteps    = 200    # Max turns per evaluation game
$SideboardEpisodes    = 30     # Sideboard episodes (only used if deck < min size)
$NumEvalGames         = 5      # Quick eval games inside sideboard finalize
$WarmupEpisodes       = 40     # Random warmup before PPO kicks in
$Iterations           = 1      # Outer iterations (1 = single-pass simulation)
$PlayWorkers          = $null  # null = auto (C++: half-cores capped at 8)

# =============================================================================
#  END OF CONFIGURATION
# =============================================================================

Set-StrictMode -Off
$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding          = [System.Text.Encoding]::UTF8

$TalisharUrl = if ($env:TALISHAR_URL)         { $env:TALISHAR_URL }         else { "http://localhost:8080/game" }
$AssetsPath  = if ($env:TALISHAR_ASSETS_PATH) { $env:TALISHAR_ASSETS_PATH } else { Join-Path $PSScriptRoot "Talishar\Assets" }

$ScriptDir = Join-Path $PSScriptRoot "scripts"
$Python    = "python"

# =============================================================================
#  ── HELPERS ──────────────────────────────────────────────────────────────────
# =============================================================================

function Resolve-DeckSource {
    <#
    .SYNOPSIS
        Turn a FaBrary URL, slug, or local path into a resolved { Slug, LocalPath } object.
    #>
    param([string]$Source, [string]$CacheDir, [string]$Label)
    New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

    # Already a local file?
    if (Test-Path $Source -PathType Leaf) {
        return @{ Slug = $null; LocalPath = (Resolve-Path $Source).Path }
    }

    # Extract slug from URL or accept bare slug
    $Slug = $Source
    if ($Source -match "fabrary\.net/decks/([A-Z0-9]+)") {
        $Slug = $Matches[1]
    }

    $OutFile = Join-Path $CacheDir "$($Slug.ToLower())_deck.json"
    return @{ Slug = $Slug; LocalPath = $OutFile }
}

function Get-FabraryDeck {
    param([string]$Slug, [string]$OutFile, [string]$Label)
    if (Test-Path $OutFile) {
        Write-Host "  [$Label] Deck already fetched  -> $OutFile"
        return $true
    }
    Write-Host "  [$Label] Fetching FaBrary deck $Slug ..."
    & $Python "$ScriptDir\fetch_fabrary_deck.py" `
        "https://fabrary.net/decks/$Slug" `
        --out $OutFile --pretty
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [$Label] Saved -> $OutFile"
        return $true
    }
    Write-Host "  [$Label] WARNING: fetch failed (exit $LASTEXITCODE)" -ForegroundColor DarkYellow
    return $false
}

function Read-DeckMeta {
    <#
    .SYNOPSIS
        Parse hero_id, hero_class, equipment_header, and card count from a deck JSON.
        Returns a hashtable with those keys plus ShortName (first underscore-segment).
    #>
    param([string]$JsonPath)
    $data = Get-Content $JsonPath -Raw | ConvertFrom-Json
    $totalCards = 0
    if ($data.deck) {
        $data.deck.PSObject.Properties | ForEach-Object { $totalCards += [int]$_.Value }
    }
    # Derive short name (e.g. "aurora_shooting_star" -> "aurora")
    $short = ($data.hero_id -split "_")[0]
    return @{
        HeroId          = $data.hero_id
        HeroClass       = $data.hero_class
        EquipmentHeader = $data.equipment_header
        Format          = if ($data.format) { $data.format } else { $Format }
        ShortName       = $short
        TotalCards      = $totalCards
        Name            = if ($data.name) { $data.name } else { $data.hero_id }
    }
}

# =============================================================================
#  ── STEP 1: resolve / fetch decks ────────────────────────────────────────────
# =============================================================================

$MatchupDir = Join-Path $PSScriptRoot "results\matchup_sims"
$DeckDir    = Join-Path $MatchupDir "decks"
New-Item -ItemType Directory -Force -Path $DeckDir | Out-Null

Write-Host ""
Write-Host "================================================================"
Write-Host "  FaB Deck Matchup Simulator"
Write-Host "================================================================"
Write-Host "  Deck 1 : $Deck1Source"
Write-Host "  Deck 2 : $Deck2Source"
Write-Host "  Format : $Format"
Write-Host ""

$Deck1Info = Resolve-DeckSource -Source $Deck1Source -CacheDir $DeckDir -Label "Deck 1"
$Deck2Info = Resolve-DeckSource -Source $Deck2Source -CacheDir $DeckDir -Label "Deck 2"

$Deck1Ok = $true
$Deck2Ok = $true

if ($Deck1Info.Slug) {
    $Deck1Ok = Get-FabraryDeck -Slug $Deck1Info.Slug -OutFile $Deck1Info.LocalPath -Label "Deck 1"
} else {
    Write-Host "  [Deck 1] Using local file: $($Deck1Info.LocalPath)"
}
if ($Deck2Info.Slug) {
    $Deck2Ok = Get-FabraryDeck -Slug $Deck2Info.Slug -OutFile $Deck2Info.LocalPath -Label "Deck 2"
} else {
    Write-Host "  [Deck 2] Using local file: $($Deck2Info.LocalPath)"
}

if (-not $Deck1Ok -or -not (Test-Path $Deck1Info.LocalPath)) {
    Write-Host "ERROR: Deck 1 JSON not available. Aborting." -ForegroundColor Red
    exit 1
}
if (-not $Deck2Ok -or -not (Test-Path $Deck2Info.LocalPath)) {
    Write-Host "ERROR: Deck 2 JSON not available. Aborting." -ForegroundColor Red
    exit 1
}

# =============================================================================
#  ── STEP 2: parse hero metadata ──────────────────────────────────────────────
# =============================================================================

$P1Meta = Read-DeckMeta -JsonPath $Deck1Info.LocalPath
$P2Meta = Read-DeckMeta -JsonPath $Deck2Info.LocalPath

# Derive output directory name from hero short names
$MatchupLabel  = "$($P1Meta.ShortName)_vs_$($P2Meta.ShortName)"
$OutDir        = Join-Path $PSScriptRoot "results\matchup_sims\$MatchupLabel"
$ResultsJson   = Join-Path $OutDir "results.json"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# Detect effective format (use deck's declared format if present)
$EffectiveFormat = $P1Meta.Format
$MinDeckSizes = @{ "silver_age" = 40; "blitz" = 40; "classic_constructed" = 60; "upf" = 60 }
$MinSize = if ($MinDeckSizes.ContainsKey($EffectiveFormat)) { $MinDeckSizes[$EffectiveFormat] } else { 40 }

Write-Host ""
Write-Host "  Deck 1 : $($P1Meta.Name)"
Write-Host "           Hero: $($P1Meta.HeroId) | Class: $($P1Meta.HeroClass)"
Write-Host "           Cards in JSON deck: $($P1Meta.TotalCards)"
if ($P1Meta.TotalCards -ge $MinSize) {
    if ($P1Meta.TotalCards -gt $MinSize) {
        Write-Host "           -> Pool ($($P1Meta.TotalCards) cards): greedy-cut to $MinSize for play" -ForegroundColor Cyan
    } else {
        Write-Host "           -> Game-ready ($($P1Meta.TotalCards) cards): sideboard skipped" -ForegroundColor Cyan
    }
} else {
    Write-Host "           -> Below minimum ($($P1Meta.TotalCards) < $MinSize): sideboard RL will run" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "  Deck 2 : $($P2Meta.Name)"
Write-Host "           Hero: $($P2Meta.HeroId) | Class: $($P2Meta.HeroClass)"
Write-Host "           Cards in JSON deck: $($P2Meta.TotalCards)"
if ($P2Meta.TotalCards -ge $MinSize) {
    if ($P2Meta.TotalCards -gt $MinSize) {
        Write-Host "           -> Pool ($($P2Meta.TotalCards) cards): greedy-cut to $MinSize for play" -ForegroundColor Cyan
    } else {
        Write-Host "           -> Game-ready ($($P2Meta.TotalCards) cards): sideboard skipped" -ForegroundColor Cyan
    }
} else {
    Write-Host "           -> Below minimum ($($P2Meta.TotalCards) < $MinSize): sideboard RL will run" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  Output dir : $OutDir"
Write-Host ""

# =============================================================================
#  ── STEP 3: build C++ engine ─────────────────────────────────────────────────
# =============================================================================

Write-Host "  Building C++ engine for $($P1Meta.ShortName) vs $($P2Meta.ShortName)..."
Write-Host ""

# The build script uses Assets deck names (PascalCase short names), not full hero IDs.
$BuildDeck1 = (Get-Culture).TextInfo.ToTitleCase($P1Meta.ShortName)
$BuildDeck2 = (Get-Culture).TextInfo.ToTitleCase($P2Meta.ShortName)

& "$PSScriptRoot\build_cpp_engine_for_matchup.ps1" `
    -Deck1     $BuildDeck1 `
    -Deck2     $BuildDeck2 `
    -Deck1Json $Deck1Info.LocalPath `
    -Deck2Json $Deck2Info.LocalPath `
    -TalisharSrc (Join-Path $PSScriptRoot "Talishar") `
    -TalisharUrl $TalisharUrl `
    -NoServer

$CppEngineBuildSucceeded = ($LASTEXITCODE -eq 0)

$CppEngineDir = $null
if ($CppEngineBuildSucceeded) {
    # Discover hashed engine dir — try both "{p1}_vs_{p2}" short-name patterns
    $CppEngineDir = Get-ChildItem (Join-Path $PSScriptRoot "results\cpp_engines") -Directory `
        -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -like "${MatchupLabel}-*" -or $_.Name -eq $MatchupLabel
        } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1 -ExpandProperty FullName

    if (-not $CppEngineDir) {
        Write-Host "  WARNING: C++ build reported success but engine directory not found." -ForegroundColor DarkYellow
        $CppEngineBuildSucceeded = $false
    } else {
        Write-Host "  Engine directory: $CppEngineDir" -ForegroundColor Green
    }
}

if (-not $CppEngineBuildSucceeded) {
    Write-Host ""
    Write-Host "  WARNING: C++ engine build failed — falling back to HTTP Talishar." -ForegroundColor DarkYellow
    Write-Host ""
} else {
    Write-Host "  C++ engine ready."
    Write-Host ""
}

# =============================================================================
#  ── STEP 4: run simulation ───────────────────────────────────────────────────
# =============================================================================

Write-Host "  Starting simulation ..."
Write-Host "  Both decks are FIXED (no deckbuilding phase)."
Write-Host "  Phase 3 (play) trains both agents then runs final evaluation."
Write-Host ""

# Sideboard args — only passed if a deck is below minimum size (needs RL sideboard)
$P1SideboardArgs = @("--p1-fixed-deck", $Deck1Info.LocalPath)
$P2SideboardArgs = @("--p2-fixed-deck", $Deck2Info.LocalPath)

$CppArgs     = if ($CppEngineBuildSucceeded) { @("--cpp-engine-dir", $CppEngineDir) } else { @() }
$WorkerArgs  = if ($null -ne $PlayWorkers)   { @("--workers", $PlayWorkers) }         else { @() }

& $Python "$ScriptDir\train_full_pipeline.py" `
    --format                     $EffectiveFormat `
    --hero-id                    $P1Meta.HeroId `
    --hero-class                 $P1Meta.HeroClass `
    --equipment-header           $P1Meta.EquipmentHeader `
    --opponent-mode              dual `
    --p2-hero-id                 $P2Meta.HeroId `
    --p2-hero-class              $P2Meta.HeroClass `
    --p2-equipment-header        $P2Meta.EquipmentHeader `
    --opponent-hero-id           $P2Meta.HeroId `
    --deckbuild-episodes         0 `
    --sideboard-episodes         $SideboardEpisodes `
    --play-episodes              $PlayEpisodes `
    --num-eval-games             $NumEvalGames `
    --warmup-episodes            $WarmupEpisodes `
    --warmup-baseline-eval-episodes 20 `
    --iterations                 $Iterations `
    --final-eval-episodes        $FinalEvalEpisodes `
    --final-eval-max-steps       $FinalEvalMaxSteps `
    --no-render-gif `
    --talishar-url               $TalisharUrl `
    --assets-path                $AssetsPath `
    --out-dir                    $OutDir `
    --results-json               $ResultsJson `
    @P1SideboardArgs `
    @P2SideboardArgs `
    @CppArgs `
    @WorkerArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: simulation exited with code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# =============================================================================
#  ── STEP 5: results summary ──────────────────────────────────────────────────
# =============================================================================

Write-Host ""
Write-Host "================================================================"
Write-Host "  Simulation Results — $MatchupLabel"
Write-Host "================================================================"

if (Test-Path $ResultsJson) {
    $results = Get-Content $ResultsJson -Raw | ConvertFrom-Json

    # Helper: format a final_eval block
    function Show-PlayerResult {
        param([string]$Label, [string]$DeckName, [psobject]$PlayerData)
        Write-Host ""
        Write-Host "  $Label ($DeckName)"
        if ($PlayerData.final_eval) {
            $fe  = $PlayerData.final_eval
            $pct = [math]::Round($fe.win_rate * 100, 1)
            $rec = "$($fe.wins)W / $($fe.losses)L"
            if ($fe.PSObject.Properties["draws"] -and $fe.draws -gt 0) {
                $rec += " / $($fe.draws)D"
            }
            Write-Host "    Win rate (final eval) : ${pct}%  ($rec over $FinalEvalEpisodes games)" -ForegroundColor Green
        } elseif ($PlayerData.win_rates -and $PlayerData.win_rates.Count -gt 0) {
            $last = $PlayerData.win_rates[-1]
            $pct  = [math]::Round($last * 100, 1)
            Write-Host "    Win rate (last iter)  : ${pct}%  (from training)" -ForegroundColor Yellow
        } else {
            Write-Host "    Win rate              : N/A" -ForegroundColor DarkGray
        }
        if ($PlayerData.active_decks) {
            $deckSizes = $PlayerData.active_decks.PSObject.Properties |
                ForEach-Object { "$($_.Name): $($_.Value) cards" }
            Write-Host "    Game deck(s)          : $($deckSizes -join ' | ')"
        }
    }

    Show-PlayerResult `
        -Label    "Deck 1 (P1)" `
        -DeckName $P1Meta.Name `
        -PlayerData $results.p1

    Show-PlayerResult `
        -Label    "Deck 2 (P2)" `
        -DeckName $P2Meta.Name `
        -PlayerData $results.p2

    # Head-to-head summary
    Write-Host ""
    $p1FinalRate = $null
    $p2FinalRate = $null
    if ($results.p1.final_eval) { $p1FinalRate = $results.p1.final_eval.win_rate }
    elseif ($results.p1.win_rates -and $results.p1.win_rates.Count -gt 0) { $p1FinalRate = $results.p1.win_rates[-1] }
    if ($results.p2.final_eval) { $p2FinalRate = $results.p2.final_eval.win_rate }
    elseif ($results.p2.win_rates -and $results.p2.win_rates.Count -gt 0) { $p2FinalRate = $results.p2.win_rates[-1] }

    if ($null -ne $p1FinalRate -and $null -ne $p2FinalRate) {
        $p1Pct = [math]::Round($p1FinalRate * 100, 1)
        $p2Pct = [math]::Round($p2FinalRate * 100, 1)
        Write-Host "  Head-to-head : P1 $p1Pct%  vs  P2 $p2Pct%"
        if ($p1FinalRate -gt $p2FinalRate) {
            Write-Host "  Verdict      : Deck 1 ($($P1Meta.Name)) wins the matchup" -ForegroundColor Green
        } elseif ($p2FinalRate -gt $p1FinalRate) {
            Write-Host "  Verdict      : Deck 2 ($($P2Meta.Name)) wins the matchup" -ForegroundColor Green
        } else {
            Write-Host "  Verdict      : Even matchup" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "  WARNING: results.json not found at $ResultsJson" -ForegroundColor DarkYellow
}

Write-Host ""
Write-Host "  Full results : $ResultsJson"
Write-Host "  Output dir   : $OutDir"
Write-Host ""
