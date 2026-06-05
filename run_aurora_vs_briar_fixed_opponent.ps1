#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Find the best Aurora deck against a specific fixed Briar opponent deck.

.DESCRIPTION
    Aurora trains all three phases (deckbuilder + sideboard + play agent).
    Briar is FIXED to a known FaBrary deck — no deckbuilding, no sideboarding.

    Because the Briar deck is already a 40-card Silver Age game deck it
    meets the minimum play size, so --p2-fixed-deck skips Phase 1 and 2
    for Briar entirely every iteration.

    Fixed opponent deck (Briar):
        https://fabrary.net/decks/01KTBBVEZE0TPDAZ74Z4D787G4

    Aurora warm-start deck:
        https://fabrary.net/decks/01KST88R7JVEQ73M82ZA0PJ9RN

.NOTES
    Set $env:TALISHAR_URL before running if your local Talishar is not on
    the default http://localhost:8080/game path.

    Set $env:FABRARY_API_KEY if you have a FaBrary API key; otherwise the
    fetch step will attempt an unauthenticated request.
#>

# ─── Configuration ────────────────────────────────────────────────────────────

$TalisharUrl = if ($env:TALISHAR_URL) { $env:TALISHAR_URL } else { "http://localhost:8080/game" }
$AssetsPath  = if ($env:TALISHAR_ASSETS_PATH) { $env:TALISHAR_ASSETS_PATH } else {
    Join-Path $PSScriptRoot "Talishar\Assets"
}
$OutDir      = Join-Path $PSScriptRoot "results\aurora_vs_briar_fixed"
$DeckDir     = Join-Path $OutDir "sage_decks"
$ResultsJson = Join-Path $OutDir "results.json"

# FaBrary deck slugs
$AuroraSlug        = "01KST88R7JVEQ73M82ZA0PJ9RN"   # Aurora warm-start (deckbuilder seed)
$BriarFixedSlug    = "01KTBBVEZE0TPDAZ74Z4D787G4"    # Briar fixed opponent deck
$AuroraDeckJson    = Join-Path $DeckDir "aurora_warmstart_deck.json"
$BriarFixedDeckJson = Join-Path $DeckDir "briar_fixed_deck.json"

# Hero identifiers (Talishar internal IDs — Silver Age Young versions)
$AuroraHeroId    = "aurora"
$AuroraHeroClass = "Runeblade"
$AuroraEquipment = "aurora star_fall aether_ironweave spellbound_creepers aether_crackers crown_of_dichotomy"

$BriarHeroId     = "briar"
$BriarHeroClass  = "Runeblade"
$BriarEquipment  = "briar star_fall aether_ironweave spellbound_creepers aether_crackers crown_of_dichotomy"

# ── Training volume ───────────────────────────────────────────────────────────
# Briar never decbuilds or sideboards so only Aurora's Phase 1/2 consume
# episodes.  Phase 3 (play) trains both agents each iteration.
$DeckbuildEpisodes    = 10    # Aurora Phase 1 episodes per iteration
$SideboardEpisodes    = 10    # Aurora Phase 2 episodes per iteration
$PlayEpisodes         = 100   # Phase 3 play episodes per iteration
$NumEvalGames         = 20    # C++ games used inside deckbuilder eval
$NumSideboardEpisodes = 10    # Sideboard episodes inside deckbuilder eval
$Iterations           = 3
$PlayWorkers          = $null  # null = auto (half cores capped at 8 for C++)

# Final evaluation (post-training)
$FinalEvalEpisodes = 100
$FinalEvalMaxSteps = 200
$GifFps            = 1.0

# ─── Setup ───────────────────────────────────────────────────────────────────

$ScriptDir = Join-Path $PSScriptRoot "scripts"
$Python    = "python"

New-Item -ItemType Directory -Force -Path $DeckDir | Out-Null

Write-Host ""
Write-Host "================================================================"
Write-Host "  Aurora (training) vs Briar (fixed deck) — Silver Age"
Write-Host "================================================================"
Write-Host "  Mode         : Aurora trains all phases; Briar is FIXED"
Write-Host "  Talishar URL : $TalisharUrl"
Write-Host "  Assets path  : $AssetsPath"
Write-Host "  Output dir   : $OutDir"
Write-Host ""

# ─── Step 1: Fetch FaBrary decks ─────────────────────────────────────────────

function Get-FabraryDeck {
    param(
        [string]$Slug,
        [string]$OutFile,
        [string]$Label
    )
    if (Test-Path $OutFile) {
        Write-Host "  [$Label] Deck already fetched -> $OutFile"
        return $true
    }
    Write-Host "  [$Label] Fetching FaBrary deck $Slug ..."
    & $Python "$ScriptDir\fetch_fabrary_deck.py" `
        "https://fabrary.net/decks/$Slug" `
        --out $OutFile `
        --pretty
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [$Label] Saved -> $OutFile"
        return $true
    } else {
        Write-Host "  [$Label] WARNING: fetch failed (exit $LASTEXITCODE)" -ForegroundColor DarkYellow
        return $false
    }
}

$AuroraFetched     = Get-FabraryDeck -Slug $AuroraSlug     -OutFile $AuroraDeckJson     -Label "Aurora (warm-start)"
$BriarFixedFetched = Get-FabraryDeck -Slug $BriarFixedSlug -OutFile $BriarFixedDeckJson -Label "Briar  (fixed opponent)"

# Aurora warm-start seed (deckbuilder starts from this pool)
$AuroraStartArgs = @()
if ($AuroraFetched -and (Test-Path $AuroraDeckJson)) {
    $AuroraStartArgs += "--p1-starting-deck"; $AuroraStartArgs += $AuroraDeckJson
}

# Briar fixed deck — passed as --p2-fixed-deck so Phase 1 & 2 are always skipped
$BriarFixedArgs = @()
if ($BriarFixedFetched -and (Test-Path $BriarFixedDeckJson)) {
    $BriarFixedArgs += "--p2-fixed-deck"; $BriarFixedArgs += $BriarFixedDeckJson
    Write-Host ""
    Write-Host "  Briar opponent deck pinned to: $BriarFixedDeckJson" -ForegroundColor Cyan
    Write-Host "  Phase 1 (deckbuilder) and Phase 2 (sideboard) will be skipped for Briar." -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "  WARNING: Briar fixed deck not available — Briar will train deckbuilder/sideboard too." -ForegroundColor DarkYellow
}

# ─── Step 2: Build C++ engine ────────────────────────────────────────────────

Write-Host ""
Write-Host "  Building C++ engine for Aurora vs Briar..."
Write-Host ""

$CppBuildArgs = @{
    Deck1       = $AuroraHeroId
    Deck2       = $BriarHeroId
    TalisharSrc = (Join-Path $PSScriptRoot "Talishar")
    TalisharUrl = $TalisharUrl
    NoServer    = $true
}
if (Test-Path $AuroraDeckJson)     { $CppBuildArgs["Deck1Json"] = $AuroraDeckJson }
if (Test-Path $BriarFixedDeckJson) { $CppBuildArgs["Deck2Json"] = $BriarFixedDeckJson }

& "$PSScriptRoot\build_cpp_engine_for_matchup.ps1" @CppBuildArgs

$CppEngineBuildSucceeded = ($LASTEXITCODE -eq 0)

$CppEngineDir = $null
if ($CppEngineBuildSucceeded) {
    $CppEngineDir = Get-ChildItem (Join-Path $PSScriptRoot "results\cpp_engines") -Directory `
        -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "aurora_vs_briar-*" -or $_.Name -eq "aurora_vs_briar" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1 -ExpandProperty FullName
    if (-not $CppEngineDir) {
        Write-Host "  WARNING: C++ build reported success but engine directory not found." -ForegroundColor DarkYellow
        $CppEngineBuildSucceeded = $false
    } else {
        Write-Host "  Engine directory: $CppEngineDir"
    }
}

if (-not $CppEngineBuildSucceeded) {
    Write-Host ""
    Write-Host "  WARNING: C++ engine build failed — falling back to HTTP Talishar." -ForegroundColor DarkYellow
    Write-Host ""
} else {
    Write-Host "  C++ engine build succeeded."
    Write-Host ""
}

# ─── Step 3: Run the full pipeline ───────────────────────────────────────────

Write-Host "  Starting train_full_pipeline.py ..."
Write-Host "  Aurora: trains Phase 1 (deckbuilder) + Phase 2 (sideboard) + Phase 3 (play)"
Write-Host "  Briar:  fixed deck — only Phase 3 (play agent) trains"
Write-Host ""

& $Python "$ScriptDir\train_full_pipeline.py" `
    --format              silver_age `
    --hero-id             $AuroraHeroId `
    --hero-class          $AuroraHeroClass `
    --equipment-header    $AuroraEquipment `
    --opponent-mode       dual `
    --p2-hero-id          $BriarHeroId `
    --p2-hero-class       $BriarHeroClass `
    --p2-equipment-header $BriarEquipment `
    --deckbuild-episodes  $DeckbuildEpisodes `
    --sideboard-episodes  $SideboardEpisodes `
    --play-episodes       $PlayEpisodes `
    --num-eval-games      $NumEvalGames `
    --num-sideboard-episodes $NumSideboardEpisodes `
    --iterations          $Iterations `
    --final-eval-episodes $FinalEvalEpisodes `
    --final-eval-max-steps $FinalEvalMaxSteps `
    --gif-fps             $GifFps `
    --talishar-url        $TalisharUrl `
    --assets-path         $AssetsPath `
    --out-dir             $OutDir `
    --results-json        $ResultsJson `
    $(if ($CppEngineBuildSucceeded) { @("--cpp-engine-dir", $CppEngineDir) } else { @() }) `
    $(if ($null -ne $PlayWorkers)   { @("--workers",         $PlayWorkers) } else { @() }) `
    @AuroraStartArgs `
    @BriarFixedArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: train_full_pipeline.py exited with code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# ─── Step 4: Summary ─────────────────────────────────────────────────────────

Write-Host ""
Write-Host "================================================================"
Write-Host "  Training complete"
Write-Host "================================================================"
if (Test-Path $ResultsJson) {
    Write-Host "  Results -> $ResultsJson"
    $results = Get-Content $ResultsJson | ConvertFrom-Json

    Write-Host ""
    Write-Host "  Aurora (P1 — trained deckbuilder + sideboard + play)"
    if ($results.p1.win_rates) {
        Write-Host ("    Training win rates : " + ($results.p1.win_rates -join ", "))
    }
    if ($results.p1.final_eval) {
        $fe  = $results.p1.final_eval
        $pct = [math]::Round($fe.win_rate * 100, 1)
        $rec = "$($fe.wins)W/$($fe.losses)L/$($fe.draws)D"
        Write-Host "    Final eval win%    : ${pct}%  ($rec)"
    }
    if ($results.p1.final_eval_gif) {
        Write-Host "    Render GIF         : $($results.p1.final_eval_gif)"
    }

    Write-Host ""
    Write-Host "  Briar (P2 — fixed deck, play agent only)"
    if ($results.p2.win_rates) {
        Write-Host ("    Training win rates : " + ($results.p2.win_rates -join ", "))
    }
    if ($results.p2.final_eval) {
        $fe  = $results.p2.final_eval
        $pct = [math]::Round($fe.win_rate * 100, 1)
        $rec = "$($fe.wins)W/$($fe.losses)L/$($fe.draws)D"
        Write-Host "    Final eval win%    : ${pct}%  ($rec)"
    }
    if ($results.p2.final_eval_gif) {
        Write-Host "    Render GIF         : $($results.p2.final_eval_gif)"
    }
}
Write-Host ""
Write-Host "  Agents + results saved to: $OutDir"
Write-Host ""
