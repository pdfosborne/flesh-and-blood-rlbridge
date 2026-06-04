#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Train & evaluate the full 3-phase RL pipeline for
    Aurora vs Briar (Young Silver Age heroes) in Silver Age.

.DESCRIPTION
    Phase 1 — Deckbuilder: optimise the 55-card registered pool for each hero
    Phase 2 — Sideboard:   select the 40-card game deck vs the opposing hero
    Phase 3 — Play:        co-evolution self-play (dual mode)

    FaBrary decks used as warm-start pools
    ----------------------------------------
    Aurora  : https://fabrary.net/decks/01KST88R7JVEQ73M82ZA0PJ9RN
    Briar   : https://fabrary.net/decks/01KGZPKM6NBVNFYEEWWS4SGFQ7

.NOTES
    Set $env:TALISHAR_URL before running if your local Talishar is not on
    the default http://localhost:8080/game path.

    Set $env:FABRARY_API_KEY if you have a FaBrary API key; otherwise the
    fetch step will attempt an unauthenticated request and fall back to
    skipping the warm-start if the API is not accessible.
#>

# ─── Configuration ────────────────────────────────────────────────────────────

$TalisharUrl    = if ($env:TALISHAR_URL)    { $env:TALISHAR_URL }    else { "http://localhost:8080/game" }
$AssetsPath     = if ($env:TALISHAR_ASSETS_PATH) { $env:TALISHAR_ASSETS_PATH } else {
    Join-Path $PSScriptRoot "Talishar\Assets"
}
$OutDir         = Join-Path $PSScriptRoot "..\results\auroraPaulvanGijssel_vs_briarAjanell"
$DeckDir        = Join-Path $OutDir "sage_decks"
$ResultsJson    = Join-Path $OutDir "results.json"

# FaBrary deck slugs
$AuroraSlug     = "01KST88R7JVEQ73M82ZA0PJ9RN"
$BriarSlug      = "01KGZPKM6NBVNFYEEWWS4SGFQ7"
$AuroraDeckJson = Join-Path $DeckDir "aurora_PaulvanGijssel_deck.json"
$BriarDeckJson  = Join-Path $DeckDir "briar_Ajanell_deck.json"

# Hero identifiers (Talishar internal IDs)
# Aurora and Briar are the Silver Age Young hero versions (20 life, SA legal)
$AuroraHeroId      = "aurora"
$AuroraHeroClass   = "Runeblade"
# Equipment: hero  weapon  head  chest  arms  legs
# Star Fall is the 2H Lightning Runeblade weapon both SA heroes equip.
# Head/chest/arms/legs taken from the BriarSAGEPrecon (same class/talent).
$AuroraEquipment   = "aurora star_fall aether_ironweave spellbound_creepers aether_crackers crown_of_dichotomy"

# Briar is the Silver Age Young hero version (20 life, SA legal)
$BriarHeroId       = "briar"
$BriarHeroClass    = "Runeblade"
$BriarEquipment    = "briar star_fall aether_ironweave spellbound_creepers aether_crackers crown_of_dichotomy"

# Training volume  (adjust for longer/faster runs)
$DeckbuildEpisodes      = 80
$SideboardEpisodes      = 30
$PlayEpisodes           = 50
$NumEvalGames           = 3
$NumSideboardEpisodes   = 5   # sideboard runs *inside* each deckbuilder eval
$Iterations             = 3

# Final evaluation (post-training)
$FinalEvalEpisodes      = 20  # eval games with best deck + optimal policy
$FinalEvalMaxSteps      = 60
$GifFps                 = 1.0

# ─── Setup ────────────────────────────────────────────────────────────────────

$ScriptDir  = Join-Path $PSScriptRoot "scripts"
$Python     = "python"

New-Item -ItemType Directory -Force -Path $DeckDir | Out-Null

Write-Host ""
Write-Host "========================================================"
Write-Host "  Aurora vs Briar — Silver Age — 3-Phase RL Pipeline"
Write-Host "========================================================"
Write-Host "  Talishar URL : $TalisharUrl"
Write-Host "  Assets path  : $AssetsPath"
Write-Host "  Output dir   : $OutDir"
Write-Host ""

# ─── Step 1: Fetch FaBrary decks (warm-start pools) ─────────────────────────

function Get-FabraryDeck {
    param(
        [string]$Slug,
        [string]$OutFile,
        [string]$Label
    )
    if (Test-Path $OutFile) {
        Write-Host "  [$Label] Starting deck already fetched -> $OutFile"
        return $true
    }
    Write-Host ""
    Write-Host "  [$Label] Fetching FaBrary deck $Slug ..."
    & $Python "$ScriptDir\fetch_fabrary_deck.py" `
        "https://fabrary.net/decks/$Slug" `
        --out $OutFile `
        --pretty
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [$Label] Saved -> $OutFile"
        return $true
    } else {
        Write-Host "  [$Label] WARNING: fetch failed (exit $LASTEXITCODE) - running without warm-start"
        return $false
    }
}

$AuroraFetched = Get-FabraryDeck -Slug $AuroraSlug -OutFile $AuroraDeckJson -Label "Aurora"
$BriarFetched  = Get-FabraryDeck -Slug $BriarSlug  -OutFile $BriarDeckJson  -Label "Briar"

# Build optional starting-deck args
$StartingDeckArgs = @()
if ($AuroraFetched -and (Test-Path $AuroraDeckJson)) {
    $StartingDeckArgs += "--p1-starting-deck"; $StartingDeckArgs += $AuroraDeckJson
}
if ($BriarFetched -and (Test-Path $BriarDeckJson)) {
    $StartingDeckArgs += "--p2-starting-deck"; $StartingDeckArgs += $BriarDeckJson
}

# ─── Step 2: Build C++ engine for this matchup ───────────────────────────────

Write-Host ""
Write-Host "  Building C++ engine for Aurora vs Briar..."
Write-Host ""

& "$PSScriptRoot\build_cpp_engine_for_matchup.ps1" `
    -Deck1       $AuroraHeroId `
    -Deck2       $BriarHeroId `
    -TalisharSrc (Join-Path $PSScriptRoot "Talishar") `
    -TalisharUrl $TalisharUrl

$CppEngineBuildSucceeded = ($LASTEXITCODE -eq 0)
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  WARNING: C++ engine build failed (exit $LASTEXITCODE)." -ForegroundColor DarkYellow
    Write-Host "  Continuing — training will fall back to HTTP Talishar." -ForegroundColor DarkYellow
    Write-Host "  Fix errors above and re-run to get the speed benefit." -ForegroundColor DarkYellow
    Write-Host ""
} else {
    Write-Host "  C++ engine build succeeded."
    Write-Host "  Expected runtime backend: C++ engine (with automatic HTTP fallback if unavailable per matchup)."
    Write-Host ""
}

# ─── Step 3: Run the full pipeline ───────────────────────────────────────────

Write-Host ""
Write-Host "  Starting train_full_pipeline.py ..."
if ($CppEngineBuildSucceeded) {
    Write-Host "  Backend selection: C++ preferred; train_full_pipeline.py will print actual runtime backend."
} else {
    Write-Host "  Backend selection: HTTP Talishar expected; train_full_pipeline.py will print actual runtime backend."
}
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
    @StartingDeckArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: train_full_pipeline.py exited with code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# ─── Step 4: Print summary ───────────────────────────────────────────────────

Write-Host ""
Write-Host "========================================================"
Write-Host "  Training complete"
Write-Host "========================================================"
if (Test-Path $ResultsJson) {
    Write-Host "  Results -> $ResultsJson"
    $results = Get-Content $ResultsJson | ConvertFrom-Json
    Write-Host ""
    Write-Host "  Aurora (P1)"
    Write-Host ("    Training win rates : " + (($results.p1.win_rates | ForEach-Object { $_ }) -join ", "))
    if ($results.p1.final_eval) {
        $fe = $results.p1.final_eval
        $pct = [math]::Round($fe.win_rate * 100, 1)
        $rec = [string]$fe.wins + 'W/' + [string]$fe.losses + 'L/' + [string]$fe.draws + 'D'
        Write-Host ('    Final eval win%    : ' + [string]$pct + '%  (' + $rec + ')')
    }
    if ($results.p1.final_eval_gif) {
        Write-Host "    Render GIF         : $($results.p1.final_eval_gif)"
    }
    Write-Host ""
    Write-Host "  Briar (P2)"
    Write-Host ("    Training win rates : " + (($results.p2.win_rates | ForEach-Object { $_ }) -join ", "))
    if ($results.p2.final_eval) {
        $fe = $results.p2.final_eval
        $pct = [math]::Round($fe.win_rate * 100, 1)
        $rec = [string]$fe.wins + 'W/' + [string]$fe.losses + 'L/' + [string]$fe.draws + 'D'
        Write-Host ('    Final eval win%    : ' + [string]$pct + '%  (' + $rec + ')')
    }
    if ($results.p2.final_eval_gif) {
        Write-Host "    Render GIF         : $($results.p2.final_eval_gif)"
    }
}
Write-Host ""
Write-Host "  Agents saved to: $OutDir"
Write-Host ""
