#!/usr/bin/env pwsh

$ResultsDir = if ($env:RESULTS_DIR) {
    $env:RESULTS_DIR
} else {
    Join-Path $PSScriptRoot "results/matchup_sims/briar_vs_riptide"
}

$AssetsPath = if ($env:TALISHAR_ASSETS_PATH) {
    $env:TALISHAR_ASSETS_PATH
} else {
    Join-Path $PSScriptRoot "Talishar\Assets"
}

$TalisharUrl = if ($env:TALISHAR_URL) { $env:TALISHAR_URL } else { "http://localhost:8080/game" }
$Episodes    = if ($env:EPISODES)    { $env:EPISODES }    else { "10" }
$ParallelWorkers = if ($env:PARALLEL_WORKERS) { $env:PARALLEL_WORKERS } else { "4" }
$MaxSteps    = if ($env:MAX_STEPS)   { $env:MAX_STEPS }   else { "1000" }
$RenderMaxSteps = if ($env:RENDER_MAX_STEPS) { $env:RENDER_MAX_STEPS } else { "200" }
$PollSeconds = if ($env:POLL_SECONDS){ $env:POLL_SECONDS } else { "30" }
$StallNoDamageTurns = if ($env:STALL_NO_DAMAGE_TURNS) { $env:STALL_NO_DAMAGE_TURNS } else { "6" }
$StallLowHandTurns = if ($env:STALL_LOW_HAND_TURNS) { $env:STALL_LOW_HAND_TURNS } else { "3" }
$StallMaxSingleLowHandTurns = if ($env:STALL_MAX_SINGLE_LOW_HAND_TURNS) { $env:STALL_MAX_SINGLE_LOW_HAND_TURNS } else { "5" }
$StallMinAttackHand = if ($env:STALL_MIN_ATTACK_HAND) { $env:STALL_MIN_ATTACK_HAND } else { "2" }
$GifFps      = if ($env:GIF_FPS)    { $env:GIF_FPS }     else { "3" }
$Python      = if ($env:PYTHON)     { $env:PYTHON }      else { "python" }


$ScriptPath  = Join-Path $PSScriptRoot "scripts/eval_phase3_checkpoint.py"
$ForwardArgs = @($args)

if ($ForwardArgs -contains "-?" -or $ForwardArgs -contains "/?") {
    $ForwardArgs = @("--help")
}

# Auto-discover compiled C++ engine for this matchup (e.g. Briar_vs_Riptide-<hash>).
$ParityCppArgs = @()
if ($env:CPP_ENGINE_DIR) {
    $ParityCppArgs = @("--parity-cpp-engine-dir", $env:CPP_ENGINE_DIR)
} else {
    $CacheRoot = Join-Path $PSScriptRoot "results\cpp_engines"
    $DirName   = Split-Path $ResultsDir -Leaf
    if ($DirName -match '^(.+)_vs_(.+)$') {
        $d1 = (Get-Culture).TextInfo.ToTitleCase($Matches[1].ToLower())
        $d2 = (Get-Culture).TextInfo.ToTitleCase($Matches[2].ToLower())
        $Label = "${d1}_vs_${d2}"
        $CppEngineDir = Get-ChildItem $CacheRoot -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like "${Label}-*" -or $_.Name -eq $Label } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1 -ExpandProperty FullName
        if ($CppEngineDir) {
            Write-Host "  Parity C++ engine: $CppEngineDir"
            $ParityCppArgs = @("--parity-cpp-engine-dir", $CppEngineDir)
        }
    }
}

& $Python $ScriptPath `
    --results-dir  $ResultsDir `
    --assets-path  $AssetsPath `
    --talishar-url $TalisharUrl `
    --episodes     $Episodes `
    --parallel-workers $ParallelWorkers `
    --max-steps    $MaxSteps `
    --render-max-steps $RenderMaxSteps `
        --stall-no-damage-turns $StallNoDamageTurns `
        --stall-low-hand-turns $StallLowHandTurns `
        --stall-max-single-low-hand-turns $StallMaxSingleLowHandTurns `
        --stall-min-attack-hand $StallMinAttackHand `
    --watch `
    --poll-seconds $PollSeconds `
    --gif-fps      $GifFps `
    @ParityCppArgs `
    @ForwardArgs

exit $LASTEXITCODE