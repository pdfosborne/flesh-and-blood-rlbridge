#!/usr/bin/env pwsh

$ResultsDir = if ($env:RESULTS_DIR) {
    $env:RESULTS_DIR
} else {
    Join-Path $PSScriptRoot "results/full_pipeline"
}

$Episodes = if ($env:EPISODES) { $env:EPISODES } else { "100" }
$MaxSteps = if ($env:MAX_STEPS) { $env:MAX_STEPS } else { "60" }
$PollSeconds = if ($env:POLL_SECONDS) { $env:POLL_SECONDS } else { "30" }
$GifFps = if ($env:GIF_FPS) { $env:GIF_FPS } else { "3" }
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python3" }

$ScriptPath = Join-Path $PSScriptRoot "scripts/eval_phase3_checkpoint.py"
$ForwardArgs = @($args)

if ($ForwardArgs -contains "-?" -or $ForwardArgs -contains "/?") {
    $ForwardArgs = @("--help")
}

& $Python $ScriptPath `
    --results-dir $ResultsDir `
    --episodes $Episodes `
    --max-steps $MaxSteps `
    --watch `
    --poll-seconds $PollSeconds `
    --gif-fps $GifFps `
    @ForwardArgs

exit $LASTEXITCODE