# Run docker compose with optional GPU overlay when CUDA is available.
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ComposeArgs
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

. "$Root\scripts\docker-gpu-compose.ps1"
Initialize-FabDockerGpuCompose

& docker compose @ComposeArgs
exit $LASTEXITCODE
