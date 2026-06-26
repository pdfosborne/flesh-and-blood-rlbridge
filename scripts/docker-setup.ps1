# Build and start the Docker stack (Windows). Enables GPU overlay when CUDA is available.
param(
    [switch]$Foreground,
    [switch]$Logs,
    [switch]$Eval
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Get-ChildItem bin\fab-* -ErrorAction SilentlyContinue | ForEach-Object { $_.IsReadOnly = $false }

if (-not $Eval) {
    if (-not (Test-Path Talishar-FE\package.json)) {
        Write-Host "[setup] Cloning Talishar-FE..."
        git clone --depth 1 https://github.com/Talishar/Talishar-FE Talishar-FE
    }
}

. "$Root\scripts\docker-gpu-compose.ps1"
Initialize-FabDockerGpuCompose
Write-FabDockerGpuComposeNote

if ($Eval) {
    . "$Root\scripts\docker-eval-compose.ps1"
    Initialize-FabDockerEvalCompose -Mode eval
    Write-FabDockerEvalComposeNote
    $composeProfile = @("--profile", "eval")
} else {
    $env:FAB_DOCKER_STACK = "full"
    $composeProfile = @("--profile", "full")
}

Write-Host "[setup] Building and starting Docker stack..."
& docker compose @composeProfile up --build -d @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "[setup] Waiting for services..."
$urls = if ($Eval) {
    @(
        @{ Url = "http://localhost:8080/"; Label = "Talishar backend" },
        @{ Url = "http://localhost:8765/"; Label = "Web GUI" }
    )
} else {
    @(
        @{ Url = "http://localhost:8080/"; Label = "Talishar backend" },
        @{ Url = "http://localhost:5173/"; Label = "Talishar-FE" },
        @{ Url = "http://localhost:8765/"; Label = "Web GUI" }
    )
}
foreach ($entry in $urls) {
    Write-Host -NoNewline "[setup] $($entry.Label)..."
    $ok = $false
    foreach ($i in 1..120) {
        try {
            Invoke-WebRequest -Uri $entry.Url -UseBasicParsing -TimeoutSec 2 | Out-Null
            $ok = $true
            break
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    if ($ok) { Write-Host " OK" } else { Write-Host " TIMEOUT" }
}

if ($Eval -and (Test-Path docker\ready-message-eval.txt)) {
    Get-Content docker\ready-message-eval.txt
} elseif (Test-Path docker\ready-message.txt) {
    Get-Content docker\ready-message.txt
} else {
    Write-Host "Setup complete. Open http://localhost:8765"
}

if ($Logs) {
    docker compose logs -f --tail=50
}
