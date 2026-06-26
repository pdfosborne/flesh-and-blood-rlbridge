# Build and start the Docker stack (Windows). Enables GPU overlay when CUDA is available.
param(
    [switch]$Foreground,
    [switch]$Logs
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Get-ChildItem bin\fab-* -ErrorAction SilentlyContinue | ForEach-Object { $_.IsReadOnly = $false }

if (-not (Test-Path Talishar-FE\package.json)) {
    Write-Host "[setup] Cloning Talishar-FE..."
    git clone --depth 1 https://github.com/Talishar/Talishar-FE Talishar-FE
}

. "$Root\scripts\docker-gpu-compose.ps1"
Initialize-FabDockerGpuCompose
Write-FabDockerGpuComposeNote

Write-Host "[setup] Building and starting Docker stack..."
& docker compose up --build -d @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "[setup] Waiting for services (Talishar-FE can take several minutes on first run)..."
$urls = @(
    @{ Url = "http://localhost:8080/"; Label = "Talishar backend" },
    @{ Url = "http://localhost:5173/"; Label = "Talishar-FE" },
    @{ Url = "http://localhost:8765/"; Label = "Web GUI" }
)
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

if (Test-Path docker\ready-message.txt) {
    Get-Content docker\ready-message.txt
} else {
    Write-Host "Setup complete. Open http://localhost:8765"
}

if ($Logs) {
    docker compose logs -f --tail=50
}
