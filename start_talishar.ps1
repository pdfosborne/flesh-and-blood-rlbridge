#Requires -Version 5.1
# =============================================================
#  start_talishar.ps1
#  Starts the Talishar backend (Docker Compose) and the
#  Talishar-FE Vite dev server, then prints the URLs.
#
#  Usage:
#    .\start_talishar.ps1              # start everything
#    .\start_talishar.ps1 -BackendOnly # skip the FE dev server
#    .\start_talishar.ps1 -FeOnly      # skip Docker (backend already up)
#    .\start_talishar.ps1 -Down        # stop backend containers
# =============================================================

param(
    [switch]$BackendOnly,
    [switch]$FeOnly,
    [switch]$Down
)

$ErrorActionPreference = "Stop"

$RepoRoot    = $PSScriptRoot
$TalisharDir = Join-Path $RepoRoot "Talishar"
$FeDir       = Join-Path $RepoRoot "Talishar-FE"
$BackendUrl  = "http://localhost:8080"
$FeUrl       = "http://localhost:5173"

function Write-Header {
    param([string]$Msg)
    Write-Host ""
    Write-Host "=== $Msg ===" -ForegroundColor Cyan
}

# --- Tear-down mode ----------------------------------------------------------
if ($Down) {
    Write-Header "Stopping Talishar backend containers"
    Push-Location $TalisharDir
    docker compose down
    Pop-Location
    Write-Host "Backend stopped." -ForegroundColor Green
    exit 0
}

# --- Backend (Docker Compose) ------------------------------------------------
if (-not $FeOnly) {
    Write-Header "Starting Talishar backend (Docker Compose)"

    if (-not (Test-Path $TalisharDir)) {
        Write-Error "Talishar directory not found: $TalisharDir"
        exit 1
    }

    Push-Location $TalisharDir
    try {
        docker compose up -d --build
        if ($LASTEXITCODE -ne 0) {
            Write-Error "docker compose up failed (exit $LASTEXITCODE)"
            exit 1
        }
    }
    finally {
        Pop-Location
    }

    Write-Host "Backend containers started." -ForegroundColor Green
    Write-Host "  API / game engine : $BackendUrl" -ForegroundColor Yellow
    Write-Host "  phpMyAdmin        : http://localhost:5001" -ForegroundColor Yellow

    # Poll until reachable (up to 30 s)
    Write-Host "  Waiting for backend to become reachable..." -NoNewline
    $maxWait = 30
    $ready   = $false
    for ($i = 0; $i -lt $maxWait; $i++) {
        try {
            $resp = Invoke-WebRequest -Uri "$BackendUrl/" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
            if ($resp.StatusCode -lt 500) {
                $ready = $true
                break
            }
        }
        catch { }
        Start-Sleep -Seconds 1
        Write-Host -NoNewline "."
    }
    Write-Host ""
    if ($ready) {
        Write-Host "  Backend is up." -ForegroundColor Green
    }
    else {
        Write-Host "  Backend did not respond within ${maxWait}s - containers may still be initialising." -ForegroundColor Yellow
    }
}

# --- Frontend (Vite dev server) ----------------------------------------------
if (-not $BackendOnly) {
    Write-Header "Starting Talishar-FE (Vite dev server)"

    if (-not (Test-Path $FeDir)) {
        Write-Error "Talishar-FE directory not found: $FeDir"
        exit 1
    }

    # Install dependencies if needed
    $nodeModules = Join-Path $FeDir "node_modules"
    if (-not (Test-Path $nodeModules)) {
        Write-Host "  node_modules not found - running npm install..." -ForegroundColor Yellow
        Push-Location $FeDir
        npm install
        Pop-Location
    }

    # Check if Vite is already running on port 5173
    $alreadyUp = $false
    try {
        $r = Invoke-WebRequest -Uri $FeUrl -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        if ($r.StatusCode -lt 500) {
            $alreadyUp = $true
        }
    }
    catch { }

    if ($alreadyUp) {
        Write-Host "  Vite dev server already running at $FeUrl" -ForegroundColor Green
    }
    else {
        # Launch in a new PowerShell window so it keeps running after this script exits
        $feCmd = "Set-Location '$FeDir'; npm run dev"
        Start-Process powershell -ArgumentList "-NoExit", "-Command", $feCmd -WindowStyle Normal
        Write-Host "  Vite dev server launched in a new window." -ForegroundColor Green
        Write-Host "  Frontend URL : $FeUrl" -ForegroundColor Yellow

        # Wait up to 20 s for Vite to bind the port
        Write-Host "  Waiting for Vite to start..." -NoNewline
        $feReady = $false
        for ($i = 0; $i -lt 20; $i++) {
            Start-Sleep -Seconds 1
            try {
                $r = Invoke-WebRequest -Uri $FeUrl -TimeoutSec 1 -UseBasicParsing -ErrorAction Stop
                if ($r.StatusCode -lt 500) {
                    $feReady = $true
                    break
                }
            }
            catch { }
            Write-Host -NoNewline "."
        }
        Write-Host ""
        if ($feReady) {
            Write-Host "  Frontend is up." -ForegroundColor Green
        }
        else {
            Write-Host "  Frontend did not respond yet - it may need a few more seconds." -ForegroundColor Yellow
        }
    }
}

# --- Summary -----------------------------------------------------------------
Write-Header "Talishar stack ready"
if (-not $FeOnly)      { Write-Host "  Backend  : $BackendUrl" -ForegroundColor Green }
if (-not $BackendOnly) { Write-Host "  Frontend : $FeUrl"      -ForegroundColor Green }

Write-Host ""
Write-Host "Set these env vars before training:" -ForegroundColor White
Write-Host '  $env:TALISHAR_URL    = "http://localhost:8080"' -ForegroundColor DarkYellow
Write-Host '  $env:TALISHAR_FE_URL = "http://localhost:5173"' -ForegroundColor DarkYellow
Write-Host ""
Write-Host "To stop the backend containers:  .\start_talishar.ps1 -Down" -ForegroundColor DarkGray
