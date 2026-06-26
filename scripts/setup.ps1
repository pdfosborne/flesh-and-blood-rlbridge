# First-time local setup: venv, editable install, Talishar backend, then launch GUI.
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found. Install Python 3.10+ or set `$env:PYTHON."
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment in .venv"
    & $Python -m venv .venv
}

& .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[gui,cpp]"

fab-bridge init

Write-Host ""
Write-Host "Setup complete. Start the web GUI with:"
Write-Host "  .\.venv\Scripts\Activate.ps1; fab-gui"
Write-Host ""
Write-Host "Or use Docker for Talishar + GUI together:"
Write-Host "  .\scripts\docker-setup.ps1 -Eval    # eval-only (recommended)"
Write-Host "  .\scripts\docker-setup.ps1          # full stack with Talishar-FE"
