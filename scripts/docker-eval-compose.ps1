# Configure COMPOSE_FILE for eval-only Docker stack (Windows).
param(
    [ValidateSet("eval", "full")]
    [string]$Stack = "eval"
)

$Root = Split-Path -Parent $PSScriptRoot

function Initialize-FabDockerEvalCompose {
    param([string]$Mode = "eval")
    if ($Mode -eq "eval") {
        $env:FAB_DOCKER_STACK = "eval"
        if (-not $env:COMPOSE_FILE) {
            $env:COMPOSE_FILE = "docker-compose.yml:docker-compose.eval.yml"
        } elseif ($env:COMPOSE_FILE -notlike "*docker-compose.eval.yml*") {
            $env:COMPOSE_FILE = "$($env:COMPOSE_FILE):docker-compose.eval.yml"
        }
    }
}

function Write-FabDockerEvalComposeNote {
    if ($env:FAB_DOCKER_STACK -eq "eval") {
        Write-Host "[docker] Eval stack — Talishar backend + GUI (no Talishar-FE / Playwright)"
    }
}
