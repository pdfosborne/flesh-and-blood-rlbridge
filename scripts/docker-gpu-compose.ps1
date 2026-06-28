# Configure COMPOSE_FILE for optional GPU support on fab-bridge.

$Root = Split-Path -Parent $PSScriptRoot
. "$Root\scripts\docker-gpu-detect.ps1"

function Initialize-FabDockerGpuCompose {
    Test-FabDockerGpu
    if ($env:FAB_DOCKER_GPU -eq "1") {
        $env:COMPOSE_FILE = "docker-compose.yml;docker-compose.gpu.yml"
    } else {
        $env:COMPOSE_FILE = "docker-compose.yml"
    }
}

function Write-FabDockerGpuComposeNote {
    if ($env:FAB_DOCKER_GPU -eq "1") {
        Write-Host "[docker] GPU detected - fab-bridge will use CUDA PyTorch (gpus: `all`)"
    } else {
        Write-Host "[docker] No Docker GPU - fab-bridge training uses CPU PyTorch"
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    Initialize-FabDockerGpuCompose
    Write-FabDockerGpuComposeNote
    Write-Output "COMPOSE_FILE=$($env:COMPOSE_FILE)"
}
