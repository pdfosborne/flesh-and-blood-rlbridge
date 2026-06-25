# Detect whether Docker can pass an NVIDIA GPU into containers.
# Sets $env:FAB_DOCKER_GPU = "1" when a smoke test succeeds.

function Test-FabDockerGpu {
    $env:FAB_DOCKER_GPU = "0"

    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        return
    }
    try {
        docker info *> $null
    } catch {
        return
    }

    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        return
    }
    try {
        nvidia-smi *> $null
    } catch {
        return
    }

    try {
        docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi *> $null
        if ($LASTEXITCODE -eq 0) {
            $env:FAB_DOCKER_GPU = "1"
        }
    } catch {
        $env:FAB_DOCKER_GPU = "0"
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    Test-FabDockerGpu
    Write-Output $env:FAB_DOCKER_GPU
}
