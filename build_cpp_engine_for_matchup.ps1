<#
.SYNOPSIS
    Generate and build a C++ FAB engine for a specific deck matchup.

.DESCRIPTION
    Given two Talishar Assets deck names this script:
      1. Generates C++ source via generate_cpp_engine.py
      2. Checks/installs pybind11
      3. Configures and builds the CMake project
      4. Verifies the compiled fab_engine module
      5. Prints a usage summary

    The compiled engine is cached in:
        results/cpp_engines/<Deck1>_vs_<Deck2>/

    On the next training run, TalisharEngineEnvironment auto-detects
    and uses the cached engine (no code changes required).

.PARAMETER Deck1
    Talishar Assets deck name for player 1

.PARAMETER Deck2
    Talishar Assets deck name for player 2

.PARAMETER TalisharSrc
    Path to the Talishar PHP source root (default: .\Talishar)

.PARAMETER TalisharUrl
    URL of the running Talishar server (default: $env:TALISHAR_URL or http://localhost)

.PARAMETER NoServer
    Skip live Talishar game; use PHP source scan only.

.PARAMETER NoBuild
    Generate C++ source but skip cmake build.

.PARAMETER CacheDir
    Override the default cache root (default: .\results\cpp_engines)

.PARAMETER PipelineJson
    Path to a results JSON from train_full_pipeline.py or
    train_three_phase_pipeline.py. Deck1/Deck2 are read automatically.

.EXAMPLE
    .\build_cpp_engine_for_matchup.ps1 -Deck1 Ira -Deck2 Ira

.EXAMPLE
    .\build_cpp_engine_for_matchup.ps1 -Deck1 BriarSAGEPrecon -Deck2 DorintheSAGEPrecon

.EXAMPLE
    .\build_cpp_engine_for_matchup.ps1 -PipelineJson results/full_pipeline/results.json -NoServer

.EXAMPLE
    .\build_cpp_engine_for_matchup.ps1 -Deck1 Ira -Deck2 Ira -NoBuild
#>

[CmdletBinding()]
param(
    [string]$Deck1        = "",
    [string]$Deck2        = "",
    [string]$TalisharSrc  = "Talishar",
    [string]$TalisharUrl  = "",
    [switch]$NoServer,
    [switch]$NoBuild,
    [string]$CacheDir     = "results\cpp_engines",
    [string]$PipelineJson = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Header {
    param([string]$text)
    Write-Host ""
    Write-Host ("=" * 62) -ForegroundColor Cyan
    Write-Host "  $text" -ForegroundColor Cyan
    Write-Host ("=" * 62) -ForegroundColor Cyan
}

function Write-Step {
    param([string]$n, [string]$text)
    Write-Host ""
    Write-Host "Step $n : $text" -ForegroundColor Yellow
}

function Write-Ok   { param([string]$t); Write-Host "  OK   $t" -ForegroundColor Green }
function Write-Warn { param([string]$t); Write-Host "  WARN $t" -ForegroundColor DarkYellow }
function Write-Fail { param([string]$t); Write-Host "  FAIL $t" -ForegroundColor Red }

# ---------------------------------------------------------------------------
# Locate repo root
# ---------------------------------------------------------------------------

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $RepoRoot

# ---------------------------------------------------------------------------
# Step 0 : Read deck names from pipeline JSON if supplied
# ---------------------------------------------------------------------------

if ($PipelineJson -ne "") {
    Write-Step "0" "Reading deck names from pipeline JSON"
    $jsonPath = Resolve-Path $PipelineJson -ErrorAction SilentlyContinue
    if (-not $jsonPath) {
        Write-Fail "Pipeline JSON not found: $PipelineJson"
        exit 1
    }
    $pj = Get-Content $jsonPath | ConvertFrom-Json

    if ($Deck1 -eq "" -and $pj.PSObject.Properties["p1"].Value.PSObject.Properties["deck_asset_name"]) {
        $Deck1 = $pj.p1.deck_asset_name
    }
    if ($Deck2 -eq "" -and $pj.PSObject.Properties["p2"].Value.PSObject.Properties["deck_asset_name"]) {
        $Deck2 = $pj.p2.deck_asset_name
    }
    if ($Deck1 -eq "" -and $pj.PSObject.Properties["hero_id"]) {
        $Deck1 = $pj.hero_id
    }
    if ($Deck2 -eq "" -and $pj.PSObject.Properties["opponent_deck_name"]) {
        $Deck2 = $pj.opponent_deck_name
    }
    Write-Ok "Deck1=$Deck1  Deck2=$Deck2"
}

# ---------------------------------------------------------------------------
# Validate required args
# ---------------------------------------------------------------------------

if ($Deck1 -eq "" -or $Deck2 -eq "") {
    Write-Fail "Deck1 and Deck2 must be specified (or supply -PipelineJson)."
    Write-Host ""
    Write-Host "Usage examples:"
    Write-Host "  .\build_cpp_engine_for_matchup.ps1 -Deck1 Ira -Deck2 Ira"
    Write-Host "  .\build_cpp_engine_for_matchup.ps1 -Deck1 BriarSAGEPrecon -Deck2 DorintheSAGEPrecon"
    exit 1
}

$MatchupKey = "${Deck1}_vs_${Deck2}"
$EngineDir  = Join-Path (Join-Path $RepoRoot $CacheDir) $MatchupKey

if ($TalisharUrl -ne "") {
    $BaseUrl = $TalisharUrl
} elseif ($env:TALISHAR_URL) {
    $BaseUrl = $env:TALISHAR_URL
} else {
    $BaseUrl = "http://localhost"
}

Write-Header "FAB C++ Engine Builder"
Write-Host "  Matchup    : $Deck1  vs  $Deck2"
Write-Host "  Engine dir : $EngineDir"
Write-Host "  PHP source : $TalisharSrc"
Write-Host "  Server URL : $BaseUrl"
if ($NoServer) { Write-Host "  Mode       : PHP scan only (--no-server)" -ForegroundColor DarkYellow }

# ---------------------------------------------------------------------------
# Step 1 : Generate C++ source
# ---------------------------------------------------------------------------

Write-Step "1" "Generating C++ source"

$genArgs = @(
    "scripts\generate_cpp_engine.py",
    "--deck1",        $Deck1,
    "--deck2",        $Deck2,
    "--talishar-src", $TalisharSrc,
    "--out",          $EngineDir
)
if ($NoServer) { $genArgs += "--no-server" }
if ($BaseUrl -ne "http://localhost") { $genArgs += @("--base-url", $BaseUrl) }

python @genArgs
if ($LASTEXITCODE -ne 0) {
    Write-Fail "generate_cpp_engine.py failed (exit $LASTEXITCODE)"
    exit 1
}
Write-Ok "C++ source written to $EngineDir"

# ---------------------------------------------------------------------------
# Step 2 : Show card stub summary
# ---------------------------------------------------------------------------

Write-Step "2" "Reviewing generated card stubs"

$manifestPath = Join-Path $EngineDir "card_manifest.json"
if (Test-Path $manifestPath) {
    $manifest = Get-Content $manifestPath | ConvertFrom-Json
    $cards   = $manifest.cards.PSObject.Properties
    $total   = ($cards | Measure-Object).Count
    $withPhp = ($cards | Where-Object { $_.Value.php_found -eq $true } | Measure-Object).Count
    $noPhp   = $total - $withPhp

    Write-Host ""
    Write-Host "  Card stubs : $total total" -ForegroundColor White
    Write-Host "  PHP logic  : $withPhp (ready to translate)" -ForegroundColor Green
    if ($noPhp -gt 0) {
        Write-Host "  No PHP     : $noPhp (manual implementation needed)" -ForegroundColor DarkYellow
    } else {
        Write-Host "  No PHP     : 0" -ForegroundColor Green
    }
    Write-Host ""
    Write-Host "  cards.h : $(Join-Path $EngineDir 'cards.h')" -ForegroundColor White
    Write-Host "  Each stub has PHP logic as comments. Remove throw once implemented."
} else {
    Write-Warn "card_manifest.json not found - skipping card count summary"
}

# ---------------------------------------------------------------------------
# Step 3 : Check / install pybind11
# ---------------------------------------------------------------------------

Write-Step "3" "Checking pybind11"

$savedEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$p11ver = python -m pybind11 --version 2>&1
$p11exit = $LASTEXITCODE
$ErrorActionPreference = $savedEAP

if ($p11exit -eq 0) {
    Write-Ok "pybind11 already installed: $p11ver"
} else {
    Write-Warn "pybind11 not found - installing"
    python -m pip install pybind11 --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "python -m pip install pybind11 failed"
        exit 1
    }
    Write-Ok "pybind11 installed"
}

# ---------------------------------------------------------------------------
# Step 4 : Build
# ---------------------------------------------------------------------------

if ($NoBuild) {
    Write-Host ""
    Write-Host "  Skipping build (-NoBuild flag set)." -ForegroundColor DarkYellow
    Write-Host "  Build manually:"
    Write-Host "    cd `"$EngineDir`""
    Write-Host "    cmake -B build -DCMAKE_BUILD_TYPE=Release ."
    Write-Host "    cmake --build build --config Release"
} else {
    Write-Step "4" "Building C++ engine"

    $cmakeCmd = Get-Command cmake -ErrorAction SilentlyContinue
    if (-not $cmakeCmd) {
        # Try common install paths not always on PATH
        $cmakeCandidates = @(
            "C:\Program Files\CMake\bin\cmake.exe",
            "C:\Program Files (x86)\CMake\bin\cmake.exe",
            "${env:ProgramFiles}\CMake\bin\cmake.exe"
        )
        foreach ($candidate in $cmakeCandidates) {
            if (Test-Path $candidate) {
                $env:PATH = "$([System.IO.Path]::GetDirectoryName($candidate));$env:PATH"
                $cmakeCmd = Get-Command cmake -ErrorAction SilentlyContinue
                Write-Ok "Found cmake at: $candidate"
                break
            }
        }
    }
    if (-not $cmakeCmd) {
        Write-Fail "cmake not found. Install from https://cmake.org/download/"
        exit 1
    }
    $cmakeVer = (cmake --version 2>&1 | Select-Object -First 1).ToString()
    Write-Ok $cmakeVer

    Write-Host "  Locating pybind11 cmake dir..."
    $pybind11Dir = ""
    $savedEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $p11cmake = python -m pybind11 --cmakedir 2>&1
    $p11exit2 = $LASTEXITCODE
    $ErrorActionPreference = $savedEAP
    if ($p11exit2 -eq 0) {
        $pybind11Dir = $p11cmake.ToString().Trim()
    } else {
        Write-Warn "pybind11 cmake dir lookup failed - installing"
        python -m pip install pybind11 --quiet
        $savedEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $p11cmake = python -m pybind11 --cmakedir 2>&1
        $p11exit2 = $LASTEXITCODE
        $ErrorActionPreference = $savedEAP
        if ($p11exit2 -ne 0) {
            Write-Fail "pybind11 --cmakedir failed: $p11cmake"
            exit 1
        }
        $pybind11Dir = $p11cmake.ToString().Trim()
    }
    Write-Ok "pybind11 cmake dir: $pybind11Dir"

    $generatorArgs = @()
    if (Get-Command ninja -ErrorAction SilentlyContinue) {
        $generatorArgs = @("-G", "Ninja")
        Write-Ok "Generator: Ninja"
    } elseif (Get-Command cl -ErrorAction SilentlyContinue) {
        # Inside a VS Developer shell already - let cmake auto-detect
        Write-Ok "Generator: auto (cl.exe found)"
    } else {
        # Not in a VS shell and no Ninja - try to locate VS and use its generator
        $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
        if (-not (Test-Path $vswhere)) {
            $vswhere = "${env:ProgramFiles}\Microsoft Visual Studio\Installer\vswhere.exe"
        }
        if (Test-Path $vswhere) {
            $vsVer  = (& $vswhere -latest -property installationVersion 2>$null | Select-Object -First 1)
            $vsMajor = 0
            if ($vsVer -and $vsVer.ToString().Trim() -ne "") {
                $vsMajorStr = ($vsVer.ToString().Trim() -split '\.')[0]
                if ($vsMajorStr -match '^\d+$') { $vsMajor = [int]$vsMajorStr }
            }
            $generatorName = switch ($vsMajor) {
                17 { "Visual Studio 17 2022" }
                16 { "Visual Studio 16 2019" }
                15 { "Visual Studio 15 2017" }
                default { "Visual Studio 17 2022" }
            }
            $generatorArgs = @("-G", $generatorName, "-A", "x64")
            Write-Ok "Generator: $generatorName (via vswhere)"
        } elseif (Get-Command g++ -ErrorAction SilentlyContinue) {
            $generatorArgs = @("-G", "MinGW Makefiles")
            Write-Ok "Generator: MinGW Makefiles"
        } else {
            Write-Warn "No compiler found in PATH."
            Write-Host ""
            Write-Host "  Options:"
            Write-Host "  1. Run this script from a VS Developer PowerShell:"
            Write-Host "     Start > Visual Studio > Developer PowerShell for VS"
            Write-Host "  2. Install Ninja + MSVC Build Tools, then re-run"
            Write-Host "  3. Install MinGW (g++) and re-run"
            Write-Host ""
            Write-Host "  Build manually once a compiler is available:"
            Write-Host "    cd `"$EngineDir`""
            Write-Host "    cmake -B build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=`"$pybind11Dir`" ."
            Write-Host "    cmake --build build --config Release"
            exit 1
        }
    }

    Push-Location $EngineDir
    try {
        # Remove stale CMakeCache so a generator change never causes an error
        $buildDir = Join-Path $EngineDir "build"
        if (Test-Path (Join-Path $buildDir "CMakeCache.txt")) {
            Write-Host "  Clearing stale CMake cache..."
            Remove-Item -Recurse -Force $buildDir -ErrorAction SilentlyContinue
        }

        Write-Host "  Configuring..."
        $cfgArgs = @("-B", "build", "-DCMAKE_BUILD_TYPE=Release", "-Dpybind11_DIR=$pybind11Dir") + $generatorArgs + @(".")
        cmake @cfgArgs 2>&1 | ForEach-Object { "    $_" }
        if ($LASTEXITCODE -ne 0) {
            Write-Fail "cmake configure failed"
            exit 1
        }

        Write-Host "  Compiling..."
        cmake --build build --config Release 2>&1 | ForEach-Object { "    $_" }
        if ($LASTEXITCODE -ne 0) {
            Write-Fail "cmake build failed - check errors above"
            exit 1
        }
    } finally {
        Pop-Location
    }

    # -------------------------------------------------------------------------
    # Step 5 : Verify
    # -------------------------------------------------------------------------

    Write-Step "5" "Verifying compiled module"

    $module = Get-ChildItem -Path $EngineDir -Filter "fab_engine*" -File |
              Where-Object { $_.Extension -in @(".pyd", ".so") } |
              Select-Object -First 1

    if (-not $module) {
        $buildPath = Join-Path $EngineDir "build"
        if (Test-Path $buildPath) {
            $module = Get-ChildItem -Path $buildPath -Recurse -Filter "fab_engine*" -File |
                      Where-Object { $_.Extension -in @(".pyd", ".so") } |
                      Select-Object -First 1
        }
    }

    if ($module) {
        Write-Ok "Module: $($module.FullName)"

        $tmpPy = [System.IO.Path]::GetTempFileName() + ".py"
        $engineFwd = $EngineDir.Replace('\', '/')
        $pyCode  = "import sys" + [char]10
        $pyCode += "sys.path.insert(0, '" + $engineFwd + "')" + [char]10
        $pyCode += "import fab_engine" + [char]10
        $pyCode += "gs = fab_engine.GameState()" + [char]10
        $pyCode += "gs.register_all_cards()" + [char]10
        $pyCode += "legal = gs.get_legal_actions()" + [char]10
        $pyCode += "print(str(len(legal)) + ' legal actions on fresh state')" + [char]10
        [System.IO.File]::WriteAllText($tmpPy, $pyCode, [System.Text.Encoding]::UTF8)

        $testOut = python $tmpPy 2>&1
        Remove-Item $tmpPy -ErrorAction SilentlyContinue

        if ($LASTEXITCODE -eq 0) {
            Write-Ok $testOut
        } else {
            Write-Warn "Import test failed (likely unimplemented card stubs):"
            Write-Host "    $testOut" -ForegroundColor DarkYellow
            Write-Host "  This is expected until cards.h stubs are fully implemented."
        }
    } else {
        Write-Fail "Compiled module not found in $EngineDir"
        exit 1
    }
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

Write-Header "Done"
Write-Host ""
Write-Host "  Engine dir : $EngineDir" -ForegroundColor White
Write-Host "  Matchup    : $MatchupKey" -ForegroundColor White
Write-Host ""
Write-Host "  TalisharEngineEnvironment will auto-detect and use this engine" -ForegroundColor Green
Write-Host "  on the next training run (no code changes needed)."
Write-Host ""
Write-Host "  To regenerate after editing cards.h:"
Write-Host "    .\build_cpp_engine_for_matchup.ps1 -Deck1 $Deck1 -Deck2 $Deck2 -NoBuild"
Write-Host ""
Write-Host "  To bypass the C++ engine and use HTTP Talishar:"
Write-Host "    TalisharEngineEnvironment(..., use_cpp_engine=False)"
Write-Host ""

Pop-Location
