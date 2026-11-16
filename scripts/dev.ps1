<#
.SYNOPSIS
    Windows counterpart of the Makefile: install, run and check the project natively.

.DESCRIPTION
    `make` targets rely on a POSIX shell (trap, kill 0, nc, `&`), so on Windows the same
    targets live here. Runs from any directory; the backend reads backend\.env, so its
    commands run from backend\, exactly as the Makefile does.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\dev.ps1 install
    powershell -ExecutionPolicy Bypass -File scripts\dev.ps1 run
    powershell -ExecutionPolicy Bypass -File scripts\dev.ps1 camera -WebcamDevice "Integrated Camera"
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("help", "install", "weights", "migrate", "run", "backend", "frontend",
        "devices", "rtsp", "webcam", "camera", "test", "lint", "typecheck", "check", "build")]
    [string]$Target = "help",

    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    # The lockfile's Windows PyTorch is the CUDA 13.0 build (NVIDIA only); it also runs
    # on CPU. Intel Iris/Arc graphics run through OpenVINO: set FALL_DEVICE=intel:gpu.
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",

    # DirectShow opens a camera by name; `dev.ps1 devices` lists them.
    [string]$WebcamDevice = "",
    [string]$WebcamSize = "1280x720",
    [int]$WebcamFps = 30,
    [int]$RtspPort = 8554,
    # Empty lets mediamtx look for mediamtx.yml next to its executable.
    [string]$MediamtxConfig = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Frontend = Join-Path $Root "frontend"
$RtspUrl = "rtsp://127.0.0.1:$RtspPort/webcam"

function Invoke-Native {
    # Run a native command in a directory and stop on a non-zero exit code, which
    # PowerShell does not do by itself for executables.
    param([string]$Dir, [string]$Exe, [string[]]$Arguments)
    Push-Location $Dir
    try {
        & $Exe @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "'$Exe $($Arguments -join ' ')' exited with status $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

function Stop-Tree {
    # `uv run` and `bun run` start child processes; stopping only the parent would
    # leave uvicorn or Vite running, so the whole tree is ended.
    param([System.Diagnostics.Process[]]$Processes)
    foreach ($process in $Processes) {
        if ($null -ne $process -and -not $process.HasExited) {
            & taskkill.exe /PID $process.Id /T /F 2>&1 | Out-Null
        }
    }
}

function Wait-Any {
    # Block until one process exits (or Ctrl-C), like `wait` after `&` in the Makefile.
    param([System.Diagnostics.Process[]]$Processes)
    while ($true) {
        foreach ($process in $Processes) {
            if ($process.HasExited) { return }
        }
        Start-Sleep -Milliseconds 300
    }
}

function Wait-Port {
    param([int]$TcpPort)
    while ($true) {
        $client = New-Object System.Net.Sockets.TcpClient
        try {
            $client.Connect("127.0.0.1", $TcpPort)
            return
        }
        catch {
            Start-Sleep -Milliseconds 200
        }
        finally {
            $client.Dispose()
        }
    }
}

function Start-Api {
    Start-Process -FilePath "uv" -WorkingDirectory $Backend -NoNewWindow -PassThru -ArgumentList @(
        "run", "uvicorn", "fall_detection.api.app:app",
        "--host", $BindHost, "--port", "$Port", "--workers", "1")
}

function Start-Dashboard {
    $env:FALL_API_TARGET = "http://${BindHost}:$Port"
    Start-Process -FilePath "bun" -WorkingDirectory $Frontend -NoNewWindow -PassThru `
        -ArgumentList @("run", "dev")
}

function Start-Rtsp {
    # mediamtx writes auto.key/auto.crt into its working directory, hence the temp dir.
    $arguments = @()
    if ($MediamtxConfig) { $arguments = @($MediamtxConfig) }
    $options = @{ FilePath = "mediamtx"; WorkingDirectory = $env:TEMP; NoNewWindow = $true; PassThru = $true }
    if ($arguments.Count -gt 0) { $options.ArgumentList = $arguments }
    Start-Process @options
}

function Start-Webcam {
    if (-not $WebcamDevice) {
        throw "set -WebcamDevice to a DirectShow camera name; 'dev.ps1 devices' lists them"
    }
    Wait-Port $RtspPort
    Start-Process -FilePath "ffmpeg" -NoNewWindow -PassThru -ArgumentList @(
        "-hide_banner", "-loglevel", "warning",
        "-f", "dshow", "-framerate", "$WebcamFps", "-video_size", $WebcamSize,
        "-i", "`"video=$WebcamDevice`"",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-g", "$WebcamFps", "-pix_fmt", "yuv420p",
        "-f", "rtsp", "-rtsp_transport", "tcp", $RtspUrl)
}

function Invoke-Together {
    param([scriptblock[]]$Starters)
    $processes = @()
    try {
        foreach ($start in $Starters) { $processes += & $start }
        Wait-Any $processes
    }
    finally {
        Stop-Tree $processes
    }
}

switch ($Target) {
    "help" {
        Get-Help $PSCommandPath -Detailed
        Write-Host "Targets: install weights migrate run backend frontend devices rtsp webcam camera test lint typecheck check build"
    }
    "install" {
        Invoke-Native $Backend "uv" @("sync", "--locked")
        Invoke-Native $Frontend "bun" @("install", "--frozen-lockfile")
    }
    "weights" {
        Invoke-Native $Root "uv" @("run", "--project", "backend", "python",
            "scripts/phase0_ml_smoke.py", "--device", $Device)
    }
    "migrate" {
        Invoke-Native $Backend "uv" @("run", "alembic", "upgrade", "head")
    }
    "run" {
        Invoke-Native $Backend "uv" @("run", "alembic", "upgrade", "head")
        Write-Host "API       http://${BindHost}:$Port"
        Write-Host "Dashboard http://localhost:5173"
        Invoke-Together @(${function:Start-Api}, ${function:Start-Dashboard})
    }
    "backend" {
        Invoke-Native $Backend "uv" @("run", "alembic", "upgrade", "head")
        Invoke-Native $Backend "uv" @("run", "uvicorn", "fall_detection.api.app:app",
            "--host", $BindHost, "--port", "$Port", "--workers", "1")
    }
    "frontend" {
        $env:FALL_API_TARGET = "http://${BindHost}:$Port"
        Invoke-Native $Frontend "bun" @("run", "dev")
    }
    "devices" {
        # FFmpeg prints the list on stderr and exits non-zero because "dummy" is not a
        # real input, so the exit code is not checked here.
        & ffmpeg -hide_banner -list_devices true -f dshow -i dummy
    }
    "rtsp" { Invoke-Together @(${function:Start-Rtsp}) }
    "webcam" { Invoke-Together @(${function:Start-Webcam}) }
    "camera" { Invoke-Together @(${function:Start-Rtsp}, ${function:Start-Webcam}) }
    "test" {
        Invoke-Native $Backend "uv" @("run", "pytest")
        Invoke-Native $Frontend "bun" @("run", "test")
    }
    "lint" {
        Invoke-Native $Backend "uv" @("run", "ruff", "check", ".")
        Invoke-Native $Backend "uv" @("run", "ruff", "format", "--check", ".")
        Invoke-Native $Frontend "bun" @("run", "lint")
    }
    "typecheck" {
        Invoke-Native $Backend "uv" @("run", "basedpyright")
        Invoke-Native $Frontend "bun" @("run", "typecheck")
    }
    "check" {
        & $PSCommandPath lint
        & $PSCommandPath typecheck
        & $PSCommandPath test
        Invoke-Native $Backend "uv" @("run", "alembic", "check")
    }
    "build" {
        Invoke-Native $Frontend "bun" @("run", "build")
    }
}
