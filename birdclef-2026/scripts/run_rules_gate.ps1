param(
    [string]$ConfigPath = "configs/rules_gate.yaml",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

Write-Host "[INFO] Running BirdCLEF rules gate..."
Write-Host "[INFO] Config: $ConfigPath"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$cmd = @(
    "-m", "src.utils.rules_gate",
    "--config", $ConfigPath
)

& $PythonExe $cmd
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    Write-Host "[FAIL] Rules gate failed with exit code $exitCode"
    exit $exitCode
}

Write-Host "[PASS] Rules gate passed."
exit 0
