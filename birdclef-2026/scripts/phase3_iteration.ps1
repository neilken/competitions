param(
    [string]$PythonExe = "python",
    [string]$BaseConfig = "configs/baseline_colab.yaml",
    [string]$AblationConfig = "configs/ablations.yaml",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$cmd = @(
    "-m", "src.training.run_ablations",
    "--base-config", $BaseConfig,
    "--ablation-config", $AblationConfig
)
if ($DryRun) {
    $cmd += "--dry-run"
}

Write-Host "[STEP] Running phase 3 iteration matrix..."
& $PythonExe $cmd
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Phase 3 iteration failed."
    exit $LASTEXITCODE
}

Write-Host "[PASS] Phase 3 iteration completed."
exit 0
