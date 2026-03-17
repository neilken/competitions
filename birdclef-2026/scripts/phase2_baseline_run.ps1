param(
    [string]$PythonExe = "python",
    [string]$BaselineConfig = "configs/baseline_colab.yaml",
    [string]$RulesConfig = "configs/rules_gate.yaml"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "[STEP] Running baseline training..."
& $PythonExe -m src.training.run_baseline --config $BaselineConfig
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Baseline training failed."
    exit $LASTEXITCODE
}

Write-Host "[STEP] Running metric parity check..."
& $PythonExe -m src.utils.metric_parity_check
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Metric parity check failed."
    exit $LASTEXITCODE
}

Write-Host "[STEP] Running deterministic inference check..."
& $PythonExe -m src.utils.determinism_check
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Determinism check failed."
    exit $LASTEXITCODE
}

Write-Host "[STEP] Running rules gate..."
& $PythonExe -m src.utils.rules_gate --config $RulesConfig
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Rules gate failed."
    exit $LASTEXITCODE
}

Write-Host "[PASS] Phase 2 baseline run completed."
exit 0
