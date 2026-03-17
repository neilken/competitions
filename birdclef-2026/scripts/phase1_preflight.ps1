param(
    [string]$PythonExe = "python",
    [string]$CvConfig = "configs/cv_policy.yaml",
    [string]$DataRoot = "/content/drive/MyDrive/birdclef-2026/data",
    [int]$SmokeSamples = 5
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "[STEP] Creating CV folds from train_soundscapes_labels..."
& $PythonExe -m src.training.create_cv_splits --config $CvConfig
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] CV split generation failed."
    exit $LASTEXITCODE
}

Write-Host "[STEP] Running data integrity check..."
& $PythonExe -m src.utils.data_integrity --data-root $DataRoot
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Data integrity check failed."
    exit $LASTEXITCODE
}

Write-Host "[STEP] Running audio smoke test..."
& $PythonExe -m src.utils.audio_smoke_test `
    --train-audio-dir (Join-Path $DataRoot "train_audio") `
    --train-soundscapes-dir (Join-Path $DataRoot "train_soundscapes") `
    --samples-per-dir $SmokeSamples
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Audio smoke test failed."
    exit $LASTEXITCODE
}

Write-Host "[PASS] Phase 1 preflight completed."
exit 0
