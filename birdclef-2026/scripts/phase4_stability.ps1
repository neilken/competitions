param(
    [string]$PythonExe = "python",
    [string]$OofCsv = "/content/drive/MyDrive/birdclef-2026/outputs/oof/baseline_colab_v1_oof.csv",
    [string]$FoldsCsv = "/content/drive/MyDrive/birdclef-2026/outputs/reports/folds.csv",
    [string]$ThresholdsCsv = "/content/drive/MyDrive/birdclef-2026/outputs/reports/class_thresholds.csv",
    [string]$SampleSubmission = "/content/drive/MyDrive/birdclef-2026/data/sample_submission.csv",
    [string]$SubmissionCsv = "/content/drive/MyDrive/birdclef-2026/outputs/submissions/submission.csv"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "[STEP] Checking fold leakage..."
& $PythonExe -m src.training.check_fold_leakage --folds-csv $FoldsCsv
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Fold leakage check failed."
    exit $LASTEXITCODE
}

Write-Host "[STEP] Optimizing class thresholds from OOF..."
& $PythonExe -m src.training.optimize_thresholds --oof-csv $OofCsv --folds-csv $FoldsCsv --output-csv $ThresholdsCsv
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Threshold optimization failed."
    exit $LASTEXITCODE
}

Write-Host "[STEP] Validating submission schema..."
& $PythonExe -m src.inference.validate_submission --sample-submission $SampleSubmission --submission $SubmissionCsv
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Submission schema validation failed."
    exit $LASTEXITCODE
}

Write-Host "[PASS] Phase 4 stability checks completed."
exit 0
