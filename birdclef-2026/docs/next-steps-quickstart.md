# BirdCLEF 2026 Quickstart (Implemented Plan)

## Colab Compatibility Notes

- Run commands from the repository root so `python -m src...` imports resolve.
- Use Linux/Colab paths in configs (already set to `/content/...` defaults).
- Install runtime dependencies before running any pipeline step.

## 0) Install Dependencies (Colab)

```bash
pip install -r requirements-colab.txt
```

Optional root setup in Colab:

```bash
cd /content
# adjust if your repo is in a different location
cd birdclef-2026
```

## 1) Build CV Splits

```bash
python -m src.training.create_cv_splits --config configs/cv_policy.yaml
```

or

```powershell
.\scripts\phase1_preflight.ps1 -DataRoot /content/drive/MyDrive/birdclef-2026/data
```

Expected output:
- `outputs/reports/folds.csv` (or Drive path in config)

## 2) Run Baseline Training

```bash
python -m src.training.run_baseline --config configs/baseline_colab.yaml
```

or

```powershell
.\scripts\phase2_baseline_run.ps1
```

Run the second model family for ensemble diversity:

```bash
python -m src.training.run_baseline --config configs/baseline_alt_colab.yaml
```

Expected output:
- Fold checkpoints under `outputs/checkpoints/`
- OOF predictions under `outputs/oof/`
- New run entry appended to `docs/trackers/experiment_registry.csv`

## 3) Generate Submission Skeleton

```bash
python -m src.inference.generate_submission_cpu \
  --sample-submission /content/drive/MyDrive/birdclef-2026/data/sample_submission.csv \
  --checkpoint /content/drive/MyDrive/birdclef-2026/outputs/checkpoints/baseline_colab_v1_fold0.pt \
  --soundscape-dir /kaggle/input/birdclef-2026/test_soundscapes \
  --output /content/drive/MyDrive/birdclef-2026/outputs/submissions/submission.csv
```

## 4) Run Rules Gate Before Submission

```bash
python -m src.utils.rules_gate --config configs/rules_gate.yaml
```

Or in PowerShell:

```powershell
.\scripts\run_rules_gate.ps1 -ConfigPath configs/rules_gate.yaml
```

## 5) Update Compliance Tracker Before External Asset Usage

Update:
- `docs/trackers/external_resource_compliance_tracker.csv`

Requirement:
- Every external resource must be approved before any experiment run that uses it.

## 6) Optional: Run Planned Ablation Matrix

```bash
python -m src.training.run_ablations --base-config configs/baseline_colab.yaml --ablation-config configs/ablations.yaml
```

or

```powershell
.\scripts\phase3_iteration.ps1
```

## 7) Optional: Blend Candidate Submissions

```bash
python -m src.inference.blend_submissions \
  --inputs sub_a.csv sub_b.csv \
  --weights 0.6 0.4 \
  --output submission.csv
```

## 8) Phase 4 Stability Checks

Leakage check:

```bash
python -m src.training.check_fold_leakage --folds-csv /content/drive/MyDrive/birdclef-2026/outputs/reports/folds.csv
```

Threshold optimization:

```bash
python -m src.training.optimize_thresholds \
  --oof-csv /content/drive/MyDrive/birdclef-2026/outputs/oof/baseline_colab_v1_oof.csv \
  --folds-csv /content/drive/MyDrive/birdclef-2026/outputs/reports/folds.csv \
  --output-csv /content/drive/MyDrive/birdclef-2026/outputs/reports/class_thresholds.csv
```

Submission validation:

```bash
python -m src.inference.validate_submission \
  --sample-submission /content/drive/MyDrive/birdclef-2026/data/sample_submission.csv \
  --submission /content/drive/MyDrive/birdclef-2026/outputs/submissions/submission.csv
```

Runtime rehearsal:

```bash
python -m src.inference.runtime_rehearsal --config configs/inference_cpu_submission.yaml --max-minutes 90
```

Log each submission:

```bash
python -m src.utils.submission_log \
  --log-csv /content/drive/MyDrive/birdclef-2026/docs/trackers/submission_log.csv \
  --set submission_id=LOCAL_DRY_RUN run_id=baseline_v1 config_id=baseline_colab_v1 rules_gate_passed=true notes=\"local rehearsal\"
```
