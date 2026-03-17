# BirdCLEF 2026 Testing and Acceptance

## Data Tests

- Verify expected artifacts exist:
  - `train_audio/`
  - `train_soundscapes/`
  - `train.csv`
  - `taxonomy.csv`
  - `sample_submission.csv`
  - `train_soundscapes_labels.csv`
  - `recording_location.txt`
- Verify local staged files match archive-derived size manifest.
- Run audio decode smoke test on random files from `train_audio` and `train_soundscapes`.

Acceptance:
- All required artifacts present.
- No size mismatches.
- No decode failures in smoke sample.

## Validation Tests

- Fold leakage checks:
  - no group overlap across train/validation within each fold
  - no missing fold assignments
- Metric parity checks:
  - local metric implementation aligns with competition metric assumptions

Acceptance:
- Leakage check passes.
- Metric check script runs with expected outputs.

## Submission Tests

- Validate `submission.csv` against `sample_submission.csv`:
  - exact same column set and order
  - no missing row IDs
  - numeric finite probabilities in `[0, 1]`
- CPU runtime rehearsal under no-internet conditions.
- Determinism check for inference with fixed checkpoint + seed.

Acceptance:
- Submission schema validator passes.
- Runtime is <= 90 minutes in rehearsal environment.
- Determinism check differences are within expected numeric tolerance.

## Go / No-Go Criteria

- Baseline end-to-end run is reproducible.
- Two model families show OOF uplift over baseline.
- Final ensemble candidate passes rules gate and submission tests.
