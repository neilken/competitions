# BirdCLEF 2026 Default CV Policy

## Objective

Use a leakage-resistant local validation strategy that better approximates leaderboard behavior under distribution shift.

## Default Split Strategy

- Split unit: 5-second soundscape segments from `train_soundscapes_labels.csv`
- Primary grouping: `group_id = <site>_<date>` parsed from filename pattern `..._SXX_YYYYMMDD_HHMMSS.ogg`
- Fold method: `GroupKFold(n_splits=5)` on `group_id`
- Randomness: deterministic ordering with fixed `seed`

## Why This Default

- Prevents same site-date acoustic context from appearing in both train and validation folds.
- Preserves temporal and location structure better than random segment-level splits.
- Reduces optimistic leakage compared to naive random CV.

## Fallback Behavior

If site/date cannot be parsed for a row:
- Use `group_id = filename` as conservative fallback.
- Emit a warning and log fallback count.

If fold assignment is impossible:
- Block experiment start and require explicit override config.

## Required Artifacts Per Split Build

- `folds.csv` with columns:
  - `row_id`
  - `filename`
  - `start`
  - `end`
  - `group_id`
  - `fold`
  - `labels`
- Split metadata:
  - total rows
  - unique groups
  - per-fold row counts
  - per-fold unique group counts
  - fallback parse count

## Operational Rule

Do not compare model runs that use different fold definitions unless explicitly documented as a CV-policy experiment.
