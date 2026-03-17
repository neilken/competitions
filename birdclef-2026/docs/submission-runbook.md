# BirdCLEF 2026 Submission Runbook

## Purpose

Operational guide for controlled, rule-compliant submission activity from baseline through final week.

## Submission Budget Policy

- Maximum allowed by competition: 5 submissions/day.
- Default operating budget: 2 to 3 high-information submissions/day.
- Use full 5/day only during tightly scoped ablation windows.
- Do not submit low-confidence runs that do not test a clear hypothesis.

## Daily Submission Flow

1. Pick hypothesis and candidate run ID.
2. Confirm experiment registry entry exists.
3. Run rules gate script.
4. Generate and validate `submission.csv`.
5. Submit to Kaggle.
6. Log submission ID, score, and notes in submission log.
7. Update hypothesis status as pass, fail, or inconclusive.

## Final Week Controls (May 24 to June 3, 2026 UTC)

- Keep exactly two final-candidate pipelines by June 1, 2026 UTC.
- Freeze training and inference code paths for final candidates.
- Re-run each final candidate from clean Colab runtime.
- Submit final entries by June 2, 2026 UTC to preserve buffer.

## No-Submit Conditions

- Rules gate fails.
- Runtime estimate exceeds CPU 90-minute budget.
- Output file name is not `submission.csv`.
- Schema mismatch versus `sample_submission.csv`.
- Missing documentation for external resources used by run.

## Required Logs

- `docs/trackers/experiment_registry.csv`
- `docs/trackers/submission_log.csv`
- External resource compliance tracker with approvals
