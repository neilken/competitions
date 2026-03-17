# BirdCLEF 2026 Rules Gate Checklist

Use this checklist before every Kaggle scored submission.

## Competition Conduct

- [ ] One Kaggle account only is being used.
- [ ] No private sharing of competition code or data outside official team boundaries.
- [ ] Team mode is still solo unless explicitly changed before the merger deadline.

## Data and External Resources

- [ ] Every external dataset/model/tool is listed in the compliance tracker.
- [ ] Every listed external resource is publicly accessible and reasonably accessible.
- [ ] External resource licenses are recorded and compatible with intended use.
- [ ] Citation and acquisition steps exist for every external resource used.
- [ ] No hidden test data has been used for training, calibration, or pseudo-labeling.

## Notebook Submission Constraints

- [ ] Final scored notebook path is CPU-only.
- [ ] End-to-end runtime is <= 90 minutes in a CPU environment.
- [ ] Internet is disabled for scored submission.
- [ ] Output file is exactly named `submission.csv`.
- [ ] Output columns and order exactly match `sample_submission.csv`.
- [ ] Submission probabilities are numeric, finite, and within `[0, 1]`.

## Reproducibility and Winner Readiness

- [ ] Run uses a tracked config ID and seed.
- [ ] Fold mapping used for model selection is archived.
- [ ] Checkpoint ID and inference config are recorded.
- [ ] Third-party code/model licenses are tracked for open-source handoff readiness.

## Go / No-Go

- [ ] All checks above pass. If any check fails, do not submit.
