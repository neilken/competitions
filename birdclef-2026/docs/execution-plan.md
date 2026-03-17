# BirdCLEF 2026 Rule-Hardened Execution Plan

## Mission

Maximize private leaderboard performance while maintaining strict rule compliance and reproducibility.

Execution assumptions:
- Solo execution
- Colab-first training and inference workflow
- 15+ hours per week
- External data/models allowed only when publicly accessible and reasonably accessible

Hard deadlines (UTC):
- Team merge and entry deadline: May 27, 2026
- Final submission deadline: June 3, 2026

## Non-Negotiable Submission Constraints

- Kaggle scored notebook must run on CPU within 90 minutes
- Internet must be disabled in scored submission
- Output filename must be `submission.csv`
- Final file schema must match `sample_submission.csv` exactly

## Phase Schedule

### Phase 1: Data Integrity and Compliance Setup (March 17 to March 24, 2026)

Deliverables:
- Verified dataset staging with archive-size integrity checks
- External resource compliance tracker started
- Default CV split policy frozen
- Rules gate checklist created and required before any scored submission

Exit criteria:
- All expected dataset artifacts verified
- Compliance tracker contains every external resource used so far
- CV split script generates reproducible folds

### Phase 2: Baseline and Scored-Notebook Skeleton (March 24 to April 3, 2026)

Deliverables:
- Baseline training scaffold with reproducible config and fold mapping
- CPU-safe submission generation scaffold
- Early dry-run submission path validated locally and in Colab
- Experiment registry logging live

Exit criteria:
- One end-to-end baseline run produces tracked artifacts
- Submission skeleton writes valid `submission.csv`

### Phase 3: Controlled Score Iteration (April 3 to May 3, 2026)

Deliverables:
- Ablation matrix for windowing, augmentation, loss, thresholding, and model head
- Second model family integrated for ensemble diversity
- External data/model usage formally approved in tracker before use

Exit criteria:
- At least two model families with measurable OOF gains over baseline
- No experiment run without registry entry and config ID

### Phase 4: Ensemble and Stability (May 3 to May 24, 2026)

Deliverables:
- Stable candidate ensembles with class-wise thresholds
- Robustness tests for rare classes and background-noise shifts
- 2 to 3 production-ready candidate pipelines with tradeoff notes

Exit criteria:
- Ensemble candidates show stable cross-fold performance
- At least two candidate submissions pass all rule gates

### Phase 5: Final Operations (May 24 to June 3, 2026)

Deliverables:
- Fixed daily submission budget strategy
- Exactly two final submission options by June 1, 2026
- Clean-runtime reproducibility rerun
- Final submissions sent by June 2, 2026 UTC buffer

Exit criteria:
- Final two candidate runs are reproducible and fully documented
- Rules gate and submission validation pass for final entries

## Operating Rules

- Do not use hidden test data for any training or pseudo-labeling flow.
- Do not run external resources in experiments until compliance tracker approval is recorded.
- Do not submit if rules gate fails.
- Prioritize stable OOF evidence over single-run leaderboard spikes.
