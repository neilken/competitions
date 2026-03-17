# BirdCLEF 2026

Project workspace for the [BirdCLEF 2026 Kaggle competition](https://www.kaggle.com/competitions/birdclef-2026).

## Competition Overview

BirdCLEF 2026 is part of the LifeCLEF 2026 benchmark series and focuses on identifying wildlife species from passive acoustic monitoring recordings. The challenge is centered on audio collected in the Pantanal wetlands, with the broader goal of improving biodiversity monitoring in a large, ecologically important, and difficult-to-monitor landscape.

This is a bioacoustics classification problem with real-world constraints:

- Field recordings are noisy, messy, and habitat-dependent
- Species labels are limited relative to the volume of audio
- Generalization matters more than fitting clean training clips
- Success depends on handling domain shift between training audio and scored soundscapes

For this repository, the goal is not just leaderboard performance. The goal is to build a reproducible, well-documented competition workflow for acoustic species detection and multi-label soundscape modeling.

## Official Snapshot

- Competition: BirdCLEF 2026 on Kaggle
- Series: LifeCLEF / CLEF 2026
- Task family: multi-taxa species identification in soundscape recordings
- Region: Pantanal wetlands
- Start date: March 11, 2026
- Entry deadline: May 27, 2026
- Team merger deadline: May 27, 2026
- Final submission deadline: June 3, 2026
- Working note submission deadline: June 17, 2026

Important note: the official organizers may still update competition details, rules, or data documentation during the competition window. This README should be treated as a living project plan rather than a frozen spec.

## Why This Competition Matters

BirdCLEF is one of the strongest applied ML benchmarks for ecological audio. It sits at the intersection of:

- machine learning for biodiversity monitoring
- weakly labeled and limited-label learning
- audio representation learning
- long-tail species classification
- domain adaptation under realistic field conditions

For a data scientist with interests in computational ecology, public health, and applied ML, this is a high-value competition because it rewards both technical rigor and scientific problem framing.

## Project Goals

This folder is intended to support a full competition workflow:

1. Understand the competition data, labels, and scoring format.
2. Build a strong and reproducible baseline.
3. Develop validation that is robust to habitat, recorder, and temporal shift.
4. Improve recall on rare or difficult species without destabilizing overall performance.
5. Produce a clean final submission pipeline and a documented retrospective.

## Repository Structure

The folder now includes the initial scaffold for data setup, future modeling work, and reference documentation:

```text
birdclef-2026/
|- README.md
|- configs/
|- data/
|  |- raw/
|  |- interim/
|  `- processed/
|- docs/
|  |- competition-details.md
|  |- competition-rules.md
|  |- data-description.md
|  `- data-setup.md
|- notebooks/
|  `- 00_setup_data_to_drive.ipynb
|- src/
|  |- features/
|  |- datasets/
|  |- models/
|  |- training/
|  |- inference/
|  `- utils/
|- outputs/
|  |- checkpoints/
|  |- oof/
|  |- submissions/
|  `- reports/
```

## Data Setup

The initial data acquisition workflow is implemented as a Colab notebook:

- [00_setup_data_to_drive.ipynb](notebooks/00_setup_data_to_drive.ipynb)
- [01_train_validate_submit_colab.ipynb](notebooks/01_train_validate_submit_colab.ipynb)
- [data-setup.md](docs/data-setup.md)

This setup uses Colab to download the Kaggle archive into runtime storage, copies the raw archive into Google Drive, extracts the visible dataset, and syncs the extracted files into `MyDrive/birdclef-2026/data/`.

## Rule-Hardened Execution Assets

The execution plan is now implemented as runnable and auditable project assets:

- Plan and operations
  - [execution-plan.md](docs/execution-plan.md)
  - [submission-runbook.md](docs/submission-runbook.md)
  - [next-steps-quickstart.md](docs/next-steps-quickstart.md)
- Compliance and gates
  - [rules-gate-checklist.md](docs/rules-gate-checklist.md)
  - [competition-rules.md](docs/competition-rules.md)
  - `python -m src.utils.rules_gate --config configs/rules_gate.yaml`
- Training and inference stack
  - `python -m src.training.run_baseline --config configs/baseline_colab.yaml`
  - `python -m src.training.run_baseline --config configs/baseline_alt_colab.yaml`
  - `python -m src.inference.run_submission_from_config --config configs/inference_cpu_submission.yaml`
  - `python -m src.inference.blend_submissions --inputs ... --output submission.csv`
  - `python -m src.inference.validate_submission --sample-submission ... --submission ...`
  - `python -m src.inference.runtime_rehearsal --config configs/inference_cpu_submission.yaml --max-minutes 90`
  - `python -m src.training.optimize_thresholds --oof-csv ... --folds-csv ... --output-csv ...`
  - `python -m src.training.check_fold_leakage --folds-csv ...`
- Validation policy and testing
  - [cv-policy.md](docs/cv-policy.md)
  - [testing-and-acceptance.md](docs/testing-and-acceptance.md)
  - `python -m src.training.create_cv_splits --config configs/cv_policy.yaml`
  - `python -m src.utils.data_integrity --data-root /content/drive/MyDrive/birdclef-2026/data`
  - `python -m src.utils.audio_smoke_test --train-audio-dir ... --train-soundscapes-dir ...`
- Trackers
  - [external_resource_compliance_tracker.csv](docs/trackers/external_resource_compliance_tracker.csv)
  - [experiment_registry.csv](docs/trackers/experiment_registry.csv)
  - [submission_log.csv](docs/trackers/submission_log.csv)
  - `python -m src.utils.submission_log --log-csv ... --set submission_id=... run_id=...`

Install runtime dependencies in Colab:

```bash
pip install -r requirements-colab.txt
```

## Problem Framing

The central modeling task is likely some combination of multi-label sound event detection and clip-level species classification on long-form environmental audio. In practice, that usually means:

- converting raw audio into spectrogram-based or embedding-based representations
- predicting species presence probabilities over time windows
- aggregating window-level predictions into submission-ready outputs
- managing class imbalance, label sparsity, and background noise

Even before modeling, the hardest part is usually evaluation design. BirdCLEF-style competitions often punish overly optimistic local validation because the private test environment differs from the training distribution in recording conditions, species frequency, geography, seasonality, or annotation density.

## Core Challenges To Expect

- Extreme class imbalance and long-tail species frequency
- Weak labels or incomplete labels in soundscape segments
- Multiple overlapping taxa in the same clip
- Domain shift across devices, habitats, and environmental conditions
- Scarce positive examples for rare species
- Background insects, wind, rain, water, and anthropogenic noise
- Potential mismatch between public leaderboard and private leaderboard behavior

## Initial Technical Strategy

### 1. Baseline First

Start with a baseline that is easy to reproduce and iterate:

- standardize sample rate and clip duration
- generate log-mel spectrograms
- train a compact CNN or audio transformer baseline
- use stratified or grouped cross-validation where possible
- produce out-of-fold predictions for error analysis

The baseline should prioritize reliability over novelty. A stable baseline is more valuable than an ambitious but poorly validated architecture.

### 2. Validation Design

Validation quality will likely determine the final placement more than model architecture alone.

Priorities:

- identify grouping variables that reduce leakage
- separate similar recordings across folds where possible
- stress-test fold performance for rare species
- compare local CV against leaderboard movement conservatively
- retain all out-of-fold predictions for calibration and diagnostics

### 3. Feature and Representation Work

Potential directions:

- log-mel spectrograms with multiple FFT and hop settings
- PCEN or other noise-robust transforms
- pretrained bioacoustics embeddings
- metadata-aware features if competition rules permit them
- multi-scale windows to capture short calls and longer vocal structure

### 4. Modeling Directions

Promising options for this competition type:

- CNN backbones on spectrogram images
- audio transformers or conformer-style architectures
- pretrained bioacoustics models with fine-tuning
- two-stage systems: candidate generation plus species reranking
- ensembles across seeds, folds, and representation types

### 5. Post-processing

This competition will likely reward careful post-processing:

- threshold optimization by class
- temporal smoothing or pooling strategies
- calibration for rare classes
- blending across heterogeneous models
- class prior adjustments if local validation supports them

## EDA Checklist

Before serious model work, answer these questions:

- How many target species are there?
- Is the task strictly birds, or multi-taxa as stated by LifeCLEF?
- What is the label format for train and submission files?
- Are labels clip-level, segment-level, or soundscape-level?
- What is the duration distribution of recordings?
- How imbalanced are the species counts?
- Are there metadata columns for site, recorder, latitude/longitude, or time?
- What proportion of samples contain heavy noise or overlapping calls?

## Modeling Roadmap

### Phase 1

- set up folder structure
- download and catalog data
- build EDA notebook
- create baseline preprocessing pipeline
- train first local baseline

### Phase 2

- improve cross-validation
- add augmentation and mixup-style strategies
- test pretrained model transfer
- run per-class threshold tuning
- perform error analysis on confusion patterns and missed rare classes

### Phase 3

- ensemble best families of models
- optimize inference and submission generation
- compare public LB movement against local CV confidence
- finalize reproducible training and inference scripts

## Experiment Tracking

Every experiment should log:

- model name and backbone
- audio preprocessing settings
- fold strategy
- training seed
- augmentation recipe
- validation metric
- leaderboard score
- notes on failure modes

Without disciplined tracking, BirdCLEF competitions become impossible to debug once multiple augmentations, window sizes, and ensemble members are in play.

## Error Analysis Priorities

The most useful review loop will be:

1. Inspect worst-performing species by recall and precision.
2. Review false positives caused by habitat noise or acoustically similar taxa.
3. Separate modeling errors from labeling ambiguity.
4. Check whether missed detections cluster by site, hour, recorder, or weather proxy.
5. Decide whether the next gain should come from data, validation, architecture, or post-processing.

## Stretch Ideas

- pseudo-labeling if unlabeled or weakly labeled audio is available
- self-supervised or contrastive pretraining on raw competition audio
- hierarchical modeling across taxa groups
- retrieval-based candidate generation from embedding space
- teacher-student distillation for smaller inference-friendly models
- class-aware augmentation for rare species support

## Deliverables For This Folder

By the end of the competition, this folder should contain:

- a clean README with final approach and lessons learned
- reproducible training configuration
- inference and submission scripts
- experiment notes
- final model summary
- post-competition retrospective

## Useful Links

- [Kaggle competition page](https://www.kaggle.com/competitions/birdclef-2026)
- [BirdCLEF 2026 task page](https://www.imageclef.org/BirdCLEF2026)
- [LifeCLEF 2026 overview](https://clef2026.clef-initiative.eu/labs/lifeclef/)

## Notes

This README is intentionally more detailed than a placeholder. It is meant to serve as the operating brief for the competition folder and can be tightened later once the actual Kaggle data files, evaluation metric, and submission format are fully inspected.
