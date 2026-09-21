# 🚀 Pareto-Frontier: Amazon ML Challenge 2026

A production-grade, modular, and leakage-safe machine learning competition repository built for the **Amazon ML Challenge 2026**.

Designed around the **Pareto principle**: maximize leaderboard score and validation fidelity while maintaining minimal computational overhead, rapid iteration cycles, and absolute reproducibility.

---

## 📂 Repository Architecture

```text
pareto-frontier/
├── configs/                     # YAML configuration files for experiments
│   ├── default.yaml
│   ├── baseline_ridge.yaml      # TF-IDF + Ridge regression
│   ├── baseline_tabular_lgb.yaml# GBDT on extracted catalog features
│   └── baseline_statistical.yaml# Sanity benchmark (category median)
├── data/
│   ├── raw/                     # Untouched official competition data (train.csv, test.csv)
│   ├── interim/                 # Cached tokenized features / preprocessed text
│   └── processed/               # Extracted tabular features & clean matrices
├── docs/
│   ├── competition_playbook.md  # Step-by-step 24-hour competition execution guide
│   ├── research/
│   │   └── previous_solutions.md# Lessons & hypotheses from 2024-2025 winning solutions
│   └── experiments/
│       └── experiment_log.md    # Formatted Markdown log tracking every CV run
├── notebooks/                   # Fast scratchpads & exploratory data analysis (EDA)
├── src/
│   ├── data/                    # Ingestion, synthetic generator, schema validation
│   │   ├── loader.py
│   │   └── validation.py
│   ├── features/                # Catalog regex parsers, unit normalizer, text statistics
│   │   ├── text_extraction.py
│   │   └── tabular_builder.py
│   ├── models/                  # Baselines: statistical, linear text, tabular GBDT
│   │   ├── statistical.py
│   │   ├── linear_text.py
│   │   └── tree_models.py
│   ├── validation/              # SMAPE metric, stratified splitters, OOF target encoder
│   │   ├── metrics.py
│   │   ├── splitters.py
│   │   └── target_encoding.py
│   ├── training/                # Leakage-free K-Fold runner, checkpointing, experiment logger
│   │   ├── trainer.py
│   │   └── logger.py
│   ├── inference/               # Checkpoint loader, prediction engine, submission validator
│   │   ├── predict.py
│   │   └── submission_validator.py
│   └── ensemble/                # Optimal linear blending & rank averaging
│       └── blender.py
├── scripts/
│   ├── run_pipeline.py          # Master CLI to train, evaluate, and generate submissions
│   └── validate_submission.py   # Standalone submission integrity checker
├── tests/                       # Pytest suite (metrics, parsers, splitters, pipeline)
├── artifacts/                   # Checkpoints, OOF CSVs, submissions, experiment history
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

---

## ⚡ Quick Start

### 1. Installation

```bash
git clone https://github.com/krish-rRay23/pareto-frontier.git
cd pareto-frontier

pip install -r requirements.txt
```

### 2. Verify End-to-End Pipeline (Synthetic Data Mode)

Before official data is released, run any baseline on automatically generated synthetic Amazon catalog data:

```bash
# Run TF-IDF + Ridge Fast Baseline
python scripts/run_pipeline.py --config configs/baseline_ridge.yaml --synthetic

# Run GBDT on Catalog Tabular Features
python scripts/run_pipeline.py --config configs/baseline_tabular_lgb.yaml --synthetic

# Run Statistical Lower-Bound Benchmark
python scripts/run_pipeline.py --config configs/baseline_statistical.yaml --synthetic
```

### 3. Run Test Suite

```bash
pytest tests/ -v
```

---

## 🧠 Core Engineering Principles

### 1. Robust Metric Engineering (`src/validation/metrics.py`)
- **SMAPE (Symmetric Mean Absolute Percentage Error)**:
  $$\text{SMAPE} = \frac{100\%}{n} \sum_{i=1}^n \frac{2 \cdot |y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}$$
  - Guarded against $0 / 0$ division (returns $0.0$ when ground truth and prediction are both $0$).
  - Evaluates both arrays and pandas Series with zero NaN risk.

### 2. Zero-Leakage Validation (`src/validation/`)
- **Stratified Regression Splits**: Bins continuous targets into quantiles to guarantee identical target distributions across all $K$ folds.
- **Strict Out-of-Fold Target Encoding**: Target encodings and scaling statistics are fit strictly on training folds and applied out-of-fold to validation and test data.
- **Group-Aware Splitting**: Prevents the same brand or category from leaking between training and validation splits.

### 3. Deep Catalog Feature Extraction (`src/features/`)
E-commerce catalog data requires domain-specific text mining before hitting complex models:
- **Unit Normalization**: Automatically identifies and converts mass (`mg`, `g`, `kg`, `oz`, `lb`) to grams and volume (`ml`, `l`, `fl oz`, `gallon`) to milliliters.
- **Pack & Multiplier Parsing**: Extracts pack counts from formats like `"Pack of 4"`, `"6 Pack"`, `"3x500g"`, or `"Set of 2"`.
- **Effective Quantity Computation**: Multiplies single-unit normalized amounts by pack count (`effective_weight_g = unit_weight * pack_count`).
- **Brand Extraction Heuristics**: Extracts brand candidates via title prefix patterns and regex tags.
- **Text Statistics & Density**: Word count, character length, uppercase letter density, digit density, punctuation density.
- **Value Indicator Keywords**: Flags binary indicators for `"combo"`, `"organic"`, `"refill"`, `"premium"`, `"warranty"`, etc.

### 4. Submission Validator (`src/inference/submission_validator.py`)
Before any submission is uploaded to the competition portal, it must pass 5 strict automated checks:
1. **Row Count Check**: Exactly matches test set row count.
2. **Column Name Verification**: Exact column names required by competition organizer.
3. **Sample ID Alignment**: Identical set of IDs and exact row ordering matching the test set.
4. **Finite Number Check**: 0 `NaN`, 0 `Inf`, 0 `-Inf`.
5. **Non-Negativity Constraint**: All predictions strictly positive ($\ge \text{floor}$).

---

## 🎯 Competition Day Workflow (T + 0:00 to 24:00)

See [docs/competition_playbook.md](docs/competition_playbook.md) for full execution protocols.

1. **Ingest**: Download raw CSVs to `data/raw/train.csv` and `data/raw/test.csv`.
2. **Validate**: Run schema validation to detect nulls, missing columns, or target distribution anomalies.
3. **Submit Benchmark**: Run `python scripts/run_pipeline.py --config configs/baseline_ridge.yaml` and submit `baseline_ridge_sub.csv` to ensure submission pipeline works.
4. **Iterate**: Run tree models, tune catalog feature extraction, test deep learning backbones.
5. **Ensemble**: Blend out-of-fold predictions using `OptimalLinearBlender` directly optimizing validation SMAPE.
6. **Final Check**: Execute `python scripts/validate_submission.py --submission artifacts/submissions/final_blend.csv --test data/raw/test.csv`.
