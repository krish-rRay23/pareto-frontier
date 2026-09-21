# Amazon ML Challenge 2026: Master Competition Playbook

## Phase 0: Pre-Hackathon Readiness (Complete)
- [x] Pre-built repository, reproducible modular package structure (`pareto-frontier`).
- [x] Robust metric computation (`SMAPE`, `MAE`, `RMSE`) with full edge-case unit tests.
- [x] Automated data schema validation & leakage-free K-Fold splitters.
- [x] Fast baseline pipelines (Statistical, TF-IDF + Ridge, GBDT on catalog features).
- [x] Submission schema validator (null checks, ID alignment, finite numbers, non-negativity).

---

## Phase 1: Problem Drop & Ingestion (T + 0:00 to 1:00)
1. **Download & Place Data**:
   - Place official raw CSVs/JSONs in `data/raw/`.
   - Never modify files inside `data/raw/`.
2. **Run Schema Validation**:
   - Execute `src/data/validation.py` on `train.csv` and `test.csv`.
   - Note down: missing values, IDs, target distribution, duplicates, unknown categories.
3. **Verify Target & Metric**:
   - Check official evaluation metric. If SMAPE, confirm formulation matches `src/validation/metrics.py`.
   - Check target bounds: are there zero or negative targets?

---

## Phase 2: Exploratory Data Analysis (EDA) & Sanitization (T + 1:00 to 2:30)
1. **Target Distribution**:
   - Check skewness: calculate kurtosis and log-transform variance.
   - Look for pricing anomalies (e.g. ₹0.00, ₹999,999.00).
2. **Catalog Text Inspection**:
   - Inspect bullet points, descriptions, titles.
   - Identify common noise tokens: HTML entities (`&amp;`, `<br>`), noisy units, emojis.
3. **Train-Test Covariate Shift**:
   - Check category overlap: what fraction of test brands/categories are unseen in train?
   - Adversarial validation (predicting train vs. test) to detect distribution drift.

---

## Phase 3: Baseline First Submission (T + 2:30 to 3:30)
1. **Statistical Baseline**:
   - Run `python scripts/run_pipeline.py --config configs/baseline_ridge.yaml`.
2. **Generate First Benchmark Submission**:
   - Generate submission to `artifacts/submissions/baseline_ridge_sub.csv`.
3. **Validate & Upload**:
   - Run `python scripts/validate_submission.py --submission artifacts/submissions/baseline_ridge_sub.csv --test data/raw/test.csv`.
   - Submit to competition portal immediately to verify leaderboard scoring pipeline.

---

## Phase 4: Feature Engineering Sprint (T + 3:30 to 8:00)
1. **Run Catalog Feature Extractor**:
   - Text parsing: numeric value, unit normalization (g, kg, ml, l, count, oz).
   - Multi-pack recognition: "Pack of 3", "2x500g", "Set of 4".
   - Brand heuristic extractor: capital casing, title prefixes.
2. **Text Statistics**:
   - Length, word count, uppercase character ratio, digit ratio.
   - Keyword indicators: "combo", "premium", "pack", "organic", "refill".
3. **Out-of-Fold Target Encoding**:
   - Apply smoothed Bayesian target encoding strictly within training folds.

---

## Phase 5: Diverse Model Exploration (T + 8:00 to 18:00)
1. **Tabular Models**:
   - LightGBM / CatBoost / HistGradientBoosting on engineered features.
2. **Text Models**:
   - TF-IDF + Ridge / BayesianRidge on full text tokens.
   - Fine-tuned transformer (DeBERTa-v3 / RoBERTa) if compute permits.
3. **Multimodal Evaluation (Only If Hypothesized Advantage Holds)**:
   - Extract image embeddings (CLIP / ViT) only for a sample fold first.
   - Verify CV gain against baseline before spending hours on feature caching.

---

## Phase 6: Ensembling & Post-Processing (T + 18:00 to 22:00)
1. **Out-of-Fold Alignment**:
   - Ensure all models produce aligned OOF predictions indexed by sample ID.
2. **Blending Strategies**:
   - Optimize non-negative linear weights or rank averaging to minimize CV SMAPE.
3. **Post-Processing**:
   - Apply floor clipping (minimum observed train target).
   - Ensure predictions are strictly positive and finite.

---

## Phase 7: Final Selection & Verification (T + 22:00 to 24:00)
1. Run final submission validation checks:
   - Exactly $N$ rows matching test set.
   - Exact column names required by organizer.
   - Exact ID order preservation.
   - 0 NaN, 0 Inf, 0 negative values.
2. Select Submissions:
   - **Submission 1**: Safest, best single cross-validated model.
   - **Submission 2**: Optimal ensemble with highest CV score and conservative post-processing.
