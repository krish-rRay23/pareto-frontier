# Experiment Tracking Log

| Experiment ID | Date / Time | Model / Architecture | CV Split | SMAPE (%) | MAE | RMSE | Notes & Hypotheses | Artifact / Submission File |
|---|---|---|---|---|---|---|---|---|
| baseline_stat | 2026-09-21 | Median / Target Median | 5-Fold Stratified | - | - | - | Naive statistical lower-bound | - |
| baseline_ridge | 2026-09-21 | TF-IDF (1-2 gram) + Ridge | 5-Fold Stratified | - | - | - | Fast sparse text linear model | - |
| baseline_tabular_lgb | 2026-09-21 | GBDT + Engineered Catalog Features | 5-Fold Stratified | - | - | - | Baseline tree model on regex + stats | - |
| baseline_ridge | 2026-09-21 23:03:29 | ridge | 5-Fold stratified (s=42) | 35.2890% | 299.7735 | 617.4104 | Fast 5-Fold Stratified TF-IDF (1-2 gram) + Ridge Regression with log1p target | baseline_ridge_sub.csv |
| baseline_tabular_lgb | 2026-09-21 23:03:44 | gbdt | 5-Fold stratified (s=42) | 24.4483% | 185.5724 | 363.7452 | 5-Fold GBDT on regex catalog features (units, pack count, text density, keywords) | baseline_gbdt_sub.csv |
