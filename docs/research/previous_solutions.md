# Amazon ML Challenge: Historical Solutions & 2026 Research Playbook

## 1. Executive Summary & Critical Mindset for 2026

The Amazon ML Challenge historically presents noisy, real-world catalog and product datasets (often involving product dimension extraction, product price prediction, entity extraction, or multimodal attribute categorization). 

> **Golden Rule for 2026**: Do NOT blindly copy past 2024–2025 models. Treat past winning solutions as **hypotheses to validate** against the official 2026 problem statement, metric, constraints, and data distribution.

---

## 2. Review of Historical Winning Approaches (2024 - 2025)

### A. 3rd Place Solution (Amazon ML Challenge 2025 - Parth Rastogi et al.)
- **Problem Type**: Price prediction / numerical attribute estimation from product title, catalog description, bullet points, and product images.
- **Key Architectures**:
  - Backbone: DeBERTa-v3-base / DeBERTa-v3-large fine-tuned on concatenated product text (`[CLS] Title [SEP] Bullets [SEP] Description [SEP]`).
  - Tabular Extractors: Regex-driven extraction of pack counts, unit amounts (ml, g, kg, count, oz), and brand proxies.
  - Tabular Modeling: LightGBM and CatBoost trained on tabular features and out-of-fold transformer embeddings.
  - Ensemble: Ridge blending and weighted average of DeBERTa + LightGBM + CatBoost predictions.
- **Key Strength**: Heavy focus on regex normalization of product quantities (e.g., distinguishing single units from multi-packs).
- **Weakness / Failure Modes**: Multimodal image branch (ResNet/ViT) provided minimal signal-to-noise improvement relative to computation cost and introduced domain shift vulnerability on missing images.

### B. Neel Deven Shah's Approach (Amazon ML Challenge 2025)
- **Key Insight**: Error analysis revealed that heavy outliers and skewed numerical distributions distorted linear loss functions.
- **Transformations**: Applying log-transform (`log1p` / `expm1`) on prices/targets stabilized gradient updates and directly improved relative percentage error metrics (such as SMAPE).
- **Validation**: 5-Fold stratified cross-validation on target bins was critical to prevent fold imbalance.

### C. Kaggle & Community Takeaways (Discussion 262376 & Kaggle Datasets)
- **Extreme Catalog Noise**: Inconsistent punctuation, HTML tags in description, mixed languages (English + Hindi/Hinglish in title strings), mixed units (e.g., "1000g" vs "1 kg").
- **Quantity & Pack Sensitivity**: A 10x error frequently occurred when models failed to parse "Pack of 10" or "Combo Pack".
- **Target Distribution**: Power-law / Pareto distribution across product price and quantities. Standard MSE directly harms percentage metrics like SMAPE or MAPE.

---

## 3. Proven Techniques vs. High-Risk Pitfalls

| Technique | Status | Competition Guidance |
|---|---|---|
| **Log Target Transform (`log1p`)** | **Validated** | Use whenever the target is positive and right-skewed; align loss function with competition metric (SMAPE / MAPE / RMSLE). |
| **Catalog Regex Extraction** | **Validated** | Parse `value`, `unit`, `pack_count`, and `total_quantity` as explicit tabular features. Tree models exploit these far faster than raw text. |
| **Text Feature Statistics** | **Validated** | Word counts, character counts, uppercase ratios, digit densities, brand candidate tokens. |
| **TF-IDF + Ridge Baseline** | **Validated** | Run in under 3 minutes; provides an immediate non-trivial baseline and sanity check for pipeline plumbing. |
| **Strict Out-of-Fold (OOF)** | **Validated** | Zero leakage: target encoding and scalers must be fit strictly on train folds. |
| **Image Embeddings (ViT / ResNet)** | **Hypothesis (Verify)** | Treat as an unverified hypothesis. Often high GPU cost for negligible metric gain. Only integrate if CV confirms statistically significant gain. |
| **Heavy Ensembling (50+ models)** | **High Risk** | High inference latency, prone to submission timeout or memory exhaustion in Docker/eval environments. |
| **External Price Lookup** | **Disallowed** | Violates competition integrity rules and causes catastrophic failure on unseen private test sets. |

---

## 4. Hypotheses & Validation Checklist for Amazon ML 2026

When Day 1 data drops, systematically test these hypotheses:

1. **H1 (Target Formulation)**: Does `log1p(target)` training with `expm1` inverse transform outperform raw target regression on validation SMAPE?
2. **H2 (Text Quality)**: Does removing HTML and lowercasing improve or degrade TF-IDF / DeBERTa performance? (Caution: Case patterns like all-caps often indicate brand names or warnings).
3. **H3 (Pack Multiplying)**: Does computing `effective_quantity = value * pack_count` correlate strongly with target values?
4. **H4 (Target Encoding)**: Does target encoding of high-cardinality brand candidates improve tree models without overfitting?
5. **H5 (Post-Processing)**: Does clipping predictions to `[train_min * 0.5, train_max * 1.5]` prevent out-of-distribution penalties?
