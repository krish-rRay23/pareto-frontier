# Amazon ML Challenge 2026 — 99.99+ Business Entity Resolution Plan

## Objective

Build a production-grade Business Entity Resolution system for the Amazon ML Challenge 2026 that is engineered toward a **Macro F0.5 / final submission score of >= 0.9999**.

This is a target, not a mathematical guarantee: the private leaderboard is evaluated on a hidden test set. Therefore, a local validation score alone is never sufficient. The system is considered successful only when its validation behavior is robust to the hidden-test failure modes already observed by the team.

The central principle is:

> **Do not try to win with a larger model. Build a system that almost never loses the correct candidate and almost never accepts an incorrect candidate.**

The challenge metric is S1-entity-level Macro F0.5, so false-positive merges and bad singleton decisions are especially expensive.

---

# Core Strategy

The final system will combine:

**high-recall retrieval + strong pairwise evidence + hard-negative learning + collective S1/S2/S3 reasoning + precision-first decision decoding**

The system should look like:

```text
Raw S1 / S2 / S3
        |
        v
Multi-view normalization
        |
        v
Country-aware partitioning
        |
        v
High-recall multi-channel retrieval
        |
        v
Candidate union + retrieval provenance
        |
        v
Pair + context feature engine
        |
        v
XGBoost + LightGBM ensemble
        |
        v
Hard-negative refinement
        |
        v
Collective / cross-source reranking
        |
        v
Calibrated probabilities
        |
        v
Expected-F0.5 decision layer
        |
        +---- margin gate
        +---- singleton gate
        +---- target exclusivity
        +---- contradiction checks
        |
        v
Final S1 -> S2/S3 matching set
```

The important point is that the classifier is only one part of the system. Retrieval, context and final decoding are equally important.

---

# Phase 1 — Build an Almost-Lossless Candidate and Evidence Layer

## 1. Multi-view normalization

Never rely on a single normalized string. Preserve the raw record and generate multiple representations:

- raw text
- Unicode-normalized text
- lowercase / accent-stripped text
- legal-form normalized text
- tokenized text
- sorted-token representation
- compact alphanumeric representation
- transliterated representation
- character n-gram representation
- structured address components
- extracted numeric tokens

The normalization layer must preserve useful information rather than aggressively deleting it.

It should handle:

- punctuation variation
- legal suffix variation
- honorifics
- abbreviations
- transliteration
- spelling errors
- word-order changes
- spacing differences
- numeric variation
- multilingual text

Learn challenge-specific transliteration / abbreviation behavior only from training data and keep those transformations cross-fitted to avoid leakage.

---

## 2. Country-aware retrieval

Country should be used as a strong search restriction because the analyzed training structure shows no cross-country ground-truth matches, but the model must remain open-set for France and must not hard-code the world to only US and India.

For each country partition, retrieve candidates through multiple independent channels.

### Retrieval channels

```text
Exact normalized name
Exact normalized address
Exact name + address
Rare-token inverted index
BM25 name
BM25 address
BM25 name + address
Character TF-IDF name
Character TF-IDF address
Name + address composite retrieval
Numeric / postal / locality retrieval
Reverse target -> S1 retrieval
Cross-script retrieval
Missing-field retrieval
Sibling / cluster expansion
```

Do not depend on a single top-K retrieval source.

The candidate set is the union of all retrieval channels.

---

## 3. Reverse retrieval is mandatory

Do both directions:

```text
S1 -> S2/S3
S2/S3 -> S1
```

Forward retrieval is useful for finding copies of an S1 entity.

Reverse retrieval is useful when a target record contains unusual or corrupted information that makes S1-side retrieval weak.

For every candidate preserve:

- retrieval channel
- retrieval rank
- retrieval score
- reciprocal rank
- best rank
- second-best rank
- score gap
- number of independent retrieval channels agreeing

This converts retrieval from a hidden preprocessing step into model evidence.

---

## 4. Missing-name / missing-address retrieval

This is a first-class subsystem because the team has already identified empty, truncated and incomplete copies as one of the largest hidden-test failure modes.

Do not apply the same retrieval rule to all records.

When name is weak or missing, emphasize:

- address tokens
- postal code
- locality
- city/state
- house number
- numeric tokens
- rare tokens
- cross-source evidence

When address is weak or missing, emphasize:

- name
- rare name tokens
- transliteration
- abbreviations
- phonetic / character similarity
- reverse retrieval
- S2/S3 support

When both are weak, use the strongest surviving evidence and increase the candidate budget.

---

## 5. Adaptive candidate budgets

Do not use one fixed candidate K for every record.

Use a small budget for obvious matches and a larger budget for ambiguous records.

Conceptually:

```text
Easy exact case            -> small K
Strong lexical agreement   -> medium K
Ambiguous name              -> large K
Missing address             -> large K
Missing name                -> large K
High collision density     -> very large K
Unseen-domain / France     -> large K
```

The only non-negotiable requirement is that candidate recall remains extremely high.

### Candidate-recall target

Engineering target:

```text
Overall candidate recall       >= 99.995%
US candidate recall            >= 99.995%
India candidate recall         >= 99.995%
France-like / unseen-domain   >= 99.99%
S2 candidate recall             >= 99.995%
S3 candidate recall             >= 99.995%
Missing-field recall            >= 99.99%
```

These are system targets, not claims that they are already achieved.

If a true pair is absent from the candidate set, no downstream model can recover it.

---

# Phase 2 — Learn the Correct Match and the Correct Competition

## 6. Feature engine

Build roughly 100–150 meaningful features, grouped by evidence family rather than blindly increasing feature count.

### Name evidence

Include:

```text
exact raw match
exact normalized match
exact legal-normalized match
exact compact match
Levenshtein similarity
Jaro-Winkler
RapidFuzz ratio
partial ratio
token-sort similarity
token-set similarity
token Jaccard
character n-gram similarity
prefix / suffix similarity
first-token match
last-token match
acronym match
token-count difference
length ratio
transliteration similarity
```

### Address evidence

Include:

```text
exact address
Levenshtein
Jaro-Winkler
RapidFuzz
token Jaccard
token overlap
postal exact match
postal prefix match
house-number exact match
house-number absolute difference
house-number relative difference
house-number digit similarity
city similarity
state similarity
locality similarity
component agreement
component conflict
missingness pattern
```

### Numeric evidence

Give numeric evidence its own feature family:

```text
number-set intersection
number-set Jaccard
exact numeric sequence match
house-number agreement
conflicting-number count
postal agreement
```

### Retrieval evidence

Include:

```text
retrieved by exact name?
retrieved by exact address?
retrieved by BM25?
retrieved by TF-IDF?
retrieved by reverse retrieval?
retrieved by cross-script retrieval?
agreement count
best rank
second-best rank
rank gap
best score
second-best score
score gap
reciprocal rank
```

### Context evidence

Include:

```text
number of competing S1 candidates
candidate density
name rarity
address rarity
token rarity
same-name frequency
same-address frequency
S1 sibling support
target competition
```

---

## 7. Model ensemble

Train two complementary tree models:

```text
XGBoost
LightGBM
```

Use entity-disjoint out-of-fold predictions.

Then tune the blend:

```text
P_pair = alpha * P_XGB + (1 - alpha) * P_LGBM
```

Only add CatBoost if it demonstrates genuinely complementary out-of-fold errors.

Do not introduce a large neural model simply because it is more complex. The main bottlenecks are candidate recall, hard negatives, distribution shift and entity-level decision logic.

---

## 8. Hard-negative mining

Random negatives are insufficient.

The training set must contain difficult false matches such as:

```text
same name / different address
same address / different name
same locality / different business
nearby house numbers
same-country collisions
high TF-IDF false positives
high BM25 false positives
reverse-retrieval false positives
same-name siblings
cross-source look-alikes
```

Run iterative mining:

```text
Train V1
    ->
score difficult candidates
    ->
collect high-confidence false positives
    ->
bucket by failure type
    ->
add to training
    ->
Train V2
    ->
repeat until improvements saturate
```

The goal is to teach the classifier the exact mistakes it is making.

---

## 9. Density-matched training

The negative distribution used for training and validation must resemble the real inference problem.

Match:

- candidate-set density
- country/source mix
- missingness
- collision rates
- difficult-name frequency
- retrieval-channel distribution
- hard-negative categories

Do not create an artificially easy validation universe and then optimize against it.

---

## 10. S2 <-> S3 collective evidence

In addition to:

```text
S1 -> S2
S1 -> S3
```

model:

```text
S2 <-> S3
```

as supporting evidence.

For candidate S1-S2 and S1-S3 relationships, derive:

```text
S2/S3 name similarity
S2/S3 address similarity
S2/S3 numeric agreement
S2/S3 rare-token overlap
S2/S3 house-number consistency
S2/S3 retrieval agreement
```

Then create a collective support score.

The purpose is not to replace S1 matching. It is to determine whether multiple source records independently tell the same business story.

---

## 11. Sibling / entity-context reasoning

For each S1 candidate, ask:

```text
Does this target agree with other strong copies associated with the same S1?
```

For example:

```text
S1-A -> T1 = strong
S1-A -> T2 = strong
S1-A -> T3 = strong

candidate T4 = medium
```

If T4 agrees with the same entity fingerprint and context, its probability should increase.

Conversely, if T4 strongly competes with another S1, its score should decrease.

This is the difference between pair classification and true entity resolution.

---

# Phase 3 — Convert Pair Scores into the Final Matching Set

## 12. Calibration

Use out-of-fold predictions for:

```text
isotonic regression
or
Platt scaling
```

Keep calibration only if it improves downstream decision quality.

Do not assume a calibrated number is automatically a better competition score.

---

## 13. Candidate competition and margin

For every target:

```text
p1 = strongest S1 candidate
p2 = second strongest S1 candidate

margin = p1 - p2
```

Do not treat:

```text
0.997 vs 0.10
```

the same as:

```text
0.997 vs 0.996
```

The first is high-confidence.

The second is ambiguous.

Use:

```text
probability
+
margin
+
retrieval agreement
+
competition density
+
cross-source support
```

to make the decision.

---

## 14. Expected-F0.5 decoding

Do not independently threshold every pair.

For each S1:

1. Rank candidate targets by calibrated probability.
2. Consider the possible match sets:
   - empty
   - top 1
   - top 2
   - top 3
   - ...
3. Estimate the expected F0.5 of each set.
4. Select the set with the strongest expected competition metric.

This directly optimizes the structure of the competition instead of optimizing a generic binary classification objective.

The decoder must allow:

```text
zero matches
one match
multiple matches
```

---

## 15. Singleton gate

Some S1 records genuinely have no matching target.

For every S1 compute:

```text
best probability
second-best probability
margin
candidate density
evidence strength
```

When evidence is insufficient:

```text
return empty match set
```

Never force top-1 matching merely because a candidate exists.

This is essential for a precision-heavy macro metric.

---

## 16. Target exclusivity

After scoring, enforce the structural constraint:

```text
One S2/S3 target -> at most one S1
One S1       -> zero / one / many S2/S3
```

If several S1s claim the same target:

```text
retain the strongest structurally consistent assignment
reject weaker competing assignments
```

This should happen after the candidate scorer has evaluated the evidence.

---

## 17. Contradiction checks

Add deterministic safety rules for obvious contradictions, such as:

```text
country conflict
strong address conflict
strong numeric conflict
impossible competition assignment
```

These should be conservative safety checks, not aggressive hand-written matching rules.

---

# Phase 4 — Make the Local Validation Resemble the Hidden Leaderboard

This phase is the difference between a good local model and a competitive submission.

## 18. Validation protocol

Do not rely on one random split.

Use:

### A. Entity-disjoint OOF

Group by S1 so no S1 appears in both training and validation.

### B. Density-matched validation

Match candidate density and negative composition to full inference.

### C. Hard-negative stress validation

Explicitly evaluate:

```text
same-name collisions
same-address collisions
nearby house numbers
missing address
missing name
both missing
typos
abbreviations
transliteration
orphan copies
truncated copies
```

### D. Domain-shift validation

Create controlled unseen-domain stress tests for France-like behavior.

### E. Public leaderboard validation

Every serious experiment is eventually judged by:

```text
public score - local score
```

Track this gap explicitly.

---

## 19. Error budget

A 0.9999 macro target leaves only roughly 0.01% average error.

Therefore:

> **Do not call a pipeline final because its average F0.5 is high.**

The final candidate must also show near-perfect behavior on the critical slices.

Minimum acceptance targets:

```text
Overall candidate recall       >= 99.995%
Pair precision                 >= 99.99%
Singleton FP rate              < 0.02% target
Major missing-field slices     >= 99.99%
France-like retrieval          >= 99.99%
No structural output errors
```

For the final model, track how many S1 entities are actually causing score loss.

The objective is to shrink the error population, not merely increase the mean model probability.

---

# Final Architecture

The final system should converge to:

```text
                         RAW DATA
                            |
                            v
                 MULTI-VIEW NORMALIZATION
                            |
                            v
                    COUNTRY PARTITION
                            |
                            v
                 MULTI-CHANNEL RETRIEVAL
                            |
      +---------------------+----------------------+
      |                     |                      |
      v                     v                      v
   FORWARD              REVERSE                CROSS-SOURCE
   RETRIEVAL            RETRIEVAL               RETRIEVAL
      |                     |                      |
      +---------------------+----------------------+
                            |
                            v
                     CANDIDATE UNION
                            |
                            v
                 RETRIEVAL-RECALL AUDIT
                            |
                            v
                  PAIR + CONTEXT FEATURES
                            |
                            v
                 XGBOOST + LIGHTGBM
                            |
                            v
                   HARD-NEGATIVE LOOP
                            |
                            v
                   OOF PROBABILITY
                     CALIBRATION
                            |
                            v
                 COLLECTIVE RERANKING
                            |
           +----------------+----------------+
           |                |                |
           v                v                v
       S1 SUPPORT       S2 <-> S3       TARGET COMPETITION
           |                |                |
           +----------------+----------------+
                            |
                            v
                 EXPECTED-F0.5 DECODER
                            |
             +--------------+--------------+
             |              |              |
             v              v              v
          MARGIN        SINGLETON      EXCLUSIVITY
           GATE            GATE            GATE
             |              |              |
             +--------------+--------------+
                            |
                            v
                    FINAL MATCH SET
                            |
                            v
                      VALID SUBMISSION
```

# Definition of Done

The work is complete only when all of the following are true:

```text
[ ] Multi-view normalization is stable and leakage-safe
[ ] Forward retrieval is implemented
[ ] Reverse target -> S1 retrieval is implemented
[ ] Missing-field retrieval is implemented
[ ] Candidate union preserves retrieval provenance
[ ] Candidate recall is >= 99.995% on the main validation population
[ ] Candidate recall is >= 99.99% on difficult slices
[ ] ~100–150 complementary pair/context features are available
[ ] XGBoost + LightGBM ensemble is trained
[ ] Hard-negative mining is iterative
[ ] Training negatives are density-matched
[ ] S2 <-> S3 collective features are tested
[ ] S1 sibling/context features are tested
[ ] OOF calibration is implemented and validated
[ ] Margin-based decisions are implemented
[ ] Singleton abstention is implemented
[ ] Target exclusivity is enforced
[ ] Expected-F0.5 set decoding is implemented
[ ] France/unseen-domain stress testing passes
[ ] Missing-name/address stress testing passes
[ ] Public-vs-local gap is tracked for every serious submission
[ ] Submission format passes all integrity checks
[ ] Final pipeline is reproducible from a clean environment
```

# Submission Philosophy

Do not submit every local improvement.

A candidate should be submitted only when it has a reason to exist:

```text
Experiment
    ->
measurable improvement
    ->
error-category improvement
    ->
stress-test improvement
    ->
submission
```

The public leaderboard is an external validation signal, not the optimization target by itself.

The final model should be the pipeline with the strongest evidence that its **retrieval, ranking and decision behavior generalizes to the hidden test distribution**.

# One-Line Strategy

> **Recover almost every true candidate, distinguish the few dangerous look-alikes, reason across S1/S2/S3 relationships, and make the final decision at the entity level rather than the pair level.**

# Target

**Engineering target: >= 0.9999 final Macro F0.5 / submission score.**

No single component is expected to produce this score. The target is achieved only if the complete retrieval + ranking + collective reasoning + decision system drives the remaining error population down to the required level.
