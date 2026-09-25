import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

# Official F0.5 metric definition
def calc_f05(precision, recall):
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def entity_f05(true_set, pred_set):
    # Singletons
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    if len(pred_set) == 0:
        return 0.0
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return calc_f05(prec, rec)

print("=== MACRO F0.5 MATHEMATICAL SENSITIVITY ANALYSIS ===")

# Scenario 1: Entity has 3 true matches.
# What happens if we predict 2 true matches (miss 1)?
true_3 = {1, 2, 3}
print(f"True matches: 3. Predict 3 TP (3/3): F0.5 = {entity_f05(true_3, {1, 2, 3}):.4f}")
print(f"True matches: 3. Predict 2 TP (2/3, 0 FP): F0.5 = {entity_f05(true_3, {1, 2}):.4f}")
print(f"True matches: 3. Predict 1 TP (1/3, 0 FP): F0.5 = {entity_f05(true_3, {1}):.4f}")
print(f"True matches: 3. Predict 3 TP + 1 FALSE MERGE (3/3, 1 FP): F0.5 = {entity_f05(true_3, {1, 2, 3, 99}):.4f}")
print(f"True matches: 3. Predict 2 TP + 1 FALSE MERGE (2/3, 1 FP): F0.5 = {entity_f05(true_3, {1, 2, 99}):.4f}")

# Scenario 2: Singleton entity (0 true matches)
print("\n--- SINGLETON ENTITY ---")
true_0 = set()
print(f"True: 0. Predict empty: F0.5 = {entity_f05(true_0, set()):.4f}")
print(f"True: 0. Predict 1 false match: F0.5 = {entity_f05(true_0, {99}):.4f}")

# Critical comparison:
# If you are 50% sure about a borderline candidate:
# Option A: omit it -> Precision stays 1.0, Recall is slightly lower.
# Option B: include it -> If it's a false positive, precision drops from 1.0 to 0.75 or 0.67, causing a MASSIVE drop in F0.5!
drop_miss = entity_f05(true_3, {1, 2, 3}) - entity_f05(true_3, {1, 2})
drop_false_merge = entity_f05(true_3, {1, 2, 3}) - entity_f05(true_3, {1, 2, 3, 99})

print(f"\nDrop from 1 missed match (FN): {drop_miss:.4f}")
print(f"Drop from 1 false merge (FP): {drop_false_merge:.4f}")
print(f"False merge penalty is {drop_false_merge / drop_miss:.2f}x WORSE than a missed match on F0.5!")
