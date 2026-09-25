"""End-to-end Entity Resolution Pipeline orchestrating blocking, features, modeling, and inference."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.blocking.blocker import (
    MultiPassBlocker,
    evaluate_blocking,
    export_candidate_pairs_tsv,
)
from src.features.consensus import compute_s2_s3_consensus
from src.features.pairwise_features import (
    build_pairwise_feature_matrix,
    precompute_record_views,
)
from src.inference.decision_policy import (
    PrecisionDecisionPolicy,
    optimize_decision_policy,
)
from src.inference.submission_validator import validate_submission_package
from src.models.classifier import LightGBMPairClassifier
from src.validation.metrics import compute_macro_f05
from src.validation.splitters import get_cv_splitter, get_entity_splits


class EntityResolutionPipeline:
    """Orchestrates end-to-end entity resolution workflow."""

    def __init__(
        self,
        max_candidates_per_entity: int = 50,
        enable_consensus_features: bool = True,
        lgb_params: Optional[Dict[str, object]] = None,
    ):
        self.max_candidates_per_entity = max_candidates_per_entity
        self.enable_consensus_features = enable_consensus_features
        self.blocker = MultiPassBlocker(max_candidates_per_entity=max_candidates_per_entity)
        self.model = LightGBMPairClassifier(**(lgb_params or {}))
        self.policy = PrecisionDecisionPolicy()
        self.oof_metrics: Dict[str, float] = {}

    def fit_and_validate(
        self,
        s1_records: List[Dict[str, str]],
        target_records: List[Dict[str, str]],
        ground_truth: Dict[str, Set[str]],
        n_splits: int = 5,
        optimize_policy: bool = True,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Fit blocker, build features, train 5-fold model, optimize policy on OOF, and return metrics."""
        if verbose:
            print(f"[1/5] Indexing {len(target_records):,} target records in Blocker...")
        self.blocker.fit_targets(target_records)

        if verbose:
            print(f"[2/5] Generating candidate pairs for {len(s1_records):,} S1 records...")
        cands_dict = self.blocker.block_all(s1_records)
        blocking_eval = evaluate_blocking(cands_dict, ground_truth, len(target_records))
        if verbose:
            print(f"      Candidate Recall: {blocking_eval['candidate_recall']*100:.2f}% | "
                  f"Avg Candidates/S1: {blocking_eval['avg_candidates_per_s1']:.2f}")

        if verbose:
            print("[3/5] Extracting pairwise features & consensus signals...")
        s1_dict = {r["entity_id"]: r for r in s1_records}
        target_dict = {r["entity_id"]: r for r in target_records}

        X, y, pairs = build_pairwise_feature_matrix(
            s1_dict=s1_dict,
            target_dict=target_dict,
            candidates_dict=cands_dict,
            ground_truth=ground_truth,
        )

        if self.enable_consensus_features and not X.empty:
            active_target_ids = {tid for _, tid in pairs}
            target_precomputed = {
                tid: precompute_record_views(target_dict[tid])
                for tid in active_target_ids
                if tid in target_dict
            }
            consensus_df = compute_s2_s3_consensus(pairs, target_dict, target_precomputed)
            X = pd.concat([X, consensus_df], axis=1)

        if verbose:
            print(f"      Extracted {X.shape[1]} features across {len(X):,} candidate pairs.")

        # 4. Out-of-fold Training & Prediction
        if verbose:
            print(f"[4/5] Training 5-Fold GroupKFold LightGBM Pair Classifier...")

        s1_ids = list(s1_dict.keys())
        entity_splits = get_entity_splits(s1_ids, n_splits=n_splits, shuffle=True, seed=42)

        oof_probs = np.zeros(len(X), dtype=float)
        pair_s1_indices = {sid: [] for sid in s1_ids}
        for idx, (sid, _) in enumerate(pairs):
            pair_s1_indices[sid].append(idx)

        for fold, (train_s1_idx, val_s1_idx) in enumerate(entity_splits, 1):
            train_s1_set = {s1_ids[i] for i in train_s1_idx}
            val_s1_set = {s1_ids[i] for i in val_s1_idx}

            train_pair_idx = [i for sid in train_s1_set for i in pair_s1_indices[sid]]
            val_pair_idx = [i for sid in val_s1_set for i in pair_s1_indices[sid]]

            if not val_pair_idx:
                continue

            X_tr, y_tr = X.iloc[train_pair_idx], y[train_pair_idx]
            X_va, y_val = X.iloc[val_pair_idx], y[val_pair_idx]

            fold_model = LightGBMPairClassifier(**self.model.params)
            fold_model.fit(X_tr, y_tr, X_val=X_va, y_val=y_val, early_stopping_rounds=30)
            oof_probs[val_pair_idx] = fold_model.predict_proba(X_va)

        # Fit final model on all data
        self.model.fit(X, y)

        # 5. Optimize Decision Policy on OOF
        if optimize_policy:
            if verbose:
                print("[5/5] Optimizing Decision Policy on OOF predictions...")
            opt_policy, opt_metrics = optimize_decision_policy(
                pairs=pairs,
                val_scores=oof_probs,
                ground_truth=ground_truth,
                all_s1_ids=s1_ids,
                verbose=False,
            )
            self.policy = opt_policy
            self.oof_metrics = {**blocking_eval, **opt_metrics}
        else:
            oof_preds = self.policy.apply(pairs, oof_probs, all_s1_ids=s1_ids)
            eval_res = compute_macro_f05(ground_truth, oof_preds)
            self.oof_metrics = {**blocking_eval, **eval_res}

        if verbose:
            print("\n" + "=" * 60)
            print(f"OOF Macro F0.5:        {self.oof_metrics['macro_f05']:.4f}")
            print(f"OOF Macro Precision:   {self.oof_metrics['macro_precision']:.4f}")
            print(f"OOF Macro Recall:      {self.oof_metrics['macro_recall']:.4f}")
            print(f"Singleton Accuracy:    {self.oof_metrics['singleton_accuracy']:.4f}")
            print(f"False Merges Count:    {self.oof_metrics['false_merges']:,}")
            print(f"Candidate Recall:      {self.oof_metrics['candidate_recall']*100:.2f}%")
            print("=" * 60)

        return self.oof_metrics

    def predict_test_and_export(
        self,
        test_s1_records: List[Dict[str, str]],
        test_target_records: List[Dict[str, str]],
        output_dir: str = "output",
        test_dir: str = "resources/student_resource/dataset/test",
        validate: bool = True,
        verbose: bool = True,
    ) -> Tuple[str, str]:
        """Generate test candidate_pairs.tsv and matching_results.tsv and run official validation."""
        os.makedirs(output_dir, exist_ok=True)
        matching_path = os.path.join(output_dir, "matching_results.tsv")
        candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

        if verbose:
            print(f"Indexing {len(test_target_records):,} test target records...")
        test_blocker = MultiPassBlocker(max_candidates_per_entity=self.max_candidates_per_entity)
        test_blocker.fit_targets(test_target_records)

        if verbose:
            print(f"Generating candidate pairs for {len(test_s1_records):,} test S1 records...")
        test_cands = test_blocker.block_all(test_s1_records)

        # Ensure all test S1 IDs appear in submission files
        source1_file = os.path.join(test_dir, "test_source1.tsv")
        if os.path.isfile(source1_file):
            from src.inference.submission_validator import read_s1_ids_from_source
            all_s1_ids = read_s1_ids_from_source(source1_file)
            test_s1_order = sorted(all_s1_ids)
        else:
            test_s1_order = [r["entity_id"] for r in test_s1_records]

        # Export candidate_pairs.tsv
        export_candidate_pairs_tsv(test_cands, candidate_path, s1_ordered_ids=test_s1_order)
        if verbose:
            print(f"Saved: {candidate_path} ({len(test_s1_order):,} rows)")

        # Extract features for test pairs
        test_s1_dict = {r["entity_id"]: r for r in test_s1_records}
        test_target_dict = {r["entity_id"]: r for r in test_target_records}

        X_test, _, test_pairs = build_pairwise_feature_matrix(
            s1_dict=test_s1_dict,
            target_dict=test_target_dict,
            candidates_dict=test_cands,
        )

        if self.enable_consensus_features and not X_test.empty:
            active_tids = {tid for _, tid in test_pairs}
            tgt_precomp = {
                tid: precompute_record_views(test_target_dict[tid])
                for tid in active_tids
                if tid in test_target_dict
            }
            cons_test = compute_s2_s3_consensus(test_pairs, test_target_dict, tgt_precomp)
            X_test = pd.concat([X_test, cons_test], axis=1)

        # Run inference
        if verbose:
            print(f"Scoring {len(X_test):,} candidate pairs with trained model...")
        test_scores = self.model.predict_proba(X_test) if not X_test.empty else np.array([])

        # Apply Decision Policy
        matches_dict = self.policy.apply(test_pairs, test_scores, all_s1_ids=test_s1_order)

        # Export matching_results.tsv
        with open(matching_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for sid in test_s1_order:
                mids = matches_dict.get(sid, set())
                m_str = ",".join(sorted(mids)) if mids else ""
                f.write(f"{sid}\t{m_str}\n")

        if verbose:
            print(f"Saved: {matching_path} ({len(test_s1_order):,} rows)")

        # Validate
        if validate:
            if verbose:
                print("Running official submission validation checks...")
            report = validate_submission_package(
                matching_path=matching_path,
                candidate_path=candidate_path,
                test_dir=test_dir,
            )
            if verbose:
                print(report.summary())
            if not report.is_valid:
                raise RuntimeError(f"Submission validation failed! Issues: {report.errors}")

        return matching_path, candidate_path
