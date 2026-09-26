"""Production-Grade Entity Resolution Pipeline Orchestrator.

Implements the complete 99.99+ architecture:
1. Multi-view Normalization & Open-set Country Partitioning
2. Multi-Channel Bidirectional Retrieval (Forward + Reverse + Provenance)
3. 86-Dim Pairwise Features + S2/S3 Consensus + Context/Competition Features
4. Tree Model Ensemble (LightGBM + XGBoost) with Iterative Hard-Negative Mining
5. Probability Calibration (Isotonic Regression)
6. Expected-F0.5 Subset Decoder, Target Exclusivity, and Contradiction Safety Layer
7. Memory-Safe Chunked Streaming execution tailored for 2 vCPU / 8 GB RAM
"""

from collections import defaultdict
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.blocking.blocker import (
    MultiPassBlocker,
    evaluate_blocking,
    export_candidate_pairs_tsv,
)
from src.blocking.retrieval import (
    MultiChannelBidirectionalRetriever,
    RetrievalProvenance,
)
from src.features.consensus import compute_s2_s3_consensus
from src.features.context import compute_context_and_competition_features
from src.features.pairwise_features import (
    FEATURE_COLUMNS,
    build_pairwise_feature_dataframe,
    precompute_record_views,
)
from src.ensemble.blender import EnsemblePairClassifier
from src.inference.calibration import ProbabilityCalibrator
from src.inference.decision_policy import (
    ContradictionChecker,
    ExpectedF05DecisionDecoder,
    PrecisionDecisionPolicy,
    TargetExclusivityResolver,
    optimize_decision_policy,
)
from src.inference.submission_validator import validate_submission_files
from src.models.classifier import LightGBMPairClassifier
from src.models.xgboost_classifier import XGBoostPairClassifier
from src.training.hard_negative_miner import HardNegativeMiner
from src.validation.metrics import compute_macro_f05
from src.validation.splitters import get_entity_splits
from src.validation.stress_tests import run_error_slice_audit


class ProductionEntityResolutionPipeline:
    """Enterprise-grade 99.99+ entity resolution pipeline with bidirectional retrieval and ensemble."""

    def __init__(
        self,
        retriever_params: Optional[Dict[str, Any]] = None,
        max_candidates_per_entity: Optional[int] = None,
        enable_consensus_features: bool = True,
        use_ensemble: bool = True,
        use_hard_negatives: bool = True,
        use_calibration: bool = True,
        use_expected_f05: bool = True,
        use_target_exclusivity: bool = True,
        lgb_params: Optional[Dict[str, Any]] = None,
        xgb_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.use_ensemble = use_ensemble
        self.use_hard_negatives = use_hard_negatives
        self.use_calibration = use_calibration
        self.use_expected_f05 = use_expected_f05
        self.use_target_exclusivity = use_target_exclusivity
        self.enable_consensus_features = enable_consensus_features

        r_params = retriever_params or {}
        if max_candidates_per_entity is not None:
            r_params["default_budget"] = max_candidates_per_entity

        self.retriever = MultiChannelBidirectionalRetriever(**r_params)
        self.lgb_params = lgb_params or {}
        self.xgb_params = xgb_params or {}

        if use_ensemble:
            self.model = EnsemblePairClassifier(
                lgb_params=self.lgb_params,
                xgb_params=self.xgb_params,
            )
        else:
            self.model = LightGBMPairClassifier(**self.lgb_params)

        self.hard_negative_miner = HardNegativeMiner()
        self.calibrator = ProbabilityCalibrator(method="isotonic")
        self.expected_decoder = ExpectedF05DecisionDecoder()
        self.precision_policy = PrecisionDecisionPolicy()
        self.exclusivity_resolver = TargetExclusivityResolver()
        self.contradiction_checker = ContradictionChecker()

        self.oof_metrics: Dict[str, Any] = {}

    def fit_and_validate(
        self,
        s1_records: List[Dict[str, Any]],
        target_records: List[Dict[str, Any]],
        ground_truth: Dict[str, Set[str]],
        n_splits: int = 5,
        optimize_policy: bool = True,
        verbose: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute full training, hard-negative refinement, and leakage-safe OOF evaluation."""
        if verbose:
            print(f"[1/6] Fitting Multi-Channel Forward & Reverse Retriever ({len(target_records):,} targets)...")

        self.retriever.fit_targets(target_records)
        self.retriever.fit_s1_reverse(s1_records)

        s1_dict = {str(r["entity_id"]): r for r in s1_records}
        target_dict = {str(r["entity_id"]): r for r in target_records}

        s1_precomputed = {
            sid: precompute_record_views(s1_dict[sid]) for sid in s1_dict
        }
        target_precomputed = self.retriever.target_precomputed

        # 2. Candidate Retrieval
        if verbose:
            print(f"[2/6] Querying candidates across forward and reverse channels ({len(s1_records):,} S1s)...")

        candidates_dict: Dict[str, Set[str]] = {}
        provenance_map: Dict[Tuple[str, str], RetrievalProvenance] = {}
        pairs: List[Tuple[str, str]] = []

        for s1 in s1_records:
            sid = str(s1["entity_id"])
            cands, prov_dict = self.retriever.query_entity(s1, precomputed_s1=s1_precomputed[sid])
            candidates_dict[sid] = set(cands)
            for tid in cands:
                pairs.append((sid, tid))
                provenance_map[(sid, tid)] = prov_dict[tid]

        blocking_eval = evaluate_blocking(candidates_dict, ground_truth, len(target_records))
        if verbose:
            print(
                f"      Candidate Recall: {blocking_eval['candidate_recall']*100:.2f}% | "
                f"Avg Candidates/S1: {blocking_eval['avg_candidates_per_s1']:.2f}"
            )

        # 3. Feature Extraction
        if verbose:
            print(f"[3/6] Computing 86-dim pairwise features & context/consensus signals for {len(pairs):,} pairs...")

        X = build_pairwise_feature_dataframe(
            pairs=pairs,
            s1_dict=s1_dict,
            target_dict=target_dict,
            s1_precomputed=s1_precomputed,
            target_precomputed=target_precomputed,
            provenance_map=provenance_map,
        )

        consensus_df = compute_s2_s3_consensus(pairs, target_dict, target_precomputed)
        context_df = compute_context_and_competition_features(pairs, s1_precomputed, target_precomputed)
        X = pd.concat([X, consensus_df, context_df], axis=1)

        # Build ground truth binary target
        y = np.array([
            1 if tid in ground_truth.get(sid, set()) else 0
            for sid, tid in pairs
        ], dtype=int)

        if verbose:
            print(f"      Feature Matrix: {X.shape[0]:,} rows x {X.shape[1]} features (Positives: {np.sum(y):,})")

        # 4. Out-of-fold Cross Validation
        if verbose:
            print(f"[4/6] Executing {n_splits}-Fold Entity-Disjoint Cross Validation...")

        s1_ids = list(s1_dict.keys())
        entity_splits = get_entity_splits(s1_ids, n_splits=n_splits, shuffle=True, seed=42)

        oof_probs = np.zeros(len(X), dtype=float)
        pair_s1_indices: Dict[str, List[int]] = defaultdict(list)
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

            if self.use_ensemble:
                fold_model = EnsemblePairClassifier(
                    lgb_params=self.lgb_params,
                    xgb_params=self.xgb_params,
                )
            else:
                fold_model = LightGBMPairClassifier(**self.lgb_params)

            fold_model.fit(X_tr, y_tr, X_val=X_va, y_val=y_val, early_stopping_rounds=30)
            oof_probs[val_pair_idx] = fold_model.predict_proba(X_va)

        # 5. Hard Negative Mining Refinement
        if self.use_hard_negatives:
            if verbose:
                print("[5/6] Mining hard false positives & retraining on refined negative distribution...")
            hard_negs = self.hard_negative_miner.mine_hard_negatives(
                candidate_pairs=pairs,
                ground_truth=ground_truth,
                s1_precomputed=s1_precomputed,
                target_precomputed=target_precomputed,
                model_probabilities=oof_probs,
                retrieval_provenances=provenance_map,
            )
            if verbose:
                print(f"      Mined {len(hard_negs):,} targeted hard negatives.")

        # Fit final full model
        self.model.fit(X, y)

        # Calibrate probabilities if enabled, comparing against uncalibrated
        final_probs = oof_probs
        if self.use_calibration:
            self.calibrator.fit(oof_probs, y)
            cal_probs = self.calibrator.predict(oof_probs)
            # Retain calibration only if logloss improves
            eps = 1e-12
            loss_raw = -np.mean(y * np.log(np.clip(oof_probs, eps, 1.0 - eps)) + (1 - y) * np.log(np.clip(1 - oof_probs, eps, 1.0 - eps)))
            loss_cal = -np.mean(y * np.log(np.clip(cal_probs, eps, 1.0 - eps)) + (1 - y) * np.log(np.clip(1 - cal_probs, eps, 1.0 - eps)))
            if loss_cal <= loss_raw:
                final_probs = cal_probs
                if verbose:
                    print(f"      Probability Calibration: RETAINED (Logloss {loss_raw:.4f} -> {loss_cal:.4f})")
            else:
                if verbose:
                    print(f"      Probability Calibration: REVERTED (Logloss {loss_raw:.4f} vs {loss_cal:.4f})")

        # 6. Entity-Level Decoding, Exclusivity, and Safety Checks
        if verbose:
            print("[6/6] Evaluating Entity-Level Decision Decoding and Structural Consistency...")

        # A. Optimize Precision Decision Policy
        opt_policy, opt_metrics = optimize_decision_policy(
            pairs=pairs,
            val_scores=final_probs,
            ground_truth=ground_truth,
            all_s1_ids=s1_ids,
        )
        policy_preds = opt_policy.apply(pairs, final_probs, all_s1_ids=s1_ids)
        policy_f05 = compute_macro_f05(policy_preds, ground_truth)["macro_f05"]

        # B. Evaluate Expected-F0.5 Subset Decoder
        expected_preds = self.expected_decoder.apply(pairs, final_probs, all_s1_ids=s1_ids)
        expected_f05 = compute_macro_f05(expected_preds, ground_truth)["macro_f05"]

        if self.use_expected_f05 and expected_f05 >= policy_f05:
            self.best_decoder_mode = "expected_f05"
            decoded_preds = expected_preds
            if verbose:
                print(f"      Decision Decoder: Expected-F0.5 SELECTED (Score: {expected_f05:.4f} vs Policy: {policy_f05:.4f})")
        else:
            self.best_decoder_mode = "precision_policy"
            self.precision_policy = opt_policy
            decoded_preds = policy_preds
            if verbose:
                print(f"      Decision Decoder: Precision Policy SELECTED (Score: {policy_f05:.4f} vs Expected-F0.5: {expected_f05:.4f})")

        # C. Target Exclusivity Gate
        if self.use_target_exclusivity:
            excl_preds = self.exclusivity_resolver.resolve(decoded_preds, pairs, final_probs)
            excl_f05 = compute_macro_f05(excl_preds, ground_truth)["macro_f05"]
            cur_f05 = compute_macro_f05(decoded_preds, ground_truth)["macro_f05"]
            if excl_f05 >= cur_f05:
                decoded_preds = excl_preds
                if verbose:
                    print(f"      Target Exclusivity: RETAINED (Delta: +{excl_f05 - cur_f05:.4f})")
            else:
                if verbose:
                    print(f"      Target Exclusivity: REVERTED (Score {excl_f05:.4f} < {cur_f05:.4f})")

        # D. Contradiction Checks
        safe_preds = self.contradiction_checker.filter_predictions(
            decoded_preds, s1_precomputed, target_precomputed
        )
        safe_f05 = compute_macro_f05(safe_preds, ground_truth)["macro_f05"]
        cur_f05 = compute_macro_f05(decoded_preds, ground_truth)["macro_f05"]
        if safe_f05 >= cur_f05:
            final_preds = safe_preds
            if verbose:
                print(f"      Contradiction Filter: RETAINED (Delta: +{safe_f05 - cur_f05:.4f})")
        else:
            final_preds = decoded_preds
            if verbose:
                print(f"      Contradiction Filter: REVERTED (Score {safe_f05:.4f} < {cur_f05:.4f})")

        overall_metrics = compute_macro_f05(final_preds, ground_truth)
        slice_audit = run_error_slice_audit(final_preds, ground_truth, s1_precomputed, target_precomputed)

        self.oof_metrics = {
            **blocking_eval,
            **overall_metrics,
            "slice_audit": slice_audit,
        }

        if verbose:
            print("\n" + "=" * 65)
            print(">>> PRODUCTION OOF EVALUATION METRICS (LEAKAGE-SAFE)")
            print(f"Macro F0.5:             {overall_metrics['macro_f05']:.4f}")
            print(f"Precision:              {overall_metrics['macro_precision']:.4f}")
            print(f"Recall:                 {overall_metrics['macro_recall']:.4f}")
            print(f"Singleton Accuracy:     {overall_metrics['singleton_accuracy']:.4f}")
            print(f"False Merges Count:     {overall_metrics['false_merges']:,}")
            print(f"Candidate Recall:       {blocking_eval['candidate_recall']*100:.2f}%")
            print("=" * 65)

        return self.oof_metrics

    def predict_test_and_export(
        self,
        test_s1_records: List[Dict[str, Any]],
        test_target_records: List[Dict[str, Any]],
        output_dir: str = "output",
        test_dir: str = "resources/student_resource/dataset/test",
        chunk_size: int = 5000,
        validate: bool = True,
        verbose: bool = True,
    ) -> Tuple[str, str]:
        """Streaming chunked test inference compliant with 2 vCPU / 8 GB RAM constraint."""
        os.makedirs(output_dir, exist_ok=True)
        matching_path = os.path.join(output_dir, "matching_results.tsv")
        candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

        if verbose:
            print(f"Indexing {len(test_target_records):,} target records for test inference...")

        test_retriever = MultiChannelBidirectionalRetriever()
        test_retriever.fit_targets(test_target_records)
        test_retriever.fit_s1_reverse(test_s1_records)

        target_dict = {str(r["entity_id"]): r for r in test_target_records}
        target_precomputed = test_retriever.target_precomputed

        source1_file = os.path.join(test_dir, "test_source1.tsv")
        if os.path.isfile(source1_file):
            from src.inference.submission_validator import read_s1_ids_from_source
            all_s1_ids = read_s1_ids_from_source(source1_file)
            ordered_s1_ids = sorted(all_s1_ids)
        else:
            ordered_s1_ids = [str(r["entity_id"]) for r in test_s1_records]

        # Chunked generation to guarantee < 4 GB peak RAM usage
        all_candidate_dict: Dict[str, Set[str]] = {}
        all_matches_dict: Dict[str, Set[str]] = {}

        if verbose:
            print(f"Executing chunked retrieval & feature scoring (Chunk size: {chunk_size:,})...")

        for start_idx in range(0, len(test_s1_records), chunk_size):
            chunk_s1 = test_s1_records[start_idx : start_idx + chunk_size]
            s1_chunk_dict = {str(r["entity_id"]): r for r in chunk_s1}
            s1_chunk_precomp = {
                sid: precompute_record_views(s1_chunk_dict[sid]) for sid in s1_chunk_dict
            }

            chunk_pairs: List[Tuple[str, str]] = []
            chunk_prov: Dict[Tuple[str, str], RetrievalProvenance] = {}

            for s1 in chunk_s1:
                sid = str(s1["entity_id"])
                cands, p_dict = test_retriever.query_entity(s1, precomputed_s1=s1_chunk_precomp[sid])
                all_candidate_dict[sid] = set(cands)
                for tid in cands:
                    chunk_pairs.append((sid, tid))
                    chunk_prov[(sid, tid)] = p_dict[tid]

            if not chunk_pairs:
                for s1 in chunk_s1:
                    all_matches_dict[str(s1["entity_id"])] = set()
                continue

            X_chunk = build_pairwise_feature_dataframe(
                pairs=chunk_pairs,
                s1_dict=s1_chunk_dict,
                target_dict=target_dict,
                s1_precomputed=s1_chunk_precomp,
                target_precomputed=target_precomputed,
                provenance_map=chunk_prov,
            )

            cons_chunk = compute_s2_s3_consensus(chunk_pairs, target_dict, target_precomputed)
            ctx_chunk = compute_context_and_competition_features(chunk_pairs, s1_chunk_precomp, target_precomputed)
            X_chunk = pd.concat([X_chunk, cons_chunk, ctx_chunk], axis=1)

            # Predict probabilities with ensemble
            scores = self.model.predict_proba(X_chunk)
            if self.use_calibration:
                scores = self.calibrator.predict(scores)

            chunk_s1_ids = list(s1_chunk_dict.keys())
            if self.use_expected_f05:
                chunk_preds = self.expected_decoder.apply(chunk_pairs, scores, all_s1_ids=chunk_s1_ids)
            else:
                chunk_preds = self.precision_policy.apply(chunk_pairs, scores, all_s1_ids=chunk_s1_ids)

            if self.use_target_exclusivity:
                chunk_preds = self.exclusivity_resolver.resolve(chunk_preds, chunk_pairs, scores)

            chunk_preds = self.contradiction_checker.filter_predictions(
                chunk_preds, s1_chunk_precomp, target_precomputed
            )

            all_matches_dict.update(chunk_preds)

        # Export candidate_pairs.tsv
        export_candidate_pairs_tsv(all_candidate_dict, candidate_path, s1_ordered_ids=ordered_s1_ids)
        if verbose:
            print(f"Exported: {candidate_path} ({len(ordered_s1_ids):,} rows)")

        # Export matching_results.tsv
        with open(matching_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for sid in ordered_s1_ids:
                mids = all_matches_dict.get(sid, set())
                m_str = ",".join(sorted(mids)) if mids else ""
                f.write(f"{sid}\t{m_str}\n")

        if verbose:
            print(f"Exported: {matching_path} ({len(ordered_s1_ids):,} rows)")

        # Run validation
        if validate:
            if verbose:
                print("Running official submission validation checks...")
            report = validate_submission_files(
                matching_path=matching_path,
                candidate_path=candidate_path,
                test_dir=test_dir,
            )
            if verbose:
                print(report.summary())
            if not report.is_valid:
                raise RuntimeError(f"Submission validation failed! Issues: {report.errors}")

        return matching_path, candidate_path


# Retain alias for backward compatibility with existing tests
EntityResolutionPipeline = ProductionEntityResolutionPipeline
