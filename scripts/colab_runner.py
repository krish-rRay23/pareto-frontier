"""High-Speed, Resumable Production Runner for Google Colab with T4 GPU.

Designed to execute the 99.99+ Entity Resolution Pipeline on full 1.73M test records:
1. Fast Local NVMe Copying (optional from Google Drive to /content for 10x I/O speed)
2. Checkpointed Model & Calibrator (skips retraining if checkpoint exists)
3. Resumable Chunked Inference (saves progress every 25,000 records to disk)
4. Fast-path exact-match acceleration for obvious candidates
5. GPU acceleration for XGBoost / LightGBM
6. Auto-concatenation and submission integrity validation
"""

import argparse
import glob
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import joblib
import numpy as np
import pandas as pd
from rich.console import Console

# Add repository root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking.retrieval import MultiChannelBidirectionalRetriever, RetrievalProvenance
from src.features.consensus import compute_s2_s3_consensus
from src.features.context import compute_context_and_competition_features
from src.features.pairwise_features import (
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
from src.inference.submission_validator import validate_submission_package, read_s1_ids_from_source
from src.data.loader import load_source_tsv, load_ground_truth

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="High-Speed Resumable Colab Runner")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/content/drive/MyDrive/ML_challenge/dataset",
        help="Path to dataset directory containing train/ and test/ folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/content/drive/MyDrive/ML_challenge/output",
        help="Path to save final submission TSVs and checkpoints",
    )
    parser.add_argument(
        "--local-scratch",
        type=str,
        default="/content/local_data",
        help="Local fast SSD directory in Colab to copy datasets for high I/O speed",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=25000,
        help="Number of S1 test records per resumable checkpoint chunk",
    )
    parser.add_argument(
        "--n-train-s1",
        type=int,
        default=50000,
        help="Number of S1 records for model training (50k is optimal for speed vs precision)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=True,
        help="Enable CUDA GPU acceleration for tree inference",
    )
    return parser.parse_args()


def resolve_dataset_path(base_path: str) -> str:
    """Handle both '/content/drive/MyDrive' and '/content/drive/My Drive' variations."""
    if os.path.exists(base_path):
        return base_path
    # Try swapping MyDrive and My Drive
    if "MyDrive" in base_path:
        alt_path = base_path.replace("MyDrive", "My Drive")
        if os.path.exists(alt_path):
            return alt_path
    elif "My Drive" in base_path:
        alt_path = base_path.replace("My Drive", "MyDrive")
        if os.path.exists(alt_path):
            return alt_path
    return base_path


def find_dataset_directories(root_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Recursively find train and test directories containing TSV files."""
    train_dir = None
    test_dir = None

    for root, dirs, files in os.walk(root_dir):
        # Look for train_source1.tsv or train ground truth
        if any("train_source1" in f for f in files) or any("train_ground_truth" in f for f in files):
            train_dir = root
        # Look for test_source1.tsv
        if any("test_source1" in f for f in files):
            test_dir = root
        if train_dir and test_dir:
            break

    return train_dir, test_dir


def copy_dataset_to_fast_ssd(src_dir: str, dst_dir: str) -> Tuple[str, str]:
    """Copy TSV files from Google Drive to Colab local SSD for 10x-20x I/O speedup."""
    src_dir = resolve_dataset_path(src_dir)
    train_dst = os.path.join(dst_dir, "train")
    test_dst = os.path.join(dst_dir, "test")

    # If already copied on local SSD, return immediately
    if (
        os.path.exists(os.path.join(train_dst, "train_source1.tsv"))
        and os.path.exists(os.path.join(test_dst, "test_source1.tsv"))
    ):
        console.print(f"[green]Dataset already present on fast local SSD: {dst_dir}[/green]")
        return train_dst, test_dst

    console.print(f"[bold cyan]>>> Locating and copying dataset from ({src_dir}) to fast local SSD ({dst_dir})...[/bold cyan]")
    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(test_dst, exist_ok=True)

    # 1. Direct path check
    train_src = os.path.join(src_dir, "train") if os.path.isdir(os.path.join(src_dir, "train")) else src_dir
    test_src = os.path.join(src_dir, "test") if os.path.isdir(os.path.join(src_dir, "test")) else src_dir

    train_files = glob.glob(os.path.join(train_src, "train*.tsv"))
    test_files = glob.glob(os.path.join(test_src, "test*.tsv"))

    # 2. If not found directly, perform recursive search
    if not train_files or not test_files:
        console.print(f"    [yellow]Files not found in direct subpaths. Searching recursively under {src_dir}...[/yellow]")
        found_train, found_test = find_dataset_directories(src_dir)
        
        # If still not found, search the parent (e.g. if pointing to ML_challenge instead of ML_challenge/dataset)
        if (not found_train or not found_test) and os.path.exists(os.path.dirname(src_dir)):
            found_train, found_test = find_dataset_directories(os.path.dirname(src_dir))

        if found_train:
            train_src = found_train
            train_files = glob.glob(os.path.join(train_src, "train*.tsv"))
            console.print(f"    [green]Found train directory:[/green] {train_src}")
        if found_test:
            test_src = found_test
            test_files = glob.glob(os.path.join(test_src, "test*.tsv"))
            console.print(f"    [green]Found test directory:[/green] {test_src}")

    if not train_files:
        raise FileNotFoundError(
            f"Could not locate 'train_source1.tsv' under '{src_dir}' or its parent directories.\n"
            f"Please ensure the dataset is uploaded to Google Drive, or sync from S3 / GCS.\n"
            f"In Colab, check: !ls -R /content/drive/MyDrive/ML_challenge"
        )

    for f in train_files:
        console.print(f"    Copying {os.path.basename(f)} to train...")
        shutil.copy2(f, train_dst)

    for f in test_files:
        console.print(f"    Copying {os.path.basename(f)} to test...")
        shutil.copy2(f, test_dst)

    console.print(f"[green]Dataset copied successfully to fast SSD! ({len(train_files)} train, {len(test_files)} test files)[/green]\n")
    return train_dst, test_dst


def get_trained_model(
    train_dir: str,
    checkpoint_dir: str,
    n_train_s1: int = 50000,
    use_gpu: bool = True,
) -> Tuple[EnsemblePairClassifier, ProbabilityCalibrator, PrecisionDecisionPolicy]:
    """Train or load checkpointed model, calibrator, and decision policy."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_file = os.path.join(checkpoint_dir, "production_ensemble_ckpt.joblib")

    if os.path.isfile(ckpt_file):
        console.print(f"[bold green]>>> Loading checkpointed model from: {ckpt_file}[/bold green]")
        saved_bundle = joblib.load(ckpt_file)
        return saved_bundle["model"], saved_bundle["calibrator"], saved_bundle["policy"]

    console.print(f"[bold yellow]>>> No checkpoint found. Training high-precision ensemble ({n_train_s1:,} S1s)...[/bold yellow]")
    s1_sample = load_source_tsv(os.path.join(train_dir, "train_source1.tsv"), max_rows=n_train_s1)
    s1_sample_ids = {r["entity_id"] for r in s1_sample}
    console.print(f"    Loaded {len(s1_sample):,} Source 1 training queries.")

    gt_all = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))
    gt_sample = {sid: gt_all[sid] for sid in s1_sample_ids if sid in gt_all}
    needed_tids = {tid for tids in gt_sample.values() for tid in tids}
    console.print(f"    Loaded ground truth ({len(gt_sample):,} labeled S1s, {len(needed_tids):,} positive targets).")

    # Fast targeted loading of targets
    console.print("    Loading reference targets for training (positive matches + hard negative pool)...")
    s2_pos = load_source_tsv(os.path.join(train_dir, "train_source2.tsv"), filter_ids=needed_tids)
    s3_pos = load_source_tsv(os.path.join(train_dir, "train_source3.tsv"), filter_ids=needed_tids)
    s2_bg = load_source_tsv(os.path.join(train_dir, "train_source2.tsv"), max_rows=100000)
    s3_bg = load_source_tsv(os.path.join(train_dir, "train_source3.tsv"), max_rows=100000)

    target_pool_dict = {}
    for r in (s2_pos + s3_pos + s2_bg + s3_bg):
        target_pool_dict[str(r["entity_id"])] = r
    all_targets = list(target_pool_dict.values())
    console.print(f"    [green]Indexed training target pool: {len(all_targets):,} entities.[/green]")

    # Index targets and retrieve pairs
    console.print("    Building multi-channel retrieval index...")
    retriever = MultiChannelBidirectionalRetriever(default_budget=40, ambiguous_budget=80)
    retriever.fit_targets(all_targets)
    retriever.fit_s1_reverse(s1_sample)

    s1_dict = {str(r["entity_id"]): r for r in s1_sample}
    target_dict = target_pool_dict
    s1_pre = {sid: precompute_record_views(s1_dict[sid]) for sid in s1_dict}
    tgt_pre = retriever.target_precomputed

    train_pairs: List[Tuple[str, str]] = []
    prov_map: Dict[Tuple[str, str], RetrievalProvenance] = {}
    console.print("    Retrieving candidate pairs for training queries...")
    for i, s1 in enumerate(s1_sample):
        sid = str(s1["entity_id"])
        cands, p_dict = retriever.query_entity(s1, precomputed_s1=s1_pre[sid])
        for tid in cands:
            train_pairs.append((sid, tid))
            prov_map[(sid, tid)] = p_dict[tid]
        if (i + 1) % 10000 == 0 or (i + 1) == len(s1_sample):
            console.print(f"        Retrieved candidates for {i+1:,}/{len(s1_sample):,} queries ({len(train_pairs):,} candidate pairs)...")

    # Build features
    console.print(f"    Extracting 86 pairwise + consensus + context features for {len(train_pairs):,} pairs...")
    X = build_pairwise_feature_dataframe(train_pairs, s1_dict, target_dict, s1_pre, tgt_pre, prov_map)
    cons_df = compute_s2_s3_consensus(train_pairs, target_dict, tgt_pre)
    ctx_df = compute_context_and_competition_features(train_pairs, s1_pre, tgt_pre)
    X = pd.concat([X, cons_df, ctx_df], axis=1)

    y = np.array([1 if tid in gt_sample.get(sid, set()) else 0 for sid, tid in train_pairs], dtype=int)

    # Configure GPU parameters
    xgb_params = {"n_estimators": 250, "max_depth": 6}
    lgb_params = {"n_estimators": 250, "max_depth": 6, "num_leaves": 31}
    if use_gpu:
        xgb_params["tree_method"] = "hist"
        xgb_params["device"] = "cuda"

    console.print("    Fitting LightGBM + XGBoost Ensemble...")
    model = EnsemblePairClassifier(lgb_params=lgb_params, xgb_params=xgb_params)
    split_idx = int(0.85 * len(X))
    model.fit(X.iloc[:split_idx], y[:split_idx], X_val=X.iloc[split_idx:], y_val=y[split_idx:])

    console.print("    Calibrating probabilities...")
    calibrator = ProbabilityCalibrator(method="isotonic")
    val_probs = model.predict_proba(X.iloc[split_idx:])
    calibrator.fit(val_probs, y[split_idx:])

    console.print("    Optimizing Precision Decision Policy...")
    policy, p_metrics = optimize_decision_policy(
        pairs=train_pairs[split_idx:],
        val_scores=val_probs,
        ground_truth=gt_sample,
        all_s1_ids=list(set(sid for sid, _ in train_pairs[split_idx:])),
    )
    console.print(f"    [green]Training complete! Best OOF Macro F0.5: {p_metrics['macro_f05']:.4f}[/green]")

    # Save checkpoint
    bundle = {"model": model, "calibrator": calibrator, "policy": policy}
    joblib.dump(bundle, ckpt_file)
    console.print(f"    Saved checkpoint to {ckpt_file}\n")
    return model, calibrator, policy


def run_resumable_inference(
    test_dir: str,
    output_dir: str,
    model: EnsemblePairClassifier,
    calibrator: ProbabilityCalibrator,
    policy: PrecisionDecisionPolicy,
    chunk_size: int = 25000,
):
    """Run chunked test inference with complete state resumption on restart."""
    os.makedirs(output_dir, exist_ok=True)
    chunks_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    console.print(f"[bold cyan]>>> Step 1: Loading Test Reference Targets (Source 2 and 3)...[/bold cyan]")
    s2_test = load_source_tsv(os.path.join(test_dir, "test_source2.tsv"))
    s3_test = load_source_tsv(os.path.join(test_dir, "test_source3.tsv"))
    test_targets = s2_test + s3_test
    console.print(f"[green]Loaded {len(test_targets):,} test target pool entities.[/green]")

    console.print(f"[bold cyan]>>> Step 2: Indexing Targets in Multi-Channel Retriever...[/bold cyan]")
    retriever = MultiChannelBidirectionalRetriever(default_budget=35, ambiguous_budget=70)
    retriever.fit_targets(test_targets)
    target_dict = {str(r["entity_id"]): r for r in test_targets}
    tgt_pre = retriever.target_precomputed

    console.print(f"[bold cyan]>>> Step 3: Loading Test Source 1 Queries...[/bold cyan]")
    s1_test = load_source_tsv(os.path.join(test_dir, "test_source1.tsv"))
    total_s1 = len(s1_test)
    console.print(f"[green]Loaded {total_s1:,} Test S1 entities to resolve.[/green]\n")

    num_chunks = int(math.ceil(total_s1 / chunk_size))
    console.print(f"[bold yellow]Total Chunks: {num_chunks} (Chunk size: {chunk_size:,} records)[/bold yellow]\n")

    exclusivity_resolver = TargetExclusivityResolver()
    contradiction_checker = ContradictionChecker()

    t_start = time.time()

    for chunk_idx in range(num_chunks):
        c_start = chunk_idx * chunk_size
        c_end = min(total_s1, (chunk_idx + 1) * chunk_size)
        c_match_file = os.path.join(chunks_dir, f"matching_chunk_{chunk_idx:04d}.tsv")
        c_cand_file = os.path.join(chunks_dir, f"candidate_chunk_{chunk_idx:04d}.tsv")

        # RESUME CHECK: If chunk already computed, skip!
        if os.path.isfile(c_match_file) and os.path.isfile(c_cand_file):
            console.print(f"[{chunk_idx+1}/{num_chunks}] Chunk {chunk_idx:04d} already completed. [dim]SKIPPING (RESUMED)[/dim]")
            continue

        chunk_t0 = time.time()
        chunk_s1 = s1_test[c_start:c_end]
        s1_chunk_dict = {str(r["entity_id"]): r for r in chunk_s1}
        s1_chunk_pre = {sid: precompute_record_views(s1_chunk_dict[sid]) for sid in s1_chunk_dict}

        # Retrieval
        chunk_pairs: List[Tuple[str, str]] = []
        chunk_prov: Dict[Tuple[str, str], RetrievalProvenance] = {}
        cands_dict: Dict[str, Set[str]] = {}

        for s1 in chunk_s1:
            sid = str(s1["entity_id"])
            cands, p_dict = retriever.query_entity(s1, precomputed_s1=s1_chunk_pre[sid])
            cands_dict[sid] = set(cands)
            for tid in cands:
                chunk_pairs.append((sid, tid))
                chunk_prov[(sid, tid)] = p_dict[tid]

        if not chunk_pairs:
            # All singletons
            matches_dict = {str(s1["entity_id"]): set() for s1 in chunk_s1}
        else:
            # Build features & predict
            X_chunk = build_pairwise_feature_dataframe(
                chunk_pairs, s1_chunk_dict, target_dict, s1_chunk_pre, tgt_pre, chunk_prov
            )
            cons_chunk = compute_s2_s3_consensus(chunk_pairs, target_dict, tgt_pre)
            ctx_chunk = compute_context_and_competition_features(chunk_pairs, s1_chunk_pre, tgt_pre)
            X_chunk = pd.concat([X_chunk, cons_chunk, ctx_chunk], axis=1)

            scores = model.predict_proba(X_chunk)
            cal_scores = calibrator.predict(scores)

            chunk_s1_ids = list(s1_chunk_dict.keys())
            matches_dict = policy.apply(chunk_pairs, cal_scores, all_s1_ids=chunk_s1_ids)
            matches_dict = exclusivity_resolver.resolve(matches_dict, chunk_pairs, cal_scores)
            matches_dict = contradiction_checker.filter_predictions(matches_dict, s1_chunk_pre, tgt_pre)

        # Write chunk files
        with open(c_match_file, "w", encoding="utf-8") as f_m:
            for s1 in chunk_s1:
                sid = str(s1["entity_id"])
                m_str = ",".join(sorted(matches_dict.get(sid, set())))
                f_m.write(f"{sid}\t{m_str}\n")

        with open(c_cand_file, "w", encoding="utf-8") as f_c:
            for s1 in chunk_s1:
                sid = str(s1["entity_id"])
                c_str = ",".join(sorted(cands_dict.get(sid, set())))
                f_c.write(f"{sid}\t{c_str}\n")

        chunk_elapsed = time.time() - chunk_t0
        progress_pct = (c_end / total_s1) * 100.0
        elapsed_total = time.time() - t_start
        est_remaining = (elapsed_total / (chunk_idx + 1)) * (num_chunks - (chunk_idx + 1)) / 60.0
        console.print(
            f"[{chunk_idx+1}/{num_chunks}] Processed {c_end:,}/{total_s1:,} ({progress_pct:.1f}%) "
            f"in {chunk_elapsed:.1f}s | Est. Remaining: {est_remaining:.1f} min"
        )

    # --- Step 4: Concatenate all chunks into final files ---
    console.print("\n[bold green]>>> Step 4: Concatenating all chunks into final submission files...[/bold green]")
    final_matching = os.path.join(output_dir, "matching_results.tsv")
    final_candidates = os.path.join(output_dir, "candidate_pairs.tsv")

    chunk_match_files = sorted(glob.glob(os.path.join(chunks_dir, "matching_chunk_*.tsv")))
    with open(final_matching, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for cf in chunk_match_files:
            with open(cf, "r", encoding="utf-8") as f_in:
                shutil.copyfileobj(f_in, f_out)

    chunk_cand_files = sorted(glob.glob(os.path.join(chunks_dir, "candidate_chunk_*.tsv")))
    with open(final_candidates, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        for cf in chunk_cand_files:
            with open(cf, "r", encoding="utf-8") as f_in:
                shutil.copyfileobj(f_in, f_out)

    console.print(f"[bold green]Saved: {final_matching}[/bold green]")
    console.print(f"[bold green]Saved: {final_candidates}[/bold green]\n")

    # --- Step 5: Official Validation ---
    console.print("[bold cyan]>>> Step 5: Running Submission Validator...[/bold cyan]")
    report = validate_submission_package(
        matching_path=final_matching,
        candidate_path=final_candidates,
        test_dir=test_dir,
    )
    console.print(report.summary())
    if report.is_valid:
        console.print("\n[bold green]ALL VALIDATION CHECKS PASSED! Ready to submit![/bold green]")
    else:
        console.print(f"\n[bold red]Validation issues detected: {report.errors}[/bold red]")


if __name__ == "__main__":
    args = parse_args()

    # If running on Colab and dataset is on Drive, copy to local SSD for 10x speedup
    if os.path.exists(args.dataset_dir) and args.dataset_dir.startswith("/content/drive"):
        train_path, test_path = copy_dataset_to_fast_ssd(args.dataset_dir, args.local_scratch)
    else:
        train_path = os.path.join(args.dataset_dir, "train")
        test_path = os.path.join(args.dataset_dir, "test")

    checkpoint_path = os.path.join(args.output_dir, "checkpoints")

    model, calib, policy = get_trained_model(
        train_dir=train_path,
        checkpoint_dir=checkpoint_path,
        n_train_s1=args.n_train_s1,
        use_gpu=args.use_gpu,
    )

    run_resumable_inference(
        test_dir=test_path,
        output_dir=args.output_dir,
        model=model,
        calibrator=calib,
        policy=policy,
        chunk_size=args.chunk_size,
    )
