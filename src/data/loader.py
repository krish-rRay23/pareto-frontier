"""High-speed streaming data loaders for Entity Resolution TSV files."""

import csv
import os
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Tuple


DELIM = "\t"


def load_source_tsv(
    filepath: str,
    max_rows: Optional[int] = None,
    filter_ids: Optional[Set[str]] = None,
) -> List[Dict[str, str]]:
    """Streamingly load records from a source TSV file (S1, S2, or S3).

    Returns a list of dicts with keys: 'entity_id', 'business_name', 'business_address', 'country'.
    Uses pandas C-engine for 10x-20x speedup over standard csv reader when available.
    """
    try:
        import pandas as pd
        df = pd.read_csv(
            filepath,
            sep=DELIM,
            nrows=max_rows,
            dtype=str,
            keep_default_na=False,
            engine="c",
            on_bad_lines="skip",
        )
        if filter_ids is not None:
            df = df[df["entity_id"].isin(filter_ids)]
        # Ensure standard columns exist
        for col in ["business_name", "business_address", "country"]:
            if col not in df.columns:
                df[col] = ""
        return df[["entity_id", "business_name", "business_address", "country"]].to_dict(orient="records")
    except Exception:
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=DELIM)
            for i, row in enumerate(reader):
                if max_rows is not None and i >= max_rows:
                    break
                eid = row["entity_id"]
                if filter_ids is not None and eid not in filter_ids:
                    continue
                records.append({
                    "entity_id": eid,
                    "business_name": row.get("business_name", ""),
                    "business_address": row.get("business_address", ""),
                    "country": row.get("country", ""),
                })
        return records


def load_ground_truth(
    filepath: str,
    filter_s1_ids: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """Load ground truth mapping: source1_entity_id -> set of matched_entity_ids."""
    gt_map: Dict[str, Set[str]] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=DELIM)
        for row in reader:
            s1_id = row["source1_entity_id"]
            if filter_s1_ids is not None and s1_id not in filter_s1_ids:
                continue
            raw = row.get("matched_entity_ids", "").strip()
            if raw:
                mids = {x.strip() for x in raw.split(",") if x.strip()}
            else:
                mids = set()
            gt_map[s1_id] = mids
            if filter_s1_ids is not None and len(gt_map) >= len(filter_s1_ids):
                break
    return gt_map


def stream_tsv_chunks(
    filepath: str, chunk_size: int = 100000
) -> Generator[List[Dict[str, str]], None, None]:
    """Stream a large TSV file in manageable memory chunks."""
    chunk = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=DELIM)
        for row in reader:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def load_benchmark_subset(
    data_dir: str = "resources/student_resource/dataset/train",
    n_s1: int = 2000,
    background_noise_ratio: int = 30,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, Set[str]]]:
    """Load a representative stratified benchmark slice for fast local model development.

    Includes:
    - n_s1 Source 1 reference records
    - Exactly all their true ground truth target records from S2 and S3
    - Realistic background non-matching target records to test blocker reduction & precision.
    """
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train_source3.tsv")

    s1_records = load_source_tsv(s1_path, max_rows=n_s1)
    s1_ids = {r["entity_id"] for r in s1_records}

    ground_truth = load_ground_truth(gt_path, filter_s1_ids=s1_ids)

    # Collect all true target IDs
    true_target_ids = set()
    for mids in ground_truth.values():
        true_target_ids.update(mids)

    # Load targets: all true targets + background noise
    target_records = []
    max_bg = n_s1 * background_noise_ratio

    found_targets = set()
    for t_path in [s2_path, s3_path]:
        bg_count = 0
        with open(t_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=DELIM)
            for row in reader:
                eid = row["entity_id"]
                if eid in true_target_ids:
                    target_records.append(row)
                    found_targets.add(eid)
                elif bg_count < max_bg // 2:
                    target_records.append(row)
                    bg_count += 1
                if len(found_targets) >= len(true_target_ids) and bg_count >= max_bg // 2:
                    break

    return s1_records, target_records, ground_truth
