"""Official ML Challenge 2026 Submission Validator."""

from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple


DELIM = "\t"
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


@dataclass
class ERValidationReport:
    """Diagnostic report for entity resolution submission files."""

    is_valid: bool = True
    matching_rows: int = 0
    candidate_rows: int = 0
    expected_s1_count: int = 0
    empty_matches_count: int = 0
    empty_candidates_count: int = 0
    matches_not_in_candidates_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"=== ML Challenge 2026 Submission Validation Report [{status}] ===",
            f"Matching Results Rows: {self.matching_rows} (Expected: {self.expected_s1_count})",
            f"Candidate Pairs Rows:  {self.candidate_rows}",
            f"Singleton (Empty) Matches: {self.empty_matches_count}",
        ]
        if self.matches_not_in_candidates_count > 0:
            lines.append(f"Matches Not In Candidate Pairs: {self.matches_not_in_candidates_count}")
        if self.errors:
            lines.append("Errors:")
            for err in self.errors:
                lines.append(f"  - [FAIL] {err}")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - [WARN] {w}")
        return "\n".join(lines)


def read_s1_ids_from_source(source1_path: str) -> Set[str]:
    """Read first column entity IDs from test_source1.tsv."""
    with open(source1_path, encoding="utf-8") as f:
        next(f, None)  # skip header
        return {line.split(DELIM, 1)[0].strip() for line in f if line.strip()}


def validate_tsv_file(
    filepath: str,
    expected_header: List[str],
    required_s1_ids: Set[str],
    col_name: str,
    errors: List[str],
) -> Optional[Dict[str, Set[str]]]:
    """Validate a results TSV file against competition formatting rules."""
    path = Path(filepath)
    if not path.is_file():
        errors.append(f"File not found: {filepath}")
        return None

    mapping: Dict[str, Set[str]] = {}
    seen = set()
    dup_rows = set()
    intra_dupes = set()
    self_matches = set()
    wrong_prefix = set()

    with open(filepath, encoding="utf-8") as f:
        header = f.readline()
        if not header:
            errors.append(f"{path.name} is empty.")
            return None

        if DELIM not in header and "," in header:
            errors.append(
                f"{path.name}: header has no TAB but contains commas. "
                "Must be TAB-separated (.tsv) using sep='\\t'."
            )
            return None

        cols = [c.strip().lower() for c in header.rstrip("\n").split(DELIM)]
        if cols != expected_header:
            errors.append(
                f"{path.name}: unexpected header {cols}. Expected {expected_header}."
            )
            return None

        for line_num, line in enumerate(f, start=2):
            s1, tab, rest = line.partition(DELIM)
            if not tab:
                if s1.strip():
                    errors.append(f"{path.name}: malformed row at line {line_num}: {line.rstrip()!r}")
                continue

            s1 = s1.strip()
            if s1 in seen:
                dup_rows.add(s1)
            seen.add(s1)

            raw_ids = rest.rstrip("\n").split(",") if rest.strip() else []
            clean_ids = [i.strip() for i in raw_ids if i.strip()]

            if len(clean_ids) != len(set(clean_ids)):
                intra_dupes.add(s1)

            id_set = set(clean_ids)
            mapping[s1] = id_set

            for mid in id_set:
                if mid.startswith("S1-"):
                    self_matches.add(mid)
                elif not mid.startswith(("S2-", "S3-")):
                    wrong_prefix.add(mid)

    if dup_rows:
        errors.append(f"{path.name}: duplicate source1_entity_id row(s): {len(dup_rows)} found.")
    if intra_dupes:
        errors.append(f"{path.name}: duplicate IDs within a {col_name} list: {len(intra_dupes)} found.")
    if self_matches:
        errors.append(f"{path.name}: contains Source 1 IDs (self-matches): {len(self_matches)} found.")
    if wrong_prefix:
        errors.append(f"{path.name}: contains IDs without S2-/S3- prefix: {len(wrong_prefix)} found.")

    missing_s1 = required_s1_ids - seen
    if missing_s1:
        errors.append(f"{path.name}: required test S1 entities missing: {len(missing_s1)} entities missing.")

    extra_s1 = seen - required_s1_ids
    if extra_s1:
        errors.append(f"{path.name}: contains S1 IDs not in test set: {len(extra_s1)} unexpected entities.")

    return mapping


def validate_submission_package(
    matching_path: str,
    candidate_path: Optional[str] = None,
    test_dir: str = "resources/student_resource/dataset/test",
    run_official_script: bool = True,
) -> ERValidationReport:
    """Run comprehensive validation on both matching_results.tsv and candidate_pairs.tsv."""
    report = ERValidationReport()
    source1_file = os.path.join(test_dir, "test_source1.tsv")

    if not os.path.isfile(source1_file):
        report.is_valid = False
        report.errors.append(f"Test source1 file not found at {source1_file}")
        return report

    required_s1 = read_s1_ids_from_source(source1_file)
    report.expected_s1_count = len(required_s1)

    # 1. Validate matching_results.tsv
    matched_mapping = validate_tsv_file(
        matching_path,
        MATCHING_HEADER,
        required_s1,
        "matched_entity_ids",
        report.errors,
    )
    if matched_mapping is not None:
        report.matching_rows = len(matched_mapping)
        report.empty_matches_count = sum(1 for m in matched_mapping.values() if len(m) == 0)

    # 2. Validate candidate_pairs.tsv (if provided)
    candidate_mapping = None
    if candidate_path and os.path.isfile(candidate_path):
        candidate_mapping = validate_tsv_file(
            candidate_path,
            CANDIDATE_HEADER,
            required_s1,
            "candidate_entity_ids",
            report.errors,
        )
        if candidate_mapping is not None:
            report.candidate_rows = len(candidate_mapping)
            report.empty_candidates_count = sum(1 for c in candidate_mapping.values() if len(c) == 0)

            # Check: Every matched ID must be in candidates
            if matched_mapping is not None:
                mismatches = 0
                for s1_id, m_set in matched_mapping.items():
                    c_set = candidate_mapping.get(s1_id, set())
                    extra = m_set - c_set
                    if extra:
                        mismatches += 1
                report.matches_not_in_candidates_count = mismatches
                if mismatches > 0:
                    report.warnings.append(
                        f"{mismatches} S1 entities have matched IDs that were not present in candidate_pairs.tsv."
                    )
    elif candidate_path:
        report.warnings.append(f"candidate_pairs.tsv not found at {candidate_path}")

    # 3. Optional: Run official validator script
    if run_official_script:
        official_validator = "resources/student_resource/utils/validate_submission.py"
        if os.path.isfile(official_validator):
            cmd = [
                sys.executable,
                official_validator,
                "--matching",
                matching_path,
                "--test-dir",
                test_dir,
            ]
            if candidate_path and os.path.isfile(candidate_path):
                cmd.extend(["--candidate", candidate_path])

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if proc.returncode != 0:
                    report.errors.append(f"Official validator failed:\n{proc.stdout}\n{proc.stderr}")
            except Exception as e:
                report.warnings.append(f"Could not execute official validator script: {e}")

    report.is_valid = len(report.errors) == 0
    return report
