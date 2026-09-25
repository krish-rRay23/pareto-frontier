"""S2-S3 cross-source consensus features for entity resolution."""

from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

from src.features.pairwise_features import jaccard_similarity


def compute_s2_s3_consensus(
    pairs: List[Tuple[str, str]],
    target_dict: Dict[str, Dict[str, str]],
    precomputed_targets: Dict[str, object],
) -> pd.DataFrame:
    """Compute consensus features between S2 and S3 candidate peers for the same S1 entity.

    Since 85% of true matches co-occur across both S2 and S3, candidates that share
    mutual high similarity with a peer candidate from the opposite source have much
    higher true match likelihood.
    """
    # Group target IDs by S1 ID and partition into S2 and S3 sets
    s1_to_s2: Dict[str, List[str]] = {}
    s1_to_s3: Dict[str, List[str]] = {}

    for sid, tid in pairs:
        if tid.startswith("S2-"):
            if sid not in s1_to_s2:
                s1_to_s2[sid] = []
            s1_to_s2[sid].append(tid)
        elif tid.startswith("S3-"):
            if sid not in s1_to_s3:
                s1_to_s3[sid] = []
            s1_to_s3[sid].append(tid)

    has_cross_peer = []
    peer_max_name_jaccard = []
    peer_max_num_match = []

    for sid, tid in pairs:
        is_s2 = tid.startswith("S2-")
        peer_tids = s1_to_s3.get(sid, []) if is_s2 else s1_to_s2.get(sid, [])

        if not peer_tids or tid not in precomputed_targets:
            has_cross_peer.append(0.0)
            peer_max_name_jaccard.append(0.0)
            peer_max_num_match.append(0.0)
            continue

        has_cross_peer.append(1.0)
        curr_pv = precomputed_targets[tid]
        curr_name_toks = curr_pv["name_tokens"]
        curr_prim_num = curr_pv["addr"]["primary_number"]

        max_jacc = 0.0
        num_match = 0.0

        for ptid in peer_tids:
            if ptid not in precomputed_targets:
                continue
            peer_pv = precomputed_targets[ptid]
            jacc = jaccard_similarity(curr_name_toks, peer_pv["name_tokens"])
            if jacc > max_jacc:
                max_jacc = jacc

            peer_num = peer_pv["addr"]["primary_number"]
            if curr_prim_num and peer_num and curr_prim_num == peer_num:
                num_match = 1.0

        peer_max_name_jaccard.append(max_jacc)
        peer_max_num_match.append(num_match)

    consensus_df = pd.DataFrame(
        {
            "has_cross_source_peer": has_cross_peer,
            "peer_max_name_jaccard": peer_max_name_jaccard,
            "peer_max_num_match": peer_max_num_match,
            "peer_consensus_score": np.array(peer_max_name_jaccard) * np.array(peer_max_num_match),
        }
    )
    return consensus_df
