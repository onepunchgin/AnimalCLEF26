#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from utils import log, save_numpy, seed_everything, timed
import config


def cosine_sim_matrix(emb):
    """Compute pairwise cosine sim. Assumes L2-normalized embeddings."""
    sim = np.clip(emb @ emb.T, -1.0, 1.0)
    np.fill_diagonal(sim, 0.0)
    return sim.astype(np.float32)


def build_features_from_sims(sim_dict: dict, n: int) -> tuple[np.ndarray, list]:
    """
    Convert {sim_name: (n,n) matrix} into a (n*n - n)/2 × num_features array
    of upper-triangular pair features.
    """
    feature_names = []
    sim_arrays = []

    # Direct sim values
    for name, sim in sim_dict.items():
        # Upper triangular (i<j)
        idx_i, idx_j = np.triu_indices(n, k=1)
        sim_vec = sim[idx_i, idx_j]
        sim_arrays.append(sim_vec)
        feature_names.append(f"sim_{name}")

    # Aggregate features
    sim_stack = np.stack(sim_arrays, axis=1)  # (n_pairs, n_sims)
    n_sims = sim_stack.shape[1]

    extra_features = []
    extra_features.append(sim_stack.min(axis=1));   feature_names.append("sim_min")
    extra_features.append(sim_stack.max(axis=1));   feature_names.append("sim_max")
    extra_features.append(sim_stack.mean(axis=1));  feature_names.append("sim_mean")
    extra_features.append(sim_stack.std(axis=1));   feature_names.append("sim_std")
    extra_features.append(sim_stack.max(axis=1) - sim_stack.min(axis=1))
    feature_names.append("sim_range")

    # Rank features: for each pair, what's its rank in column j (top-1 match etc.)
    # Compute per-row max sim across full matrix (not just upper tri) for rank context
    # Use first sim matrix as reference for rank
    ref_sim = list(sim_dict.values())[0]
    # For pair (i, j): rank of sim[i,j] in row i (0 = best)
    rank_matrix = np.zeros_like(ref_sim, dtype=np.int32)
    for i in range(n):
        order = np.argsort(-ref_sim[i])  # descending
        rank_matrix[i, order] = np.arange(n)
    rank_i_in_j = rank_matrix[idx_i, idx_j].astype(np.float32) / n
    rank_j_in_i = rank_matrix[idx_j, idx_i].astype(np.float32) / n
    extra_features.append(rank_i_in_j); feature_names.append("rank_i_in_j_norm")
    extra_features.append(rank_j_in_i); feature_names.append("rank_j_in_i_norm")
    extra_features.append(np.minimum(rank_i_in_j, rank_j_in_i))
    feature_names.append("rank_min_norm")

    full_feats = np.column_stack([sim_stack] + [
        np.asarray(f).reshape(-1, 1) for f in extra_features
    ])
    return full_feats.astype(np.float32), feature_names


def build_pair_labels(df: pd.DataFrame, n: int) -> np.ndarray:
    """Same-identity labels for upper-triangular pairs."""
    labels = df["_id"].astype("category").cat.codes.values
    idx_i, idx_j = np.triu_indices(n, k=1)
    return (labels[idx_i] == labels[idx_j]).astype(np.int32)


def collect_sim_dict_for_split(split: str, df: pd.DataFrame) -> dict:
    """
    Load all available embeddings for a split (val/test) and compute
    cosine similarity matrices.
    """
    n = len(df)
    sims = {}

    # Try v4 cached embeddings
    candidates = {
        "megad":     config.V4_FEATURES / f"megadesc_{split}_turtles.npy",
        "miew":      config.V4_FEATURES / f"miew_{split}_turtles.npy",
        "dinov2":    config.V4_FEATURES / f"dinov2_{split}_turtles.npy",
        "arcface_v2": config.V4_FEATURES / f"arcface_v2_{split}_turtles.npy",
    }
    for name, path in candidates.items():
        if path.exists():
            emb = np.load(path)
            if emb.shape[0] == n:
                sims[name] = cosine_sim_matrix(emb)
                log.info(f"  loaded {name}: {emb.shape}")
            else:
                log.warning(f"  {name}: shape mismatch {emb.shape[0]} vs {n}")

    # v5 stage 1 masked
    masked_candidates = {
        "megad_masked": config.V5_FEATURES_DIR / f"megadesc_masked_{split}_turtles.npy",
        "miew_masked":  config.V5_FEATURES_DIR / f"miew_masked_{split}_turtles.npy",
    }
    for name, path in masked_candidates.items():
        if path.exists():
            emb = np.load(path)
            if emb.shape[0] == n:
                sims[name] = cosine_sim_matrix(emb)
                log.info(f"  loaded {name}: {emb.shape}")

    # v5 stage 2 ArcFace
    arcface_path = config.V5_FEATURES_DIR / f"stage2c_arcface_{split}_turtles.npy"
    if arcface_path.exists():
        emb = np.load(arcface_path)
        if emb.shape[0] == n:
            sims["stage2_arcface"] = cosine_sim_matrix(emb)
            log.info(f"  loaded stage2_arcface: {emb.shape}")

    # v4 step 8 WildFusion (test only)
    if split == "test":
        wf_path = config.V4_SIMILARITY / "step8_test_vs_test_fused.npy"
        if wf_path.exists():
            wf_sim = np.load(wf_path)
            if wf_sim.shape == (n, n):
                np.fill_diagonal(wf_sim, 0.0)
                sims["wildfusion"] = np.clip(wf_sim, 0.0, 1.0).astype(np.float32)
                log.info(f"  loaded wildfusion: {wf_sim.shape}")

    return sims


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    seed_everything(config.RANDOM_SEED)

    val_df = pd.read_csv(config.V4_TURTLE_VAL_CSV)
    test_df = pd.read_csv(config.V4_TURTLE_TEST_CSV)

    # Val
    log.info("=" * 60)
    log.info(f"Val pair features: {len(val_df)} images = "
             f"{len(val_df)*(len(val_df)-1)//2} pairs")
    log.info("=" * 60)
    sims_val = collect_sim_dict_for_split("val", val_df)
    if not sims_val:
        log.error("no val embeddings found")
        return 1

    with timed("build val features"):
        val_feats, feature_names = build_features_from_sims(sims_val, len(val_df))
        val_labels = build_pair_labels(val_df, len(val_df))

    log.info(f"  val features: {val_feats.shape}, n_features={val_feats.shape[1]}")
    log.info(f"  positive pairs: {val_labels.sum()}/{len(val_labels)} "
             f"({100*val_labels.mean():.2f}%)")

    save_numpy(val_feats, config.V5_PAIR_FEATURES_DIR / "val_pair_features.npy")
    save_numpy(val_labels, config.V5_PAIR_FEATURES_DIR / "val_pair_labels.npy")

    # Test
    log.info("")
    log.info("=" * 60)
    log.info(f"Test pair features: {len(test_df)} images = "
             f"{len(test_df)*(len(test_df)-1)//2} pairs")
    log.info("=" * 60)
    sims_test = collect_sim_dict_for_split("test", test_df)
    if not sims_test:
        log.error("no test embeddings found")
        return 1

    # Use only sim sources available for both val and test
    common_sources = set(sims_val.keys()) & set(sims_test.keys())
    log.info(f"Common sim sources: {sorted(common_sources)}")

    sims_val_common = {k: sims_val[k] for k in sorted(common_sources)}
    sims_test_common = {k: sims_test[k] for k in sorted(common_sources)}

    with timed("rebuild val with common sources"):
        val_feats, feature_names = build_features_from_sims(
            sims_val_common, len(val_df))
        save_numpy(val_feats,
                    config.V5_PAIR_FEATURES_DIR / "val_pair_features.npy")

    with timed("build test features"):
        test_feats, _ = build_features_from_sims(sims_test_common, len(test_df))

    log.info(f"  test features: {test_feats.shape}")
    save_numpy(test_feats, config.V5_PAIR_FEATURES_DIR / "test_pair_features.npy")

    # Save feature names
    with open(config.V5_PAIR_FEATURES_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    log.info(f"feature names saved")

    log.info("")
    log.info(f"Pair features built: {len(feature_names)} features each")
    log.info(f"  val: {val_feats.shape}, labels: {val_labels.sum()} pos / "
             f"{len(val_labels)} total")
    log.info(f"  test: {test_feats.shape}")
    log.info("Next: stage3b_train_xgboost.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
