#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from utils import (log, save_numpy, seed_everything, timed,
                     write_submission, synthesize_nonscored,
                     cluster_agg, build_test_matched_val_indices)
import config


def fuse_sims(sims, weights):
    total = sum(weights)
    fused = sum((w/total) * s for s, w in zip(sims, weights))
    np.fill_diagonal(fused, 0.0)
    return np.clip(fused, 0.0, 1.0)


def tune_threshold(val_emb, val_df, n_splits=5):
    threshold_grid = np.arange(0.20, 0.80, 0.025).tolist()
    results = []
    for split_i in range(n_splits):
        keep = build_test_matched_val_indices(val_df, seed=42 + split_i)
        sub_emb = val_emb[keep]
        sub_labels = val_df.iloc[keep]["_id"].astype("category").cat.codes.values
        sim = np.clip(sub_emb @ sub_emb.T, 0.0, 1.0)
        np.fill_diagonal(sim, 0.0)
        for t in threshold_grid:
            cids = cluster_agg(sim, t)
            ari = adjusted_rand_score(sub_labels, cids)
            results.append({"split": split_i, "t": round(t, 3), "ari": ari})
    res = pd.DataFrame(results)
    agg = res.groupby("t").agg(
        mean_ari=("ari", "mean"), std_ari=("ari", "std"),
    ).reset_index().sort_values("mean_ari", ascending=False)
    log.info("\n" + agg.head(10).to_string(index=False))
    best = agg.iloc[0]
    return float(best["t"]), float(best["mean_ari"])


def tune_fusion(val_a, val_b, val_df, n_splits=3):
    """Tune (w_b, threshold) for fusion of two embedding sets."""
    weight_grid = [0.25, 0.4, 0.5, 0.6, 0.75]
    threshold_grid = np.arange(0.30, 0.80, 0.025).tolist()
    results = []
    for split_i in range(n_splits):
        keep = build_test_matched_val_indices(val_df, seed=42 + split_i)
        sub_a = val_a[keep]
        sub_b = val_b[keep]
        sub_labels = val_df.iloc[keep]["_id"].astype("category").cat.codes.values
        sim_a = sub_a @ sub_a.T
        sim_b = sub_b @ sub_b.T
        for w_b in weight_grid:
            fused = fuse_sims([sim_a, sim_b], [1 - w_b, w_b])
            for t in threshold_grid:
                cids = cluster_agg(fused, t)
                ari = adjusted_rand_score(sub_labels, cids)
                results.append({"split": split_i, "w_b": w_b,
                                  "t": round(t, 3), "ari": ari})
    res = pd.DataFrame(results)
    agg = res.groupby(["w_b", "t"]).agg(
        mean_ari=("ari", "mean"), std_ari=("ari", "std"),
    ).reset_index().sort_values("mean_ari", ascending=False)
    log.info("\n" + agg.head(10).to_string(index=False))
    best = agg.iloc[0]
    return float(best["w_b"]), float(best["t"]), float(best["mean_ari"])


def build_and_submit(turtle_cluster_ids, test_df, lizard_df, sample, out_name):
    turtle_sub = pd.DataFrame({
        "image_id": test_df["image_id"].astype(str).tolist(),
        "cluster": [f"cluster_SeaTurtleID2022_{c:04d}" for c in turtle_cluster_ids],
    })
    lizard_sub = pd.DataFrame({
        "image_id": lizard_df["image_id"].astype(str).tolist(),
        "cluster": [f"cluster_TexasHornedLizards_{i:04d}" for i in range(len(lizard_df))],
    })
    nonscored = synthesize_nonscored(sample)
    sub = pd.concat([turtle_sub, lizard_sub, nonscored], ignore_index=True)
    sub = sub.drop_duplicates(subset="image_id", keep="first")
    out_path = config.V5_SUBMISSIONS_DIR / out_name
    write_submission(sub, out_path, sample_submission=sample)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    seed_everything(config.RANDOM_SEED)

    val_df = pd.read_csv(config.V4_TURTLE_VAL_CSV)
    test_df = pd.read_csv(config.V4_TURTLE_TEST_CSV)
    lizard_df = pd.read_csv(config.V4_LIZARD_TEST_CSV)
    sample = pd.read_csv(config.COMPETITION_SAMPLE_SUB)

    # Load ArcFace embeddings
    val_arcface = np.load(config.V5_FEATURES_DIR / "stage2c_arcface_val_turtles.npy")
    test_arcface = np.load(config.V5_FEATURES_DIR / "stage2c_arcface_test_turtles.npy")
    log.info(f"ArcFace val {val_arcface.shape}, test {test_arcface.shape}")

    # Strategy 1: ArcFace alone
    log.info("")
    log.info("=" * 60)
    log.info("Strategy 1 — ArcFace alone")
    log.info("=" * 60)
    best_t, val_ari = tune_threshold(val_arcface, val_df)
    log.info(f"BEST threshold={best_t}, val_ari={val_ari:.4f}")

    sim = np.clip(test_arcface @ test_arcface.T, 0.0, 1.0)
    np.fill_diagonal(sim, 0.0)
    save_numpy(sim.astype(np.float32),
                config.V5_SIMILARITY_DIR / "stage2d_arcface_only_sim.npy")
    cluster_ids = cluster_agg(sim, best_t)
    log.info(f"test: {len(test_df)} → {len(np.unique(cluster_ids))} clusters")
    out_path = build_and_submit(cluster_ids, test_df, lizard_df, sample,
                                  "stage2d_arcface_only.csv")

    # Strategy 2: ArcFace + MIEW fusion
    miew_val_path = config.V4_FEATURES / "miew_val_turtles.npy"
    miew_test_path = config.V4_FEATURES / "miew_test_turtles.npy"
    if miew_val_path.exists() and miew_test_path.exists():
        log.info("")
        log.info("=" * 60)
        log.info("Strategy 2 — ArcFace + MIEW fusion")
        log.info("=" * 60)
        val_miew = np.load(miew_val_path)
        test_miew = np.load(miew_test_path)
        log.info(f"MIEW val {val_miew.shape}, test {test_miew.shape}")

        best_w, best_t2, val_ari2 = tune_fusion(val_arcface, val_miew, val_df)
        log.info(f"BEST w_miew={best_w}, t={best_t2}, val_ari={val_ari2:.4f}")

        sim_a = test_arcface @ test_arcface.T
        sim_b = test_miew @ test_miew.T
        fused = fuse_sims([sim_a, sim_b], [1 - best_w, best_w])
        save_numpy(fused.astype(np.float32),
                    config.V5_SIMILARITY_DIR / "stage2d_arcface_miew_sim.npy")
        cids2 = cluster_agg(fused, best_t2)
        log.info(f"test: {len(test_df)} → {len(np.unique(cids2))} clusters")
        out_path2 = build_and_submit(cids2, test_df, lizard_df, sample,
                                       "stage2d_arcface_miew_fusion.csv")
    else:
        log.warning("v4 MIEW embeddings missing; skipping fusion strategy")

    log.info("")
    log.info("Stage 2d done. Submit ONE of:")
    log.info(f"  stage2d_arcface_only.csv (val ARI {val_ari:.4f})")
    if miew_val_path.exists():
        log.info(f"  stage2d_arcface_miew_fusion.csv (val ARI {val_ari2:.4f})")
    log.info("Decision rule:")
    log.info("  Best Kaggle >= 0.22 → ArcFace works, proceed to stage 3")
    log.info("  Kaggle 0.18-0.22 → marginal, your call")
    log.info("  Kaggle < 0.18 → ArcFace not adding value; stage 3 may not help either")
    return 0


if __name__ == "__main__":
    sys.exit(main())
