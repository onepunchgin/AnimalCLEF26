#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from utils import (log, save_numpy, seed_everything, timed,
                     write_submission, synthesize_nonscored,
                     cluster_agg)
import config


def predict_pair_probs(booster, X):
    import xgboost as xgb
    return booster.predict(xgb.DMatrix(X))


def build_test_sim_from_probs(probs: np.ndarray, n: int) -> np.ndarray:
    """Convert per-pair probs to (n,n) symmetric matrix."""
    sim = np.zeros((n, n), dtype=np.float32)
    idx_i, idx_j = np.triu_indices(n, k=1)
    sim[idx_i, idx_j] = probs
    sim[idx_j, idx_i] = probs
    return sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="stage3b_xgboost.json")
    parser.add_argument("--out-name", default="stage3c_xgboost_cluster.csv")
    args = parser.parse_args()

    seed_everything(config.RANDOM_SEED)

    import xgboost as xgb
    model_path = config.V5_MODELS_DIR / args.model
    if not model_path.exists():
        log.error(f"missing model {model_path}; run stage3b first")
        return 1
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    log.info(f"loaded model: {model_path}")

    test_feat_path = config.V5_PAIR_FEATURES_DIR / "test_pair_features.npy"
    if not test_feat_path.exists():
        log.error(f"missing {test_feat_path}; run stage3a first")
        return 1
    X_test = np.load(test_feat_path)
    log.info(f"test pair features: {X_test.shape}")

    test_df = pd.read_csv(config.V4_TURTLE_TEST_CSV)
    n_test = len(test_df)

    with timed("XGBoost predict on test pairs"):
        test_probs = predict_pair_probs(booster, X_test)
    log.info(f"test probs: shape={test_probs.shape}, "
             f"min={test_probs.min():.4f}, max={test_probs.max():.4f}, "
             f"mean={test_probs.mean():.4f}")
    log.info(f"high-prob pairs (>0.5): {(test_probs > 0.5).sum()}")
    log.info(f"high-prob pairs (>0.8): {(test_probs > 0.8).sum()}")

    # Build sim matrix
    test_sim = build_test_sim_from_probs(test_probs, n_test)
    save_numpy(test_sim,
                config.V5_SIMILARITY_DIR / "stage3c_xgboost_sim.npy")

    # Tune threshold on val (probs from val-pair features)
    log.info("")
    log.info("Tuning threshold on val pair probabilities...")
    val_feat_path = config.V5_PAIR_FEATURES_DIR / "val_pair_features.npy"
    val_label_path = config.V5_PAIR_FEATURES_DIR / "val_pair_labels.npy"
    X_val = np.load(val_feat_path)
    y_val = np.load(val_label_path)

    val_probs = predict_pair_probs(booster, X_val)
    log.info(f"val probs vs labels: high-prob accuracy = "
             f"{((val_probs > 0.5) == y_val).mean():.4f}")

    # Build val sim and cluster
    val_df = pd.read_csv(config.V4_TURTLE_VAL_CSV)
    n_val = len(val_df)
    val_sim = build_test_sim_from_probs(val_probs, n_val)

    # Use full val (not subsampled) for tuning since pair-level signal is fine
    # But sweep thresholds
    from sklearn.metrics import adjusted_rand_score
    val_labels = val_df["_id"].astype("category").cat.codes.values

    best_t = None
    best_ari = -1
    log.info("Threshold sweep on full val:")
    for t in np.arange(0.05, 0.95, 0.05):
        cids = cluster_agg(val_sim, t)
        ari = adjusted_rand_score(val_labels, cids)
        n_c = len(np.unique(cids))
        marker = " ← best" if ari > best_ari else ""
        if ari > best_ari:
            best_t = float(t)
            best_ari = float(ari)
        log.info(f"  t={t:.2f}  n_clusters={n_c:4d}  ARI={ari:.4f}{marker}")

    log.info(f"BEST: t={best_t}, val_ari={best_ari:.4f}")

    # Apply to test
    cluster_ids = cluster_agg(test_sim, best_t)
    n_clusters = len(np.unique(cluster_ids))
    log.info(f"test: {n_test} → {n_clusters} clusters")

    # Submit
    lizard_df = pd.read_csv(config.V4_LIZARD_TEST_CSV)
    sample = pd.read_csv(config.COMPETITION_SAMPLE_SUB)
    turtle_sub = pd.DataFrame({
        "image_id": test_df["image_id"].astype(str).tolist(),
        "cluster": [f"cluster_SeaTurtleID2022_{c:04d}" for c in cluster_ids],
    })
    lizard_sub = pd.DataFrame({
        "image_id": lizard_df["image_id"].astype(str).tolist(),
        "cluster": [f"cluster_TexasHornedLizards_{i:04d}" for i in range(len(lizard_df))],
    })
    nonscored = synthesize_nonscored(sample)
    submission = pd.concat([turtle_sub, lizard_sub, nonscored], ignore_index=True)
    submission = submission.drop_duplicates(subset="image_id", keep="first")

    out_path = config.V5_SUBMISSIONS_DIR / args.out_name
    write_submission(submission, out_path, sample_submission=sample)
    log.info(f"Submission: {out_path}")
    log.info(f"Val ARI: {best_ari:.4f}")
    log.info("Decision rule:")
    log.info("  Kaggle >= 0.25 → XGBoost helps, proceed to stage 4 ensemble")
    log.info("  Kaggle < 0.25 → Stage 4 may not help much; consider stopping")
    return 0


if __name__ == "__main__":
    sys.exit(main())
