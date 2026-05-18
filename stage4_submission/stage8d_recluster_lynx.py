#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch

import config
from utils import cluster_agg, log, save_numpy, seed_everything, timed

from stage2b_train_arcface import MegaDescArcFace, build_eval_transforms
from stage7b_extract_and_cluster import (
    calibrate_match_threshold,
    db_match,
    extract_embeddings,
    l2norm,
    load_species_data,
)


SPECIES = "LynxID2025"


# =====================================================================
# Pick best checkpoint
# =====================================================================

def pick_best_checkpoint() -> tuple[Path, dict]:
    stage8c_path = config.V5_CKPTS_DIR / "stage8c_lynx_extended_best.pth"
    stage7a_path = (
        config.V5_CKPTS_DIR / f"stage7a_{SPECIES.lower()}_arcface_best.pth"
    )

    if not stage7a_path.exists():
        log.error(f"missing baseline checkpoint: {stage7a_path}")
        raise SystemExit(1)

    ckpt7 = torch.load(stage7a_path, map_location="cpu", weights_only=False)
    val7 = float(ckpt7.get("val_ari", 0.0))

    if stage8c_path.exists():
        ckpt8 = torch.load(stage8c_path, map_location="cpu", weights_only=False)
        val8 = float(ckpt8.get("val_ari", 0.0))
        if val8 > val7:
            log.info(f"Using stage8c (val_ari={val8:.4f} > stage7a {val7:.4f})")
            return stage8c_path, {
                "chosen": "stage8c",
                "stage8c_val_ari": val8,
                "stage7a_val_ari": val7,
            }
        log.info(f"Keeping stage7a (val_ari={val7:.4f} >= stage8c {val8:.4f})")
        return stage7a_path, {
            "chosen": "stage7a",
            "stage8c_val_ari": val8,
            "stage7a_val_ari": val7,
        }

    log.info(f"stage8c not found; using stage7a (val_ari={val7:.4f})")
    return stage7a_path, {
        "chosen": "stage7a",
        "stage8c_val_ari": None,
        "stage7a_val_ari": val7,
    }


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=10,
                        help="K for K-NN weighted-vote DB matching")
    parser.add_argument("--cluster-threshold", type=float, default=None,
                        help="Override agglomerative threshold for unmatched. "
                             "If unset, uses the chosen ckpt's val_threshold.")
    args = parser.parse_args()

    seed_everything(config.RANDOM_SEED)
    device = config.DEVICE

    log.info("=" * 70)
    log.info("Stage 8d -- Re-cluster lynx test with best checkpoint")
    log.info("=" * 70)
    log.info(f"device={device}  K={args.K}")

    # ------------------------------------------------------------------
    # 1) Pick best ckpt
    # ------------------------------------------------------------------
    ckpt_path, ckpt_info = pick_best_checkpoint()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = int(ckpt["num_classes"])
    use_subcenter = bool(ckpt.get("use_subcenter", True))
    val_ari_at_save = ckpt.get("val_ari", "unknown")
    val_t_at_save = ckpt.get("val_threshold", None)
    log.info(f"  checkpoint: {ckpt_path.name}")
    log.info(f"  num_classes={num_classes}  subcenter={use_subcenter}")
    log.info(f"  val_ari@save={val_ari_at_save}  val_t@save={val_t_at_save}")

    model = MegaDescArcFace(
        num_classes=num_classes, use_subcenter=use_subcenter,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    transform = build_eval_transforms()

    # ------------------------------------------------------------------
    # 2) Load competition train (DB) + test for LynxID2025
    # ------------------------------------------------------------------
    train_df, test_df = load_species_data(SPECIES)
    for p in train_df["_img"].head(5).tolist():
        if not Path(p).exists():
            log.error(f"missing train image (sanity): {p}")
            return 1
    for p in test_df["_img"].head(5).tolist():
        if not Path(p).exists():
            log.error(f"missing test image (sanity): {p}")
            return 1

    db_paths = train_df["_img"].astype(str).tolist()
    test_paths = test_df["_img"].astype(str).tolist()

    db_emb_path = config.V5_FEATURES_DIR / f"stage8d_{SPECIES}_db_emb.npy"
    test_emb_path = config.V5_FEATURES_DIR / f"stage8d_{SPECIES}_test_emb.npy"

    with timed(f"extract DB embeddings ({len(db_paths)} imgs)"):
        db_emb = extract_embeddings(model, db_paths, transform, device)
    db_emb = l2norm(db_emb)
    save_numpy(db_emb, db_emb_path)
    log.info(f"  db_emb shape={db_emb.shape}")

    with timed(f"extract TEST embeddings ({len(test_paths)} imgs)"):
        test_emb = extract_embeddings(model, test_paths, transform, device)
    test_emb = l2norm(test_emb)
    save_numpy(test_emb, test_emb_path)
    log.info(f"  test_emb shape={test_emb.shape}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    db_ids = train_df["_id"].astype(str).to_numpy()
    log.info(f"db identities: {len(np.unique(db_ids))} unique  "
             f"({len(db_ids)} images)")

    # ------------------------------------------------------------------
    # 3) Test x DB cosine similarity
    # ------------------------------------------------------------------
    with timed("compute test x db cosine sim"):
        sim_td = (test_emb @ db_emb.T).astype(np.float32)
        sim_td = np.clip(sim_td, 0.0, 1.0)
    log.info(f"sim_td: shape={sim_td.shape}  mean={sim_td.mean():.4f}  "
             f"max={sim_td.max():.4f}  "
             f"p95={float(np.percentile(sim_td, 95)):.4f}  "
             f"p99={float(np.percentile(sim_td, 99)):.4f}")

    # ------------------------------------------------------------------
    # 4) Calibrate t_match
    # ------------------------------------------------------------------
    with timed("calibrate t_match"):
        t_match, match_diag = calibrate_match_threshold(
            db_emb, db_ids,
            seeds=[101, 202, 303, 404, 505], K=args.K,
        )

    # ------------------------------------------------------------------
    # 5) DB-guided matching
    # ------------------------------------------------------------------
    n_test = test_emb.shape[0]
    with timed("apply DB-guided matching"):
        assignments, top_sims, best_scores = db_match(
            sim_td, db_ids, t_match, K=args.K,
        )
        n_matched = sum(1 for a in assignments if a is not None)
        log.info(f"matched {n_matched}/{n_test} test images to DB "
                 f"identities at t_match={t_match:.3f}")
        if n_matched > 0:
            uniq_match_ids = sorted({a for a in assignments if a is not None})
            log.info(f"  matched to {len(uniq_match_ids)} distinct DB identities")
        if top_sims:
            log.info(f"top_sim distribution:  min={min(top_sims):.4f}  "
                     f"median={float(np.median(top_sims)):.4f}  "
                     f"max={max(top_sims):.4f}")

    # ------------------------------------------------------------------
    # 6) Cluster matched + agglomerate unmatched
    # ------------------------------------------------------------------
    if args.cluster_threshold is not None:
        t_cluster = float(args.cluster_threshold)
        log.info(f"using user-supplied t_cluster={t_cluster:.3f}")
    elif val_t_at_save is not None and val_t_at_save != "unknown":
        t_cluster = float(val_t_at_save)
        log.info(f"using checkpoint val_threshold as t_cluster={t_cluster:.3f}")
    else:
        t_cluster = 0.55
        log.warning(f"no checkpoint val_threshold; falling back to "
                    f"t_cluster={t_cluster:.3f}")

    with timed("build cluster labels"):
        cluster_labels = np.full(n_test, -1, dtype=np.int64)
        db_id_to_cluster: dict[str, int] = {}
        next_cluster = 0
        for i, a in enumerate(assignments):
            if a is None:
                continue
            if a not in db_id_to_cluster:
                db_id_to_cluster[a] = next_cluster
                next_cluster += 1
            cluster_labels[i] = db_id_to_cluster[a]
        n_db_clusters = next_cluster
        log.info(f"matched -> {n_db_clusters} DB-anchored clusters")

        unmatched_idx = np.array(
            [i for i in range(n_test) if assignments[i] is None],
            dtype=np.int64,
        )
        log.info(f"unmatched test images: {len(unmatched_idx)}")

        if len(unmatched_idx) > 1:
            sub_emb = test_emb[unmatched_idx]
            sub_sim = (sub_emb @ sub_emb.T).astype(np.float32)
            sub_sim = np.clip(sub_sim, 0.0, 1.0)
            sub_sim = (sub_sim + sub_sim.T) / 2.0
            np.fill_diagonal(sub_sim, 0.0)
            sub_clusters = cluster_agg(sub_sim, t_cluster)
            uniq_sub = np.unique(sub_clusters)
            log.info(f"unmatched clustered into {len(uniq_sub)} clusters "
                     f"at t_cluster={t_cluster:.3f}")
            remap = {c: next_cluster + idx for idx, c in enumerate(uniq_sub)}
            for local_i, gi in enumerate(unmatched_idx):
                cluster_labels[gi] = remap[sub_clusters[local_i]]
            next_cluster += len(uniq_sub)
        elif len(unmatched_idx) == 1:
            cluster_labels[unmatched_idx[0]] = next_cluster
            next_cluster += 1

    n_clusters = int(np.unique(cluster_labels).size)
    log.info(f"final n_clusters={n_clusters}  "
             f"(db-anchored={n_db_clusters}, "
             f"new={n_clusters - n_db_clusters})")

    counts = pd.Series(cluster_labels).value_counts()
    log.info(f"cluster size distribution: "
             f"min={counts.min()}, median={counts.median():.0f}, "
             f"max={counts.max()}, top5={counts.head(5).tolist()}")

    # ------------------------------------------------------------------
    # 7) Write submission
    # ------------------------------------------------------------------
    out_csv = config.V5_SUBMISSIONS_DIR / "stage8d_lynx_clusters.csv"
    sub = pd.DataFrame({
        "image_id": test_df["image_id"].astype(str).tolist(),
        "cluster": [
            f"cluster_{SPECIES}_{int(c):04d}"
            for c in cluster_labels
        ],
    })
    sub.to_csv(out_csv, index=False)
    log.info(f"wrote {out_csv}  ({len(sub)} rows)")

    diag = {
        "checkpoint": ckpt_info,
        "species": SPECIES,
        "n_db": int(len(db_ids)),
        "n_db_identities": int(len(np.unique(db_ids))),
        "n_test": int(n_test),
        "K": int(args.K),
        "t_match": float(t_match),
        "t_cluster": float(t_cluster),
        "n_matched_to_db": int(n_matched),
        "n_db_anchored_clusters": int(n_db_clusters),
        "n_unmatched": int(len(unmatched_idx)),
        "n_clusters_final": int(n_clusters),
        "match_calibration": match_diag,
        "ckpt_val_ari": (
            float(val_ari_at_save)
            if isinstance(val_ari_at_save, (int, float))
            else None
        ),
        "ckpt_val_threshold": (
            float(val_t_at_save)
            if isinstance(val_t_at_save, (int, float))
            else None
        ),
    }
    diag_path = (
        config.V5_SUBMISSIONS_DIR / "stage8d_lynx_clusters_diag.json"
    )
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2, default=float)
    log.info(f"wrote diagnostics to {diag_path}")

    log.info("Stage 8d complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
