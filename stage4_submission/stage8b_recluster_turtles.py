#!/usr/bin/env python

from __future__ import annotations

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
from stage5a_db_guided_turtles import (
    calibrate_match_threshold,
    calibrate_cluster_threshold,
    db_match,
    krerank,
    l2norm,
    load_emb,
)


# =====================================================================
# Helpers
# =====================================================================

@torch.inference_mode()
def extract_embeddings(model, paths: list[str], transform, device) -> np.ndarray:
    from PIL import Image
    model.eval()
    bs = config.FEATURE_BATCH_SIZE
    embs: list[np.ndarray] = []
    n = len(paths)
    for s in range(0, n, bs):
        batch_paths = paths[s:s + bs]
        batch = torch.stack([
            transform(Image.open(p).convert("RGB")) for p in batch_paths
        ]).to(device, non_blocking=True)
        embs.append(model.extract_embeddings(batch).cpu().numpy())
        if (s // bs) % 20 == 0:
            log.info(f"    extracted {min(s + bs, n)}/{n}")
    return np.concatenate(embs, axis=0).astype(np.float32)


def pick_best_checkpoint() -> tuple[Path, dict]:
    """Return (path, info) for the better of stage8a vs stage2b."""
    stage8a_path = config.V5_CKPTS_DIR / "stage8a_turtle_extended_best.pth"
    stage2b_path = config.V5_CKPTS_DIR / "stage2b_arcface_best.pth"

    if not stage2b_path.exists():
        log.error(f"missing baseline checkpoint: {stage2b_path}")
        raise SystemExit(1)

    ckpt2 = torch.load(stage2b_path, map_location="cpu", weights_only=False)
    val2 = float(ckpt2.get("val_ari", 0.0))

    if stage8a_path.exists():
        ckpt8 = torch.load(stage8a_path, map_location="cpu", weights_only=False)
        val8 = float(ckpt8.get("val_ari", 0.0))
        if val8 > val2:
            log.info(f"Using stage8a (val_ari={val8:.4f} > stage2b {val2:.4f})")
            return stage8a_path, {
                "chosen": "stage8a",
                "stage8a_val_ari": val8,
                "stage2b_val_ari": val2,
            }
        log.info(f"Keeping stage2b (val_ari={val2:.4f} >= stage8a {val8:.4f})")
        return stage2b_path, {
            "chosen": "stage2b",
            "stage8a_val_ari": val8,
            "stage2b_val_ari": val2,
        }

    log.info(f"stage8a not found; using stage2b (val_ari={val2:.4f})")
    return stage2b_path, {
        "chosen": "stage2b",
        "stage8a_val_ari": None,
        "stage2b_val_ari": val2,
    }


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    seed_everything(config.RANDOM_SEED)
    device = config.DEVICE

    log.info("=" * 70)
    log.info("Stage 8b -- Re-cluster turtles with best ArcFace checkpoint")
    log.info("=" * 70)

    # ------------------------------------------------------------------
    # 1) Pick best checkpoint
    # ------------------------------------------------------------------
    ckpt_path, ckpt_info = pick_best_checkpoint()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = int(ckpt["num_classes"])
    use_subcenter = bool(ckpt.get("use_subcenter", True))
    log.info(f"  checkpoint: {ckpt_path.name}")
    log.info(f"  num_classes={num_classes}  subcenter={use_subcenter}")

    model = MegaDescArcFace(
        num_classes=num_classes, use_subcenter=use_subcenter,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    transform = build_eval_transforms()

    # ------------------------------------------------------------------
    # 2) Load v4 turtle test/db CSVs
    # ------------------------------------------------------------------
    test_df = pd.read_csv(config.V4_TURTLE_TEST_CSV)
    db_df = pd.read_csv(config.V4_TURTLE_DB_CSV)
    if "_img" not in test_df.columns or "_img" not in db_df.columns:
        log.error("v4 turtle CSVs are missing the `_img` column")
        return 1
    log.info(f"turtle test: {len(test_df)} images")
    log.info(f"turtle DB:   {len(db_df)} images, "
             f"{db_df['_id'].nunique()} identities")

    test_paths = test_df["_img"].astype(str).tolist()
    db_paths = db_df["_img"].astype(str).tolist()

    # ------------------------------------------------------------------
    # 3) Extract embeddings (fresh for the chosen checkpoint)
    # ------------------------------------------------------------------
    test_emb_path = config.V5_FEATURES_DIR / "stage8b_arcface_test_turtles.npy"
    db_emb_path = config.V5_FEATURES_DIR / "stage8b_arcface_db_turtles.npy"

    with timed(f"extract turtle TEST embeddings ({len(test_paths)} imgs)"):
        arc_test_raw = extract_embeddings(model, test_paths, transform, device)
    arc_test = l2norm(arc_test_raw)
    save_numpy(arc_test, test_emb_path)
    log.info(f"  arc_test shape={arc_test.shape}")

    with timed(f"extract turtle DB embeddings ({len(db_paths)} imgs)"):
        arc_db_raw = extract_embeddings(model, db_paths, transform, device)
    arc_db = l2norm(arc_db_raw)
    save_numpy(arc_db, db_emb_path)
    log.info(f"  arc_db shape={arc_db.shape}")

    # Free model GPU memory before heavy numpy work
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4) Load v4 MegaD/MIEW/DINOv2 embeddings (same as stage 5a)
    # ------------------------------------------------------------------
    feat4 = config.V4_FEATURES
    mega_test = load_emb(feat4 / "megadesc_test_turtles.npy", "megadesc_test")
    miew_test = load_emb(feat4 / "miew_test_turtles.npy", "miew_test")
    dino_test = load_emb(feat4 / "dinov2_test_turtles.npy", "dinov2_test")
    mega_db = load_emb(feat4 / "megadesc_db_turtles.npy", "megadesc_db")
    miew_db = load_emb(feat4 / "miew_db_turtles.npy", "miew_db")

    n_test = arc_test.shape[0]
    n_db = arc_db.shape[0]
    if (mega_test.shape[0] != n_test or miew_test.shape[0] != n_test
            or dino_test.shape[0] != n_test):
        log.error("v4 test embedding row count mismatch")
        return 1
    if mega_db.shape[0] != n_db or miew_db.shape[0] != n_db:
        log.error("v4 db embedding row count mismatch")
        return 1

    db_ids = db_df["_id"].astype(str).to_numpy()

    # ------------------------------------------------------------------
    # 5) Fusion (same weights as stage 5a)
    # ------------------------------------------------------------------
    W_TEST_TEST = {
        "megadesc": 0.25,
        "miew": 0.40,
        "arcface": 0.25,
        "dinov2": 0.10,
    }
    W_TEST_DB = {
        "megadesc": 0.25,
        "miew": 0.40,
        "arcface": 0.25,
    }
    s_tt = sum(W_TEST_TEST.values())
    s_td = sum(W_TEST_DB.values())
    W_TEST_TEST = {k: v / s_tt for k, v in W_TEST_TEST.items()}
    W_TEST_DB = {k: v / s_td for k, v in W_TEST_DB.items()}
    log.info(f"test-test fusion weights: {W_TEST_TEST}")
    log.info(f"test-db   fusion weights: {W_TEST_DB}")

    with timed("compute fused test x test sim"):
        sim_tt = (
            W_TEST_TEST["megadesc"] * (mega_test @ mega_test.T)
            + W_TEST_TEST["miew"] * (miew_test @ miew_test.T)
            + W_TEST_TEST["arcface"] * (arc_test @ arc_test.T)
            + W_TEST_TEST["dinov2"] * (dino_test @ dino_test.T)
        ).astype(np.float32)
        np.fill_diagonal(sim_tt, 0.0)
        sim_tt = np.clip(sim_tt, 0.0, 1.0)
        log.info(f"sim_tt: shape={sim_tt.shape}  mean={sim_tt.mean():.4f}  "
                 f"max(off-diag)={sim_tt.max():.4f}")

    with timed("compute fused test x db sim"):
        sim_td = (
            W_TEST_DB["megadesc"] * (mega_test @ mega_db.T)
            + W_TEST_DB["miew"] * (miew_test @ miew_db.T)
            + W_TEST_DB["arcface"] * (arc_test @ arc_db.T)
        ).astype(np.float32)
        sim_td = np.clip(sim_td, 0.0, 1.0)
        log.info(f"sim_td: shape={sim_td.shape}  mean={sim_td.mean():.4f}  "
                 f"max={sim_td.max():.4f}")

    # ------------------------------------------------------------------
    # 6) Calibrate t_match (DB cross-val) and t_cluster (DB pair recovery)
    # ------------------------------------------------------------------
    db_blocks = [
        (mega_db, W_TEST_DB["megadesc"]),
        (miew_db, W_TEST_DB["miew"]),
        (arc_db, W_TEST_DB["arcface"]),
    ]
    with timed("calibrate t_match"):
        t_match, match_diag = calibrate_match_threshold(
            db_blocks, db_ids,
            seeds=[101, 202, 303, 404, 505], K=10,
        )
    with timed("calibrate t_cluster"):
        t_cluster, cluster_diag = calibrate_cluster_threshold(db_blocks, db_ids)

    # ------------------------------------------------------------------
    # 7) DB-guided matching with abstention
    # ------------------------------------------------------------------
    with timed("apply DB-guided matching"):
        K = 10
        assignments, top_sims, best_scores = db_match(
            sim_td, db_ids, t_match, K=K,
        )
        n_matched = sum(1 for a in assignments if a is not None)
        log.info(f"matched {n_matched}/{n_test} test images to DB "
                 f"identities at t_match={t_match:.3f}")
        if n_matched > 0:
            uniq_match_ids = sorted({a for a in assignments if a is not None})
            log.info(f"  matched to {len(uniq_match_ids)} distinct DB identities")
        log.info(f"top_sim distribution:  min={min(top_sims):.4f}  "
                 f"median={float(np.median(top_sims)):.4f}  "
                 f"max={max(top_sims):.4f}")

    # ------------------------------------------------------------------
    # 8) K-reciprocal re-ranking on test x test
    # ------------------------------------------------------------------
    with timed("k-reciprocal re-ranking on test x test"):
        sim_tt_rerank = krerank(sim_tt, k=20, lambda_value=0.3)
        log.info(f"reranked sim_tt: mean={sim_tt_rerank.mean():.4f}  "
                 f"max={sim_tt_rerank.max():.4f}")
        save_numpy(
            sim_tt_rerank,
            config.V5_SIMILARITY_DIR / "stage8b_test_tt_reranked.npy",
        )
        save_numpy(
            sim_td,
            config.V5_SIMILARITY_DIR / "stage8b_test_db_fused.npy",
        )

    # ------------------------------------------------------------------
    # 9) Build cluster labels (DB-anchored + agglomerate unmatched)
    # ------------------------------------------------------------------
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
            sub_sim = sim_tt_rerank[np.ix_(unmatched_idx, unmatched_idx)]
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

    # ------------------------------------------------------------------
    # 10) Write per-species turtle CSV + diagnostics
    # ------------------------------------------------------------------
    out_csv = config.V5_SUBMISSIONS_DIR / "stage8b_turtle_clusters.csv"
    turtle_sub = pd.DataFrame({
        "image_id": test_df["image_id"].astype(str).tolist(),
        "cluster": [
            f"cluster_SeaTurtleID2022_{int(c) + 1:04d}"
            for c in cluster_labels
        ],
    })
    turtle_sub.to_csv(out_csv, index=False)
    log.info(f"wrote {out_csv}  ({len(turtle_sub)} rows)")

    counts = pd.Series(cluster_labels).value_counts()
    log.info(f"cluster size distribution: "
             f"min={counts.min()}, median={counts.median():.0f}, "
             f"max={counts.max()}, top5={counts.head(5).tolist()}")

    diag = {
        "checkpoint": ckpt_info,
        "n_test": int(n_test),
        "n_db": int(n_db),
        "n_db_identities": int(len(np.unique(db_ids))),
        "fusion_weights_test_test": W_TEST_TEST,
        "fusion_weights_test_db": W_TEST_DB,
        "t_match": float(t_match),
        "t_cluster": float(t_cluster),
        "K": 10,
        "krerank_k": 20,
        "krerank_lambda": 0.3,
        "n_matched_to_db": int(n_matched),
        "n_db_anchored_clusters": int(n_db_clusters),
        "n_unmatched": int(len(unmatched_idx)),
        "n_clusters_final": int(n_clusters),
        "match_calibration": match_diag,
        "cluster_calibration": cluster_diag,
    }
    diag_path = (config.V5_SUBMISSIONS_DIR
                 / "stage8b_turtle_clusters_diag.json")
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2, default=float)
    log.info(f"wrote diagnostics to {diag_path}")

    log.info("Stage 8b complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
