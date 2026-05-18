#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch

import config
from utils import cluster_agg, log, save_numpy, seed_everything, timed

from stage2b_train_arcface import MegaDescArcFace, build_eval_transforms


# =====================================================================
# Constants
# =====================================================================

COMP_BASE = config.COMPETITION_ROOT
SUPPORTED_SPECIES = ("LynxID2025", "SalamanderID2025")


# =====================================================================
# Helpers
# =====================================================================

def l2norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return x / n


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


# =====================================================================
# Calibration: t_match via DB leave-one-out
# =====================================================================

def calibrate_match_threshold(
    db_emb: np.ndarray,
    db_ids: np.ndarray,
    seeds: list[int],
    K: int = 10,
) -> tuple[float, dict]:
    """
    Leave-one-out cross-validation on the DB:
      For each id with >= 2 imgs, hold one out as a query, treat the rest as
      gallery, run K-NN weighted-vote matching, sweep top_sim threshold.

    Returns (t_match, diagnostics).
    """
    log.info("calibrating t_match via leave-one-out DB cross validation")
    unique_ids = np.array(sorted(set(db_ids.tolist())))
    log.info(f"  DB has {len(unique_ids)} unique identities, "
             f"{len(db_ids)} images")

    eligible = []
    for uid in unique_ids:
        members = np.where(db_ids == uid)[0]
        if len(members) >= 2:
            eligible.append(uid)
    log.info(f"  eligible identities (>=2 imgs): {len(eligible)}")

    if not eligible:
        log.warning("no eligible DB ids for calibration; falling back to t_match=0.55")
        return 0.55, {"rule": "fallback", "n_eligible": 0}

    records = []  # rows: (best_score, top_sim, correct)

    for seed in seeds:
        rng = np.random.default_rng(seed)
        query_idx_list = []
        for uid in eligible:
            members = np.where(db_ids == uid)[0]
            q = rng.choice(members, size=1)[0]
            query_idx_list.append(int(q))
        query_idx = np.array(query_idx_list, dtype=np.int64)
        gallery_mask = np.ones(len(db_ids), dtype=bool)
        gallery_mask[query_idx] = False
        gallery_idx = np.where(gallery_mask)[0]
        gallery_ids = db_ids[gallery_idx]

        sim_qg = (db_emb[query_idx] @ db_emb[gallery_idx].T).astype(np.float32)
        sim_qg = np.clip(sim_qg, 0.0, 1.0)

        if sim_qg.shape[1] < K:
            top_k = np.argsort(-sim_qg, axis=1)
        else:
            part = np.argpartition(-sim_qg, K, axis=1)[:, :K]
            top_k = np.take_along_axis(
                part,
                np.argsort(-np.take_along_axis(sim_qg, part, axis=1), axis=1),
                axis=1,
            )

        for qi in range(len(query_idx)):
            top_idx = top_k[qi]
            top_sim = sim_qg[qi, top_idx]
            top_ids = gallery_ids[top_idx]
            scores = defaultdict(float)
            for s, gid in zip(top_sim, top_ids):
                scores[gid] += float(s)
            best_id, best_score = max(scores.items(), key=lambda x: x[1])
            true_id = db_ids[query_idx[qi]]
            records.append((float(best_score), float(top_sim[0]),
                            bool(best_id == true_id)))

    records = np.array(
        records,
        dtype=[("score", "f4"), ("top_sim", "f4"), ("correct", "?")],
    )
    log.info(f"  collected {len(records)} cross-val records over "
             f"{len(seeds)} seeds")

    sweep = np.arange(0.30, 0.951, 0.025)
    rows = []
    n_total = len(records)
    for t in sweep:
        keep = records["top_sim"] >= t
        n_kept = int(keep.sum())
        n_correct = int((keep & records["correct"]).sum())
        precision = n_correct / max(n_kept, 1)
        recall = n_correct / max(n_total, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        rows.append({
            "t": float(t),
            "kept": n_kept,
            "correct": n_correct,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    df = pd.DataFrame(rows)
    high_prec = df[df["precision"] >= 0.90]
    if len(high_prec) > 0:
        chosen = high_prec.sort_values("recall", ascending=False).iloc[0]
        rule = "precision>=0.90 with max recall"
    else:
        chosen = df.sort_values("f1", ascending=False).iloc[0]
        rule = "max F1 (no t reached precision>=0.90)"

    t_star = float(chosen["t"])
    log.info(f"  selection rule: {rule}")
    log.info(f"  chosen t_match={t_star:.3f}  "
             f"precision={chosen['precision']:.3f}  "
             f"recall={chosen['recall']:.3f}  "
             f"f1={chosen['f1']:.3f}  kept={int(chosen['kept'])}/{n_total}")
    log.info("  threshold sweep (every 4th row):")
    for _, r in df.iloc[::4].iterrows():
        log.info(f"    t={r['t']:.3f}  p={r['precision']:.3f}  "
                 f"r={r['recall']:.3f}  f1={r['f1']:.3f}  "
                 f"kept={int(r['kept'])}")

    diag = {
        "t_match": t_star,
        "rule": rule,
        "sweep": df.to_dict(orient="records"),
        "n_records": int(n_total),
        "n_eligible_ids": int(len(eligible)),
        "K": int(K),
    }
    return t_star, diag


# =====================================================================
# DB-guided matching
# =====================================================================

def db_match(
    test_db_sim: np.ndarray,
    db_ids: np.ndarray,
    t_match: float,
    K: int = 10,
) -> tuple[list[str | None], list[float], list[float]]:
    """K-NN weighted-vote with abstention. Returns (assignments, top_sims, scores)."""
    n_test = test_db_sim.shape[0]
    assignments: list[str | None] = []
    top_sims: list[float] = []
    best_scores: list[float] = []
    for i in range(n_test):
        order = np.argsort(-test_db_sim[i])[:K]
        sims = test_db_sim[i, order]
        ids = db_ids[order]
        scores = defaultdict(float)
        for s, gid in zip(sims, ids):
            scores[gid] += float(s)
        best_id, best_score = max(scores.items(), key=lambda x: x[1])
        top_sim = float(sims[0])
        top_sims.append(top_sim)
        best_scores.append(best_score)
        assignments.append(str(best_id) if top_sim >= t_match else None)
    return assignments, top_sims, best_scores


# =====================================================================
# Data loading
# =====================================================================

def load_species_data(species: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load competition train (DB) + test rows for the given species."""
    meta_path = COMP_BASE / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv missing at {meta_path}")
    meta = pd.read_csv(meta_path)

    train = meta[(meta["dataset"] == species) & (meta["split"] == "train")].copy()
    test = meta[(meta["dataset"] == species) & (meta["split"] == "test")].copy()
    if len(train) == 0:
        raise RuntimeError(f"no training rows for species={species}")
    if len(test) == 0:
        raise RuntimeError(f"no test rows for species={species}")
    train["_img"] = train["path"].apply(lambda p: str(COMP_BASE / p))
    train["_id"] = train["identity"].astype(str)
    test["_img"] = test["path"].apply(lambda p: str(COMP_BASE / p))
    log.info(f"[{species}] db (train) rows: {len(train)}  "
             f"({train['_id'].nunique()} ids)")
    log.info(f"[{species}] test rows: {len(test)}")
    return train, test


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True, choices=SUPPORTED_SPECIES)
    parser.add_argument("--K", type=int, default=10,
                        help="K for K-NN weighted-vote DB matching")
    parser.add_argument("--cluster-threshold", type=float, default=None,
                        help="Override agglomerative threshold for unmatched. "
                             "If unset, uses checkpoint val_threshold.")
    args = parser.parse_args()

    seed_everything(config.RANDOM_SEED)
    device = config.DEVICE
    species = args.species

    log.info("=" * 70)
    log.info(f"Stage 7b -- Extract + DB-guided cluster {species}")
    log.info("=" * 70)
    log.info(f"device={device}  K={args.K}")

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    ckpt_path = (
        config.V5_CKPTS_DIR / f"stage7a_{species.lower()}_arcface_best.pth"
    )
    if not ckpt_path.exists():
        log.error(f"missing checkpoint: {ckpt_path}")
        log.error(f"run stage7a_train_species_arcface.py --species {species} first")
        return 1
    log.info(f"loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = int(ckpt["num_classes"])
    use_subcenter = bool(ckpt.get("use_subcenter", True))
    val_ari_at_save = ckpt.get("val_ari", "unknown")
    val_t_at_save = ckpt.get("val_threshold", None)
    log.info(f"  num_classes={num_classes}  subcenter={use_subcenter}")
    log.info(f"  val_ari@save={val_ari_at_save}  val_t@save={val_t_at_save}")

    # ------------------------------------------------------------------
    # Build model + load weights
    # ------------------------------------------------------------------
    model = MegaDescArcFace(
        num_classes=num_classes, use_subcenter=use_subcenter,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    transform = build_eval_transforms()

    # ------------------------------------------------------------------
    # Load competition train (DB) + test rows
    # ------------------------------------------------------------------
    train_df, test_df = load_species_data(species)

    # Sanity check files exist
    for p in train_df["_img"].head(5).tolist():
        if not Path(p).exists():
            log.error(f"missing train image (first 5 sanity): {p}")
            return 1
    for p in test_df["_img"].head(5).tolist():
        if not Path(p).exists():
            log.error(f"missing test image (first 5 sanity): {p}")
            return 1

    # ------------------------------------------------------------------
    # Extract embeddings
    # ------------------------------------------------------------------
    db_paths = train_df["_img"].astype(str).tolist()
    test_paths = test_df["_img"].astype(str).tolist()

    db_emb_path = config.V5_FEATURES_DIR / f"stage7b_{species}_db_emb.npy"
    test_emb_path = config.V5_FEATURES_DIR / f"stage7b_{species}_test_emb.npy"

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

    # Free model memory before heavy numpy work
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    db_ids = train_df["_id"].astype(str).to_numpy()
    log.info(f"db identities: {len(np.unique(db_ids))} unique  "
             f"({len(db_ids)} images)")

    # ------------------------------------------------------------------
    # Test x DB cosine similarity
    # ------------------------------------------------------------------
    with timed("compute test x db cosine sim"):
        sim_td = (test_emb @ db_emb.T).astype(np.float32)
        sim_td = np.clip(sim_td, 0.0, 1.0)
    log.info(f"sim_td: shape={sim_td.shape}  mean={sim_td.mean():.4f}  "
             f"max={sim_td.max():.4f}  "
             f"p95={float(np.percentile(sim_td, 95)):.4f}  "
             f"p99={float(np.percentile(sim_td, 99)):.4f}")

    # ------------------------------------------------------------------
    # Calibrate t_match
    # ------------------------------------------------------------------
    with timed("calibrate t_match"):
        t_match, match_diag = calibrate_match_threshold(
            db_emb, db_ids,
            seeds=[101, 202, 303, 404, 505],
            K=args.K,
        )

    # ------------------------------------------------------------------
    # Run K-NN DB matching
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
    # Cluster matched + unmatched
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
    # Write submission CSV (per-species)
    # ------------------------------------------------------------------
    out_csv = (
        config.V5_SUBMISSIONS_DIR / f"stage7b_{species}_clusters.csv"
    )
    sub = pd.DataFrame({
        "image_id": test_df["image_id"].astype(str).tolist(),
        "cluster": [
            f"cluster_{species}_{int(c):04d}"
            for c in cluster_labels
        ],
    })
    sub.to_csv(out_csv, index=False)
    log.info(f"wrote {out_csv}  ({len(sub)} rows)")

    # ------------------------------------------------------------------
    # Diagnostics JSON
    # ------------------------------------------------------------------
    diag = {
        "species": species,
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
        config.V5_SUBMISSIONS_DIR / f"stage7b_{species}_clusters_diag.json"
    )
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2, default=float)
    log.info(f"wrote diagnostics to {diag_path}")

    log.info("")
    log.info(f"Stage 7b complete for {species}.")
    log.info("Next: run stage7b for the other species, then stage7c_final_submission.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
