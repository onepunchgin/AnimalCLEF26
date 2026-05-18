#!/usr/bin/env python

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import config
from utils import cluster_agg, log, save_numpy, seed_everything, timed


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def l2norm(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization, safe against zero rows."""
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return x / n


def load_emb(path: Path, name: str) -> np.ndarray:
    arr = np.load(path)
    arr = l2norm(arr)
    log.info(f"loaded {name}: shape={arr.shape}  path={path}")
    return arr


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_match_threshold(
    db_emb_fused_blocks: list[tuple[np.ndarray, float]],
    db_ids: np.ndarray,
    seeds: list[int],
    K: int = 10,
) -> tuple[float, dict]:
    """
    Simulate sparse-query scenario: for each DB identity, hold out 1 image as
    'query', use the rest as 'gallery'. Run K-NN weighted-vote matching on
    fused similarities and sweep threshold.

    Returns (t_star, diagnostics).
    """
    log.info("calibrating t_match via leave-one-out DB cross validation")
    rng_master = np.random.default_rng(0)
    unique_ids = np.array(sorted(set(db_ids.tolist())))
    log.info(f"  DB has {len(unique_ids)} unique identities, "
             f"{len(db_ids)} images")

    records = []  # rows: (best_score, top_sim, correct)

    for seed in seeds:
        rng = np.random.default_rng(seed)
        query_idx_list = []
        for uid in unique_ids:
            members = np.where(db_ids == uid)[0]
            if len(members) < 2:
                continue
            q = rng.choice(members, size=1)[0]
            query_idx_list.append(int(q))
        query_idx = np.array(query_idx_list, dtype=np.int64)
        gallery_mask = np.ones(len(db_ids), dtype=bool)
        gallery_mask[query_idx] = False
        gallery_idx = np.where(gallery_mask)[0]
        gallery_ids = db_ids[gallery_idx]

        # Fused sim (query x gallery)
        sim_qg = np.zeros((len(query_idx), len(gallery_idx)), dtype=np.float32)
        for emb, w in db_emb_fused_blocks:
            sim_qg += w * (emb[query_idx] @ emb[gallery_idx].T).astype(np.float32)

        # K nearest neighbors per query
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

    # Sweep threshold using top_sim (matches deployment criterion)
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

    # Choose t_star: highest precision >= 0.90 with maximum recall;
    # fall back to F1-max.
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
        "K": int(K),
    }
    return t_star, diag


def calibrate_cluster_threshold(
    db_emb_fused_blocks: list[tuple[np.ndarray, float]],
    db_ids: np.ndarray,
) -> tuple[float, dict]:
    """
    For each DB identity with >= 3 images, hold out 2 images as 'queries' and
    keep the rest in a small mixed gallery. Sweep an agglomerative threshold
    that should cluster the 2 same-identity queries together while keeping
    distractor identities separated.

    Strategy: build a synthetic test-like set per seed, sweep threshold,
    measure pairwise same-identity precision and recall, select the t that
    maximizes pair F1.
    """
    log.info("calibrating t_cluster via DB pair-recovery sweep")
    unique_ids = np.array(sorted(set(db_ids.tolist())))
    eligible = []
    for uid in unique_ids:
        members = np.where(db_ids == uid)[0]
        if len(members) >= 3:
            eligible.append(uid)
    log.info(f"  eligible identities (>=3 imgs): {len(eligible)}")

    seeds = [0, 1, 2]
    sweep = np.arange(0.10, 0.701, 0.025)
    sweep_rows = {float(t): {"tp": 0, "fp": 0, "fn": 0} for t in sweep}

    rng_master = np.random.default_rng(123)

    for seed in seeds:
        rng = np.random.default_rng(seed)
        # Build a synthetic 'test' set: 2 images from each eligible id +
        # 1 image from each non-eligible id (singletons), capped to 500 imgs.
        chosen_idx = []
        chosen_ids = []
        for uid in eligible:
            members = np.where(db_ids == uid)[0]
            picked = rng.choice(members, size=2, replace=False)
            chosen_idx.extend(picked.tolist())
            chosen_ids.extend([uid, uid])
        # Add singletons for diversity
        non_elig = [u for u in unique_ids if u not in set(eligible)]
        rng.shuffle(non_elig)
        for uid in non_elig:
            members = np.where(db_ids == uid)[0]
            picked = rng.choice(members, size=1)
            chosen_idx.append(int(picked[0]))
            chosen_ids.append(uid)
            if len(chosen_idx) >= 500:
                break

        chosen_idx = np.array(chosen_idx, dtype=np.int64)
        chosen_ids_arr = np.array(chosen_ids)

        # Fused sim within this set
        sim = np.zeros((len(chosen_idx), len(chosen_idx)), dtype=np.float32)
        for emb, w in db_emb_fused_blocks:
            sub = emb[chosen_idx]
            sim += w * (sub @ sub.T).astype(np.float32)
        np.fill_diagonal(sim, 0.0)
        sim = np.clip(sim, 0.0, 1.0)

        # Ground-truth same-identity pair mask (upper triangle only)
        n = len(chosen_idx)
        same_id = chosen_ids_arr[:, None] == chosen_ids_arr[None, :]
        triu = np.triu(np.ones((n, n), dtype=bool), k=1)

        for t in sweep:
            cluster_ids = cluster_agg(sim, float(t))
            same_cluster = cluster_ids[:, None] == cluster_ids[None, :]
            tp = int(((same_cluster & same_id) & triu).sum())
            fp = int(((same_cluster & ~same_id) & triu).sum())
            fn = int(((~same_cluster & same_id) & triu).sum())
            sweep_rows[float(t)]["tp"] += tp
            sweep_rows[float(t)]["fp"] += fp
            sweep_rows[float(t)]["fn"] += fn

    rows = []
    for t in sweep:
        r = sweep_rows[float(t)]
        prec = r["tp"] / max(r["tp"] + r["fp"], 1)
        rec = r["tp"] / max(r["tp"] + r["fn"], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append({
            "t": float(t),
            "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
            "precision": prec, "recall": rec, "f1": f1,
        })
    df = pd.DataFrame(rows)
    chosen = df.sort_values("f1", ascending=False).iloc[0]
    t_cluster = float(chosen["t"])

    log.info(f"  chosen t_cluster={t_cluster:.3f}  "
             f"pair_precision={chosen['precision']:.3f}  "
             f"pair_recall={chosen['recall']:.3f}  "
             f"pair_f1={chosen['f1']:.3f}")
    log.info("  cluster threshold sweep (every 3rd row):")
    for _, r in df.iloc[::3].iterrows():
        log.info(f"    t={r['t']:.3f}  p={r['precision']:.3f}  "
                 f"r={r['recall']:.3f}  f1={r['f1']:.3f}")

    diag = {
        "t_cluster": t_cluster,
        "sweep": df.to_dict(orient="records"),
    }
    return t_cluster, diag


# ---------------------------------------------------------------------------
# K-reciprocal re-ranking
# ---------------------------------------------------------------------------

def krerank(sim: np.ndarray, k: int = 20, lambda_value: float = 0.3) -> np.ndarray:
    """
    K-reciprocal re-ranking on an NxN similarity matrix (Zhong et al. 2017).
    Vectorized via boolean indicator matrices for N up to a few thousand.
    """
    N = sim.shape[0]
    sim = sim.astype(np.float32, copy=False)
    k_eff = min(k, N - 1)
    k_half = max(min(k // 2, N - 1), 1)

    # Top-k indices per row (excluding self, since diagonal should be 0)
    top_k = np.argsort(-sim, axis=1)[:, :k_eff]
    top_half = np.argsort(-sim, axis=1)[:, :k_half]

    rows_full = np.repeat(np.arange(N), k_eff)
    rows_half = np.repeat(np.arange(N), k_half)
    indicator = np.zeros((N, N), dtype=bool)
    indicator[rows_full, top_k.ravel()] = True
    indicator_half = np.zeros((N, N), dtype=bool)
    indicator_half[rows_half, top_half.ravel()] = True

    reciprocal = indicator & indicator.T
    reciprocal_half = indicator_half & indicator_half.T

    expanded = reciprocal.copy()
    # Expand: for each i, for each j in R(i), if R_half(j) overlaps R(i)
    # by >= 2/3 of |R_half(j)|, then add R_half(j) to R(i).
    for i in range(N):
        Ri = reciprocal[i]
        for j in np.where(Ri)[0]:
            R_j_half = reciprocal_half[j]
            n_half = int(R_j_half.sum())
            if n_half == 0:
                continue
            overlap = int((Ri & R_j_half).sum())
            if overlap >= (2.0 / 3.0) * n_half:
                expanded[i] |= R_j_half
    # symmetrize
    expanded = expanded | expanded.T

    R = expanded.astype(np.float32)
    intersection = R @ R.T
    sizes = R.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - intersection
    jaccard = np.where(union > 0, intersection / union, 0.0).astype(np.float32)
    np.fill_diagonal(jaccard, 0.0)

    reranked = (1.0 - lambda_value) * sim + lambda_value * jaccard
    np.fill_diagonal(reranked, 0.0)
    return np.clip(reranked.astype(np.float32), 0.0, 1.0)


# ---------------------------------------------------------------------------
# DB matching
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    seed_everything(config.RANDOM_SEED)

    # ------------------------------------------------------------------
    # Step 1: Load and L2-normalize embeddings
    # ------------------------------------------------------------------
    with timed("load embeddings"):
        feat4 = config.V4_FEATURES
        feat5 = config.V5_FEATURES_DIR

        mega_test = load_emb(feat4 / "megadesc_test_turtles.npy", "megadesc_test")
        miew_test = load_emb(feat4 / "miew_test_turtles.npy", "miew_test")
        dino_test = load_emb(feat4 / "dinov2_test_turtles.npy", "dinov2_test")
        arc_test = load_emb(feat5 / "stage2c_arcface_test_turtles.npy",
                             "arcface_test")

        mega_db = load_emb(feat4 / "megadesc_db_turtles.npy", "megadesc_db")
        miew_db = load_emb(feat4 / "miew_db_turtles.npy", "miew_db")
        arc_db = load_emb(feat5 / "stage2c_arcface_db_turtles.npy",
                           "arcface_db")

        n_test = mega_test.shape[0]
        n_db = mega_db.shape[0]
        log.info(f"n_test={n_test}  n_db={n_db}")

        if (miew_test.shape[0] != n_test or dino_test.shape[0] != n_test
                or arc_test.shape[0] != n_test):
            log.error("test embedding row counts do not match")
            return 1
        if miew_db.shape[0] != n_db or arc_db.shape[0] != n_db:
            log.error("db embedding row counts do not match")
            return 1

    # ------------------------------------------------------------------
    # Load CSVs
    # ------------------------------------------------------------------
    test_df = pd.read_csv(config.V4_TURTLE_TEST_CSV)
    db_df = pd.read_csv(config.V4_TURTLE_DB_CSV)
    if len(test_df) != n_test:
        log.error(f"test_df rows ({len(test_df)}) != embedding rows "
                  f"({n_test})")
        return 1
    if len(db_df) != n_db:
        log.error(f"db_df rows ({len(db_df)}) != embedding rows ({n_db})")
        return 1
    db_ids = db_df["_id"].astype(str).to_numpy()
    log.info(f"db identities: {len(np.unique(db_ids))} unique  "
             f"({len(db_ids)} images)")

    # ------------------------------------------------------------------
    # Fusion weights (per spec)
    # ------------------------------------------------------------------
    W_TEST_TEST = {
        "megadesc": 0.25,
        "miew": 0.40,
        "arcface": 0.25,
        "dinov2": 0.10,
    }
    W_TEST_DB = {  # renormalized over the 3 models we have
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

    # ------------------------------------------------------------------
    # Step 2: fused similarities
    # ------------------------------------------------------------------
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
    # Step 3: calibrate t_match using DB cross-validation
    # ------------------------------------------------------------------
    db_blocks = [
        (mega_db, W_TEST_DB["megadesc"]),
        (miew_db, W_TEST_DB["miew"]),
        (arc_db, W_TEST_DB["arcface"]),
    ]
    with timed("calibrate t_match"):
        t_match, match_diag = calibrate_match_threshold(
            db_blocks, db_ids,
            seeds=[101, 202, 303, 404, 505],
            K=10,
        )

    # ------------------------------------------------------------------
    # Step 4: calibrate t_cluster on db pair recovery
    # ------------------------------------------------------------------
    with timed("calibrate t_cluster"):
        t_cluster, cluster_diag = calibrate_cluster_threshold(db_blocks, db_ids)

    # ------------------------------------------------------------------
    # Step 5: K-NN weighted vote DB matching (test side)
    # ------------------------------------------------------------------
    with timed("apply DB-guided matching"):
        K = 10
        assignments, top_sims, best_scores = db_match(sim_td, db_ids, t_match, K)
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
    # Step 6: K-reciprocal re-ranking on test x test
    # ------------------------------------------------------------------
    with timed("k-reciprocal re-ranking on test x test"):
        sim_tt_rerank = krerank(sim_tt, k=20, lambda_value=0.3)
        log.info(f"reranked sim_tt: mean={sim_tt_rerank.mean():.4f}  "
                 f"max={sim_tt_rerank.max():.4f}")
        save_numpy(
            sim_tt_rerank,
            config.V5_SIMILARITY_DIR / "stage5a_test_tt_reranked.npy",
        )
        save_numpy(
            sim_td,
            config.V5_SIMILARITY_DIR / "stage5a_test_db_fused.npy",
        )

    # ------------------------------------------------------------------
    # Step 7: cluster matched + cluster unmatched independently
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
    # Step 8: save submission (turtle-only) + diagnostics
    # ------------------------------------------------------------------
    out_csv = config.V5_SUBMISSIONS_DIR / "stage5a_db_guided_turtles.csv"
    turtle_sub = pd.DataFrame({
        "image_id": test_df["image_id"].astype(str).tolist(),
        "cluster": [
            f"cluster_SeaTurtleID2022_{int(c) + 1:04d}"
            for c in cluster_labels
        ],
    })
    turtle_sub.to_csv(out_csv, index=False)
    log.info(f"wrote {out_csv}  ({len(turtle_sub)} rows)")

    # Per-cluster diagnostics for any large groupings (sanity)
    counts = pd.Series(cluster_labels).value_counts()
    log.info(f"cluster size distribution: "
             f"min={counts.min()}, median={counts.median():.0f}, "
             f"max={counts.max()}, top5={counts.head(5).tolist()}")

    diag = {
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
                 / "stage5a_db_guided_turtles_diag.json")
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2, default=float)
    log.info(f"wrote diagnostics to {diag_path}")
    log.info("stage5a complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
