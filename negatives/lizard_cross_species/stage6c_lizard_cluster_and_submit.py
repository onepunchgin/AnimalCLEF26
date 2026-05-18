#!/usr/bin/env python

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from utils import (
    cluster_agg,
    log,
    seed_everything,
    synthesize_nonscored,
    timed,
    write_submission,
)
import config


# =====================================================================
# Helpers
# =====================================================================

def l2norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return x / n


def cosine_sim(emb: np.ndarray) -> np.ndarray:
    sim = (emb @ emb.T).astype(np.float32)
    sim = np.clip(sim, 0.0, 1.0)
    sim = (sim + sim.T) / 2.0
    np.fill_diagonal(sim, 0.0)
    return sim


def sweep_val_threshold(val_emb: np.ndarray, val_codes: np.ndarray):
    """Sweep agglomerative threshold; return best (t, ari) and full curve."""
    sim = cosine_sim(val_emb)
    curve = []
    best_t, best_ari = 0.5, -1.0
    for t in np.arange(0.20, 0.901, 0.025):
        t = float(round(float(t), 4))
        cl = cluster_agg(sim, t)
        ari = float(adjusted_rand_score(val_codes, cl))
        n_clusters = int(pd.Series(cl).nunique())
        n_singletons = int((pd.Series(cl).value_counts() == 1).sum())
        curve.append({
            "t": t,
            "ari": ari,
            "n_clusters": n_clusters,
            "n_singletons": n_singletons,
        })
        if ari > best_ari:
            best_ari, best_t = ari, t
    return best_t, best_ari, curve


# =====================================================================
# Main: cluster Texas test + write final submission
# =====================================================================

def cluster_texas(t_chosen: float):
    """Load Texas embeddings, cluster, write CSV + diag. Returns out_csv path."""
    feat = config.V5_FEATURES_DIR

    texas_emb_path = feat / "stage6_lizard_texas_test_emb.npy"
    if not texas_emb_path.exists():
        log.error(f"missing Texas embeddings: {texas_emb_path}")
        log.error("run stage6b_extract_lizard_embeddings.py first")
        return None, None

    texas_emb = l2norm(np.load(texas_emb_path))
    log.info(f"loaded Texas test embeddings: {texas_emb.shape}")

    if not config.V4_LIZARD_TEST_CSV.exists():
        log.error(f"missing {config.V4_LIZARD_TEST_CSV}")
        return None, None
    texas_df = pd.read_csv(config.V4_LIZARD_TEST_CSV)
    if len(texas_df) != texas_emb.shape[0]:
        log.error(f"row mismatch: csv {len(texas_df)} vs emb {texas_emb.shape[0]}")
        return None, None

    sim = cosine_sim(texas_emb)
    log.info(f"texas sim: mean={sim.mean():.4f}  max={sim.max():.4f}  "
             f"p95={float(np.percentile(sim, 95)):.4f}  "
             f"p99={float(np.percentile(sim, 99)):.4f}")

    # Sweep around chosen threshold for diagnostics.
    sweep_grid = sorted({round(float(x), 4) for x in (
        list(np.arange(max(0.10, t_chosen - 0.10),
                       min(0.99, t_chosen + 0.10) + 1e-6, 0.02))
        + [t_chosen]
    )})
    sweep_rows = []
    for t in sweep_grid:
        cl = cluster_agg(sim, float(t))
        sizes = pd.Series(cl).value_counts()
        sweep_rows.append({
            "t": float(t),
            "n_clusters": int(len(sizes)),
            "n_singletons": int((sizes == 1).sum()),
            "max_cluster_size": int(sizes.max()),
        })
    log.info("Texas-side sweep around chosen threshold:")
    for r in sweep_rows:
        marker = " <- chosen" if abs(r["t"] - t_chosen) < 1e-9 else ""
        log.info(f"  t={r['t']:.4f}  n_clusters={r['n_clusters']:>3d}  "
                 f"singletons={r['n_singletons']:>3d}  "
                 f"max_size={r['max_cluster_size']}{marker}")

    with timed(f"agglomerative clustering at t={t_chosen:.4f}"):
        clusters = cluster_agg(sim, float(t_chosen))
        sizes = pd.Series(clusters).value_counts()
        n_clusters = int(len(sizes))
        n_singletons = int((sizes == 1).sum())
        max_size = int(sizes.max())
    log.info(f"final lizard clustering: t={t_chosen:.4f}  "
             f"n_clusters={n_clusters}  singletons={n_singletons}  "
             f"max_cluster_size={max_size}")

    # Map raw cluster ids to a stable 0..n_clusters-1 ordering.
    uniq = pd.Series(clusters).unique()
    remap = {c: i for i, c in enumerate(sorted(uniq.tolist()))}
    out = pd.DataFrame({
        "image_id": texas_df["image_id"].astype(str).tolist(),
        "cluster": [
            f"cluster_TexasHornedLizards_{remap[c]:04d}" for c in clusters
        ],
    })
    out_csv = config.V5_SUBMISSIONS_DIR / "stage6c_texas_lizard_clusters.csv"
    out.to_csv(out_csv, index=False)
    log.info(f"wrote {out_csv}  ({len(out)} rows)")

    return out_csv, {
        "t_chosen": float(t_chosen),
        "n_clusters": n_clusters,
        "n_singletons": n_singletons,
        "max_cluster_size": max_size,
        "sweep": sweep_rows,
        "texas_sim_stats": {
            "mean": float(sim.mean()),
            "max": float(sim.max()),
            "p50": float(np.percentile(sim, 50)),
            "p90": float(np.percentile(sim, 90)),
            "p95": float(np.percentile(sim, 95)),
            "p99": float(np.percentile(sim, 99)),
        },
    }


def write_final(lizard_csv: Path) -> int:
    """Combine turtle (stage 5a) + lizard (stage 6c) + nonscored singletons."""
    turtle_csv = config.V5_SUBMISSIONS_DIR / "stage5a_db_guided_turtles.csv"
    sample_csv = config.COMPETITION_SAMPLE_SUB

    if not turtle_csv.exists():
        log.error(f"missing turtle submission: {turtle_csv}")
        return 1
    if not lizard_csv.exists():
        log.error(f"missing lizard submission: {lizard_csv}")
        return 1
    if not sample_csv.exists():
        log.error(f"missing sample submission: {sample_csv}")
        return 1

    turtle_sub = pd.read_csv(turtle_csv)
    lizard_sub = pd.read_csv(lizard_csv)
    sample = pd.read_csv(sample_csv)

    log.info(f"turtle rows: {len(turtle_sub)}  "
             f"unique clusters: {turtle_sub['cluster'].nunique()}")
    log.info(f"lizard rows: {len(lizard_sub)}  "
             f"unique clusters: {lizard_sub['cluster'].nunique()}")

    nonscored = synthesize_nonscored(sample)
    log.info(f"nonscored rows: {len(nonscored)} "
             f"(singletons for LynxID2025 + SalamanderID2025)")

    submission = pd.concat([turtle_sub, lizard_sub, nonscored],
                           ignore_index=True)
    submission = submission.drop_duplicates(subset="image_id", keep="first")

    out_path = config.V5_SUBMISSIONS_DIR / "stage6d_final.csv"
    write_submission(submission, out_path, sample_submission=sample)
    log.info(f"Final submission: {out_path}")
    return 0


def main() -> int:
    seed_everything(config.RANDOM_SEED)

    feat = config.V5_FEATURES_DIR
    val_emb_path = feat / "stage6_lizard_balearic_val_emb.npy"
    val_codes_path = feat / "stage6_lizard_balearic_val_codes.npy"

    if not val_emb_path.exists():
        log.error(f"missing val embeddings: {val_emb_path}")
        log.error("run stage6b_extract_lizard_embeddings.py first")
        return 1
    if not val_codes_path.exists():
        log.error(f"missing val codes: {val_codes_path}")
        return 1

    val_emb = l2norm(np.load(val_emb_path))
    val_codes = np.load(val_codes_path)
    log.info(f"loaded val embeddings: {val_emb.shape}  "
             f"unique labels: {len(np.unique(val_codes))}")

    # ------------------------------------------------------------------
    # 1) Threshold calibration on val
    # ------------------------------------------------------------------
    log.info("")
    log.info("=" * 70)
    log.info("Threshold calibration on BalearicLizard val")
    log.info("=" * 70)
    with timed("val threshold sweep"):
        best_t, best_ari, curve = sweep_val_threshold(val_emb, val_codes)

    log.info("val ARI sweep:")
    for r in curve:
        marker = " <- chosen" if abs(r["t"] - best_t) < 1e-9 else ""
        log.info(f"  t={r['t']:.4f}  ARI={r['ari']:+.4f}  "
                 f"n_clusters={r['n_clusters']:>4d}  "
                 f"singletons={r['n_singletons']:>4d}{marker}")
    log.info(f"BEST: t={best_t:.4f}  val_ARI={best_ari:+.4f}")

    # ------------------------------------------------------------------
    # 2) Cluster Texas test at calibrated threshold
    # ------------------------------------------------------------------
    log.info("")
    log.info("=" * 70)
    log.info("Cluster TexasHornedLizard test at calibrated threshold")
    log.info("=" * 70)
    out_csv, texas_diag = cluster_texas(best_t)
    if out_csv is None:
        return 1

    # ------------------------------------------------------------------
    # 3) Save full diagnostics JSON
    # ------------------------------------------------------------------
    diag = {
        "calibration": {
            "best_t": float(best_t),
            "best_val_ari": float(best_ari),
            "sweep": curve,
            "n_val_images": int(val_emb.shape[0]),
            "n_val_identities": int(len(np.unique(val_codes))),
        },
        "texas": texas_diag,
    }
    diag_path = (config.V5_SUBMISSIONS_DIR
                 / "stage6c_texas_lizard_clusters_diag.json")
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2, default=float)
    log.info(f"wrote diagnostics to {diag_path}")

    # ------------------------------------------------------------------
    # 4) Build final submission (stage6d_final.csv)
    # ------------------------------------------------------------------
    log.info("")
    log.info("=" * 70)
    log.info("Build final submission (turtles + lizards + nonscored)")
    log.info("=" * 70)
    rc = write_final(out_csv)
    if rc != 0:
        return rc

    log.info("")
    log.info("Stage 6c complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
