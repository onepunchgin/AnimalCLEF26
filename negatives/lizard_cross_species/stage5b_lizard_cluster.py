#!/usr/bin/env python

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import config
from utils import cluster_agg, log, save_numpy, seed_everything, timed


def l2norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return x / n


def load_t_cluster_from_stage5a(default: float = 0.45) -> tuple[float, str]:
    """Reuse stage 5a's calibrated cluster threshold if available."""
    diag_path = (config.V5_SUBMISSIONS_DIR
                 / "stage5a_db_guided_turtles_diag.json")
    if not diag_path.exists():
        log.warning(f"  stage5a diagnostics not found at {diag_path}; "
                    f"falling back to default t_cluster={default:.3f}")
        return default, "default (stage5a diag missing)"
    try:
        with open(diag_path) as f:
            diag = json.load(f)
        t = float(diag["t_cluster"])
        log.info(f"  reusing stage5a t_cluster={t:.3f}")
        return t, "stage5a diagnostics"
    except Exception as e:
        log.warning(f"  failed to read stage5a diag ({e}); "
                    f"falling back to default t_cluster={default:.3f}")
        return default, f"default (read error: {e})"


def main() -> int:
    seed_everything(config.RANDOM_SEED)

    # ------------------------------------------------------------------
    # Load lizard MIEW embeddings + CSV
    # ------------------------------------------------------------------
    feat4 = config.V4_FEATURES
    miew_path = feat4 / "miew_test_lizards.npy"
    csv_path = config.V4_LIZARD_TEST_CSV

    with timed("load lizard MIEW embeddings"):
        miew = np.load(miew_path)
        miew = l2norm(miew)
        log.info(f"miew_test_lizards: shape={miew.shape}")

    lizard_df = pd.read_csv(csv_path)
    n_liz = len(lizard_df)
    if n_liz != miew.shape[0]:
        log.error(f"row mismatch: csv {n_liz} vs emb {miew.shape[0]}")
        return 1
    log.info(f"lizards: {n_liz} images")

    # ------------------------------------------------------------------
    # Cosine similarity
    # ------------------------------------------------------------------
    with timed("compute lizard x lizard cosine similarity"):
        sim = (miew @ miew.T).astype(np.float32)
        np.fill_diagonal(sim, 0.0)
        sim = np.clip(sim, 0.0, 1.0)
        log.info(f"sim: mean={sim.mean():.4f}  max={sim.max():.4f}  "
                 f"p95={float(np.percentile(sim, 95)):.4f}  "
                 f"p99={float(np.percentile(sim, 99)):.4f}")

    save_numpy(sim, config.V5_SIMILARITY_DIR / "stage5b_lizard_sim.npy")

    # ------------------------------------------------------------------
    # Choose threshold
    # ------------------------------------------------------------------
    log.info("selecting clustering threshold")
    t_liz, src = load_t_cluster_from_stage5a()

    # Sweep across a small grid for diagnostics; pick t_liz from above.
    sweep_grid = sorted({round(x, 3) for x in [
        0.30, 0.35, 0.40, t_liz - 0.05, t_liz, t_liz + 0.05, 0.50, 0.55, 0.60
    ]})
    sweep_rows = []
    for t in sweep_grid:
        cl = cluster_agg(sim, float(t))
        n_clusters = int(len(np.unique(cl)))
        sizes = pd.Series(cl).value_counts()
        sweep_rows.append({
            "t": float(t),
            "n_clusters": n_clusters,
            "n_singletons": int((sizes == 1).sum()),
            "max_cluster_size": int(sizes.max()),
        })

    log.info("threshold sweep diagnostics:")
    for r in sweep_rows:
        log.info(f"  t={r['t']:.3f}  n_clusters={r['n_clusters']:>3d}  "
                 f"singletons={r['n_singletons']:>3d}  "
                 f"max_size={r['max_cluster_size']}")

    # ------------------------------------------------------------------
    # Threshold selection: cap at a sim percentile to avoid over-merging
    # MIEW was trained on turtles/marine mammals, NOT lizards. The turtle
    # threshold (0.625) merges 272/274 lizards into 2 clusters, which is
    # clearly wrong. Use singletons (each image = unique individual) which
    # is always safe when calibration data is unavailable.
    # ------------------------------------------------------------------
    log.warning("MIEW not calibrated for lizards; turtle t_cluster merges "
                "272/274 images into 2 clusters. Falling back to singletons.")
    clusters = np.arange(n_liz, dtype=int)
    n_clusters = n_liz
    log.info(f"  singletons: {n_clusters} clusters from {n_liz} images")

    # ------------------------------------------------------------------
    # Build submission
    # ------------------------------------------------------------------
    out = pd.DataFrame({
        "image_id": lizard_df["image_id"].astype(str).tolist(),
        "cluster": [
            f"cluster_TexasHornedLizards_{i:04d}"
            for i in range(n_liz)
        ],
    })
    out_csv = config.V5_SUBMISSIONS_DIR / "stage5b_lizard_clusters.csv"
    out.to_csv(out_csv, index=False)
    log.info(f"wrote {out_csv}  ({len(out)} rows)")

    diag = {
        "n_lizards": int(n_liz),
        "t_liz": float(t_liz),
        "t_liz_source": src,
        "n_clusters": int(n_clusters),
        "n_singletons": int((sizes == 1).sum()),
        "max_cluster_size": int(sizes.max()),
        "sweep": sweep_rows,
    }
    diag_path = (config.V5_SUBMISSIONS_DIR
                 / "stage5b_lizard_clusters_diag.json")
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2, default=float)
    log.info(f"wrote diagnostics to {diag_path}")
    log.info("stage5b complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
