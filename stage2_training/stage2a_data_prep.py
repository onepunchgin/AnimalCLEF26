#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from utils import log, seed_everything
import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-masked", action="store_true",
                        help="use stage 1 masked images instead of raw")
    args = parser.parse_args()

    seed_everything(config.RANDOM_SEED)

    train_df = pd.read_csv(config.V4_TURTLE_TRAIN_CSV)
    val_df = pd.read_csv(config.V4_TURTLE_VAL_CSV)

    log.info(f"train: {len(train_df)} images, {train_df['_id'].nunique()} identities")
    log.info(f"val: {len(val_df)} images, {val_df['_id'].nunique()} identities")
    log.info(f"shared identities (should be 0): "
             f"{len(set(train_df['_id']) & set(val_df['_id']))}")

    # Distribution of images per identity (matters for PK sampling)
    train_counts = Counter(train_df["_id"])
    counts_arr = np.array(list(train_counts.values()))
    log.info("")
    log.info("Train images-per-identity distribution:")
    log.info(f"  min={counts_arr.min()}, max={counts_arr.max()}, "
             f"median={int(np.median(counts_arr))}, mean={counts_arr.mean():.1f}")
    log.info(f"  identities with <4 images: {(counts_arr < 4).sum()}")
    log.info(f"  identities with <8 images: {(counts_arr < 8).sum()}")
    log.info(f"  identities with >20 images: {(counts_arr > 20).sum()}")

    # PK sampling feasibility
    min_p_per_batch = 8
    K_options = [4, 6, 8]
    log.info("")
    log.info("PK sampling feasibility:")
    for K in K_options:
        n_eligible = (counts_arr >= K).sum()
        log.info(f"  K={K}: {n_eligible}/{len(counts_arr)} identities have ≥K images")

    # Save config for stage 2b
    cfg = {
        "use_masked": args.use_masked,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_identities": int(train_df["_id"].nunique()),
        "min_imgs_per_id": int(counts_arr.min()),
        "max_imgs_per_id": int(counts_arr.max()),
        "median_imgs_per_id": int(np.median(counts_arr)),
        "recommended_K": 4,  # number of imgs per identity per batch
        "recommended_P": 8,  # number of identities per batch
        "batch_size": 32,    # P*K
    }

    out_path = config.V5_MODELS_DIR / "stage2a_data_config.json"
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    log.info(f"Config saved: {out_path}")
    log.info("")
    log.info(f"Recommended training batch size: {cfg['batch_size']} "
             f"(P={cfg['recommended_P']} × K={cfg['recommended_K']})")
    log.info("Next: stage2b_train_arcface.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
