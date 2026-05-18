#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch

from utils import log, save_numpy, seed_everything, timed
import config

# Reuse model class from stage2b
sys.path.insert(0, str(Path(__file__).parent))
from stage2b_train_arcface import (MegaDescArcFace, build_eval_transforms,
                                       extract_val_embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="stage2b_arcface_best.pth")
    parser.add_argument("--use-masked", action="store_true",
                        help="must match training; auto-detected from ckpt if available")
    args = parser.parse_args()

    seed_everything(config.RANDOM_SEED)

    ckpt_path = config.V5_CKPTS_DIR / args.ckpt
    if not ckpt_path.exists():
        log.error(f"missing {ckpt_path}")
        return 1

    log.info(f"loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
    num_classes = ckpt["num_classes"]
    use_subcenter = ckpt.get("use_subcenter", True)
    use_masked = ckpt.get("use_masked", args.use_masked)
    log.info(f"  num_classes={num_classes}, subcenter={use_subcenter}, "
             f"masked={use_masked}")
    log.info(f"  val_ari at save = {ckpt.get('val_ari', 'unknown')}")

    model = MegaDescArcFace(num_classes=num_classes,
                              use_subcenter=use_subcenter).to(config.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    transform = build_eval_transforms()

    # Process each split
    splits = {
        "val": config.V4_TURTLE_VAL_CSV,
        "test": config.V4_TURTLE_TEST_CSV,
        "db": config.V4_TURTLE_DB_CSV,
    }
    for name, csv_path in splits.items():
        if not csv_path.exists():
            log.warning(f"missing {csv_path}, skipping {name}")
            continue
        df = pd.read_csv(csv_path)
        log.info("")
        log.info(f"=== Extracting {name}: {len(df)} images ===")
        with timed(f"extract {name}"):
            embs = extract_val_embeddings(
                model, df, transform, config.DEVICE, use_masked)
        out_path = config.V5_FEATURES_DIR / f"stage2c_arcface_{name}_turtles.npy"
        save_numpy(embs, out_path)
        log.info(f"  shape={embs.shape}, "
                 f"norm range [{np.linalg.norm(embs, axis=1).min():.3f}, "
                 f"{np.linalg.norm(embs, axis=1).max():.3f}]")

    log.info("")
    log.info("Stage 2c complete. Next: stage2d_eval_arcface.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
