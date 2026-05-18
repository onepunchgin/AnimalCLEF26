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

from stage2b_train_arcface import MegaDescArcFace, build_eval_transforms
from stage6a_train_lizard_arcface import (
    BALEARIC_META,
    BALEARIC_ROOT,
    extract_lizard_embeddings,
    remap_path,
)


CKPT_NAME_DEFAULT = "stage6a_lizard_arcface_best.pth"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=CKPT_NAME_DEFAULT)
    args = parser.parse_args()

    seed_everything(config.RANDOM_SEED)
    device = config.DEVICE

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    ckpt_path = config.V5_CKPTS_DIR / args.ckpt
    if not ckpt_path.exists():
        log.error(f"missing checkpoint: {ckpt_path}")
        log.error("run stage6a_train_lizard_arcface.py first")
        return 1
    log.info(f"loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = int(ckpt["num_classes"])
    use_subcenter = bool(ckpt.get("use_subcenter", True))
    val_ari_at_save = ckpt.get("val_ari", "unknown")
    val_t_at_save = ckpt.get("val_threshold", "unknown")
    log.info(f"  num_classes={num_classes}  subcenter={use_subcenter}")
    log.info(f"  val_ari@save={val_ari_at_save}  val_t@save={val_t_at_save}")

    val_ids = ckpt.get("lizard_val_ids")
    if val_ids is None:
        log.error("checkpoint missing 'lizard_val_ids' — re-run stage6a")
        return 1
    val_ids = list(val_ids)
    log.info(f"  lizard_val_ids: {len(val_ids)} identities")

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
    # 1) BalearicLizard val embeddings
    # ------------------------------------------------------------------
    if not BALEARIC_META.exists():
        log.error(f"missing BalearicLizard metadata: {BALEARIC_META}")
        return 1
    meta = pd.read_csv(BALEARIC_META)
    val_set = set(val_ids)
    val_df = meta[meta["id"].isin(val_set)].reset_index(drop=True)
    log.info(f"BalearicLizard val: {len(val_df)} images "
             f"across {val_df['id'].nunique()} identities")

    val_paths = [str(remap_path(p)) for p in val_df["path"].tolist()]
    missing = [p for p in val_paths if not Path(p).exists()]
    if missing:
        log.error(f"missing {len(missing)}/{len(val_paths)} val image files. "
                  f"first 3: {missing[:3]}")
        return 1

    with timed("extract balearic val embeddings"):
        val_embs = extract_lizard_embeddings(model, val_paths, transform, device)
    log.info(f"  shape={val_embs.shape}  "
             f"norm range [{np.linalg.norm(val_embs, axis=1).min():.3f}, "
             f"{np.linalg.norm(val_embs, axis=1).max():.3f}]")

    val_id_strs = np.asarray(val_df["id"].astype(str).tolist())
    sorted_val_ids = sorted(val_set)
    val_id_to_code = {iid: i for i, iid in enumerate(sorted_val_ids)}
    val_codes = np.asarray(
        [val_id_to_code[i] for i in val_df["id"].tolist()], dtype=np.int64,
    )

    save_numpy(val_embs, config.V5_FEATURES_DIR / "stage6_lizard_balearic_val_emb.npy")
    save_numpy(val_id_strs, config.V5_FEATURES_DIR / "stage6_lizard_balearic_val_ids.npy")
    save_numpy(val_codes, config.V5_FEATURES_DIR / "stage6_lizard_balearic_val_codes.npy")

    # ------------------------------------------------------------------
    # 2) TexasHornedLizard test embeddings
    # ------------------------------------------------------------------
    if not config.V4_LIZARD_TEST_CSV.exists():
        log.error(f"missing lizard test CSV: {config.V4_LIZARD_TEST_CSV}")
        return 1
    texas_df = pd.read_csv(config.V4_LIZARD_TEST_CSV)
    if "_img" not in texas_df.columns:
        log.error("expected '_img' column in lizard test CSV")
        return 1
    texas_paths = [str(p) for p in texas_df["_img"].astype(str).tolist()]
    log.info(f"TexasHornedLizard test: {len(texas_paths)} images")

    missing = [p for p in texas_paths if not Path(p).exists()]
    if missing:
        log.error(f"missing {len(missing)}/{len(texas_paths)} test image files. "
                  f"first 3: {missing[:3]}")
        return 1

    with timed("extract texas test embeddings"):
        texas_embs = extract_lizard_embeddings(model, texas_paths, transform, device)
    log.info(f"  shape={texas_embs.shape}  "
             f"norm range [{np.linalg.norm(texas_embs, axis=1).min():.3f}, "
             f"{np.linalg.norm(texas_embs, axis=1).max():.3f}]")

    save_numpy(texas_embs, config.V5_FEATURES_DIR / "stage6_lizard_texas_test_emb.npy")

    log.info("")
    log.info("Stage 6b complete. Next: stage6c_lizard_cluster_and_submit.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
