#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score

from utils import (log, save_numpy, seed_everything, timed,
                     write_submission, synthesize_nonscored,
                     cluster_agg, build_test_matched_val_indices)
import config


def get_masked_path(orig_path: str, image_id: str, split: str) -> str:
    """Get the masked-image path corresponding to original image."""
    masked = config.V5_MASKED_IMAGES_DIR / split / f"{image_id}.jpg"
    if not masked.exists():
        # Fall back to original if mask wasn't generated
        return orig_path
    return str(masked)


def extract_megadesc(image_paths, cache_path, label=""):
    if cache_path.exists():
        log.info(f"[cache] MegaD {label}")
        return np.load(cache_path)

    import timm
    import torchvision.transforms as T
    from PIL import Image

    log.info(f"extracting MegaD for {len(image_paths)} {label} images")
    model = timm.create_model(config.MEGADESC_HF_HUB_ID, num_classes=0,
                                pretrained=True).to(config.DEVICE).eval()
    tf = T.Compose([
        T.Resize(config.MEGADESC_INPUT_SIZE), T.ToTensor(),
        T.Normalize(config.MEGADESC_NORM_MEAN, config.MEGADESC_NORM_STD),
    ])
    embs = []
    bs = config.FEATURE_BATCH_SIZE
    with torch.inference_mode():
        for s in range(0, len(image_paths), bs):
            batch = torch.stack([tf(Image.open(p).convert("RGB"))
                                  for p in image_paths[s:s+bs]]).to(config.DEVICE)
            embs.append(model(batch).cpu().numpy())
            if (s // bs) % 10 == 0:
                log.info(f"  {label} {s+len(batch)}/{len(image_paths)}")
    embs = np.concatenate(embs, axis=0)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-8)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embs)
    del model
    torch.cuda.empty_cache()
    return embs


def extract_miew(image_paths, cache_path, label=""):
    if cache_path.exists():
        log.info(f"[cache] MIEW {label}")
        return np.load(cache_path)

    from transformers import AutoModel
    import torchvision.transforms as T
    from PIL import Image

    log.info(f"extracting MIEW for {len(image_paths)} {label} images")
    model = AutoModel.from_pretrained(config.MIEW_HF_HUB_ID,
                                         trust_remote_code=True)
    model = model.to(config.DEVICE).eval()
    tf = T.Compose([
        T.Resize((440, 440)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    embs = []
    bs = max(1, config.FEATURE_BATCH_SIZE // 2)
    with torch.inference_mode():
        for s in range(0, len(image_paths), bs):
            batch = torch.stack([tf(Image.open(p).convert("RGB"))
                                  for p in image_paths[s:s+bs]]).to(config.DEVICE)
            out = model(batch)
            emb = out if not isinstance(out, dict) else \
                  out.get("embedding", out.get("features", list(out.values())[0]))
            embs.append(emb.cpu().numpy())
            if (s // bs) % 10 == 0:
                log.info(f"  {label} {s+len(batch)}/{len(image_paths)}")
    embs = np.concatenate(embs, axis=0)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-8)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embs)
    del model
    torch.cuda.empty_cache()
    return embs


def fuse_sims(sims, weights):
    total = sum(weights)
    fused = sum((w/total) * s for s, w in zip(sims, weights))
    np.fill_diagonal(fused, 0.0)
    return np.clip(fused, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-name", default="stage1d_masked_fusion.csv")
    parser.add_argument("--w-miew", type=float, default=0.6,
                        help="step 3a optimal weight")
    parser.add_argument("--threshold", type=float, default=0.675,
                        help="step 3a optimal threshold")
    args = parser.parse_args()

    seed_everything(config.RANDOM_SEED)

    val_df = pd.read_csv(config.V4_TURTLE_VAL_CSV)
    test_df = pd.read_csv(config.V4_TURTLE_TEST_CSV)

    # Build masked-image paths
    log.info("Resolving masked-image paths...")
    val_paths = []
    for _, row in val_df.iterrows():
        iid = str(row["image_id"]) if "image_id" in val_df.columns \
              else Path(row["_img"]).stem
        val_paths.append(get_masked_path(row["_img"], iid, "val_turtle"))
    test_paths = []
    for _, row in test_df.iterrows():
        iid = str(row["image_id"]) if "image_id" in test_df.columns \
              else Path(row["_img"]).stem
        test_paths.append(get_masked_path(row["_img"], iid, "test_turtle"))

    n_val_masked = sum(1 for p in val_paths if "masked_images" in p)
    n_test_masked = sum(1 for p in test_paths if "masked_images" in p)
    log.info(f"  val: {n_val_masked}/{len(val_paths)} using masked variants")
    log.info(f"  test: {n_test_masked}/{len(test_paths)} using masked variants")
    if n_test_masked < len(test_paths) * 0.95:
        log.warning(f"  fewer than 95% of test images have masks; "
                     "consider re-running stage1b first")

    # Extract embeddings on masked variants
    val_md = extract_megadesc(
        val_paths, config.V5_FEATURES_DIR / "megadesc_masked_val_turtles.npy", "val")
    test_md = extract_megadesc(
        test_paths, config.V5_FEATURES_DIR / "megadesc_masked_test_turtles.npy", "test")
    val_miew = extract_miew(
        val_paths, config.V5_FEATURES_DIR / "miew_masked_val_turtles.npy", "val")
    test_miew = extract_miew(
        test_paths, config.V5_FEATURES_DIR / "miew_masked_test_turtles.npy", "test")

    # Tune (we know step 3a's optimum, but verify)
    log.info("")
    log.info("Tuning on test-matched val with masked embeddings...")
    weight_grid = [0.40, 0.50, 0.60, 0.70]
    threshold_grid = np.arange(0.55, 0.80, 0.025).tolist()

    results = []
    for split_i in range(3):
        keep = build_test_matched_val_indices(val_df, seed=42 + split_i)
        sub_md = val_md[keep]
        sub_miew = val_miew[keep]
        sub_labels = val_df.iloc[keep]["_id"].astype("category").cat.codes.values

        for w_miew in weight_grid:
            sim = fuse_sims([sub_md @ sub_md.T, sub_miew @ sub_miew.T],
                              [1 - w_miew, w_miew])
            best_t_ari = -1
            best_t = None
            for t in threshold_grid:
                cids = cluster_agg(sim, t)
                ari = adjusted_rand_score(sub_labels, cids)
                if ari > best_t_ari:
                    best_t_ari = ari
                    best_t = t
            results.append({"split": split_i, "w_miew": w_miew,
                             "threshold": best_t, "ari": best_t_ari})

    res = pd.DataFrame(results)
    agg = res.groupby("w_miew").agg(
        mean_ari=("ari", "mean"), std_ari=("ari", "std"),
        median_t=("threshold", "median"),
    ).reset_index().sort_values("mean_ari", ascending=False)
    log.info("\n" + agg.to_string(index=False))
    best = agg.iloc[0]
    best_w_miew = float(best["w_miew"])
    best_t = float(best["median_t"])
    log.info(f"BEST (masked): w_miew={best_w_miew}, t={best_t}, "
             f"val_ARI={best['mean_ari']:.4f}")

    # Apply to test
    sim_md_t = test_md @ test_md.T
    sim_miew_t = test_miew @ test_miew.T
    fused = fuse_sims([sim_md_t, sim_miew_t],
                        [1 - best_w_miew, best_w_miew])
    save_numpy(fused.astype(np.float32),
                config.V5_SIMILARITY_DIR / "stage1d_masked_fusion_sim.npy")

    cluster_ids = cluster_agg(fused, best_t)
    n_clusters = len(np.unique(cluster_ids))
    log.info(f"test: {len(test_df)} → {n_clusters} clusters")

    # Build submission
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
    log.info("")
    log.info(f"Submission: {out_path}")
    log.info(f"Val ARI (masked): {best['mean_ari']:.4f}")
    log.info("")
    log.info("Compare to v4 step 3a baseline (Kaggle 0.17, val 0.88).")
    log.info("Decision rule:")
    log.info("  Kaggle >= 0.19 → segmentation helps, proceed to stage 2")
    log.info("  Kaggle 0.15-0.18 → marginal, your call")
    log.info("  Kaggle < 0.15 → segmentation hurt, revert to raw for stage 2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
