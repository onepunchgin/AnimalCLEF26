#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image

from utils import log, seed_everything, timed
import config


def load_sam_predictor(backend: str):
    """Load SAM predictor based on chosen backend."""
    if backend == "sam2":
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        return SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

    elif backend == "hf_sam":
        from transformers import SamModel, SamProcessor
        model = SamModel.from_pretrained("facebook/sam-vit-base").to(config.DEVICE)
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        return (model, processor)

    raise ValueError(f"unknown backend: {backend}")


def predict_mask_sam2(predictor, image_pil: Image.Image) -> tuple[np.ndarray, float]:
    """SAM2 prediction with center point prompt."""
    img_arr = np.array(image_pil)
    h, w = img_arr.shape[:2]
    predictor.set_image(img_arr)
    point_coords = np.array([[w//2, h//2]])
    point_labels = np.array([1])
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    best_idx = int(scores.argmax())
    return masks[best_idx].astype(bool), float(scores[best_idx])


def predict_mask_hf(predictor, image_pil: Image.Image) -> tuple[np.ndarray, float]:
    """HuggingFace SAM prediction with center point prompt."""
    model, processor = predictor
    w, h = image_pil.size
    inputs = processor(
        image_pil, input_points=[[[w//2, h//2]]], return_tensors="pt",
    ).to(config.DEVICE)
    with torch.inference_mode():
        outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    # Flatten scores across all dims (shape varies by transformers version)
    scores_flat = outputs.iou_scores.cpu().numpy().flatten()
    best_idx = int(np.argmax(scores_flat))
    # masks[0]: (num_points, num_masks, H, W) or (num_masks, H, W)
    masks_per_point = masks[0][0] if masks[0].ndim == 4 else masks[0]
    best_idx = min(best_idx, masks_per_point.shape[0] - 1)
    best_mask = masks_per_point[best_idx].numpy().astype(bool)
    return best_mask, float(scores_flat[best_idx])


def process_dataset(predictor, backend: str, df: pd.DataFrame,
                     out_dir: Path, label: str):
    """Run SAM on every image in df, save masks, return quality report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    quality_rows = []
    skipped = 0
    t0 = time.time()

    for enum_i, (i, row) in enumerate(df.iterrows()):
        img_path = row["_img"]
        # Use image_id if present, else use stem of file
        if "image_id" in df.columns:
            img_id = str(row["image_id"])
        else:
            img_id = Path(img_path).stem
        out_path = out_dir / f"{img_id}.png"

        # Skip if already processed
        if out_path.exists():
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            if backend == "sam2":
                mask, score = predict_mask_sam2(predictor, img)
            else:
                mask, score = predict_mask_hf(predictor, img)

            coverage = float(mask.sum() / mask.size)
            # Quality flags
            too_small = coverage < 0.05
            too_large = coverage > 0.95

            # Save as PNG (0/255)
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img.save(out_path)

            quality_rows.append({
                "image_id": img_id,
                "img_path": img_path,
                "score": score,
                "coverage": coverage,
                "too_small": too_small,
                "too_large": too_large,
            })

            if (enum_i + 1) % 100 == 0 or (enum_i + 1) == len(df):
                elapsed = time.time() - t0
                rate = (enum_i + 1 - skipped) / max(1, elapsed)
                eta_sec = (len(df) - enum_i - 1) / max(1, rate)
                log.info(f"  {label} {enum_i+1}/{len(df)}  "
                         f"({rate:.1f} img/s, ETA {eta_sec/60:.1f} min)")

        except Exception as e:
            log.error(f"  failed on {img_id}: {e}")
            quality_rows.append({
                "image_id": img_id, "img_path": img_path,
                "score": -1, "coverage": -1,
                "too_small": False, "too_large": False, "error": str(e),
            })

    log.info(f"  {label}: {len(df) - skipped} processed, {skipped} skipped (already done)")
    return pd.DataFrame(quality_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "sam2", "hf_sam"])
    parser.add_argument("--datasets", nargs="+",
                        default=["test_turtle", "db_turtle", "test_lizard"],
                        help="which datasets to process")
    args = parser.parse_args()

    seed_everything(config.RANDOM_SEED)

    # Auto-detect backend
    if args.backend == "auto":
        try:
            import sam2  # noqa: F401
            args.backend = "sam2"
        except ImportError:
            args.backend = "hf_sam"
    log.info(f"Using SAM backend: {args.backend}")

    log.info("Loading predictor...")
    with timed("predictor load"):
        predictor = load_sam_predictor(args.backend)

    # Datasets to process
    dataset_specs = {
        "test_turtle": (config.V4_TURTLE_TEST_CSV,
                          config.V5_MASKS_DIR / "test_turtle"),
        "db_turtle":   (config.V4_TURTLE_DB_CSV,
                          config.V5_MASKS_DIR / "db_turtle"),
        "test_lizard": (config.V4_LIZARD_TEST_CSV,
                          config.V5_MASKS_DIR / "test_lizard"),
        "val_turtle":  (config.V4_TURTLE_VAL_CSV,
                          config.V5_MASKS_DIR / "val_turtle"),
    }

    all_quality = {}
    for ds_name in args.datasets:
        if ds_name not in dataset_specs:
            log.warning(f"unknown dataset: {ds_name}, skipping")
            continue
        csv_path, mask_dir = dataset_specs[ds_name]
        if not csv_path.exists():
            log.error(f"missing CSV: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        log.info("")
        log.info("=" * 60)
        log.info(f"Processing {ds_name}: {len(df)} images")
        log.info("=" * 60)
        with timed(f"SAM on {ds_name}"):
            quality = process_dataset(predictor, args.backend, df,
                                        mask_dir, label=ds_name)
        all_quality[ds_name] = quality

        # Save quality report per dataset
        quality.to_csv(mask_dir / "_quality_report.csv", index=False)
        n_small = quality["too_small"].sum() if "too_small" in quality.columns else 0
        n_large = quality["too_large"].sum() if "too_large" in quality.columns else 0
        log.info(f"  quality: {len(quality)} masks, "
                 f"{n_small} too small, {n_large} too large")
        if "score" in quality.columns:
            valid = quality[quality["score"] > 0]
            if len(valid) > 0:
                log.info(f"  mean SAM IoU score: {valid['score'].mean():.3f}")
                log.info(f"  mean coverage: {valid['coverage'].mean()*100:.1f}%")

    log.info("")
    log.info("Stage 1b complete. Next: stage1c_apply_masks.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
