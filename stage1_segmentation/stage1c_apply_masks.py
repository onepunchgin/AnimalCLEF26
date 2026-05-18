#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from PIL import Image

from utils import log, seed_everything, timed
import config


def apply_mask_to_image(img: Image.Image, mask: np.ndarray,
                          bg_color: tuple = (128, 128, 128)) -> Image.Image:
    """Apply binary mask, fill background with bg_color."""
    img_arr = np.array(img)
    h, w = img_arr.shape[:2]

    # Resize mask if dimensions don't match (rare, but defensive)
    if mask.shape != (h, w):
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.resize((w, h), Image.NEAREST)
        mask = np.array(mask_img) > 127

    # Background fill
    bg = np.full_like(img_arr, fill_value=bg_color)
    out = np.where(mask[..., None], img_arr, bg).astype(np.uint8)
    return Image.fromarray(out)


def process_dataset(df: pd.DataFrame, mask_dir: Path,
                     out_dir: Path, label: str):
    """Apply masks to all images in df, save masked variants."""
    out_dir.mkdir(parents=True, exist_ok=True)
    skipped = 0
    failed = 0
    t0 = time.time()

    for enum_i, (i, row) in enumerate(df.iterrows()):
        img_path = row["_img"]
        if "image_id" in df.columns:
            img_id = str(row["image_id"])
        else:
            img_id = Path(img_path).stem
        mask_path = mask_dir / f"{img_id}.png"
        out_path = out_dir / f"{img_id}.jpg"

        if out_path.exists():
            skipped += 1
            continue
        if not mask_path.exists():
            log.warning(f"  no mask for {img_id}, skipping")
            failed += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            mask_img = Image.open(mask_path).convert("L")
            mask = np.array(mask_img) > 127

            masked = apply_mask_to_image(img, mask)
            masked.save(out_path, quality=92)
        except Exception as e:
            log.error(f"  failed {img_id}: {e}")
            failed += 1

        if (enum_i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (enum_i + 1 - skipped) / max(1, elapsed)
            log.info(f"  {label} {enum_i+1}/{len(df)} ({rate:.0f} img/s)")

    log.info(f"  {label}: {len(df) - skipped - failed} new, "
             f"{skipped} cached, {failed} failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["test_turtle", "db_turtle", "test_lizard",
                                 "val_turtle"])
    args = parser.parse_args()

    seed_everything(config.RANDOM_SEED)

    dataset_specs = {
        "test_turtle": (config.V4_TURTLE_TEST_CSV,
                          config.V5_MASKS_DIR / "test_turtle",
                          config.V5_MASKED_IMAGES_DIR / "test_turtle"),
        "db_turtle":   (config.V4_TURTLE_DB_CSV,
                          config.V5_MASKS_DIR / "db_turtle",
                          config.V5_MASKED_IMAGES_DIR / "db_turtle"),
        "test_lizard": (config.V4_LIZARD_TEST_CSV,
                          config.V5_MASKS_DIR / "test_lizard",
                          config.V5_MASKED_IMAGES_DIR / "test_lizard"),
        "val_turtle":  (config.V4_TURTLE_VAL_CSV,
                          config.V5_MASKS_DIR / "val_turtle",
                          config.V5_MASKED_IMAGES_DIR / "val_turtle"),
    }

    for ds_name in args.datasets:
        if ds_name not in dataset_specs:
            continue
        csv_path, mask_dir, out_dir = dataset_specs[ds_name]
        if not csv_path.exists():
            log.error(f"missing {csv_path}")
            continue
        if not mask_dir.exists():
            log.error(f"missing masks dir {mask_dir}; run stage1b first")
            continue

        df = pd.read_csv(csv_path)
        log.info("")
        log.info(f"=== {ds_name}: {len(df)} images ===")
        with timed(f"apply masks {ds_name}"):
            process_dataset(df, mask_dir, out_dir, label=ds_name)

    log.info("")
    log.info("Stage 1c complete. Next: stage1d_eval_segmentation.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
