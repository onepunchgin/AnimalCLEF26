#!/usr/bin/env python

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image

from utils import log


def test_sam2():
    """Try to load SAM2 from official Facebook repo."""
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.build_sam import build_sam2

        log.info("SAM2 imports successful")
        return "sam2", build_sam2, SAM2ImagePredictor
    except ImportError as e:
        log.warning(f"SAM2 not installed: {e}")
        log.warning("Install with: pip install sam2")
        return None, None, None


def test_sam_hf():
    """Try SAM via HuggingFace transformers (more accessible)."""
    try:
        from transformers import SamModel, SamProcessor
        log.info("HuggingFace SAM imports successful")
        return "hf_sam", SamModel, SamProcessor
    except ImportError as e:
        log.warning(f"HuggingFace SAM not available: {e}")
        return None, None, None


def main():
    import config
    log.info("=" * 60)
    log.info("Stage 1a — SAM Installation Verification")
    log.info("=" * 60)

    # Try SAM2 first (newer, faster, better)
    backend, builder, predictor_cls = test_sam2()

    if backend is None:
        # Fall back to SAM v1 via HuggingFace
        backend, model_cls, processor_cls = test_sam_hf()

    if backend is None:
        log.error("")
        log.error("No SAM backend available. Install one of:")
        log.error("  pip install sam2")
        log.error("  pip install transformers (for HuggingFace SAM)")
        log.error("  pip install segment-anything")
        return 1

    log.info(f"Backend: {backend}")

    # Test on a real turtle image
    import pandas as pd
    test_df = pd.read_csv(config.V4_TURTLE_TEST_CSV)
    sample_img_path = test_df["_img"].iloc[0]
    log.info(f"Testing on: {sample_img_path}")
    img = Image.open(sample_img_path).convert("RGB")
    log.info(f"  image size: {img.size}")

    if backend == "sam2":
        import torch
        log.info("Loading SAM2 model (this downloads ~700MB on first run)...")
        try:
            checkpoint = "facebook/sam2-hiera-large"
            predictor = predictor_cls.from_pretrained(checkpoint)
            predictor.set_image(np.array(img))
            # Auto-mask via center point prompt
            h, w = np.array(img).shape[:2]
            point_coords = np.array([[w//2, h//2]])
            point_labels = np.array([1])
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            log.info(f"  generated {len(masks)} masks, scores: {scores}")
            best_mask = masks[scores.argmax()]
            log.info(f"  best mask: shape={best_mask.shape}, "
                     f"coverage={best_mask.sum()/best_mask.size*100:.1f}%")
        except Exception as e:
            log.error(f"SAM2 inference failed: {e}")
            return 1

    elif backend == "hf_sam":
        import torch
        log.info("Loading HuggingFace SAM (sam-vit-base, ~375MB)...")
        try:
            model = model_cls.from_pretrained("facebook/sam-vit-base").to(config.DEVICE)
            processor = processor_cls.from_pretrained("facebook/sam-vit-base")
            h, w = img.size[1], img.size[0]
            inputs = processor(img, input_points=[[[w//2, h//2]]],
                                 return_tensors="pt").to(config.DEVICE)
            with torch.inference_mode():
                outputs = model(**inputs)
            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )
            scores = outputs.iou_scores.cpu().numpy()
            log.info(f"  HF SAM masks: {len(masks)} sets, scores: {scores}")
            best = masks[0][0][scores[0].argmax()].numpy()
            log.info(f"  best mask coverage: {best.sum()/best.size*100:.1f}%")
        except Exception as e:
            log.error(f"HF SAM inference failed: {e}")
            return 1

    log.info("")
    log.info("✓ SAM is working. You can proceed to stage1b.")
    log.info(f"  Backend chosen: {backend}")
    log.info(f"  Save this for stage1b: SAM_BACKEND='{backend}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
