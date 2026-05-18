from pathlib import Path
import torch

# =====================================================================
# Roots — EDIT THESE for your machine
# =====================================================================

V4_ROOT = Path("/home/prouser1/Downloads/AnimalCLEF/pipeline_v4")
V5_ROOT = Path("/home/prouser1/Downloads/AnimalCLEF/pipeline_v5")
DATA_ROOT = Path("/home/prouser1/Downloads/AnimalCLEF/data")

COMPETITION_ROOT = DATA_ROOT / "competition" / "animal-clef-2026"
WR10K_ROOT = DATA_ROOT / "wildlifereid10k"

# =====================================================================
# v4 caches we reuse (read-only references)
# =====================================================================

V4_FEATURES = V4_ROOT / "artifacts" / "features"
V4_SIMILARITY = V4_ROOT / "artifacts" / "similarity"
V4_SUBMISSIONS = V4_ROOT / "artifacts" / "submissions"

# Specific v4 artifacts we reuse
V4_BEST_FUSION_SIM = V4_SIMILARITY / "step3a_test_turtle_fused_sim.npy"
V4_TURTLE_TRAIN_CSV = V4_FEATURES / "subset_turtle_train.csv"
V4_TURTLE_VAL_CSV = V4_FEATURES / "subset_turtle_val.csv"
V4_TURTLE_TEST_CSV = V4_FEATURES / "subset_turtle_query.csv"
V4_TURTLE_DB_CSV = V4_FEATURES / "subset_turtle_db.csv"
V4_LIZARD_TEST_CSV = V4_FEATURES / "subset_lizard_query.csv"

# =====================================================================
# v5 outputs
# =====================================================================

V5_ARTIFACTS = V5_ROOT / "artifacts"
V5_MASKS_DIR = V5_ARTIFACTS / "masks"           # PNG masks per image
V5_MASKED_IMAGES_DIR = V5_ARTIFACTS / "masked_images"  # masked image variants
V5_FEATURES_DIR = V5_ARTIFACTS / "features"      # v5 embeddings
V5_SIMILARITY_DIR = V5_ARTIFACTS / "similarity"  # v5 sim matrices
V5_CKPTS_DIR = V5_ARTIFACTS / "ckpts"            # ArcFace checkpoints
V5_PAIR_FEATURES_DIR = V5_ARTIFACTS / "pair_features"  # XGBoost features
V5_MODELS_DIR = V5_ARTIFACTS / "models"          # XGBoost models
V5_SUBMISSIONS_DIR = V5_ROOT / "submission"
V5_LOGS_DIR = V5_ROOT / "submission"

for _d in (V5_ARTIFACTS, V5_MASKS_DIR, V5_MASKED_IMAGES_DIR, V5_FEATURES_DIR,
           V5_SIMILARITY_DIR, V5_CKPTS_DIR, V5_PAIR_FEATURES_DIR,
           V5_MODELS_DIR, V5_SUBMISSIONS_DIR, V5_LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

COMPETITION_SAMPLE_SUB = COMPETITION_ROOT / "sample_submission.csv"

# =====================================================================
# Models
# =====================================================================

MEGADESC_HF_HUB_ID = "hf-hub:BVRA/MegaDescriptor-L-384"
MEGADESC_INPUT_SIZE = (384, 384)
MEGADESC_NORM_MEAN = (0.485, 0.456, 0.406)
MEGADESC_NORM_STD = (0.229, 0.224, 0.225)

MIEW_HF_HUB_ID = "conservationxlabs/miewid-msv3"
DINOV2_HF_HUB_ID = "facebook/dinov2-large"

# SAM2 / SAM
SAM2_MODEL_ID = "facebook/sam2-hiera-large"      # heavier, more accurate
SAM2_MODEL_ID_SMALL = "facebook/sam2-hiera-small" # faster, less accurate

# =====================================================================
# Compute
# =====================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 8
FEATURE_BATCH_SIZE = 32
RANDOM_SEED = 42

# Memory / GPU
# Use garbage_collection_threshold (NOT max_split_size_mb which broke step 8)
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6")
