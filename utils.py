from __future__ import annotations

import contextlib
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s\t%(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("v5")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@contextlib.contextmanager
def timed(label: str):
    log.info(f"[start]    {label}")
    t0 = time.time()
    yield
    dt = time.time() - t0
    if dt > 60:
        log.info(f"[done]     {label} in {dt/60:.1f} min")
    else:
        log.info(f"[done]     {label} in {dt:.1f}s")


def save_numpy(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    log.info(f"[saved]    {path.name}  shape={arr.shape}  dtype={arr.dtype}  "
             f"({arr.nbytes/1e6:.1f} MB)")


def write_submission(submission: pd.DataFrame, out_path: Path,
                      sample_submission: pd.DataFrame):
    """
    Validate and write submission CSV.
    Normalizes image_id to int64 to match sample_submission dtype.
    """
    submission = submission.copy()
    sample = sample_submission.copy()

    # Normalize dtypes
    try:
        submission["image_id"] = submission["image_id"].astype(np.int64)
    except (ValueError, OverflowError):
        log.warning("could not normalize image_id to int64; keeping str")
        submission["image_id"] = submission["image_id"].astype(str)
        sample["image_id"] = sample["image_id"].astype(str)
    else:
        sample["image_id"] = sample["image_id"].astype(np.int64)

    # Validate
    sample_ids = set(sample["image_id"])
    sub_ids = set(submission["image_id"])
    missing = sample_ids - sub_ids
    extra = sub_ids - sample_ids
    if missing or extra:
        raise ValueError(
            f"submission ids don't match sample. missing={len(missing)}, extra={len(extra)}"
        )

    # Reorder to match sample
    sub_indexed = submission.set_index("image_id").loc[sample["image_id"]]
    out = pd.DataFrame({
        "image_id": sample["image_id"].values,
        "cluster": sub_indexed["cluster"].values,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    log.info(f"[written]  submission {out_path.name}  ({len(out)} rows)")

    # Diagnostic per-species cluster counts
    for prefix in ("SeaTurtleID2022", "TexasHornedLizards",
                   "LynxID2025", "SalamanderID2025"):
        mask = out["cluster"].str.contains(f"cluster_{prefix}_")
        if mask.any():
            n_imgs = mask.sum()
            n_clusters = out.loc[mask, "cluster"].nunique()
            log.info(f"           cluster_{prefix}_*  {n_imgs} images, "
                     f"{n_clusters} unique clusters")


def synthesize_nonscored(sample: pd.DataFrame) -> pd.DataFrame:
    """Build singleton submissions for LynxID2025 + SalamanderID2025."""
    rows = []
    for _, row in sample.iterrows():
        sc = str(row["cluster"])
        for prefix in ("LynxID2025", "SalamanderID2025"):
            if f"cluster_{prefix}_" in sc:
                rows.append((row["image_id"], f"cluster_{prefix}_{row['image_id']}"))
                break
    return pd.DataFrame(rows, columns=["image_id", "cluster"])


def load_v4_csv(path):
    """Load a v4 subset CSV with verification."""
    if not path.exists():
        raise FileNotFoundError(f"v4 CSV missing: {path}")
    df = pd.read_csv(path)
    log.info(f"loaded {path.name}: {len(df)} rows, "
             f"identities={df['_id'].nunique() if '_id' in df.columns else 'n/a'}")
    return df


def cluster_agg(sim: np.ndarray, threshold: float, linkage_method: str = "average"):
    """Standard agglomerative clustering on similarity matrix."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    sim = np.clip(sim, 0.0, 1.0)
    dist = 1.0 - sim
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=linkage_method)
    return fcluster(Z, t=threshold, criterion="distance")


def build_test_matched_val_indices(val_df: pd.DataFrame, seed: int):
    """Build a single 'test-matched' val subsample index list."""
    rng = np.random.default_rng(seed)
    val_i = val_df.reset_index(drop=False).rename(columns={"index": "_o"})
    unique_ids = val_i["_id"].unique()
    rng.shuffle(unique_ids)
    n_new = int(len(unique_ids) * 0.3)
    new_ids = set(unique_ids[:n_new])
    keep = []
    for uid in unique_ids:
        rows = val_i[val_i["_id"] == uid]
        if uid in new_ids:
            keep.extend(rows.sample(n=1, random_state=seed)["_o"].tolist())
        else:
            keep.extend(rows.sample(n=min(5, len(rows)),
                                      random_state=seed)["_o"].tolist())
    return sorted(keep)
