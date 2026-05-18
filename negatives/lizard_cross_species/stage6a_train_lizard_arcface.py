#!/usr/bin/env python

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import DataLoader, Dataset

from utils import log, seed_everything, timed
import config

# Reuse pieces from stage2b
from stage2b_train_arcface import (
    MegaDescArcFace,
    PKSampler,
    build_eval_transforms,
    build_optimizer,
    build_train_transforms,
    ramp,
    set_trainable,
)


# =====================================================================
# Paths + globals
# =====================================================================

BALEARIC_ROOT = Path("/home/prouser1/Downloads/AnimalCLEF/data/balearic_lizard")
BALEARIC_META = BALEARIC_ROOT / "curt_metadata.csv"
BALEARIC_IMG_SUBDIR = "images-segmented"

OUT_NAME_DEFAULT = "stage6a_lizard_arcface_best.pth"


# =====================================================================
# Dataset
# =====================================================================

class LizardDataset(Dataset):
    """BalearicLizard images, label = integer class id (mapped from string id).

    Path resolution:
        metadata `path`: data/images/<id>/<stem>.jpg
        actual file:     images-segmented/<id>/<stem>.png
    We replace the prefix and swap the extension to .png.
    """
    def __init__(self, df: pd.DataFrame, transform, identity_to_class: dict):
        self.transform = transform
        self.paths: list[str] = []
        self.labels: list[int] = []
        for _, row in df.iterrows():
            rel = str(row["path"]).replace("data/images/", f"{BALEARIC_IMG_SUBDIR}/")
            rel = str(Path(rel).with_suffix(".png"))
            self.paths.append(str(BALEARIC_ROOT / rel))
            self.labels.append(int(identity_to_class[row["id"]]))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def remap_path(meta_path: str) -> Path:
    """Convert metadata path string to absolute filesystem path."""
    rel = str(meta_path).replace("data/images/", f"{BALEARIC_IMG_SUBDIR}/")
    rel = str(Path(rel).with_suffix(".png"))
    return BALEARIC_ROOT / rel


# =====================================================================
# Identity-based split
# =====================================================================

def identity_split(df: pd.DataFrame, train_frac: float, seed: int):
    """80/20 split by identity (no leakage)."""
    ids = sorted(df["id"].unique().tolist())
    rng = np.random.default_rng(seed)
    perm = ids.copy()
    rng.shuffle(perm)
    n_train = int(round(len(perm) * train_frac))
    train_ids = sorted(perm[:n_train])
    val_ids = sorted(perm[n_train:])
    return train_ids, val_ids


# =====================================================================
# Validation: extract embeddings + best threshold ARI
# =====================================================================

@torch.inference_mode()
def extract_lizard_embeddings(model, paths: list[str], transform, device) -> np.ndarray:
    from PIL import Image
    model.eval()
    bs = config.FEATURE_BATCH_SIZE
    embs: list[np.ndarray] = []
    for s in range(0, len(paths), bs):
        batch_paths = paths[s:s + bs]
        batch = torch.stack([
            transform(Image.open(p).convert("RGB")) for p in batch_paths
        ]).to(device, non_blocking=True)
        embs.append(model.extract_embeddings(batch).cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


def best_val_ari(embs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Sweep agglomerative threshold; return (best_ari, best_t)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    sim = np.clip(embs @ embs.T, 0.0, 1.0)
    dist = 1.0 - sim
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    best_ari = -1.0
    best_t = 0.5
    for t in np.arange(0.20, 0.901, 0.025):
        cids = fcluster(Z, t=float(t), criterion="distance")
        ari = adjusted_rand_score(labels, cids)
        if ari > best_ari:
            best_ari = float(ari)
            best_t = float(t)
    return best_ari, best_t


# =====================================================================
# Phase schedule (shorter than stage2b)
# =====================================================================

PHASES = [
    # name,          n_ep, bb_frac, lr_h,  lr_b,  s_st, s_en, m_st, m_en
    ("phase0_head",    2,   0.00,  1e-3,   0.0,   16,   16,  0.0,  0.0),
    ("phase1_top10",   5,   0.10,  1e-3,  1e-5,   16,   32,  0.0,  0.1),
    ("phase2_top30",  10,   0.30,  5e-4,  1e-5,   32,   64,  0.1,  0.3),
    ("phase3_all",     4,   1.00,  1e-4,  3e-6,   64,   64,  0.3,  0.3),
    ("phase4_cool",    2,   1.00,  1e-5,  1e-6,   64,   64,  0.3,  0.3),
]
TOTAL_EPOCHS = sum(p[1] for p in PHASES)


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-subcenter", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction,
                        default=True, help="mixed precision (faster)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=2)
    parser.add_argument("--out-name", default=OUT_NAME_DEFAULT)
    parser.add_argument("--check-images", action="store_true",
                        help="verify all image paths exist before training")
    parser.add_argument("--train-frac", type=float, default=0.80)
    args = parser.parse_args()

    seed_everything(config.RANDOM_SEED)
    device = config.DEVICE

    # ------------------------------------------------------------------
    # Load + filter metadata
    # ------------------------------------------------------------------
    if not BALEARIC_META.exists():
        log.error(f"missing metadata CSV: {BALEARIC_META}")
        return 1
    meta = pd.read_csv(BALEARIC_META)
    log.info(f"loaded metadata: {len(meta)} rows  "
             f"unique ids={meta['id'].nunique()}")

    # Drop singletons (need >= 2 images per identity for metric learning).
    counts = meta.groupby("id").size()
    multi_ids = set(counts[counts >= 2].index)
    n_drop = len(meta) - meta["id"].isin(multi_ids).sum()
    meta = meta[meta["id"].isin(multi_ids)].reset_index(drop=True)
    log.info(f"after dropping singletons: {len(meta)} rows  "
             f"unique ids={meta['id'].nunique()}  (dropped {n_drop} singleton rows)")

    # ------------------------------------------------------------------
    # Identity split
    # ------------------------------------------------------------------
    train_ids, val_ids = identity_split(meta, args.train_frac, config.RANDOM_SEED)
    log.info(f"identity split: {len(train_ids)} train ids / "
             f"{len(val_ids)} val ids  (no overlap)")
    assert not (set(train_ids) & set(val_ids)), "train/val identity overlap!"

    train_df = meta[meta["id"].isin(train_ids)].reset_index(drop=True)
    val_df = meta[meta["id"].isin(val_ids)].reset_index(drop=True)
    log.info(f"  train: {len(train_df)} images, {train_df['id'].nunique()} ids")
    log.info(f"  val:   {len(val_df)} images, {val_df['id'].nunique()} ids")

    identity_to_class = {iid: i for i, iid in enumerate(sorted(train_ids))}
    num_classes = len(identity_to_class)
    log.info(f"num_classes (train) = {num_classes}")

    # ------------------------------------------------------------------
    # Optional path verification
    # ------------------------------------------------------------------
    if args.check_images:
        log.info("--check-images: verifying every path on disk...")
        missing = []
        for _, row in meta.iterrows():
            p = remap_path(row["path"])
            if not p.exists():
                missing.append(str(p))
        if missing:
            log.error(f"missing {len(missing)} of {len(meta)} images. "
                      f"first 5: {missing[:5]}")
            return 1
        log.info(f"  all {len(meta)} images present on disk.")

    # ------------------------------------------------------------------
    # Build datasets / loaders
    # ------------------------------------------------------------------
    train_transform = build_train_transforms()
    eval_transform = build_eval_transforms()

    train_ds = LizardDataset(train_df, train_transform, identity_to_class)

    # PKSampler requires every label have at least num_instances samples;
    # but the sampler tolerates fewer (samples with replacement). We still
    # warn if many are below the threshold.
    label_counts = pd.Series(train_ds.labels).value_counts()
    low = (label_counts < args.num_instances).sum()
    log.info(f"  PK sampler: P={args.batch_size // args.num_instances} "
             f"K={args.num_instances}  "
             f"({low}/{len(label_counts)} train ids have < K images, "
             f"these will be sampled with replacement)")

    sampler = PKSampler(train_ds.labels, args.batch_size, args.num_instances)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    # Pre-resolve val image paths + integer labels (relative to val ids).
    val_paths = [str(remap_path(p)) for p in val_df["path"].tolist()]
    val_id_to_code = {iid: i for i, iid in enumerate(sorted(val_ids))}
    val_label_codes = np.asarray(
        [val_id_to_code[i] for i in val_df["id"].tolist()], dtype=np.int64,
    )
    log.info(f"  val: {len(val_paths)} images, "
             f"{len(set(val_label_codes.tolist()))} unique labels")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = MegaDescArcFace(
        num_classes=num_classes,
        use_subcenter=args.use_subcenter,
    ).to(device)

    # ------------------------------------------------------------------
    # Baseline val ARI (pretrained MegaD, no fine-tune)
    # ------------------------------------------------------------------
    log.info("")
    log.info("Baseline val ARI (pretrained MegaD, no fine-tune)...")
    with timed("baseline val embedding extraction"):
        baseline_embs = extract_lizard_embeddings(
            model, val_paths, eval_transform, device,
        )
    baseline_ari, baseline_t = best_val_ari(baseline_embs, val_label_codes)
    log.info(f"  baseline ARI = {baseline_ari:.4f} @ t={baseline_t:.3f}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_ari = baseline_ari
    ckpt_path = config.V5_CKPTS_DIR / args.out_name
    config.V5_CKPTS_DIR.mkdir(parents=True, exist_ok=True)

    scaler = torch.amp.GradScaler("cuda") if (args.use_amp and device == "cuda") else None

    log.info("")
    log.info(f"curriculum: {len(PHASES)} phases, total {TOTAL_EPOCHS} epochs")

    global_epoch = 0
    for phase_name, n_ep, bb_frac, lr_h, lr_b, s_st, s_en, m_st, m_en in PHASES:
        log.info("")
        log.info("=" * 70)
        log.info(f"{phase_name}: {n_ep} ep, bb={bb_frac*100:.0f}%, "
                 f"lr_h={lr_h:.0e}, lr_b={lr_b:.0e}, "
                 f"s={s_st}->{s_en}, m={m_st}->{m_en}")
        log.info("=" * 70)

        set_trainable(model, bb_frac)
        optimizer = build_optimizer(model, lr_h, lr_b)
        steps_per_epoch = len(train_loader)
        total_steps = max(1, steps_per_epoch * n_ep)
        warmup_steps = max(1, int(0.1 * total_steps))

        prev_loss = None
        step = 0
        for ep in range(n_ep):
            cur_s = ramp(ep, max(1, n_ep - 1), s_st, s_en)
            cur_m = ramp(ep, max(1, n_ep - 1), m_st, m_en)
            model.train()
            t0 = time.time()
            total_loss = 0.0
            total_correct = 0
            total_n = 0

            for imgs, labels in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # LR schedule (warmup then cosine)
                if step < warmup_steps:
                    lr_mult = step / max(1, warmup_steps)
                else:
                    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                    lr_mult = 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * prog))
                for g in optimizer.param_groups:
                    base = lr_h if g.get("name") == "head" else lr_b
                    g["lr"] = base * lr_mult
                step += 1

                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        logits, _ = model(imgs, labels=labels, s=cur_s, m=cur_m)
                        loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
                    if not torch.isfinite(loss).item():
                        log.error(f"NaN at ep {ep}; aborting phase")
                        break
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_norm=2.0,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits, _ = model(imgs, labels=labels, s=cur_s, m=cur_m)
                    loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
                    if not torch.isfinite(loss).item():
                        log.error(f"NaN at ep {ep}; aborting phase")
                        break
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_norm=2.0,
                    )
                    optimizer.step()

                total_loss += loss.item() * imgs.size(0)
                total_correct += (logits.argmax(dim=1) == labels).sum().item()
                total_n += imgs.size(0)

            ep_loss = total_loss / max(1, total_n)
            ep_acc = total_correct / max(1, total_n)
            dt = time.time() - t0
            log.info(f"  [ep {global_epoch:2d}] {phase_name} "
                     f"loss={ep_loss:.3f} acc={ep_acc:.3f} "
                     f"s={cur_s:.0f} m={cur_m:.2f} ({dt:.0f}s)")

            if prev_loss and ep_loss > 2.0 * prev_loss and ep_loss > 3.0:
                log.warning(f"  !! loss doubled: {prev_loss:.2f} -> {ep_loss:.2f}")
            prev_loss = ep_loss

            is_end = (ep == n_ep - 1)
            if (global_epoch % args.val_every == 0) or is_end:
                t_val = time.time()
                val_embs = extract_lizard_embeddings(
                    model, val_paths, eval_transform, device,
                )
                val_ari, val_t = best_val_ari(val_embs, val_label_codes)
                log.info(f"    [val @ ep {global_epoch}] ARI={val_ari:.4f} "
                         f"@ t={val_t:.3f}  (baseline {baseline_ari:.4f}) "
                         f"({time.time()-t_val:.0f}s)")
                if val_ari > best_ari:
                    best_ari = val_ari
                    torch.save({
                        "epoch": global_epoch,
                        "phase": phase_name,
                        "model_state_dict": model.state_dict(),
                        "val_ari": val_ari,
                        "val_threshold": val_t,
                        "num_classes": num_classes,
                        "use_subcenter": args.use_subcenter,
                        "use_masked": False,
                        "lizard_train_ids": list(sorted(train_ids)),
                        "lizard_val_ids": list(sorted(val_ids)),
                    }, ckpt_path)
                    log.info(f"    [ckpt] saved (val_ARI={val_ari:.4f})")

            global_epoch += 1

    log.info("")
    log.info(f"Done. Best val ARI: {best_ari:.4f} (baseline {baseline_ari:.4f})")
    log.info(f"  improvement: {best_ari - baseline_ari:+.4f}")
    log.info(f"  checkpoint: {ckpt_path}")
    log.info("Next: stage6b_extract_lizard_embeddings.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
