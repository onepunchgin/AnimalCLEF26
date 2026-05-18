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

# Reuse the (battle-tested) pieces from stage 2b
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
# Constants
# =====================================================================

COMP_BASE = config.COMPETITION_ROOT
SUPPORTED_SPECIES = ("LynxID2025", "SalamanderID2025")


# =====================================================================
# Phase schedules (per species)
# =====================================================================

# LynxID2025: 77 ids, ~2957 imgs, median 17 imgs/id  -- rich data, longer
PHASES_LYNX = [
    # name,           n_ep, bb_frac, lr_h,  lr_b,  s_st, s_en, m_st, m_en
    ("phase0_head",     3,   0.00,  1e-3,   0.0,   16,   16,  0.0,  0.0),
    ("phase1_top10",    8,   0.10,  1e-3,  1e-5,   16,   32,  0.0,  0.1),
    ("phase2_top30",   12,   0.30,  5e-4,  1e-5,   32,   64,  0.1,  0.3),
    ("phase3_all",      5,   1.00,  1e-4,  3e-6,   64,   64,  0.3,  0.3),
    ("phase4_cool",     2,   1.00,  1e-5,  1e-6,   64,   64,  0.3,  0.3),
]  # 30 epochs total

# SalamanderID2025: 587 ids, ~1388 imgs, median 1 img/id -- sparse, shorter
PHASES_SALAMANDER = [
    # name,           n_ep, bb_frac, lr_h,  lr_b,  s_st, s_en, m_st, m_en
    ("phase0_head",     2,   0.00,  1e-3,   0.0,   16,   16,  0.0,  0.0),
    ("phase1_top10",    5,   0.10,  1e-3,  1e-5,   16,   32,  0.0,  0.1),
    ("phase2_top30",    8,   0.30,  5e-4,  1e-5,   32,   64,  0.1,  0.3),
    ("phase3_all",      3,   1.00,  1e-4,  3e-6,   64,   64,  0.3,  0.3),
    ("phase4_cool",     2,   1.00,  1e-5,  1e-6,   64,   64,  0.3,  0.3),
]  # 20 epochs total


def get_phases(species: str):
    if species == "LynxID2025":
        return PHASES_LYNX
    if species == "SalamanderID2025":
        return PHASES_SALAMANDER
    raise ValueError(f"unsupported species: {species}")


# =====================================================================
# Data loading
# =====================================================================

def load_species_train(species: str, min_imgs_per_id: int = 2):
    """
    Load training metadata for a species, filter to identities with enough
    images for metric learning, and resolve absolute image paths.

    Returns
    -------
    train_filtered : pd.DataFrame
        Rows whose `_id` has >= min_imgs_per_id images. Used for
        train + val splits.
    train_full : pd.DataFrame
        All training rows for the species (including singletons). Useful
        as a reference DB at clustering time.
    """
    meta_path = COMP_BASE / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv missing at {meta_path}")
    meta = pd.read_csv(meta_path)

    train = meta[(meta["dataset"] == species) & (meta["split"] == "train")].copy()
    if len(train) == 0:
        raise RuntimeError(f"no training rows for species={species}")
    train["_img"] = train["path"].apply(lambda p: str(COMP_BASE / p))
    train["_id"] = train["identity"].astype(str)

    counts = train["_id"].value_counts()
    valid_ids = counts[counts >= min_imgs_per_id].index
    train_filtered = train[train["_id"].isin(valid_ids)].reset_index(drop=True)

    n_total = len(train)
    n_kept = len(train_filtered)
    n_total_ids = train["_id"].nunique()
    n_kept_ids = train_filtered["_id"].nunique()
    log.info(
        f"[{species}] train rows: {n_total} ({n_total_ids} ids) "
        f"-> {n_kept} ({n_kept_ids} ids with >= {min_imgs_per_id} imgs)"
    )
    return train_filtered, train


def identity_split(df: pd.DataFrame, train_frac: float, seed: int):
    """80/20 split by identity (no overlap)."""
    ids = sorted(df["_id"].unique().tolist())
    rng = np.random.default_rng(seed)
    perm = ids.copy()
    rng.shuffle(perm)
    n_train = int(round(len(perm) * train_frac))
    train_ids = sorted(perm[:n_train])
    val_ids = sorted(perm[n_train:])
    return train_ids, val_ids


# =====================================================================
# Dataset
# =====================================================================

class SpeciesDataset(Dataset):
    """Generic competition-species dataset that uses absolute `_img` paths."""

    def __init__(self, df: pd.DataFrame, transform, identity_to_class: dict):
        self.transform = transform
        self.paths: list[str] = df["_img"].astype(str).tolist()
        self.labels: list[int] = [
            int(identity_to_class[i]) for i in df["_id"].tolist()
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


# =====================================================================
# Validation utilities
# =====================================================================

@torch.inference_mode()
def extract_embeddings(model, paths: list[str], transform, device) -> np.ndarray:
    from PIL import Image
    model.eval()
    bs = config.FEATURE_BATCH_SIZE
    embs: list[np.ndarray] = []
    for s in range(0, len(paths), bs):
        batch = torch.stack([
            transform(Image.open(p).convert("RGB")) for p in paths[s:s + bs]
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
# Main
# =====================================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True, choices=SUPPORTED_SPECIES)
    parser.add_argument("--use-subcenter", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction,
                        default=True, help="mixed precision (faster)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=2)
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--min-imgs-per-id", type=int, default=2)
    args = parser.parse_args()

    seed_everything(config.RANDOM_SEED)
    device = config.DEVICE
    species = args.species
    PHASES = get_phases(species)
    TOTAL_EPOCHS = sum(p[1] for p in PHASES)

    log.info("=" * 70)
    log.info(f"Stage 7a -- Train ArcFace on {species}")
    log.info("=" * 70)
    log.info(f"device={device}  amp={args.use_amp}  subcenter={args.use_subcenter}")
    log.info(f"batch_size={args.batch_size}  K={args.num_instances}  "
             f"val_every={args.val_every}")

    # ------------------------------------------------------------------
    # Load + filter metadata (drop singletons for metric learning)
    # ------------------------------------------------------------------
    train_filtered, train_full = load_species_train(species, args.min_imgs_per_id)

    # ------------------------------------------------------------------
    # Identity split
    # ------------------------------------------------------------------
    train_ids, val_ids = identity_split(
        train_filtered, args.train_frac, config.RANDOM_SEED,
    )
    log.info(f"identity split: {len(train_ids)} train ids / "
             f"{len(val_ids)} val ids  (no overlap)")
    assert not (set(train_ids) & set(val_ids)), "train/val identity overlap!"

    train_df = train_filtered[
        train_filtered["_id"].isin(train_ids)
    ].reset_index(drop=True)
    val_df = train_filtered[
        train_filtered["_id"].isin(val_ids)
    ].reset_index(drop=True)
    log.info(f"  train: {len(train_df)} images, {train_df['_id'].nunique()} ids")
    log.info(f"  val:   {len(val_df)} images, {val_df['_id'].nunique()} ids")

    if len(train_df) == 0:
        log.error("empty train split -- aborting")
        return 1
    if len(val_df) == 0:
        log.error("empty val split -- aborting")
        return 1

    identity_to_class = {iid: i for i, iid in enumerate(sorted(train_ids))}
    num_classes = len(identity_to_class)
    log.info(f"num_classes (train) = {num_classes}")

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    train_transform = build_train_transforms()
    eval_transform = build_eval_transforms()

    train_ds = SpeciesDataset(train_df, train_transform, identity_to_class)
    label_counts = pd.Series(train_ds.labels).value_counts()
    low = (label_counts < args.num_instances).sum()
    log.info(f"  PK sampler: P={args.batch_size // args.num_instances} "
             f"K={args.num_instances}  "
             f"({low}/{len(label_counts)} train ids have < K images, "
             f"sampled with replacement)")

    sampler = PKSampler(train_ds.labels, args.batch_size, args.num_instances)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    # Pre-resolve val image paths and integer labels (within val ids only)
    val_paths = val_df["_img"].astype(str).tolist()
    val_id_to_code = {iid: i for i, iid in enumerate(sorted(val_ids))}
    val_label_codes = np.asarray(
        [val_id_to_code[i] for i in val_df["_id"].tolist()], dtype=np.int64,
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
        baseline_embs = extract_embeddings(
            model, val_paths, eval_transform, device,
        )
    baseline_ari, baseline_t = best_val_ari(baseline_embs, val_label_codes)
    log.info(f"  baseline ARI = {baseline_ari:.4f} @ t={baseline_t:.3f}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_ari = baseline_ari
    best_t = baseline_t
    ckpt_path = (
        config.V5_CKPTS_DIR / f"stage7a_{species.lower()}_arcface_best.pth"
    )
    config.V5_CKPTS_DIR.mkdir(parents=True, exist_ok=True)

    scaler = (
        torch.amp.GradScaler("cuda")
        if (args.use_amp and device == "cuda")
        else None
    )

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

                # LR schedule (warmup -> cosine)
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
                val_embs = extract_embeddings(
                    model, val_paths, eval_transform, device,
                )
                val_ari, val_t = best_val_ari(val_embs, val_label_codes)
                log.info(f"    [val @ ep {global_epoch}] ARI={val_ari:.4f} "
                         f"@ t={val_t:.3f}  (baseline {baseline_ari:.4f}) "
                         f"({time.time()-t_val:.0f}s)")
                if val_ari > best_ari:
                    best_ari = val_ari
                    best_t = val_t
                    torch.save({
                        "epoch": global_epoch,
                        "phase": phase_name,
                        "model_state_dict": model.state_dict(),
                        "val_ari": val_ari,
                        "val_threshold": val_t,
                        "num_classes": num_classes,
                        "use_subcenter": args.use_subcenter,
                        "use_masked": False,
                        "species": species,
                        "train_ids": list(sorted(train_ids)),
                        "val_ids": list(sorted(val_ids)),
                    }, ckpt_path)
                    log.info(f"    [ckpt] saved (val_ARI={val_ari:.4f})")

            global_epoch += 1

    log.info("")
    log.info(f"Done. Best val ARI: {best_ari:.4f} @ t={best_t:.3f} "
             f"(baseline {baseline_ari:.4f})")
    log.info(f"  improvement: {best_ari - baseline_ari:+.4f}")
    log.info(f"  checkpoint: {ckpt_path}")
    log.info(f"Next: stage7b_extract_and_cluster.py --species {species}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
