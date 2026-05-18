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
from torch.utils.data import DataLoader

from utils import log, seed_everything, timed
import config

from stage2b_train_arcface import (
    MegaDescArcFace,
    PKSampler,
    build_eval_transforms,
    build_optimizer,
    build_train_transforms,
    ramp,
    set_trainable,
)
from stage7a_train_species_arcface import (
    SpeciesDataset,
    best_val_ari,
    extract_embeddings,
    get_phases,
    identity_split,
    load_species_train,
)


# =====================================================================
# Paths
# =====================================================================

CZECH_BASE = Path("/home/prouser1/Downloads/AnimalCLEF/data/czech_lynx")
SPECIES = "LynxID2025"


# =====================================================================
# CzechLynx auto-discovery
# =====================================================================

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".webp")


def _discover_via_csv(base: Path) -> pd.DataFrame | None:
    """Try to locate a CSV with an identity column and resolve image paths.

    Returns a DataFrame with columns `_img` and `_id`, or None if no usable
    CSV is found.
    """
    id_col_candidates = ("identity", "id", "ID", "label", "class", "name")
    path_col_candidates = ("path", "image", "image_name", "filename", "file",
                            "img", "image_path", "Photo_name")

    csvs = sorted(base.rglob("*.csv"))
    for csv in csvs:
        try:
            df = pd.read_csv(csv)
        except Exception as e:
            log.warning(f"  could not read {csv}: {e}")
            continue
        cols = list(df.columns)
        id_col = next((c for c in id_col_candidates if c in cols), None)
        if id_col is None:
            continue
        path_col = next((c for c in path_col_candidates if c in cols), None)
        log.info(f"  CSV {csv.name}: id_col='{id_col}'  "
                 f"path_col='{path_col}'  rows={len(df)}")

        rows: list[dict] = []
        n_missing = 0
        for _, r in df.iterrows():
            ident = str(r[id_col])

            img_path: Path | None = None
            if path_col is not None:
                rel = str(r[path_col])
                # Try several plausible bases: the CSV's parent, the dataset
                # root, and the dataset root + 'images'.
                tries = [
                    csv.parent / rel,
                    base / rel,
                    base / "images" / rel,
                ]
                # Also try the same path with each common extension if
                # the CSV value is missing one.
                if Path(rel).suffix == "":
                    for ext in _IMG_EXTS:
                        tries.extend([
                            csv.parent / (rel + ext),
                            base / (rel + ext),
                            base / "images" / (rel + ext),
                        ])
                for t in tries:
                    if t.exists():
                        img_path = t
                        break

            # Fall back: search rglob by stem inside the dataset
            if img_path is None and path_col is not None:
                stem = Path(str(r[path_col])).stem
                if stem:
                    matches = list(base.rglob(stem + ".*"))
                    if matches:
                        img_path = matches[0]

            if img_path is None or not img_path.exists():
                n_missing += 1
                continue
            rows.append({"_img": str(img_path), "_id": f"czech_{ident}"})

        if rows:
            log.info(f"  resolved {len(rows)} images from {csv.name}  "
                     f"({n_missing} could not be resolved)")
            return pd.DataFrame(rows, columns=["_img", "_id"])
        log.warning(f"  CSV {csv.name} matched columns but resolved 0 paths")

    return None


def _discover_via_directories(base: Path) -> pd.DataFrame | None:
    """Each immediate subdir of `base` is treated as one identity.

    All image files anywhere beneath that subdir are collected.
    """
    if not base.exists():
        return None
    subdirs = [d for d in base.iterdir() if d.is_dir()]
    if not subdirs:
        return None

    rows: list[dict] = []
    for d in subdirs:
        # Skip plausibly non-identity directories.
        if d.name.lower() in {"metadata", "annotations", "labels"}:
            continue
        imgs: list[Path] = []
        for ext in _IMG_EXTS:
            imgs.extend(d.rglob(f"*{ext}"))
        for img in imgs:
            rows.append({"_img": str(img), "_id": f"czech_{d.name}"})

    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["_img", "_id"])
    log.info(f"  directory-based: {len(df)} images, "
             f"{df['_id'].nunique()} identities")
    return df


def discover_czech_lynx() -> pd.DataFrame | None:
    """Auto-discover CzechLynx structure. Returns None if nothing usable."""
    if not CZECH_BASE.exists():
        log.error(f"CzechLynx root missing: {CZECH_BASE}")
        return None

    log.info(f"Discovering CzechLynx layout under {CZECH_BASE}...")
    log.info(f"  contents: {[p.name for p in CZECH_BASE.iterdir()]}")

    df = _discover_via_csv(CZECH_BASE)
    if df is not None and len(df) > 0:
        return df

    df = _discover_via_directories(CZECH_BASE)
    if df is not None and len(df) > 0:
        return df

    log.error("Could not auto-discover CzechLynx structure")
    return None


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-subcenter",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-amp",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="mixed precision (faster)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=2)
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--min-imgs-per-id", type=int, default=2)
    parser.add_argument("--out-name", default="stage8c_lynx_extended_best.pth")
    args = parser.parse_args()

    seed_everything(config.RANDOM_SEED)
    device = config.DEVICE
    PHASES = get_phases(SPECIES)
    TOTAL_EPOCHS = sum(p[1] for p in PHASES)

    log.info("=" * 70)
    log.info("Stage 8c -- Lynx ArcFace re-training (LynxID2025 + CzechLynx)")
    log.info("=" * 70)
    log.info(f"device={device}  amp={args.use_amp}  subcenter={args.use_subcenter}")

    # ------------------------------------------------------------------
    # 1) LynxID2025 competition data + identity split (same seed as 7a)
    # ------------------------------------------------------------------
    lynx_filtered, _ = load_species_train(SPECIES, args.min_imgs_per_id)
    train_ids, val_ids = identity_split(
        lynx_filtered, args.train_frac, config.RANDOM_SEED,
    )
    log.info(f"LynxID2025 identity split: {len(train_ids)} train ids / "
             f"{len(val_ids)} val ids  (no overlap)")
    assert not (set(train_ids) & set(val_ids)), "train/val identity overlap!"

    lynx_train_df = lynx_filtered[
        lynx_filtered["_id"].isin(train_ids)
    ].reset_index(drop=True)
    lynx_val_df = lynx_filtered[
        lynx_filtered["_id"].isin(val_ids)
    ].reset_index(drop=True)
    log.info(f"  comp train: {len(lynx_train_df)} images, "
             f"{lynx_train_df['_id'].nunique()} ids")
    log.info(f"  comp val:   {len(lynx_val_df)} images, "
             f"{lynx_val_df['_id'].nunique()} ids")

    # Add the comp_ prefix so the namespace is shared with czech_
    lynx_train_df = lynx_train_df.copy()
    lynx_train_df["_id"] = lynx_train_df["_id"].apply(lambda x: f"comp_{x}")

    # ------------------------------------------------------------------
    # 2) CzechLynx (auto-discover)
    # ------------------------------------------------------------------
    czech_df = discover_czech_lynx()
    if czech_df is None or len(czech_df) == 0:
        log.error("CzechLynx data not available -- cannot run stage 8c")
        log.error("Wait for the download to finish, then re-run.")
        return 1
    log.info(f"CzechLynx: {len(czech_df)} images, "
             f"{czech_df['_id'].nunique()} identities")

    # Drop CzechLynx identities with < min_imgs_per_id images (they cannot
    # contribute to metric learning).
    czc_counts = czech_df["_id"].value_counts()
    keep_czech_ids = set(czc_counts[czc_counts >= args.min_imgs_per_id].index)
    n_drop = len(czech_df) - czech_df["_id"].isin(keep_czech_ids).sum()
    czech_df = czech_df[czech_df["_id"].isin(keep_czech_ids)].reset_index(drop=True)
    log.info(f"  after dropping czech singletons: {len(czech_df)} images, "
             f"{czech_df['_id'].nunique()} ids ({n_drop} rows dropped)")

    # ------------------------------------------------------------------
    # 3) Combined train (no leakage onto comp val ids)
    # ------------------------------------------------------------------
    train_df = pd.concat([lynx_train_df, czech_df], ignore_index=True)
    log.info(f"Combined train: {len(train_df)} images, "
             f"{train_df['_id'].nunique()} identities  "
             f"(comp={len(lynx_train_df)}, czech={len(czech_df)})")

    identity_to_class = {
        iid: i for i, iid in enumerate(sorted(train_df["_id"].unique()))
    }
    num_classes = len(identity_to_class)
    log.info(f"num_classes (combined train) = {num_classes}")

    # ------------------------------------------------------------------
    # 4) Stage 7a baseline val ARI for comparison
    # ------------------------------------------------------------------
    baseline_ckpt_path = (
        config.V5_CKPTS_DIR / f"stage7a_{SPECIES.lower()}_arcface_best.pth"
    )
    if baseline_ckpt_path.exists():
        baseline_ckpt = torch.load(
            baseline_ckpt_path, map_location="cpu", weights_only=False,
        )
        baseline_val_ari = float(baseline_ckpt.get("val_ari", 0.0))
        log.info(f"stage7a baseline val ARI (from ckpt): "
                 f"{baseline_val_ari:.4f}")
    else:
        baseline_val_ari = 0.0
        log.warning(f"stage7a checkpoint missing at {baseline_ckpt_path}; "
                    f"baseline_val_ari set to 0.0")

    # ------------------------------------------------------------------
    # 5) Datasets / loaders
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

    # Val: only LynxID2025 val identities (so ARI is comparable to stage 7a)
    val_paths = lynx_val_df["_img"].astype(str).tolist()
    val_id_to_code = {iid: i for i, iid in enumerate(sorted(val_ids))}
    val_label_codes = np.asarray(
        [val_id_to_code[i] for i in lynx_val_df["_id"].tolist()],
        dtype=np.int64,
    )
    log.info(f"  val: {len(val_paths)} images, "
             f"{len(set(val_label_codes.tolist()))} unique labels")

    # ------------------------------------------------------------------
    # 6) Model
    # ------------------------------------------------------------------
    model = MegaDescArcFace(
        num_classes=num_classes, use_subcenter=args.use_subcenter,
    ).to(device)

    # ------------------------------------------------------------------
    # 7) Pre-train val ARI
    # ------------------------------------------------------------------
    log.info("")
    log.info("Pre-train val ARI (pretrained MegaD, no fine-tune)...")
    with timed("pre-train val embedding extraction"):
        pre_embs = extract_embeddings(
            model, val_paths, eval_transform, device,
        )
    pre_ari, pre_t = best_val_ari(pre_embs, val_label_codes)
    log.info(f"  pretrained ARI = {pre_ari:.4f} @ t={pre_t:.3f}")

    # ------------------------------------------------------------------
    # 8) Training loop
    # ------------------------------------------------------------------
    best_ari = pre_ari
    best_t = pre_t
    ckpt_path = config.V5_CKPTS_DIR / args.out_name
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
                        loss = F.cross_entropy(logits, labels,
                                               label_smoothing=0.1)
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
                         f"@ t={val_t:.3f}  (stage7a baseline "
                         f"{baseline_val_ari:.4f}) "
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
                        "species": SPECIES,
                        "train_ids": list(sorted(train_ids)),
                        "val_ids": list(sorted(val_ids)),
                        "stage7a_baseline_val_ari": baseline_val_ari,
                        "n_train_images_combined": int(len(train_df)),
                        "n_train_identities_combined": int(num_classes),
                    }, ckpt_path)
                    log.info(f"    [ckpt] saved (val_ARI={val_ari:.4f})")

            global_epoch += 1

    # ------------------------------------------------------------------
    # 9) Final comparison
    # ------------------------------------------------------------------
    log.info("")
    log.info("=" * 70)
    log.info("FINAL COMPARISON")
    log.info("=" * 70)
    log.info(f"EXTENDED MODEL val ARI:  {best_ari:.4f} @ t={best_t:.3f}")
    log.info(f"STAGE7A BASELINE val ARI: {baseline_val_ari:.4f}")
    log.info(f"  delta: {best_ari - baseline_val_ari:+.4f}")
    if best_ari > baseline_val_ari:
        log.info("  -> USE extended model "
                 "(stage8c beats stage7a on the comparable val split)")
    else:
        log.info("  -> KEEP stage7a model "
                 "(extended model did NOT improve over stage7a baseline)")
    log.info(f"  checkpoint: {ckpt_path}")
    log.info("Next: stage8d_recluster_lynx.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
