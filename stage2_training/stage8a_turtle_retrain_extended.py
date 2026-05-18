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

from utils import (
    build_test_matched_val_indices,
    log,
    seed_everything,
    timed,
)
import config

# Reuse the (battle-tested) pieces from stage 2b
from stage2b_train_arcface import (
    MegaDescArcFace,
    PKSampler,
    TurtleDataset,
    build_eval_transforms,
    build_optimizer,
    build_train_transforms,
    evaluate_val_ari,
    extract_val_embeddings,
    ramp,
    set_trainable,
    PHASES,
)


# =====================================================================
# Paths
# =====================================================================

COMP_BASE = config.COMPETITION_ROOT
ZAKYNTHOS_BASE = Path("/home/prouser1/Downloads/AnimalCLEF/data/zakynthos_turtles")
AMVRAKIKOS_BASE = Path("/home/prouser1/Downloads/AnimalCLEF/data/amvrakikos_turtles")
REUNION_BASE = Path("/home/prouser1/Downloads/AnimalCLEF/data/reunion_turtles")


# =====================================================================
# Combined turtle training data
# =====================================================================

def load_all_turtle_train() -> pd.DataFrame:
    """Build a combined train DataFrame with columns `_img` and `_id`.

    Identity strings are namespaced with a per-dataset prefix to avoid
    cross-dataset collisions (e.g. `comp_SeaTurtleID2022_t002`, `zak_42`).
    """
    rows: list[dict] = []

    # 1. SeaTurtleID2022 competition train (already used in stage2b)
    meta_path = COMP_BASE / "metadata.csv"
    if not meta_path.exists():
        log.error(f"missing competition metadata: {meta_path}")
        return pd.DataFrame(columns=["_img", "_id"])
    meta = pd.read_csv(meta_path)
    comp = meta[(meta["dataset"] == "SeaTurtleID2022")
                & (meta["split"] == "train")]
    n_comp = 0
    for _, r in comp.iterrows():
        img = COMP_BASE / r["path"]
        rows.append({"_img": str(img), "_id": f"comp_{r['identity']}"})
        n_comp += 1
    log.info(f"  comp SeaTurtleID2022 train: {n_comp} rows")

    # 2. ZakynthosTurtles
    n_zak = 0
    n_zak_missing = 0
    zak_csv = ZAKYNTHOS_BASE / "annotations.csv"
    if zak_csv.exists():
        zak = pd.read_csv(zak_csv)
        for _, r in zak.iterrows():
            img = ZAKYNTHOS_BASE / "images" / str(r["path"])
            if img.exists():
                rows.append({"_img": str(img), "_id": f"zak_{r['identity']}"})
                n_zak += 1
            else:
                n_zak_missing += 1
        log.info(f"  zakynthos turtles: {n_zak} rows  "
                 f"({n_zak_missing} missing on disk)")
    else:
        log.warning(f"  zakynthos annotations.csv missing: {zak_csv}")

    # 3. AmvrakikosTurtles - parse identity from filename prefix
    n_amv = 0
    n_amv_missing = 0
    amv_csv = AMVRAKIKOS_BASE / "annotations.csv"
    if amv_csv.exists():
        amv = pd.read_csv(amv_csv)
        seen: set[str] = set()
        for _, r in amv.iterrows():
            img_name = str(r["image_name"])
            identity = img_name.split("_")[0]
            img = AMVRAKIKOS_BASE / "images" / img_name
            key = str(img)
            if key in seen:
                continue
            if img.exists():
                seen.add(key)
                rows.append({"_img": key, "_id": f"amv_{identity}"})
                n_amv += 1
            else:
                n_amv_missing += 1
        log.info(f"  amvrakikos turtles: {n_amv} rows  "
                 f"({n_amv_missing} missing on disk)")
    else:
        log.warning(f"  amvrakikos annotations.csv missing: {amv_csv}")

    # 4. ReunionTurtles - parse year from Date, build nested path
    n_reu = 0
    n_reu_missing = 0
    n_reu_baddate = 0
    reu_csv = REUNION_BASE / "data.csv"
    if reu_csv.exists():
        reu = pd.read_csv(reu_csv)
        for _, r in reu.iterrows():
            try:
                year = str(pd.to_datetime(r["Date"], dayfirst=False).year)
            except Exception:
                n_reu_baddate += 1
                continue
            img = (
                REUNION_BASE / str(r["Species"]) / str(r["Turtle_ID"])
                / year / str(r["Photo_name"])
            )
            if img.exists():
                rows.append({
                    "_img": str(img),
                    "_id": f"reu_{r['Turtle_ID']}",
                })
                n_reu += 1
            else:
                n_reu_missing += 1
        log.info(f"  reunion turtles: {n_reu} rows  "
                 f"({n_reu_missing} missing, {n_reu_baddate} bad-date)")
    else:
        log.warning(f"  reunion data.csv missing: {reu_csv}")

    df = pd.DataFrame(rows, columns=["_img", "_id"])
    log.info(f"Combined turtle train: {len(df)} images, "
             f"{df['_id'].nunique()} identities  "
             f"(comp={n_comp}, zak={n_zak}, amv={n_amv}, reu={n_reu})")
    return df


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-masked", action="store_true",
                        help="train on masked-image variants where available")
    parser.add_argument("--use-subcenter",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-amp",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="mixed precision training (faster)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=2)
    parser.add_argument("--out-name", default="stage8a_turtle_extended_best.pth")
    args = parser.parse_args()

    seed_everything(config.RANDOM_SEED)
    device = config.DEVICE

    log.info("=" * 70)
    log.info("Stage 8a -- Turtle ArcFace re-training (extended dataset)")
    log.info("=" * 70)
    log.info(f"device={device}  amp={args.use_amp}  subcenter={args.use_subcenter}")

    # ------------------------------------------------------------------
    # 1) Build combined turtle training set
    # ------------------------------------------------------------------
    log.info("")
    log.info("Loading combined turtle training data...")
    train_df = load_all_turtle_train()
    if len(train_df) == 0:
        log.error("combined turtle train set is empty -- aborting")
        return 1

    # Drop any singletons (PK sampler tolerates them via replacement, but we
    # log how many there are for diagnostics).
    counts = train_df["_id"].value_counts()
    n_singleton_ids = int((counts == 1).sum())
    log.info(f"  identity image-count distribution: "
             f"min={counts.min()}, median={counts.median():.0f}, "
             f"max={counts.max()}, singletons={n_singleton_ids}")

    unique_ids = sorted(train_df["_id"].unique())
    identity_to_class = {iid: i for i, iid in enumerate(unique_ids)}
    num_classes = len(unique_ids)

    # ------------------------------------------------------------------
    # 2) Validation = same competition val as stage 2b (comparable ARI)
    # ------------------------------------------------------------------
    val_df = pd.read_csv(config.V4_TURTLE_VAL_CSV)
    log.info(f"val (stage2b-comparable): {len(val_df)} images, "
             f"{val_df['_id'].nunique()} identities")
    log.info(f"train: {len(train_df)} images, {num_classes} identities, "
             f"masked={args.use_masked}")

    # ------------------------------------------------------------------
    # 3) Stage 2b baseline val ARI (load from checkpoint if present)
    # ------------------------------------------------------------------
    baseline_ckpt_path = config.V5_CKPTS_DIR / "stage2b_arcface_best.pth"
    if baseline_ckpt_path.exists():
        baseline_ckpt = torch.load(
            baseline_ckpt_path, map_location="cpu", weights_only=False,
        )
        baseline_val_ari = float(baseline_ckpt.get("val_ari", 0.0))
        log.info(f"stage2b baseline val ARI (from ckpt): {baseline_val_ari:.4f}")
    else:
        baseline_val_ari = 0.0
        log.warning(f"stage2b checkpoint missing at {baseline_ckpt_path}; "
                    f"baseline_val_ari set to 0.0")

    # ------------------------------------------------------------------
    # 4) Datasets / loaders
    # ------------------------------------------------------------------
    train_transform = build_train_transforms()
    eval_transform = build_eval_transforms()

    train_ds = TurtleDataset(
        train_df, train_transform, identity_to_class,
        use_masked=args.use_masked,
    )
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

    # ------------------------------------------------------------------
    # 5) Model
    # ------------------------------------------------------------------
    model = MegaDescArcFace(
        num_classes=num_classes, use_subcenter=args.use_subcenter,
    ).to(device)

    # ------------------------------------------------------------------
    # 6) Pre-train baseline val ARI for THIS model (sanity)
    # ------------------------------------------------------------------
    log.info("")
    log.info("Pre-train val ARI (pretrained MegaD, no fine-tune)...")
    with timed("pre-train val embedding extraction"):
        pre_embs = extract_val_embeddings(
            model, val_df, eval_transform, device, args.use_masked,
        )
    pre_ari = evaluate_val_ari(pre_embs, val_df)
    log.info(f"  pretrained val ARI = {pre_ari:.4f}")

    # ------------------------------------------------------------------
    # 7) Training loop
    # ------------------------------------------------------------------
    best_ari = pre_ari
    ckpt_path = config.V5_CKPTS_DIR / args.out_name
    config.V5_CKPTS_DIR.mkdir(parents=True, exist_ok=True)

    scaler = (
        torch.amp.GradScaler("cuda")
        if (args.use_amp and device == "cuda")
        else None
    )

    total_epochs = sum(p[1] for p in PHASES)
    log.info("")
    log.info(f"curriculum: {len(PHASES)} phases, total {total_epochs} epochs")

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

                # LR warmup -> cosine
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
                val_embs = extract_val_embeddings(
                    model, val_df, eval_transform, device, args.use_masked,
                )
                val_ari = evaluate_val_ari(val_embs, val_df)
                log.info(f"    [val @ ep {global_epoch}] ARI={val_ari:.4f} "
                         f"(stage2b baseline {baseline_val_ari:.4f}) "
                         f"({time.time()-t_val:.0f}s)")
                if val_ari > best_ari:
                    best_ari = val_ari
                    torch.save({
                        "epoch": global_epoch,
                        "phase": phase_name,
                        "model_state_dict": model.state_dict(),
                        "val_ari": val_ari,
                        "num_classes": num_classes,
                        "use_subcenter": args.use_subcenter,
                        "use_masked": args.use_masked,
                        "stage2b_baseline_val_ari": baseline_val_ari,
                        "n_train_images": int(len(train_df)),
                        "n_train_identities": int(num_classes),
                    }, ckpt_path)
                    log.info(f"    [ckpt] saved (val_ARI={val_ari:.4f})")

            global_epoch += 1

    # ------------------------------------------------------------------
    # 8) Final comparison + recommendation
    # ------------------------------------------------------------------
    log.info("")
    log.info("=" * 70)
    log.info("FINAL COMPARISON")
    log.info("=" * 70)
    log.info(f"EXTENDED MODEL val ARI:  {best_ari:.4f}")
    log.info(f"STAGE2B BASELINE val ARI: {baseline_val_ari:.4f}")
    log.info(f"  delta: {best_ari - baseline_val_ari:+.4f}")
    if best_ari > baseline_val_ari:
        log.info("  -> USE extended model  "
                 "(stage8a beats stage2b on the comparable val split)")
    else:
        log.info("  -> KEEP stage2b model  "
                 "(extended model did NOT improve over stage2b baseline)")
    log.info(f"  checkpoint: {ckpt_path}")
    log.info("Next: stage8b_recluster_turtles.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
