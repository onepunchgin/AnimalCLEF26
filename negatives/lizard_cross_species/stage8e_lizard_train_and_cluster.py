#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
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
from torch.utils.data import DataLoader

from utils import cluster_agg, log, save_numpy, seed_everything, timed
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
from stage6a_train_lizard_arcface import (
    BALEARIC_IMG_SUBDIR,
    LizardDataset,
    PHASES,
    best_val_ari,
    extract_lizard_embeddings,
    identity_split,
    remap_path,
)


# =====================================================================
# Paths / constants
# =====================================================================

BALEARIC_BASE = Path("/home/prouser1/Downloads/AnimalCLEF/data/balearic_lizard")
BALEARIC_META = BALEARIC_BASE / "curt_metadata.csv"
TOTAL_EPOCHS = sum(p[1] for p in PHASES)


# =====================================================================
# Download-completeness check
# =====================================================================

def verify_download() -> bool:
    """Return True if the BalearicLizard download appears complete."""
    if not BALEARIC_BASE.exists():
        log.error(f"BalearicLizard root missing: {BALEARIC_BASE}")
        return False

    img_dir = BALEARIC_BASE / BALEARIC_IMG_SUBDIR
    if not img_dir.exists():
        log.error(
            f"BalearicLizard not fully downloaded -- missing {img_dir}. "
            f"Found: {[p.name for p in BALEARIC_BASE.iterdir()]}"
        )
        return False

    n_subdirs = sum(1 for p in img_dir.iterdir() if p.is_dir())
    if n_subdirs < 100:
        log.error(
            f"BalearicLizard appears partially downloaded: "
            f"only {n_subdirs} identity subdirs in {img_dir}."
        )
        log.error("Wait for the download to complete before running stage 8e.")
        return False

    if not BALEARIC_META.exists():
        log.error(f"BalearicLizard metadata CSV missing: {BALEARIC_META}")
        return False

    log.info(f"BalearicLizard download check: OK "
             f"({n_subdirs} identity dirs in {img_dir.name})")
    return True


# =====================================================================
# Cluster Texas test
# =====================================================================

def cosine_sim(emb: np.ndarray) -> np.ndarray:
    sim = (emb @ emb.T).astype(np.float32)
    sim = np.clip(sim, 0.0, 1.0)
    sim = (sim + sim.T) / 2.0
    np.fill_diagonal(sim, 0.0)
    return sim


def sweep_val_threshold(val_emb: np.ndarray, val_codes: np.ndarray):
    """Sweep agglomerative threshold; return best (t, ari) and full curve."""
    sim = cosine_sim(val_emb)
    curve = []
    best_t, best_ari = 0.5, -1.0
    for t in np.arange(0.20, 0.901, 0.025):
        t = float(round(float(t), 4))
        cl = cluster_agg(sim, t)
        ari = float(adjusted_rand_score(val_codes, cl))
        n_clusters = int(pd.Series(cl).nunique())
        n_singletons = int((pd.Series(cl).value_counts() == 1).sum())
        curve.append({
            "t": t,
            "ari": ari,
            "n_clusters": n_clusters,
            "n_singletons": n_singletons,
        })
        if ari > best_ari:
            best_ari, best_t = ari, t
    return best_t, best_ari, curve


def cluster_texas(t_chosen: float, texas_emb: np.ndarray,
                  texas_df: pd.DataFrame) -> tuple[Path, dict]:
    sim = cosine_sim(texas_emb)
    log.info(f"texas sim: mean={sim.mean():.4f}  max={sim.max():.4f}  "
             f"p95={float(np.percentile(sim, 95)):.4f}  "
             f"p99={float(np.percentile(sim, 99)):.4f}")

    sweep_grid = sorted({round(float(x), 4) for x in (
        list(np.arange(max(0.10, t_chosen - 0.10),
                       min(0.99, t_chosen + 0.10) + 1e-6, 0.02))
        + [t_chosen]
    )})
    sweep_rows = []
    for t in sweep_grid:
        cl = cluster_agg(sim, float(t))
        sizes = pd.Series(cl).value_counts()
        sweep_rows.append({
            "t": float(t),
            "n_clusters": int(len(sizes)),
            "n_singletons": int((sizes == 1).sum()),
            "max_cluster_size": int(sizes.max()),
        })
    log.info("Texas-side sweep around chosen threshold:")
    for r in sweep_rows:
        marker = " <- chosen" if abs(r["t"] - t_chosen) < 1e-9 else ""
        log.info(f"  t={r['t']:.4f}  n_clusters={r['n_clusters']:>3d}  "
                 f"singletons={r['n_singletons']:>3d}  "
                 f"max_size={r['max_cluster_size']}{marker}")

    with timed(f"agglomerative clustering at t={t_chosen:.4f}"):
        clusters = cluster_agg(sim, float(t_chosen))
        sizes = pd.Series(clusters).value_counts()
        n_clusters = int(len(sizes))
        n_singletons = int((sizes == 1).sum())
        max_size = int(sizes.max())
    log.info(f"final lizard clustering: t={t_chosen:.4f}  "
             f"n_clusters={n_clusters}  singletons={n_singletons}  "
             f"max_cluster_size={max_size}")

    uniq = pd.Series(clusters).unique()
    remap = {c: i for i, c in enumerate(sorted(uniq.tolist()))}
    out = pd.DataFrame({
        "image_id": texas_df["image_id"].astype(str).tolist(),
        "cluster": [
            f"cluster_TexasHornedLizards_{remap[c]:04d}" for c in clusters
        ],
    })
    out_csv = config.V5_SUBMISSIONS_DIR / "stage8e_lizard_clusters.csv"
    out.to_csv(out_csv, index=False)
    log.info(f"wrote {out_csv}  ({len(out)} rows)")

    return out_csv, {
        "t_chosen": float(t_chosen),
        "n_clusters": n_clusters,
        "n_singletons": n_singletons,
        "max_cluster_size": max_size,
        "sweep": sweep_rows,
        "texas_sim_stats": {
            "mean": float(sim.mean()),
            "max": float(sim.max()),
            "p50": float(np.percentile(sim, 50)),
            "p90": float(np.percentile(sim, 90)),
            "p95": float(np.percentile(sim, 95)),
            "p99": float(np.percentile(sim, 99)),
        },
    }


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
    parser.add_argument("--out-name", default="stage8e_lizard_arcface_best.pth")
    parser.add_argument("--check-images", action="store_true",
                        help="verify all image paths exist before training")
    args = parser.parse_args()

    seed_everything(config.RANDOM_SEED)
    device = config.DEVICE

    log.info("=" * 70)
    log.info("Stage 8e -- BalearicLizard train + Texas cluster")
    log.info("=" * 70)

    # ------------------------------------------------------------------
    # 1) Verify BalearicLizard is fully downloaded
    # ------------------------------------------------------------------
    if not verify_download():
        return 1

    # ------------------------------------------------------------------
    # 2) Load + filter metadata
    # ------------------------------------------------------------------
    meta = pd.read_csv(BALEARIC_META)
    log.info(f"loaded metadata: {len(meta)} rows  "
             f"unique ids={meta['id'].nunique()}")

    counts = meta.groupby("id").size()
    multi_ids = set(counts[counts >= 2].index)
    n_drop = len(meta) - meta["id"].isin(multi_ids).sum()
    meta = meta[meta["id"].isin(multi_ids)].reset_index(drop=True)
    log.info(f"after dropping singletons: {len(meta)} rows  "
             f"unique ids={meta['id'].nunique()}  (dropped {n_drop})")

    # ------------------------------------------------------------------
    # 3) Identity split
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
    # 4) Optional path verification
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
    # 5) Datasets / loaders
    # ------------------------------------------------------------------
    train_transform = build_train_transforms()
    eval_transform = build_eval_transforms()

    train_ds = LizardDataset(train_df, train_transform, identity_to_class)
    label_counts = pd.Series(train_ds.labels).value_counts()
    low = (label_counts < args.num_instances).sum()
    log.info(f"  PK sampler: P={args.batch_size // args.num_instances} "
             f"K={args.num_instances}  "
             f"({low}/{len(label_counts)} train ids have < K images)")

    sampler = PKSampler(train_ds.labels, args.batch_size, args.num_instances)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    val_paths = [str(remap_path(p)) for p in val_df["path"].tolist()]
    val_id_to_code = {iid: i for i, iid in enumerate(sorted(val_ids))}
    val_label_codes = np.asarray(
        [val_id_to_code[i] for i in val_df["id"].tolist()], dtype=np.int64,
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
        pre_embs = extract_lizard_embeddings(
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
                val_embs = extract_lizard_embeddings(
                    model, val_paths, eval_transform, device,
                )
                val_ari, val_t = best_val_ari(val_embs, val_label_codes)
                log.info(f"    [val @ ep {global_epoch}] ARI={val_ari:.4f} "
                         f"@ t={val_t:.3f}  (pre-train {pre_ari:.4f}) "
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
                        "lizard_train_ids": list(sorted(train_ids)),
                        "lizard_val_ids": list(sorted(val_ids)),
                    }, ckpt_path)
                    log.info(f"    [ckpt] saved (val_ARI={val_ari:.4f})")

            global_epoch += 1

    log.info("")
    log.info(f"Done training. Best val ARI: {best_ari:.4f} @ t={best_t:.3f} "
             f"(pretrained {pre_ari:.4f})")
    log.info(f"  improvement: {best_ari - pre_ari:+.4f}")
    log.info(f"  checkpoint: {ckpt_path}")

    # ------------------------------------------------------------------
    # 9) Reload best checkpoint for embedding extraction
    # ------------------------------------------------------------------
    log.info("")
    log.info("=" * 70)
    log.info("Extract embeddings for val + Texas test")
    log.info("=" * 70)
    if not ckpt_path.exists():
        log.error(f"best checkpoint not saved: {ckpt_path}")
        log.error("training never improved over baseline -- aborting cluster step")
        return 1

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MegaDescArcFace(
        num_classes=int(ckpt["num_classes"]),
        use_subcenter=bool(ckpt.get("use_subcenter", True)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Val embeddings (for threshold calibration)
    with timed("extract balearic val embeddings"):
        val_embs = extract_lizard_embeddings(
            model, val_paths, eval_transform, device,
        )
    save_numpy(
        val_embs,
        config.V5_FEATURES_DIR / "stage8e_lizard_balearic_val_emb.npy",
    )
    save_numpy(
        val_label_codes,
        config.V5_FEATURES_DIR / "stage8e_lizard_balearic_val_codes.npy",
    )

    # Texas test embeddings
    if not config.V4_LIZARD_TEST_CSV.exists():
        log.error(f"missing lizard test CSV: {config.V4_LIZARD_TEST_CSV}")
        return 1
    texas_df = pd.read_csv(config.V4_LIZARD_TEST_CSV)
    if "_img" not in texas_df.columns:
        log.error("expected '_img' column in lizard test CSV")
        return 1
    texas_paths = [str(p) for p in texas_df["_img"].astype(str).tolist()]
    log.info(f"TexasHornedLizard test: {len(texas_paths)} images")

    missing = [p for p in texas_paths if not Path(p).exists()]
    if missing:
        log.error(f"missing {len(missing)}/{len(texas_paths)} test image files. "
                  f"first 3: {missing[:3]}")
        return 1

    with timed("extract texas test embeddings"):
        texas_embs = extract_lizard_embeddings(
            model, texas_paths, eval_transform, device,
        )
    save_numpy(
        texas_embs,
        config.V5_FEATURES_DIR / "stage8e_lizard_texas_test_emb.npy",
    )

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 10) Calibrate threshold on val + cluster Texas
    # ------------------------------------------------------------------
    log.info("")
    log.info("=" * 70)
    log.info("Threshold calibration on BalearicLizard val")
    log.info("=" * 70)
    with timed("val threshold sweep"):
        cal_t, cal_ari, curve = sweep_val_threshold(val_embs, val_label_codes)

    log.info("val ARI sweep:")
    for r in curve:
        marker = " <- chosen" if abs(r["t"] - cal_t) < 1e-9 else ""
        log.info(f"  t={r['t']:.4f}  ARI={r['ari']:+.4f}  "
                 f"n_clusters={r['n_clusters']:>4d}  "
                 f"singletons={r['n_singletons']:>4d}{marker}")
    log.info(f"BEST: t={cal_t:.4f}  val_ARI={cal_ari:+.4f}")

    log.info("")
    log.info("=" * 70)
    log.info("Cluster TexasHornedLizard test")
    log.info("=" * 70)
    out_csv, texas_diag = cluster_texas(cal_t, texas_embs, texas_df)

    diag = {
        "ckpt_val_ari": float(ckpt.get("val_ari", 0.0)),
        "ckpt_val_threshold": float(ckpt.get("val_threshold", cal_t)),
        "calibration": {
            "best_t": float(cal_t),
            "best_val_ari": float(cal_ari),
            "sweep": curve,
            "n_val_images": int(val_embs.shape[0]),
            "n_val_identities": int(len(np.unique(val_label_codes))),
        },
        "texas": texas_diag,
    }
    diag_path = (
        config.V5_SUBMISSIONS_DIR / "stage8e_lizard_clusters_diag.json"
    )
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2, default=float)
    log.info(f"wrote diagnostics to {diag_path}")

    log.info("Stage 8e complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
