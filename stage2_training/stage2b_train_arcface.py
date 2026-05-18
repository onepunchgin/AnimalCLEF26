#!/usr/bin/env python

from __future__ import annotations

import argparse
import math
import sys
import time
import json
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import Dataset, DataLoader, Sampler

from utils import (log, save_numpy, seed_everything, timed,
                     build_test_matched_val_indices)
import config


# =====================================================================
# SubCenter ArcFace head (more robust than standard ArcFace for noisy labels)
# =====================================================================

class SubCenterArcFaceHead(nn.Module):
    """
    SubCenter ArcFace: each class has K=3 sub-centers; sample is matched
    to the closest sub-center, then ArcFace margin applied.
    Reduces sensitivity to label noise / intra-class variation.
    """
    def __init__(self, feature_dim: int, num_classes: int, K: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.K = K
        # Weight: (K * num_classes, feature_dim)
        W = torch.randn(K * num_classes, feature_dim)
        W = F.normalize(W, dim=1)
        self.weight = nn.Parameter(W)

    def forward(self, features, labels=None, s=64.0, m=0.5):
        W = F.normalize(self.weight, dim=1)
        # (B, K*C)
        cos_all = features @ W.T
        # Reshape to (B, C, K), max over K → (B, C)
        B = cos_all.size(0)
        cos_per_class = cos_all.view(B, self.num_classes, self.K)
        cos_theta, _ = cos_per_class.max(dim=2)
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if labels is None or m == 0.0:
            return s * cos_theta

        theta = torch.acos(cos_theta)
        tgt_theta = theta.gather(1, labels.view(-1, 1)).squeeze(1)
        tgt_with_m = torch.cos(tgt_theta + m)
        logits = cos_theta.clone()
        logits.scatter_(1, labels.view(-1, 1), tgt_with_m.to(logits.dtype).view(-1, 1))
        return s * logits


# =====================================================================
# Backbone wrapper
# =====================================================================

class MegaDescArcFace(nn.Module):
    def __init__(self, num_classes: int, use_subcenter: bool = True,
                  pretrained_path: Path = None):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            config.MEGADESC_HF_HUB_ID, num_classes=0, pretrained=True,
        )
        self.feature_dim = 1536

        # Optionally load custom pretrained backbone (e.g., from earlier v5 run)
        if pretrained_path and pretrained_path.exists():
            log.info(f"loading pretrained backbone from {pretrained_path}")
            ckpt = torch.load(pretrained_path, map_location="cpu",
                                weights_only=False)
            backbone_state = {k.replace("backbone.", ""): v
                                for k, v in ckpt["model_state_dict"].items()
                                if k.startswith("backbone.")}
            self.backbone.load_state_dict(backbone_state, strict=False)

        if use_subcenter:
            self.head = SubCenterArcFaceHead(self.feature_dim, num_classes, K=3)
        else:
            from torch.nn import Parameter
            W = torch.randn(num_classes, self.feature_dim)
            W = F.normalize(W, dim=1)
            self.head_weight = Parameter(W)
            self.head = lambda features, labels=None, s=64.0, m=0.5: \
                self._simple_arcface(features, labels, s, m)
        self.use_subcenter = use_subcenter

    def _simple_arcface(self, features, labels, s, m):
        W = F.normalize(self.head_weight, dim=1)
        cos_theta = (features @ W.T).clamp(-1+1e-7, 1-1e-7)
        if labels is None or m == 0.0:
            return s * cos_theta
        theta = torch.acos(cos_theta)
        tgt_theta = theta.gather(1, labels.view(-1, 1)).squeeze(1)
        tgt_with_m = torch.cos(tgt_theta + m)
        logits = cos_theta.clone()
        logits.scatter_(1, labels.view(-1, 1), tgt_with_m.to(logits.dtype).view(-1, 1))
        return s * logits

    def extract_embeddings(self, x):
        feat = self.backbone(x)
        return F.normalize(feat, dim=1)

    def forward(self, x, labels=None, s=64.0, m=0.5):
        feat = self.backbone(x)
        feat_norm = F.normalize(feat, dim=1)
        logits = self.head(feat_norm, labels, s=s, m=m)
        return logits, feat_norm


# =====================================================================
# Strong augmentation
# =====================================================================

def build_train_transforms():
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((420, 420)),  # bigger then random crop
        T.RandomResizedCrop(config.MEGADESC_INPUT_SIZE, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3,
                                       saturation=0.3, hue=0.05)], p=0.7),
        T.RandomApply([T.GaussianBlur(kernel_size=5,
                                        sigma=(0.1, 1.5))], p=0.3),
        T.RandomApply([T.RandomAffine(degrees=10, translate=(0.05, 0.05),
                                        scale=(0.95, 1.05))], p=0.3),
        T.ToTensor(),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        T.Normalize(config.MEGADESC_NORM_MEAN, config.MEGADESC_NORM_STD),
    ])


def build_eval_transforms():
    import torchvision.transforms as T
    return T.Compose([
        T.Resize(config.MEGADESC_INPUT_SIZE), T.ToTensor(),
        T.Normalize(config.MEGADESC_NORM_MEAN, config.MEGADESC_NORM_STD),
    ])


# =====================================================================
# Dataset + PK sampler
# =====================================================================

class TurtleDataset(Dataset):
    def __init__(self, df, transform, identity_to_class, use_masked=False):
        self.transform = transform
        self.use_masked = use_masked
        # Resolve paths (raw or masked)
        self.paths = []
        for _, row in df.iterrows():
            if use_masked:
                iid = str(row["image_id"]) if "image_id" in df.columns \
                      else Path(row["_img"]).stem
                masked = config.V5_MASKED_IMAGES_DIR / "db_turtle" / f"{iid}.jpg"
                if not masked.exists():
                    # train/val are subsets of db_turtle, masks are stored there
                    masked = config.V5_MASKED_IMAGES_DIR / "val_turtle" / f"{iid}.jpg"
                self.paths.append(str(masked) if masked.exists() else row["_img"])
            else:
                self.paths.append(row["_img"])
        self.labels = df["_id"].map(identity_to_class).astype(int).tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


class PKSampler(Sampler):
    """P identities × K samples per batch. Critical for ArcFace."""
    def __init__(self, labels, batch_size, num_instances=4):
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.K = num_instances
        assert batch_size % num_instances == 0
        self.P = batch_size // num_instances
        self.by_label = defaultdict(list)
        for i, lbl in enumerate(labels):
            self.by_label[lbl].append(i)
        self.unique_labels = sorted(self.by_label.keys())

    def __iter__(self):
        rng = np.random.default_rng()
        pools = {l: rng.permutation(self.by_label[l]).tolist()
                 for l in self.unique_labels}
        order = rng.permutation(self.unique_labels).tolist()
        while True:
            avail = [l for l in order if pools[l]]
            if len(avail) < self.P:
                break
            chosen = rng.choice(avail, self.P, replace=False)
            for lbl in chosen:
                pool = pools[lbl]
                if len(pool) >= self.K:
                    take = pool[:self.K]
                    pools[lbl] = pool[self.K:]
                else:
                    take = rng.choice(self.by_label[lbl], self.K,
                                        replace=True).tolist()
                    pools[lbl] = []
                for idx in take:
                    yield idx

    def __len__(self):
        return sum(len(v) for v in self.by_label.values()) // self.batch_size


# =====================================================================
# Training utilities
# =====================================================================

def set_trainable(model, backbone_frac):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in (model.head.parameters() if hasattr(model.head, "parameters")
              else [model.head_weight]):
        if hasattr(p, "requires_grad"):
            p.requires_grad = True
    if backbone_frac > 0:
        params = list(model.backbone.named_parameters())
        n_unfreeze = int(len(params) * backbone_frac)
        for name, p in params[-n_unfreeze:]:
            p.requires_grad = True


def build_optimizer(model, lr_head, lr_backbone, wd_head=0.05, wd_backbone=0.01):
    head_params = []
    if hasattr(model.head, "parameters"):
        head_params = [p for p in model.head.parameters() if p.requires_grad]
    elif hasattr(model, "head_weight") and model.head_weight.requires_grad:
        head_params = [model.head_weight]

    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": lr_head,
                       "weight_decay": wd_head, "name": "head"})
    if bb_params:
        groups.append({"params": bb_params, "lr": lr_backbone,
                       "weight_decay": wd_backbone, "name": "backbone"})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


def ramp(epoch, total, start, end):
    if total <= 0:
        return end
    frac = min(1.0, max(0.0, epoch / total))
    return start + (end - start) * frac


@torch.inference_mode()
def extract_val_embeddings(model, val_df, transform, device, use_masked=False):
    from PIL import Image
    model.eval()
    paths = []
    for _, row in val_df.iterrows():
        if use_masked:
            iid = str(row["image_id"]) if "image_id" in val_df.columns \
                  else Path(row["_img"]).stem
            masked = config.V5_MASKED_IMAGES_DIR / "val_turtle" / f"{iid}.jpg"
            paths.append(str(masked) if masked.exists() else row["_img"])
        else:
            paths.append(row["_img"])
    embs = []
    bs = config.FEATURE_BATCH_SIZE
    for s in range(0, len(paths), bs):
        batch = torch.stack([transform(Image.open(p).convert("RGB"))
                              for p in paths[s:s+bs]]).to(device)
        embs.append(model.extract_embeddings(batch).cpu().numpy())
    return np.concatenate(embs, axis=0)


def evaluate_val_ari(embs, val_df, n_splits=3):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    aris = []
    for split_i in range(n_splits):
        keep = build_test_matched_val_indices(val_df, seed=42 + split_i)
        sub_emb = embs[keep]
        sub_labels = val_df.iloc[keep]["_id"].astype("category").cat.codes.values
        sim = np.clip(sub_emb @ sub_emb.T, 0.0, 1.0)
        np.fill_diagonal(sim, 0.0)
        dist = 1.0 - sim
        dist = (dist + dist.T) / 2.0
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average")
        best = -1
        for t in np.arange(0.20, 0.80, 0.05):
            cids = fcluster(Z, t=t, criterion="distance")
            ari = adjusted_rand_score(sub_labels, cids)
            if ari > best:
                best = ari
        aris.append(best)
    return float(np.mean(aris))


# =====================================================================
# Phase definitions
# =====================================================================

PHASES = [
    # name, n_epochs, bb_frac, lr_head, lr_bb, s_start, s_end, m_start, m_end
    ("phase0_head",       3, 0.00, 1e-3, 0.0,  16, 16, 0.0, 0.0),
    ("phase1_top10",      8, 0.10, 1e-3, 1e-5, 16, 32, 0.0, 0.1),
    ("phase2_top30",     15, 0.30, 5e-4, 1e-5, 32, 64, 0.1, 0.3),
    ("phase3_all",        5, 1.00, 1e-4, 3e-6, 64, 64, 0.3, 0.3),
    ("phase4_cooldown",   2, 1.00, 1e-5, 1e-6, 64, 64, 0.3, 0.3),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-masked", action="store_true")
    parser.add_argument("--use-subcenter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True,
                        help="mixed precision training (faster)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=2)
    parser.add_argument("--out-name", default="stage2b_arcface_best.pth")
    args = parser.parse_args()

    seed_everything(config.RANDOM_SEED)
    device = config.DEVICE

    train_df = pd.read_csv(config.V4_TURTLE_TRAIN_CSV)
    val_df = pd.read_csv(config.V4_TURTLE_VAL_CSV)
    unique_ids = sorted(train_df["_id"].unique())
    identity_to_class = {iid: i for i, iid in enumerate(unique_ids)}
    num_classes = len(unique_ids)

    log.info(f"train: {len(train_df)} images, {num_classes} identities, "
             f"masked={args.use_masked}")
    log.info(f"val: {len(val_df)} images, subcenter={args.use_subcenter}, "
             f"amp={args.use_amp}")

    train_transform = build_train_transforms()
    eval_transform = build_eval_transforms()
    train_ds = TurtleDataset(train_df, train_transform, identity_to_class,
                                use_masked=args.use_masked)
    sampler = PKSampler(train_ds.labels, args.batch_size, args.num_instances)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                sampler=sampler, num_workers=config.NUM_WORKERS,
                                pin_memory=True)

    model = MegaDescArcFace(num_classes=num_classes,
                              use_subcenter=args.use_subcenter).to(device)

    # Baseline val ARI
    log.info("")
    log.info("Baseline val ARI (pretrained MegaD, no fine-tune)...")
    baseline_embs = extract_val_embeddings(
        model, val_df, eval_transform, device, args.use_masked)
    baseline_ari = evaluate_val_ari(baseline_embs, val_df)
    log.info(f"  baseline ARI = {baseline_ari:.4f}")

    best_ari = baseline_ari
    ckpt_path = config.V5_CKPTS_DIR / args.out_name
    config.V5_CKPTS_DIR.mkdir(parents=True, exist_ok=True)

    scaler = torch.amp.GradScaler("cuda") if args.use_amp else None

    global_epoch = 0
    for phase_name, n_ep, bb_frac, lr_h, lr_b, s_st, s_en, m_st, m_en in PHASES:
        log.info("")
        log.info("=" * 70)
        log.info(f"{phase_name}: {n_ep} ep, bb={bb_frac*100:.0f}%, "
                 f"lr_h={lr_h:.0e}, lr_b={lr_b:.0e}, s={s_st}→{s_en}, m={m_st}→{m_en}")
        log.info("=" * 70)

        set_trainable(model, bb_frac)
        optimizer = build_optimizer(model, lr_h, lr_b)
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * n_ep
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

                if args.use_amp:
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
                        max_norm=2.0)
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
                        max_norm=2.0)
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
                log.warning(f"  !! loss doubled: {prev_loss:.2f} → {ep_loss:.2f}")
            prev_loss = ep_loss

            is_end = (ep == n_ep - 1)
            if (global_epoch % args.val_every == 0) or is_end:
                t_val = time.time()
                val_embs = extract_val_embeddings(
                    model, val_df, eval_transform, device, args.use_masked)
                val_ari = evaluate_val_ari(val_embs, val_df)
                log.info(f"    [val @ ep {global_epoch}] ARI={val_ari:.4f} "
                         f"(baseline {baseline_ari:.4f}) "
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
                    }, ckpt_path)
                    log.info(f"    [ckpt] saved (val_ARI={val_ari:.4f})")

            global_epoch += 1

    log.info("")
    log.info(f"Done. Best val ARI: {best_ari:.4f} (baseline {baseline_ari:.4f})")
    log.info(f"  improvement: {best_ari - baseline_ari:+.4f}")
    log.info(f"  checkpoint: {ckpt_path}")
    log.info("Next: stage2c_extract_embeddings.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
