#!/usr/bin/env python

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import config
from utils import cluster_agg, log, save_numpy, seed_everything, timed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def l2norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return x / n


def load_emb(path: Path, name: str) -> np.ndarray:
    arr = np.load(path)
    arr = l2norm(arr)
    log.info(f"loaded {name}: shape={arr.shape}  path={path.name}")
    return arr


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_megadesc_lizards(image_paths: list[str], cache_path: Path) -> np.ndarray:
    """Extract MegaDescriptor-L-384 embeddings for the 274 lizard images."""
    if cache_path.exists():
        log.info(f"[cache]    megadesc_lizards from {cache_path.name}")
        return np.load(cache_path)

    import torch
    import timm
    import torchvision.transforms as T
    from PIL import Image

    with timed("load MegaDescriptor"):
        model = timm.create_model(
            config.MEGADESC_HF_HUB_ID, num_classes=0, pretrained=True
        )
        model = model.to(config.DEVICE).eval()

    tf = T.Compose([
        T.Resize(config.MEGADESC_INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(config.MEGADESC_NORM_MEAN, config.MEGADESC_NORM_STD),
    ])

    embs: list[np.ndarray] = []
    bs = config.FEATURE_BATCH_SIZE
    n = len(image_paths)
    log.info(f"extracting megadesc on {n} lizard images, batch={bs}")
    with torch.inference_mode():
        for start in range(0, n, bs):
            batch_paths = image_paths[start:start + bs]
            tensors = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                tensors.append(tf(img))
            batch = torch.stack(tensors).to(config.DEVICE)
            feats = model(batch)
            embs.append(feats.detach().cpu().float().numpy())
            if (start // bs) % 4 == 0:
                log.info(f"  megadesc batch {start}..{min(start+bs, n)}/{n}")

    out = np.concatenate(embs, axis=0).astype(np.float32)
    out = l2norm(out)
    save_numpy(out, cache_path)

    # Free GPU memory
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return out


def extract_dinov2_lizards(image_paths: list[str], cache_path: Path) -> np.ndarray:
    """Extract DINOv2-large embeddings (CLS token) for the lizard images."""
    if cache_path.exists():
        log.info(f"[cache]    dinov2_lizards from {cache_path.name}")
        return np.load(cache_path)

    import torch
    import torchvision.transforms as T
    from transformers import AutoModel
    from PIL import Image

    with timed("load DINOv2-large"):
        model = AutoModel.from_pretrained(config.DINOV2_HF_HUB_ID)
        model = model.to(config.DEVICE).eval()

    # Use 518x518 (the DINOv2-large default fine-tuning resolution).
    # Standard ImageNet normalization.
    tf = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    embs: list[np.ndarray] = []
    bs = max(config.FEATURE_BATCH_SIZE // 2, 4)  # 518x518 is heavier
    n = len(image_paths)
    log.info(f"extracting dinov2 on {n} lizard images, batch={bs}, size=518")
    with torch.inference_mode():
        for start in range(0, n, bs):
            batch_paths = image_paths[start:start + bs]
            tensors = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                tensors.append(tf(img))
            batch = torch.stack(tensors).to(config.DEVICE)
            out = model(pixel_values=batch)
            # CLS token from last_hidden_state
            cls = out.last_hidden_state[:, 0, :]
            embs.append(cls.detach().cpu().float().numpy())
            if (start // bs) % 4 == 0:
                log.info(f"  dinov2 batch {start}..{min(start+bs, n)}/{n}")

    out_arr = np.concatenate(embs, axis=0).astype(np.float32)
    out_arr = l2norm(out_arr)
    save_numpy(out_arr, cache_path)

    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return out_arr


# ---------------------------------------------------------------------------
# Percentile-matching calibration
# ---------------------------------------------------------------------------

def sample_turtle_db_fused_sim(
    db_embs: list[tuple[np.ndarray, float]],
    n_pairs: int = 50000,
    seed: int = 42,
) -> np.ndarray:
    """
    Sample random off-diagonal upper-triangle pairs (i<j) from the turtle DB
    and return their fused similarity scores.
    """
    N = db_embs[0][0].shape[0]
    rng = np.random.default_rng(seed)
    # Oversample a bit to ensure we end up with >= n_pairs valid pairs.
    n_sample = int(n_pairs * 2.5)
    i_idx = rng.integers(0, N, n_sample)
    j_idx = rng.integers(0, N, n_sample)
    mask = i_idx < j_idx
    i_idx = i_idx[mask][:n_pairs]
    j_idx = j_idx[mask][:n_pairs]
    log.info(f"  sampled {len(i_idx)} unique-pair indices from "
             f"turtle DB (N={N})")

    fused = np.zeros(len(i_idx), dtype=np.float32)
    for emb, w in db_embs:
        # Per-pair dot product
        s = (emb[i_idx] * emb[j_idx]).sum(axis=1)
        fused += w * s.astype(np.float32)
    fused = np.clip(fused, 0.0, 1.0)
    return fused


def lizard_pairwise_fused_sim(
    lizard_embs: list[tuple[np.ndarray, float]],
) -> np.ndarray:
    """Full (N x N) fused similarity over the 274 lizards."""
    N = lizard_embs[0][0].shape[0]
    sim = np.zeros((N, N), dtype=np.float32)
    for emb, w in lizard_embs:
        sim += w * (emb @ emb.T).astype(np.float32)
    np.fill_diagonal(sim, 0.0)
    sim = np.clip(sim, 0.0, 1.0)
    return sim


def upper_tri_values(sim: np.ndarray) -> np.ndarray:
    n = sim.shape[0]
    iu = np.triu_indices(n, k=1)
    return sim[iu].astype(np.float32)


def calibrate_lizard_threshold_via_percentile(
    lizard_sim_full: np.ndarray,
    turtle_db_sim_sample: np.ndarray,
    t_match_turtle: float,
) -> tuple[float, float]:
    """
    Find the percentile p* of t_match_turtle in the turtle-DB sim distribution,
    then apply the same percentile to the lizard fused sim distribution.

    Returns (t_lizard, p_star).
    """
    # Fraction of turtle pairs with sim < t_match_turtle.
    p_star = float((turtle_db_sim_sample < t_match_turtle).mean())
    log.info(f"  t_match_turtle={t_match_turtle:.4f}")
    log.info(f"  turtle-DB fused sim distribution (n={len(turtle_db_sim_sample)}):")
    log.info(f"    min={turtle_db_sim_sample.min():.4f}  "
             f"max={turtle_db_sim_sample.max():.4f}  "
             f"mean={turtle_db_sim_sample.mean():.4f}")
    for q in (50, 75, 90, 95, 99, 99.5, 99.9):
        log.info(f"    p{q}={float(np.percentile(turtle_db_sim_sample, q)):.4f}")
    log.info(f"  percentile of t_match_turtle in turtle-DB sim: "
             f"{p_star * 100:.4f}%")

    liz_vals = upper_tri_values(lizard_sim_full)
    log.info(f"  lizard fused sim distribution (n={len(liz_vals)}):")
    log.info(f"    min={liz_vals.min():.4f}  max={liz_vals.max():.4f}  "
             f"mean={liz_vals.mean():.4f}")
    for q in (50, 75, 90, 95, 99, 99.5, 99.9):
        log.info(f"    p{q}={float(np.percentile(liz_vals, q)):.4f}")

    t_lizard = float(np.percentile(liz_vals, p_star * 100.0))
    log.info(f"  applying percentile p*={p_star * 100:.4f}% to lizard sim "
             f"-> t_lizard={t_lizard:.4f}")

    return t_lizard, p_star


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    seed_everything(config.RANDOM_SEED)

    # ------------------------------------------------------------------
    # Load lizard CSV + MIEW embeddings
    # ------------------------------------------------------------------
    feat4 = config.V4_FEATURES
    feat5 = config.V5_FEATURES_DIR

    lizard_df = pd.read_csv(config.V4_LIZARD_TEST_CSV)
    n_liz = len(lizard_df)
    log.info(f"lizards: {n_liz} images from {config.V4_LIZARD_TEST_CSV.name}")

    if "_img" not in lizard_df.columns:
        log.error("expected '_img' column in lizard CSV")
        return 1
    lizard_paths = lizard_df["_img"].astype(str).tolist()

    miew_liz = load_emb(feat4 / "miew_test_lizards.npy", "miew_lizards")
    if miew_liz.shape[0] != n_liz:
        log.error(f"row mismatch: csv {n_liz} vs miew {miew_liz.shape[0]}")
        return 1

    # ------------------------------------------------------------------
    # Extract / cache MegaDescriptor + DINOv2 for lizards
    # ------------------------------------------------------------------
    with timed("extract MegaDescriptor for lizards"):
        mega_liz = extract_megadesc_lizards(
            lizard_paths,
            cache_path=feat5 / "megadesc_lizard_test.npy",
        )
        mega_liz = l2norm(mega_liz)
        log.info(f"  megadesc_lizards: shape={mega_liz.shape}")

    with timed("extract DINOv2 for lizards"):
        dino_liz = extract_dinov2_lizards(
            lizard_paths,
            cache_path=feat5 / "dinov2_lizard_test.npy",
        )
        dino_liz = l2norm(dino_liz)
        log.info(f"  dinov2_lizards: shape={dino_liz.shape}")

    if mega_liz.shape[0] != n_liz or dino_liz.shape[0] != n_liz:
        log.error("extracted lizard embedding row counts do not match CSV")
        return 1

    # ------------------------------------------------------------------
    # Stage 5a turtle threshold for percentile matching
    # ------------------------------------------------------------------
    stage5a_diag_path = (config.V5_SUBMISSIONS_DIR
                         / "stage5a_db_guided_turtles_diag.json")
    if not stage5a_diag_path.exists():
        log.error(f"stage5a diagnostics not found at {stage5a_diag_path}")
        return 1
    with open(stage5a_diag_path) as f:
        stage5a_diag = json.load(f)
    t_match_turtle = float(stage5a_diag["t_match"])
    log.info(f"reusing stage5a t_match_turtle={t_match_turtle:.4f}")

    # ------------------------------------------------------------------
    # Fusion weights
    #   Lizards: we have miew + megadesc + dinov2 only.
    #   Turtle DB: only megadesc + miew + arcface are cached on disk
    #              (no DB DINOv2). The percentile match is
    #              representation-specific; we use the same models on each
    #              side where possible. Concretely, we'll calibrate using
    #              the three lizard-side models on the lizard distribution,
    #              and the three turtle-DB models that exist on the turtle
    #              side. The percentile-matching argument is that the
    #              *empirical rank* of the chosen threshold transfers, even
    #              if the underlying components differ slightly.
    # ------------------------------------------------------------------
    LIZ_W = {
        "miew": 0.40,
        "megadesc": 0.35,
        "dinov2": 0.25,
    }
    TURTLE_DB_W = {
        "miew": 0.40,
        "megadesc": 0.35,
        "arcface": 0.25,
    }
    s = sum(LIZ_W.values())
    LIZ_W = {k: v / s for k, v in LIZ_W.items()}
    s = sum(TURTLE_DB_W.values())
    TURTLE_DB_W = {k: v / s for k, v in TURTLE_DB_W.items()}
    log.info(f"lizard fusion weights:    {LIZ_W}")
    log.info(f"turtle-DB fusion weights: {TURTLE_DB_W}")

    # ------------------------------------------------------------------
    # Lizard fused pairwise sim (274 x 274)
    # ------------------------------------------------------------------
    with timed("compute lizard fused (274 x 274) similarity"):
        liz_blocks = [
            (miew_liz, LIZ_W["miew"]),
            (mega_liz, LIZ_W["megadesc"]),
            (dino_liz, LIZ_W["dinov2"]),
        ]
        sim_liz = lizard_pairwise_fused_sim(liz_blocks)
        log.info(f"sim_liz: shape={sim_liz.shape}  "
                 f"mean={sim_liz.mean():.4f}  "
                 f"max={sim_liz.max():.4f}  "
                 f"p95={float(np.percentile(sim_liz, 95)):.4f}  "
                 f"p99={float(np.percentile(sim_liz, 99)):.4f}")

    save_numpy(sim_liz, config.V5_SIMILARITY_DIR
               / "stage5b_lizard_calibrated_sim.npy")

    # ------------------------------------------------------------------
    # Turtle DB fused sim sample
    # ------------------------------------------------------------------
    with timed("load turtle-DB embeddings + sample fused similarity"):
        miew_db = load_emb(feat4 / "miew_db_turtles.npy", "miew_db")
        mega_db = load_emb(feat4 / "megadesc_db_turtles.npy", "megadesc_db")
        arc_db = load_emb(feat5 / "stage2c_arcface_db_turtles.npy", "arcface_db")
        if not (miew_db.shape[0] == mega_db.shape[0] == arc_db.shape[0]):
            log.error("turtle-DB embedding row counts disagree")
            return 1

        db_blocks = [
            (miew_db, TURTLE_DB_W["miew"]),
            (mega_db, TURTLE_DB_W["megadesc"]),
            (arc_db, TURTLE_DB_W["arcface"]),
        ]
        turtle_sample = sample_turtle_db_fused_sim(
            db_blocks, n_pairs=50000, seed=config.RANDOM_SEED,
        )

    # ------------------------------------------------------------------
    # Percentile-matching calibration
    # ------------------------------------------------------------------
    with timed("percentile-match calibration"):
        t_liz, p_star = calibrate_lizard_threshold_via_percentile(
            sim_liz, turtle_sample, t_match_turtle,
        )

    log.info("")
    log.info(f"--- calibration result ---")
    log.info(f"  turtle t_match    = {t_match_turtle:.4f}")
    log.info(f"  percentile in DB  = {p_star * 100:.4f}%")
    log.info(f"  lizard t_lizard   = {t_liz:.4f}")
    log.info(f"  rationale: pairs below t_match in turtle-DB sim "
             f"should correspond to non-matches; we apply the same "
             f"non-match fraction to the lizard distribution to pick a "
             f"comparable cluster threshold.")

    # ------------------------------------------------------------------
    # Sweep around the calibrated threshold
    # ------------------------------------------------------------------
    sweep_grid = sorted({round(x, 4) for x in np.arange(
        max(0.10, t_liz - 0.10), min(0.99, t_liz + 0.10) + 1e-6, 0.02
    ).tolist() + [t_liz]})
    sweep_rows = []
    for t in sweep_grid:
        cl = cluster_agg(sim_liz, float(t))
        sizes = pd.Series(cl).value_counts()
        sweep_rows.append({
            "t": float(t),
            "n_clusters": int(len(sizes)),
            "n_singletons": int((sizes == 1).sum()),
            "max_cluster_size": int(sizes.max()),
        })

    log.info("threshold sweep around t_liz:")
    for r in sweep_rows:
        marker = " <- chosen" if abs(r["t"] - t_liz) < 1e-9 else ""
        log.info(f"  t={r['t']:.4f}  n_clusters={r['n_clusters']:>3d}  "
                 f"singletons={r['n_singletons']:>3d}  "
                 f"max_size={r['max_cluster_size']}{marker}")

    # ------------------------------------------------------------------
    # Final clustering
    # ------------------------------------------------------------------
    with timed("agglomerative clustering at t_liz"):
        clusters = cluster_agg(sim_liz, float(t_liz))
        sizes = pd.Series(clusters).value_counts()
        n_clusters = int(len(sizes))
        n_singletons = int((sizes == 1).sum())
        max_size = int(sizes.max())
        log.info(f"final clustering: t={t_liz:.4f}  "
                 f"n_clusters={n_clusters}  singletons={n_singletons}  "
                 f"max_cluster_size={max_size}")

    # ------------------------------------------------------------------
    # Build submission
    # ------------------------------------------------------------------
    # Map raw cluster ids to a stable 0..n_clusters-1 ordering for
    # deterministic naming.
    uniq = pd.Series(clusters).unique()
    remap = {c: i for i, c in enumerate(sorted(uniq.tolist()))}
    out = pd.DataFrame({
        "image_id": lizard_df["image_id"].astype(str).tolist(),
        "cluster": [
            f"cluster_TexasHornedLizards_{remap[c]:04d}"
            for c in clusters
        ],
    })
    out_csv = config.V5_SUBMISSIONS_DIR / "stage5b_lizard_calibrated.csv"
    out.to_csv(out_csv, index=False)
    log.info(f"wrote {out_csv}  ({len(out)} rows)")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    diag = {
        "n_lizards": int(n_liz),
        "fusion_weights_lizard": LIZ_W,
        "fusion_weights_turtle_db": TURTLE_DB_W,
        "t_match_turtle": float(t_match_turtle),
        "p_star": float(p_star),
        "t_liz": float(t_liz),
        "n_clusters": n_clusters,
        "n_singletons": n_singletons,
        "max_cluster_size": max_size,
        "lizard_sim_stats": {
            "mean": float(sim_liz.mean()),
            "max": float(sim_liz.max()),
            "p50": float(np.percentile(sim_liz, 50)),
            "p90": float(np.percentile(sim_liz, 90)),
            "p95": float(np.percentile(sim_liz, 95)),
            "p99": float(np.percentile(sim_liz, 99)),
        },
        "turtle_db_sim_sample_stats": {
            "n": int(len(turtle_sample)),
            "mean": float(turtle_sample.mean()),
            "max": float(turtle_sample.max()),
            "p50": float(np.percentile(turtle_sample, 50)),
            "p90": float(np.percentile(turtle_sample, 90)),
            "p95": float(np.percentile(turtle_sample, 95)),
            "p99": float(np.percentile(turtle_sample, 99)),
        },
        "sweep": sweep_rows,
    }
    diag_path = (config.V5_SUBMISSIONS_DIR
                 / "stage5b_lizard_calibrated_diag.json")
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2, default=float)
    log.info(f"wrote diagnostics to {diag_path}")
    log.info("stage5b_lizard_calibrated complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
