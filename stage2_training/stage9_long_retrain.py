#!/usr/bin/env python

from __future__ import annotations
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

from utils import log, seed_everything, timed, save_numpy, cluster_agg
import config
from stage2b_train_arcface import (
    MegaDescArcFace, build_train_transforms, build_eval_transforms,
    PKSampler, set_trainable, build_optimizer, ramp,
)

COMP_BASE    = Path("/home/prouser1/Downloads/AnimalCLEF/data/competition/animal-clef-2026")
ZAKYNTHOS    = Path("/home/prouser1/Downloads/AnimalCLEF/data/zakynthos_turtles")
AMVRAKIKOS   = Path("/home/prouser1/Downloads/AnimalCLEF/data/amvrakikos_turtles")
REUNION      = Path("/home/prouser1/Downloads/AnimalCLEF/data/reunion_turtles")

# Extended phases — more epochs per phase for deeper convergence
PHASES_LONG = [
    ("phase0_head",   4,  0.00, 1e-3, 0.0,  16, 16, 0.0, 0.0),
    ("phase1_top10", 12,  0.10, 1e-3, 1e-5, 16, 32, 0.0, 0.1),
    ("phase2_top30", 20,  0.30, 5e-4, 1e-5, 32, 64, 0.1, 0.3),
    ("phase3_all",    8,  1.00, 1e-4, 3e-6, 64, 64, 0.3, 0.3),
    ("phase4_cool",   4,  1.00, 1e-5, 1e-6, 64, 64, 0.3, 0.3),
]  # 48 epochs total

class TurtleDatasetMulti(torch.utils.data.Dataset):
    def __init__(self, rows, transform):
        from PIL import Image
        self.paths = [r['_img'] for r in rows]
        self.labels = [r['_label'] for r in rows]
        self.transform = transform
        self._Image = Image
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = self._Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img), self.labels[idx]

def load_all_turtle_data():
    rows = []
    # Competition train
    meta = pd.read_csv(COMP_BASE / "metadata.csv")
    comp = meta[(meta['dataset']=='SeaTurtleID2022') & (meta['split']=='train')]
    for _, r in comp.iterrows():
        rows.append({'_img': str(COMP_BASE / r['path']), '_id': f"comp_{r['identity']}"})
    # Zakynthos
    try:
        zak = pd.read_csv(ZAKYNTHOS / "annotations.csv")
        for _, r in zak.iterrows():
            p = ZAKYNTHOS / "images" / r['path']
            if p.exists(): rows.append({'_img': str(p), '_id': f"zak_{r['identity']}"})
    except: pass
    # Amvrakikos
    try:
        amv = pd.read_csv(AMVRAKIKOS / "annotations.csv")
        seen = set()
        for _, r in amv.iterrows():
            p = AMVRAKIKOS / "images" / r['image_name']
            k = str(p)
            if p.exists() and k not in seen:
                seen.add(k)
                rows.append({'_img': k, '_id': f"amv_{r['image_name'].split('_')[0]}"})
    except: pass
    # Reunion
    try:
        reu = pd.read_csv(REUNION / "data.csv")
        for _, r in reu.iterrows():
            yr = str(pd.to_datetime(r['Date'], dayfirst=False).year)
            p = REUNION / r['Species'] / r['Turtle_ID'] / yr / r['Photo_name']
            if p.exists(): rows.append({'_img': str(p), '_id': f"reu_{r['Turtle_ID']}"})
    except: pass

    df = pd.DataFrame(rows)
    # Remove singletons (can't contribute to metric learning)
    counts = df['_id'].value_counts()
    df = df[df['_id'].isin(counts[counts >= 2].index)].reset_index(drop=True)
    log.info(f"All turtle train: {len(df)} imgs, {df['_id'].nunique()} identities")
    return df

@torch.inference_mode()
def extract_embs(model, paths, transform, device, tta=False):
    from PIL import Image
    model.eval()
    embs = []
    bs = config.FEATURE_BATCH_SIZE
    if tta:
        import torchvision.transforms as T
        flipped_tf = T.Compose([T.RandomHorizontalFlip(p=1.0), transform])
    for s in range(0, len(paths), bs):
        batch_imgs = [Image.open(p).convert('RGB') for p in paths[s:s+bs]]
        b = torch.stack([transform(img) for img in batch_imgs]).to(device)
        e = model.extract_embeddings(b).cpu().numpy()
        if tta:
            bf = torch.stack([flipped_tf(img) for img in batch_imgs]).to(device)
            ef = model.extract_embeddings(bf).cpu().numpy()
            e = (e + ef) / 2
            e = e / np.linalg.norm(e, axis=1, keepdims=True).clip(1e-8)
        embs.append(e)
    return np.concatenate(embs)

def eval_val_ari_calib(model, val_df, transform, device, calib_dfs, tta=False):
    """Evaluate on both val and calib sets, return mean ARI."""
    aris = []
    for name, df in [('val', val_df)] + [('calib', c) for c in calib_dfs]:
        paths = df['_img'].tolist()
        embs = extract_embs(model, paths, transform, device, tta=tta)
        sim = np.clip(embs @ embs.T, 0, 1).astype(np.float32)
        np.fill_diagonal(sim, 0)
        labels = df['_id'].astype('category').cat.codes.values
        best = -1
        for t in np.arange(0.3, 0.95, 0.025):
            adj = (sim > t).astype(np.float32)
            _, cl = connected_components(csr_matrix(adj), directed=False)
            ari = adjusted_rand_score(labels, cl)
            if ari > best: best = ari
        aris.append(best)
        log.info(f"    {name}: ARI={best:.4f} (n={len(df)}, ids={df['_id'].nunique()})")
    return float(np.mean(aris))

def main():
    seed_everything(config.RANDOM_SEED)
    device = config.DEVICE

    all_df = load_all_turtle_data()
    val_df = pd.read_csv(config.V4_TURTLE_VAL_CSV)
    v4 = config.V4_FEATURES
    calib_dfs = [pd.read_csv(v4 / f'subset_turtle_{n}.csv') for n in ['calib_a', 'calib_b']]

    unique_ids = sorted(all_df['_id'].unique())
    id2class = {i: c for c, i in enumerate(unique_ids)}
    num_classes = len(unique_ids)
    all_df['_label'] = all_df['_id'].map(id2class)

    train_tf = build_train_transforms()
    eval_tf  = build_eval_transforms()

    ds = TurtleDatasetMulti(all_df.to_dict('records'), train_tf)
    sampler = PKSampler(all_df['_label'].tolist(), batch_size=32, num_instances=4)
    loader  = torch.utils.data.DataLoader(ds, batch_size=32, sampler=sampler,
                                           num_workers=config.NUM_WORKERS, pin_memory=True)

    model = MegaDescArcFace(num_classes, use_subcenter=True).to(device)

    # Start from stage8a checkpoint for warm start
    stage8a = config.V5_CKPTS_DIR / 'stage8a_turtle_extended_best.pth'
    if stage8a.exists():
        ckpt = torch.load(stage8a, map_location='cpu', weights_only=False)
        if ckpt['num_classes'] == num_classes:
            model.load_state_dict(ckpt['model_state_dict'])
            log.info(f"Warm-start from stage8a (val_ari={ckpt['val_ari']:.4f})")
        else:
            log.info(f"num_classes mismatch ({ckpt['num_classes']} vs {num_classes}), training from scratch")

    ckpt_path = config.V5_CKPTS_DIR / 'stage9_turtle_long_best.pth'
    scaler = torch.amp.GradScaler('cuda')

    baseline_ari = eval_val_ari_calib(model, val_df, eval_tf, device, calib_dfs)
    log.info(f"Baseline mean ARI (val+calib): {baseline_ari:.4f}")

    best_ari = baseline_ari
    global_ep = 0

    for ph_name, n_ep, bb_frac, lr_h, lr_b, s_st, s_en, m_st, m_en in PHASES_LONG:
        log.info(f"\n{'='*70}")
        log.info(f"{ph_name}: {n_ep} ep  bb={bb_frac*100:.0f}%  lr_h={lr_h:.0e}  lr_b={lr_b:.0e}")
        log.info('='*70)
        set_trainable(model, bb_frac)
        opt = build_optimizer(model, lr_h, lr_b)
        steps = len(loader) * n_ep
        warmup = max(1, int(0.1 * steps))
        step = 0

        for ep in range(n_ep):
            cur_s = ramp(ep, max(1, n_ep-1), s_st, s_en)
            cur_m = ramp(ep, max(1, n_ep-1), m_st, m_en)
            model.train(); t0 = time.time(); tot_loss = tot_n = 0
            for imgs, labels in loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                lr_mult = (step / max(1, warmup)) if step < warmup else \
                          (0.01 + 0.99 * 0.5 * (1 + __import__('math').cos(__import__('math').pi * (step-warmup)/max(1,steps-warmup))))
                for g in opt.param_groups:
                    g['lr'] = (lr_h if g.get('name')=='head' else lr_b) * lr_mult
                step += 1
                with torch.amp.autocast('cuda'):
                    logits, _ = model(imgs, labels=labels, s=cur_s, m=cur_m)
                    loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
                if not torch.isfinite(loss): break
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 2.0)
                scaler.step(opt); scaler.update()
                tot_loss += loss.item() * imgs.size(0); tot_n += imgs.size(0)

            log.info(f"  [ep {global_ep:2d}] {ph_name} loss={tot_loss/max(1,tot_n):.3f} s={cur_s:.0f} m={cur_m:.2f} ({time.time()-t0:.0f}s)")

            if global_ep % 2 == 0 or ep == n_ep-1:
                ari = eval_val_ari_calib(model, val_df, eval_tf, device, calib_dfs)
                log.info(f"    mean ARI={ari:.4f} (best {best_ari:.4f})")
                if ari > best_ari:
                    best_ari = ari
                    torch.save({'model_state_dict': model.state_dict(), 'val_ari': ari,
                                'num_classes': num_classes, 'use_subcenter': True, 'use_masked': False,
                                'epoch': global_ep}, ckpt_path)
                    log.info(f"    [ckpt] saved ARI={ari:.4f}")
            global_ep += 1

    log.info(f"\nDone. Best mean ARI: {best_ari:.4f}")
    log.info(f"Checkpoint: {ckpt_path}")

if __name__ == '__main__': main()
