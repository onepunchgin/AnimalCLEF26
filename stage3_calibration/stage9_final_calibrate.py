#!/usr/bin/env python

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np, pandas as pd, torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import adjusted_rand_score

import config
from utils import log, seed_everything, cluster_agg, write_submission, synthesize_nonscored
from stage2b_train_arcface import MegaDescArcFace, build_eval_transforms

seed_everything(config.RANDOM_SEED)

# ── load best available model ──────────────────────────────────────────────────
def load_best_model():
    stage9 = config.V5_CKPTS_DIR / 'stage9_turtle_long_best.pth'
    stage8 = config.V5_CKPTS_DIR / 'stage8a_turtle_extended_best.pth'
    # stage9: 48 epochs, 9524 images, val ARI 0.9421 >> stage8a 0.858.
    # Stored val_ari metrics use incompatible scales, so always prefer stage9.
    if stage9.exists():
        c9 = torch.load(stage9, map_location='cpu', weights_only=False)
        log.info(f'Using stage9 (48ep on 9524 imgs, val_ARI_peak=0.9421, stored={c9.get("val_ari",0):.4f})')
        ckpt_path = stage9
    else:
        log.info('stage9 not found, using stage8a')
        ckpt_path = stage8
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = MegaDescArcFace(ckpt['num_classes'], ckpt['use_subcenter']).to(config.DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt['val_ari'], str(ckpt_path.name)

def tta_embs(model, paths, n_tta=3):
    tf0 = build_eval_transforms()
    tf1 = T.Compose([T.RandomHorizontalFlip(p=1.0)] + list(tf0.transforms))
    tf2 = T.Compose([T.Resize((420,420)), T.CenterCrop(config.MEGADESC_INPUT_SIZE),
                     T.ToTensor(), T.Normalize(config.MEGADESC_NORM_MEAN, config.MEGADESC_NORM_STD)])
    tfs = [tf0, tf1, tf2][:n_tta]
    all_embs = []
    with torch.inference_mode():
        for tf in tfs:
            embs = []
            for s in range(0, len(paths), config.FEATURE_BATCH_SIZE):
                b = torch.stack([tf(Image.open(p).convert('RGB')) for p in paths[s:s+config.FEATURE_BATCH_SIZE]]).to(config.DEVICE)
                embs.append(model.extract_embeddings(b).cpu().numpy())
            all_embs.append(np.concatenate(embs))
    avg = np.mean(all_embs, axis=0)
    return avg / np.linalg.norm(avg, axis=1, keepdims=True).clip(1e-8)

def extract_miew(paths):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(config.MIEW_HF_HUB_ID, trust_remote_code=True).to(config.DEVICE).eval()
    tf = T.Compose([T.Resize((440,440)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    embs = []
    bs = max(1, config.FEATURE_BATCH_SIZE//2)
    with torch.inference_mode():
        for s in range(0, len(paths), bs):
            b = torch.stack([tf(Image.open(p).convert('RGB')) for p in paths[s:s+bs]]).to(config.DEVICE)
            out = m(b)
            e = out if not isinstance(out,dict) else out.get('embedding', list(out.values())[0])
            embs.append(e.cpu().numpy())
    del m; torch.cuda.empty_cache()
    return np.concatenate(embs)

def l2(x): return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)

def krerank(sim, k=20, lam=0.3):
    N = sim.shape[0]
    top_k = np.argsort(-sim, axis=1)[:, :k]
    ind = np.zeros((N,N), dtype=bool)
    for i in range(N): ind[i, top_k[i]] = True
    recip = ind & ind.T
    top_h = np.argsort(-sim, axis=1)[:, :k//2]
    indh = np.zeros((N,N), dtype=bool)
    for i in range(N): indh[i, top_h[i]] = True
    recip_h = indh & indh.T
    exp = recip.copy()
    for i in range(N):
        for j in np.where(recip[i])[0]:
            Rj = np.where(recip_h[j])[0]
            if len(Rj) > 0 and recip[i][Rj].sum() >= 2/3*len(Rj):
                exp[i,Rj]=True; exp[Rj,i]=True
    R = exp.astype(np.float32)
    inter = R @ R.T; sz = R.sum(1)
    union = sz[:,None]+sz[None,:]-inter
    jac = np.where(union>0, inter/union, 0.0)
    np.fill_diagonal(jac, 0)
    rr = (1-lam)*sim + lam*jac.astype(np.float32)
    np.fill_diagonal(rr, 0)
    return np.clip(rr, 0, 1)

def main():
    model, model_ari, model_name = load_best_model()
    log.info(f'Model: {model_name}  stored_ari={model_ari:.4f}')
    v4 = config.V4_FEATURES

    # ── Calibrate on calib_a and calib_b with TTA + re-ranking ───────────────
    log.info('=== Calibrating on calib_a + calib_b ===')
    best_params = {'ari': -1}
    for cal_name in ['calib_a', 'calib_b']:
        df = pd.read_csv(v4 / f'subset_turtle_{cal_name}.csv')
        labels = df['_id'].astype('category').cat.codes.values
        paths = df['_img'].tolist()
        log.info(f'Extracting TTA ArcFace for {cal_name} ({len(paths)} imgs)...')
        arc = tta_embs(model, paths, n_tta=3)
        cache_p = config.V5_FEATURES_DIR / f'calib_{cal_name}_tta_embs.npy'
        np.save(str(cache_p), arc)
        log.info(f'Extracting MIEW for {cal_name}...')
        miew = l2(extract_miew(paths))
        np.save(str(config.V5_FEATURES_DIR / f'calib_{cal_name}_miew.npy'), miew)
        sim_arc  = np.clip(arc @ arc.T, 0,1).astype(np.float32); np.fill_diagonal(sim_arc,0)
        sim_miew = np.clip(miew @ miew.T,0,1).astype(np.float32); np.fill_diagonal(sim_miew,0)
        log.info(f'{cal_name}: grid search TTA+RR fusion...')
        for w_miew in [0.4, 0.5, 0.6, 0.7]:
            fused = (1-w_miew)*sim_arc + w_miew*sim_miew
            np.fill_diagonal(fused, 0)
            fused_rr = krerank(fused.astype(np.float32), k=20, lam=0.3)
            best_t_here, best_ari_here, best_nc = 0,-1,0
            for t in np.arange(0.30, 0.92, 0.01):
                cids = cluster_agg(fused_rr, float(t))
                ari = adjusted_rand_score(labels, cids)
                if ari > best_ari_here:
                    best_ari_here, best_t_here, best_nc = ari, float(t), len(np.unique(cids))
            log.info(f'  {cal_name} w={w_miew:.1f}: t={best_t_here:.3f} ARI={best_ari_here:.4f} nc={best_nc}')
            if best_ari_here > best_params['ari']:
                best_params = {'ari': best_ari_here, 'w_miew': w_miew, 't': best_t_here,
                               'nc': best_nc, 'cal': cal_name}
    log.info(f'BEST params: {best_params}')

    # Average thresholds across both calib sets at best w_miew
    w = best_params['w_miew']
    ts = []
    for cal_name in ['calib_a', 'calib_b']:
        df = pd.read_csv(v4 / f'subset_turtle_{cal_name}.csv')
        labels = df['_id'].astype('category').cat.codes.values
        arc = np.load(str(config.V5_FEATURES_DIR / f'calib_{cal_name}_tta_embs.npy'))
        miew = np.load(str(config.V5_FEATURES_DIR / f'calib_{cal_name}_miew.npy'))
        sim_arc = np.clip(arc @ arc.T,0,1).astype(np.float32); np.fill_diagonal(sim_arc,0)
        sim_miew = np.clip(miew@miew.T,0,1).astype(np.float32); np.fill_diagonal(sim_miew,0)
        fused = (1-w)*sim_arc + w*sim_miew; np.fill_diagonal(fused,0)
        fused_rr = krerank(fused.astype(np.float32))
        best_t, best_ari, best_nc = 0,-1,0
        for t in np.arange(0.30, 0.92, 0.01):
            cids = cluster_agg(fused_rr, float(t))
            ari = adjusted_rand_score(labels, cids)
            if ari > best_ari:
                best_ari, best_t, best_nc = ari, float(t), len(np.unique(cids))
        log.info(f'  {cal_name} @ w={w}: t={best_t:.3f} ARI={best_ari:.4f} nc={best_nc}')
        ts.append(best_t)
    t_final = float(np.mean(ts))
    log.info(f'Final threshold (avg across calib): t={t_final:.3f}  w_miew={w}')

    # ── Apply to test ─────────────────────────────────────────────────────────
    test_df = pd.read_csv(config.V4_TURTLE_TEST_CSV)
    log.info(f'Extracting TTA ArcFace for test ({len(test_df)} imgs)...')
    arc_test = tta_embs(model, test_df['_img'].tolist(), n_tta=3)
    miew_test = l2(np.load(str(v4 / 'miew_test_turtles.npy')))
    sim_arc_t  = np.clip(arc_test @ arc_test.T, 0,1).astype(np.float32); np.fill_diagonal(sim_arc_t,0)
    sim_miew_t = np.clip(miew_test @ miew_test.T,0,1).astype(np.float32); np.fill_diagonal(sim_miew_t,0)
    fused_t = (1-w)*sim_arc_t + w*sim_miew_t; np.fill_diagonal(fused_t,0)
    fused_rr_t = krerank(fused_t.astype(np.float32))
    cluster_ids = cluster_agg(fused_rr_t, t_final)
    n_c = len(np.unique(cluster_ids))
    log.info(f'Test: {n_c} clusters at t={t_final:.3f} w_miew={w}')

    # ── Build submission ──────────────────────────────────────────────────────
    sample = pd.read_csv(config.COMPETITION_SAMPLE_SUB)
    lizard_df = pd.read_csv(config.V4_LIZARD_TEST_CSV)
    lynx = pd.read_csv(config.V5_SUBMISSIONS_DIR / 'stage8d_lynx_clusters.csv')
    salm = pd.read_csv(config.V5_SUBMISSIONS_DIR / 'stage7b_SalamanderID2025_clusters.csv')
    turtle_sub = pd.DataFrame({
        'image_id': test_df['image_id'].astype(str),
        'cluster': [f'cluster_SeaTurtleID2022_{c+1:04d}' for c in cluster_ids]
    })
    lizard_sub = pd.DataFrame({
        'image_id': lizard_df['image_id'].astype(str),
        'cluster': [f'cluster_TexasHornedLizards_{i:04d}' for i in range(len(lizard_df))]
    })
    nonscored = synthesize_nonscored(sample)
    sub = pd.concat([turtle_sub, lizard_sub, lynx, salm, nonscored], ignore_index=True)
    sub = sub.drop_duplicates(subset='image_id', keep='first')
    out = config.V5_SUBMISSIONS_DIR / 'stage9_FINAL_submission1.csv'
    write_submission(sub, out, sample_submission=sample)
    log.info(f'=== SUBMISSION 1 READY: {out.name} ===')
    log.info(f'    Turtles: {n_c} clusters | t={t_final:.3f} | w_miew={w} | TTA+RR | {model_name}')
    log.info(f'    Best calib ARI: {best_params["ari"]:.4f}')
    
    # Also generate t±0.03 variants for submission 2 choice
    for dt in [-0.03, +0.03]:
        t2 = round(t_final + dt, 3)
        cids2 = cluster_agg(fused_rr_t, t2)
        n2 = len(np.unique(cids2))
        sub2 = sub.copy()
        sub2.loc[sub2['image_id'].isin(test_df['image_id'].astype(str)),
                 'cluster'] = [f'cluster_SeaTurtleID2022_{c+1:04d}' for c in cids2]
        out2 = config.V5_SUBMISSIONS_DIR / f'stage9_FINAL_sub2_t{int(t2*1000)}.csv'
        write_submission(sub2, out2, sample_submission=sample)
        log.info(f'Variant t={t2:.3f}: {n2} clusters -> {out2.name}')

if __name__ == '__main__': main()
