#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

from utils import log, seed_everything, timed
import config


def train_xgboost(X, y, params, n_estimators=500, early_stopping=50):
    """Train one XGBoost model with early stopping on a held-out fold."""
    import xgboost as xgb

    # 80/20 split for early stopping
    rng = np.random.default_rng(42)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    pos_train = rng.choice(pos_idx, int(0.8 * len(pos_idx)), replace=False)
    pos_val = np.setdiff1d(pos_idx, pos_train)
    neg_train = rng.choice(neg_idx, int(0.8 * len(neg_idx)), replace=False)
    neg_val = np.setdiff1d(neg_idx, neg_train)

    train_idx = np.concatenate([pos_train, neg_train])
    val_idx = np.concatenate([pos_val, neg_val])

    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
    dval = xgb.DMatrix(X[val_idx], label=y[val_idx])

    booster = xgb.train(
        params=params, dtrain=dtrain, num_boost_round=n_estimators,
        evals=[(dval, "val")], early_stopping_rounds=early_stopping,
        verbose_eval=50,
    )
    return booster, booster.best_iteration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-name", default="stage3b_xgboost.json")
    args = parser.parse_args()

    seed_everything(config.RANDOM_SEED)

    feat_path = config.V5_PAIR_FEATURES_DIR / "val_pair_features.npy"
    label_path = config.V5_PAIR_FEATURES_DIR / "val_pair_labels.npy"
    if not (feat_path.exists() and label_path.exists()):
        log.error("missing pair features; run stage3a first")
        return 1

    X = np.load(feat_path)
    y = np.load(label_path)
    log.info(f"X: {X.shape}, y: {y.shape}, pos rate: {y.mean()*100:.3f}%")

    # Class imbalance: scale_pos_weight = neg_count / pos_count
    n_pos = y.sum()
    n_neg = (y == 0).sum()
    spw = n_neg / max(1, n_pos)
    log.info(f"scale_pos_weight = {spw:.1f}")

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",  # AUC-PR is meaningful for imbalanced data
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": float(spw),
        "tree_method": "hist",
        "device": "cuda" if config.DEVICE == "cuda" else "cpu",
    }

    log.info(f"params: {params}")

    with timed("XGBoost training"):
        booster, best_iter = train_xgboost(X, y, params)

    out_path = config.V5_MODELS_DIR / args.out_name
    log.info(f"Best iteration from early stopping: {best_iter}")
    log.info("Retraining final model on full dataset...")
    import xgboost as xgb
    dtrain_full = xgb.DMatrix(X, label=y)
    final_booster = xgb.train(params, dtrain_full, num_boost_round=best_iter + 1, verbose_eval=False)
    final_booster.save_model(str(out_path))
    log.info(f"model saved: {out_path}")

    # Quick CV check (3 folds on val)
    log.info("")
    log.info("=== CV evaluation (3-fold on val) ===")
    import xgboost as xgb
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    auprcs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        dtrain = xgb.DMatrix(X[tr_idx], label=y[tr_idx])
        dval = xgb.DMatrix(X[va_idx], label=y[va_idx])
        booster_cv = xgb.train(params, dtrain, num_boost_round=200,
                                  evals=[(dval, "val")],
                                  early_stopping_rounds=30, verbose_eval=False)
        preds = booster_cv.predict(dval)
        auc = roc_auc_score(y[va_idx], preds)
        auprc = average_precision_score(y[va_idx], preds)
        aucs.append(auc)
        auprcs.append(auprc)
        log.info(f"  fold {fold}: AUC={auc:.4f}, AUPRC={auprc:.4f}")
    log.info(f"  mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    log.info(f"  mean AUPRC: {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")

    # Feature importance
    importance = booster.get_score(importance_type="gain")
    feat_names_path = config.V5_PAIR_FEATURES_DIR / "feature_names.json"
    if feat_names_path.exists():
        with open(feat_names_path) as f:
            feat_names = json.load(f)
        log.info("")
        log.info("Feature importance (top 10 by gain):")
        named = []
        for k, v in importance.items():
            idx = int(k.replace("f", ""))
            if idx < len(feat_names):
                named.append((feat_names[idx], v))
            else:
                named.append((k, v))
        named.sort(key=lambda x: -x[1])
        for name, gain in named[:10]:
            log.info(f"  {name}: {gain:.2f}")

    log.info("")
    log.info("Stage 3b done. Next: stage3c_apply_cluster.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
