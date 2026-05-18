#!/usr/bin/env python

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

import config
from utils import log, seed_everything, write_submission


# =====================================================================
# Helpers
# =====================================================================

def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing {label} submission: {path}")
    df = pd.read_csv(path)
    log.info(f"loaded {label}: {len(df)} rows, "
             f"unique clusters: {df['cluster'].nunique()}  "
             f"({path.name})")
    return df


def pick_lizard_csv() -> Path:
    """Prefer stage6c, fall back to stage5b."""
    candidates = [
        config.V5_SUBMISSIONS_DIR / "stage6c_texas_lizard_clusters.csv",
        config.V5_SUBMISSIONS_DIR / "stage5b_lizard_clusters.csv",
    ]
    for p in candidates:
        if p.exists():
            log.info(f"lizard source: {p.name}")
            return p
    raise FileNotFoundError(
        "no lizard sub-submission found. Looked for:\n"
        + "\n".join(f"  - {p}" for p in candidates)
    )


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    seed_everything(config.RANDOM_SEED)

    log.info("=" * 70)
    log.info("Stage 7c -- Final submission (4 species)")
    log.info("=" * 70)

    sample_csv = config.COMPETITION_SAMPLE_SUB
    if not sample_csv.exists():
        log.error(f"missing sample submission: {sample_csv}")
        return 1
    sample = pd.read_csv(sample_csv)
    log.info(f"sample submission: {len(sample)} rows")

    turtle_csv = (
        config.V5_SUBMISSIONS_DIR / "stage5a_db_guided_turtles.csv"
    )
    lizard_csv = pick_lizard_csv()
    lynx_csv = (
        config.V5_SUBMISSIONS_DIR / "stage7b_LynxID2025_clusters.csv"
    )
    salamander_csv = (
        config.V5_SUBMISSIONS_DIR / "stage7b_SalamanderID2025_clusters.csv"
    )

    try:
        turtle = load_csv(turtle_csv, "turtle")
        lizard = load_csv(lizard_csv, "lizard")
        lynx = load_csv(lynx_csv, "lynx")
        salamander = load_csv(salamander_csv, "salamander")
    except FileNotFoundError as e:
        log.error(str(e))
        return 1

    log.info("")
    log.info("combining sub-submissions ...")
    submission = pd.concat(
        [turtle, lizard, lynx, salamander], ignore_index=True,
    )
    n_before = len(submission)
    submission = submission.drop_duplicates(subset="image_id", keep="first")
    n_after = len(submission)
    if n_before != n_after:
        log.warning(f"dropped {n_before - n_after} duplicate image_id rows")
    log.info(f"combined rows: {n_after}")

    out_path = config.V5_SUBMISSIONS_DIR / "stage7c_final.csv"
    write_submission(submission, out_path, sample_submission=sample)
    log.info(f"Final submission: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
