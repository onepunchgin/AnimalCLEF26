#!/usr/bin/env python

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

import config
from utils import log, seed_everything, write_submission


# Ordered fallback chain per species. Earliest existing entry wins.
SOURCES: dict[str, list[str]] = {
    "turtle": [
        "stage8b_turtle_clusters.csv",
        "stage5a_db_guided_turtles.csv",
    ],
    "lizard": [
        "stage8e_lizard_clusters.csv",
        "stage6c_texas_lizard_clusters.csv",
        "stage5b_lizard_clusters.csv",
    ],
    "lynx": [
        "stage8d_lynx_clusters.csv",
        "stage7b_LynxID2025_clusters.csv",
    ],
    "salamander": [
        "stage7b_SalamanderID2025_clusters.csv",
    ],
}


def pick_source(species: str, candidates: list[str]) -> Path:
    for name in candidates:
        p = config.V5_SUBMISSIONS_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"no source CSV found for {species}. Looked for:\n"
        + "\n".join(f"  - {n}" for n in candidates)
    )


def load_csv(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info(f"loaded {label}: {len(df)} rows, "
             f"unique clusters: {df['cluster'].nunique()}  "
             f"(source: {path.name})")
    return df


def main() -> int:
    seed_everything(config.RANDOM_SEED)

    log.info("=" * 70)
    log.info("Stage 8f -- Final submission (4 species, best-of stage 8 vs prior)")
    log.info("=" * 70)

    sample_csv = config.COMPETITION_SAMPLE_SUB
    if not sample_csv.exists():
        log.error(f"missing sample submission: {sample_csv}")
        return 1
    sample = pd.read_csv(sample_csv)
    log.info(f"sample submission: {len(sample)} rows")

    chosen_sources: dict[str, str] = {}
    dataframes: list[pd.DataFrame] = []
    for species, candidates in SOURCES.items():
        try:
            path = pick_source(species, candidates)
        except FileNotFoundError as e:
            log.error(str(e))
            return 1
        chosen_sources[species] = path.name
        log.info(f"  -> {species:<10s} source: {path.name}")
        dataframes.append(load_csv(path, species))

    log.info("")
    log.info("Source summary (chosen for each species):")
    for species, name in chosen_sources.items():
        log.info(f"  {species:<10s} <- {name}")

    log.info("")
    log.info("combining sub-submissions ...")
    submission = pd.concat(dataframes, ignore_index=True)
    n_before = len(submission)
    submission = submission.drop_duplicates(subset="image_id", keep="first")
    n_after = len(submission)
    if n_before != n_after:
        log.warning(f"dropped {n_before - n_after} duplicate image_id rows")
    log.info(f"combined rows: {n_after}")

    out_path = config.V5_SUBMISSIONS_DIR / "stage8f_final.csv"
    write_submission(submission, out_path, sample_submission=sample)
    log.info(f"Final submission: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
