# AnimalCLEF26
Code, calibration sets, training logs, and the artefacts behind each negative result obtained by me for the AnimalCLEF2026 challenge are all archived here!
```markdown
# Wildlife Individual Re-Identification Pipeline

Companion code for:
**"Matching the Baseline, and Why That Is Worth Writing About:
A Post-Mortem of an AnimalCLEF 2026 Re-Identification Pipeline"**,
CLEF 2026 Working Notes (CEUR Workshop Proceedings, vol. XXXX).

## What this is

A four-stage open-world wildlife re-ID pipeline for AnimalCLEF 2026
(SeaTurtleID2022, TexasHornedLizards, LynxID2025, SalamanderID2025).
Officially-ranked private ARI **0.20371**; highest private (unofficial,
excluded by Kaggle top-5-public auto-selection) **0.20565**. Both sit
at the organisers' MegaDescriptor + MIEW + DBSCAN baseline of ≈0.20.

We publish this pipeline as a methodological reference, not a
leaderboard winner. The repository is structured so every component
is independently inspectable, including the failure modes.

## Results

| Submission | Private ARI | Public ARI | Officially ranked? |
|---|---|---|---|
| stage7c_final            | **0.2057** | 0.1668 | No (excluded by top-5-public) |
| stage8f_final_fixed      | 0.2037     | **0.1742** | Yes |
| stage2d_arcface_miew     | 0.1885     | 0.1740 | Yes |
| stage4c_final            | 0.1798     | 0.1696 | Yes |
| *Organisers' baseline*   | *≈0.20*    | --- | --- |

`step3a_miew_fusion` and `step6a_three_backbone` (both 0.1841 private)
were produced by an older internal pipeline and are omitted from this
repository; `stage4c_final` reproduces a comparable result with the
released code.

## What works (positive findings)

1. **Phase 0 head warmup**: three-epoch head-only training eliminates
   training divergence (58% → 0% across 27 runs) at zero extra compute.
2. **MIEW similarity fusion** (w=0.6 for turtles): +0.084 DMCS ARI for
   zero training compute.
3. **All-species clustering**: clustering even weak lynx/salamander
   models rather than emitting singletons adds 8.1% private ARI.

## What works conditionally

4. **Density-matched calibration (DMCS)** with
   τ\*(ρ) = 0.728 − 0.057 log ρ (R² = 0.94).
   The framework is correct; we mis-estimated test density in this
   competition.

## What does not transfer to private (negative findings)

5. Cross-species transfer from BalearicLizard (*P. lilfordi*) to
   TexasHornedLizards (*P. cornutum*): zero discriminative signal.
6. XGBoost meta-classifier on validation pair features: val ARI 0.993,
   DMCS ARI 0.712 (in-sample overfit).
7. Extended 48-epoch training: val ARI 0.942, DMCS ARI 0.783
   (identity-density distribution shift in model selection).
8. CzechLynx augmentation: improved val ARI but reduced private ARI.

See `negatives/` for runnable scripts and preserved logs for each.

## Quick start

```bash
conda create -n wildlifereid python=3.10
conda activate wildlifereid
pip install -r requirements.txt

# Train turtle ArcFace with the four-phase curriculum
python stage2_training/stage2b_train_arcface.py

# Generate the officially-ranked submission
python stage4_submission/stage8f_final_submission.py
```

For the DMCS calibration framework on your own survey:

```python
import math
def predicted_threshold(rho_target: float) -> float:
    """tau*(rho) = 0.728 - 0.057 * log(rho).  Eq. (3) of the paper."""
    return 0.728 - 0.057 * math.log(rho_target)

tau = predicted_threshold(rho_target=2.0)  # sparse-survey example
```

## Reproducing the paper

See `docs/REPRODUCIBILITY.md`. The compiled paper PDF is in
`paper/clef_v4_combined.pdf`; the source is `paper/clef_v4_combined.tex`.

## Citation

```bibtex
@inproceedings{yourname2026wildlife,
  title     = {Matching the Baseline, and Why That Is Worth Writing About: A Post-Mortem of an AnimalCLEF 2026 Re-Identification Pipeline},
  author    = {[Your Full Name]},
  booktitle = {CLEF 2026 Working Notes},
  publisher = {CEUR Workshop Proceedings},
  year      = {2026}
}
```

## License

MIT
```
