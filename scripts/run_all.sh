#!/usr/bin/env bash
# Full pipeline for the manuscript, in order. Run from the repository root with the environment activated.
# Each step is resumable: finished pretraining runs and feature files are skipped.
# Approximate cost on one A100 (40 GB): ~0.15-0.30 GPU-hours per pretraining run, ~120 runs in total.
set -euo pipefail
cd "$(dirname "$0")/.."

LOCS="Ames Crawfordsville Lincoln MOValley Scottsbluff"

# 1. Data: match files to ground truth, rotate/crop each plot into three 224 px tiles, write genotype folds.
python -m satuav.manifest

# 2. Handcrafted features (band statistics + vegetation indices) for every plot, and the ImageNet-encoder
#    baseline that `evaluate main` also probes (not reported in the manuscript).
python -m satuav.extract handcrafted
python -m satuav.extract imnet

# 3. Leave-one-location-out pretraining (5 held-out locations x 3 seeds per model) and embedding extraction.
#    rgb    : crossmodal (sat RGB + UAV RGB) and Sat-MAE on satellite RGB        -> XM-UAV, XM-Sat, Sat-MAE
#    main, seeds : the same with all six satellite bands                        -> infrared-band section
#    uavmae : UAV-only MAE                                                      -> UAV-MAE
#    uavctl : UAV-only controls (per-flight normalisation, matched masking)     -> UAV-MAE_flight, UAV-MAE_mask
#    ablate : masking ablations (Supplementary Table S2; also runs further ablations not reported in the paper)
python -m satuav.sweep rgb main seeds uavmae uavctl
python -m satuav.sweep ablate   # optional, Supplementary Table S2 only

# 4. UAV-input embeddings of the crossmodal encoders (XM-UAV) and joint embeddings (XM-Sat+UAV).
for s in 0 1 2; do
  for L in $LOCS; do
    python -m satuav.extract latent "xmodalrgb_s${s}_${L}" "xmodal_s${s}_${L}"
  done
done

# 5. Linear probes in the three evaluation settings (all-field P, within-site A, cross-site B_z).
python -m satuav.evaluate main rgb latent uavmae uavctl latefuse
python -m satuav.evaluate pooledsite
python -m satuav.evaluate pooledsite pooledsite_tp
python -m satuav.evaluate ablate   # optional, Supplementary Table S2 only

# 6. Every number in the manuscript (Tables 3-6, S3, S4) with paired t-tests -> results/paper_*.csv
python -m satuav.paper_stats

# 7. Supplementary material: masking ablation table and the cloud-shadow field figure.
python -m satuav.report ablate
python -m satuav.field_demo --loc Ames --tp 2
