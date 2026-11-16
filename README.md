# Multi-platform self-supervised learning for crop trait estimation from UAV and satellite imagery

*Multi-Platform Self-Supervised Learning with Crossmodal Pretraining for Crop Trait
Estimation from UAV and Satellite Imagery*.

<p align="center">
<img src="assets/overview.png" width=100% height=100% class="center">
</p>



The framework has two components, trained and evaluated with the same architecture, initialisation, data and schedule:

- **Unimodal masked autoencoders** for UAV RGB imagery (UAV-MAE) and for satellite imagery (Sat-MAE).
- **A crossmodal masked autoencoder** (MultiMAE-style) trained on co-registered satellite–UAV plot pairs. The masking
  is biased toward the satellite modality (Dirichlet α = (0.9, 0.1)), so the model mostly predicts UAV content from
  the satellite view. At inference the encoder takes either modality alone (XM-UAV, XM-Sat) or both (XM-Sat+UAV).

Every encoder is pretrained leave-one-location-out (5 held-out maize locations × 3 seeds) and evaluated on plot yield
with frozen ridge probes in three settings: all-field (one probe over all locations), within-site (genotype-grouped
5-fold CV at the held-out location) and cross-site (trained on the four other locations, new genotypes).

## Setup

```bash
git clone <this repository> satuav && cd satuav
git clone https://github.com/EPFL-VILAB/MultiMAE.git && (cd MultiMAE && git checkout 66910f5)   # used unmodified
mkdir -p MultiMAE/weights
# download multimae-b_98_rgb+-depth-semseg_1600e_multivit-afff3f8c.pth from the MultiMAE README into MultiMAE/weights/
```

Python 3.8 with CUDA 11.3 (the MultiMAE code requires `timm==0.4.12`, which does not run on recent PyTorch):

```bash
uv python install 3.8 && uv venv --python 3.8 .venv && source .venv/bin/activate
uv pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
uv pip install -r requirements.txt
```

## Data

The dataset is *Crop performance, aerial, and satellite data from multistate maize yield trials* (2022 season).
Place it at `MultiMAE/data/` (or set `SATUAV_DATA`):

```
MultiMAE/data/
  Sat/<Location>/TP1..TP6/*.TIF      Pléiades Neo plot crops, 6 bands (R, G, B, NIR, red edge, deep blue), 30 cm
  UAV/<Location>/TP1..TP3/*.PNG      UAV RGB plot crops
  HYBRID_HIPS_V3.5_ALLPLOTS.csv      ground truth (genotype, nitrogen, yield, ...)
  DateofCollection.xlsx              acquisition dates
```

Locations: Ames, Crawfordsville, Lincoln, MOValley (Missouri Valley), Scottsbluff. Outputs (`data/`, `runs/`,
`features/`, `results/`, ...) are written to the repository root, or to `SATUAV_OUT` if set.

## Reproducing the manuscript

`scripts/run_all.sh` runs the full pipeline in order; every step skips work that is already done.

| Step | Command | Output |
|---|---|---|
| Manifest, plot tiles, genotype folds | `python -m satuav.manifest` | `data/` |
| Handcrafted features | `python -m satuav.extract handcrafted` | `features/handcrafted/` |
| LOLO pretraining + embeddings | `python -m satuav.sweep rgb main seeds uavmae uavctl` | `runs/`, `features/` |
| UAV-input crossmodal embeddings | `python -m satuav.extract latent <run> ...` | `features/<run>__lat/` |
| Linear probes | `python -m satuav.evaluate main rgb latent uavmae uavctl latefuse pooledsite` | `results/*.csv` |
| All manuscript numbers + t-tests | `python -m satuav.paper_stats` | `results/paper_*.csv` |

`satuav.paper_stats` writes:

- `paper_main.csv`: Table 3 (absolute R² and Δ with paired t-tests, all-field / within-site / cross-site)
- `paper_per_location.csv`: Table 4 (R² at each held-out location)
- `paper_comparisons.csv`: Table 5 (UAV-only controls), Table 6 and text (infrared bands), joint vs separate input,
  feature-level fusion, and features of UAV images generated from satellite input
- `paper_timepoints.csv`, `paper_ranking.csv`: Supplementary Tables S3 and S4

Statistics: for each comparison, scores are averaged over seeds at each held-out location, and Δ is tested with a
two-sided paired t-test over the five locations (df = 4; \* p < 0.05).

Pretraining runs are defined by the YAML files in `configs/` (written by `satuav.sweep`); a single run can be started
with `python -m satuav.pretrain configs/xmodalrgb_s0_Ames.yaml`.

| Config prefix | Model | Used for |
|---|---|---|
| `xmodalrgb_s{0,1,2}_<loc>` | crossmodal, satellite RGB + UAV RGB | XM-UAV, XM-Sat, XM-Sat+UAV |
| `satmaergb_s{0,1,2}_<loc>` | satellite-only MAE, RGB | Sat-MAE |
| `uavmae_s{0,1,2}_<loc>` | UAV-only MAE | UAV-MAE |
| `uavmaeF_*`, `uavmaeM_*` | UAV-only MAE, per-flight normalisation / matched masking | controls |
| `xmodal_s*`, `satmae_s*` | the same with all six satellite bands | infrared-band section |
| `abl-a*`, `abl-p0_*` | masking ablations (seed 0, 3 held-out locations) | Supplementary Table S2 |

## Package layout

| Module | Purpose |
|---|---|
| `paths.py` | paths and constants (override with `SATUAV_MULTIMAE`, `SATUAV_DATA`, `SATUAV_OUT`) |
| `manifest.py` | file/ground-truth matching, rotation-aligned 224 px tiles, QC flags, genotype folds |
| `data.py` | tile loading, per-fold normalisation statistics, datasets |
| `model.py` | MultiMAE wrapper: per-modality patch embeddings and decoders, initialisation from released weights |
| `pretrain.py` | pretraining loop (biased Dirichlet masking, masked-patch/valid-pixel MSE, augmentation) |
| `sweep.py` | run definitions (configs) and resumable pretrain → extract driver |
| `extract.py` | handcrafted features and encoder embeddings per plot |
| `probe.py` | ridge probes with nested CV in the all-field (P), within-site (A) and cross-site (B_z) settings |
| `evaluate.py` | probe job sets for each group of runs |
| `paper_stats.py` | every number in the manuscript, with paired t-tests |
| `ideas.py`, `report.py`, `summary.py` | result loading helpers and the masking-ablation table |
| `field_demo.py` | field-level satellite → UAV prediction figure (Supplementary Figure S2) |

Some modules contain options for exploratory runs that are not part of the manuscript (for example the `trans`,
`fix` and `ctx` sweep groups); they are kept for completeness and are not needed to reproduce the reported results.
