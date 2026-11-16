"""Paths used by every module.

By default everything is resolved relative to the repository root (the directory that contains this package):

  <repo>/MultiMAE/                  clone of https://github.com/EPFL-VILAB/MultiMAE (unmodified)
  <repo>/MultiMAE/data/             the dataset (Sat/, UAV/, HYBRID_HIPS_V3.5_ALLPLOTS.csv, DateofCollection.xlsx)
  <repo>/MultiMAE/weights/          the released MultiMAE ViT-B weights
  <repo>/data, runs, features, results, figs, paper_tables   outputs written by this code

Override with environment variables if your layout differs:
  SATUAV_MULTIMAE  path to the MultiMAE clone
  SATUAV_DATA      path to the dataset directory
  SATUAV_OUT       base directory for all outputs (default: the repository root)
"""
import os

EXP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MULTIMAE = os.environ.get('SATUAV_MULTIMAE', os.path.join(EXP, 'MultiMAE'))
DATA = os.environ.get('SATUAV_DATA', os.path.join(MULTIMAE, 'data'))
OUT = os.environ.get('SATUAV_OUT', EXP)

CSV = os.path.join(DATA, 'HYBRID_HIPS_V3.5_ALLPLOTS.csv')
DATES = os.path.join(DATA, 'DateofCollection.xlsx')
WEIGHTS = os.path.join(MULTIMAE, 'weights', 'multimae-b_98_rgb+-depth-semseg_1600e_multivit-afff3f8c.pth')

EDATA = os.path.join(OUT, 'data')
CACHE = os.path.join(EDATA, 'cache')
MANIFEST = os.path.join(EDATA, 'manifest.parquet')
SPLITS = os.path.join(EDATA, 'splits')
RUNS = os.path.join(OUT, 'runs')
FEATS = os.path.join(OUT, 'features')
RESULTS = os.path.join(OUT, 'results')
FIGS = os.path.join(OUT, 'figs')
TABLES = os.path.join(OUT, 'paper_tables')

LOCS = ['Ames', 'Crawfordsville', 'Lincoln', 'MOValley', 'Scottsbluff']
N_LOCS = ['Ames', 'Crawfordsville', 'Lincoln', 'Scottsbluff']  # locations with 3 N levels

# Satellite band order in the Pleiades Neo plot crops: R, G, B, NIR, RedEdge, DeepBlue.
SAT_BANDS = ['red', 'green', 'blue', 'nir', 'rededge', 'deepblue']
TILE = 224
N_TILES = 3

if MULTIMAE not in os.sys.path:
    os.sys.path.insert(0, MULTIMAE)
