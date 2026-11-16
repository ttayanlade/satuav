"""Downstream probes on frozen features (Section 6.2 / 6.5).

Regimes
  A     : within held-out location L, 5-fold genotype-grouped CV (shared genotype->fold map).
  B_raw : train on labelled plots of the 4 pretraining locations, test on L; raw targets, raw features.
  B_z   : as B_raw with targets and features z-scored within each location (relative performance).
In B, genotypes of the test fold are removed from the training locations (new site AND new genotypes).
  B_zg  : as B_z but the test-fold genotypes are kept at the training locations (new site, seen genotypes).
  B_zc  : as B_z, but L and the training locations all flew the same UAV camera (SAME_CAM; L must be one of them).
  B_zx  : as B_zc (same held-out L), but trained on the 2 sites flown with other cameras (OTHER_CAM).
  A_r   : as A, but plot-level random 5-fold split (a genotype's other plots can be in training); inner CV random too.
  B_zr  : as B_z, but plot-level random folds at L and no genotypes removed from the training sites
          (the training set no longer depends on the fold, so the pooled result equals B_zg).
Hyper-parameters are chosen by nested CV on the training split only (A: genotype groups; B: location groups).
"""
import argparse
import ctypes
import gc
import json
import os
import warnings
import zlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from . import data as D
from .paths import EDATA, FEATS, LOCS, N_LOCS, RESULTS, SPLITS

SAME_CAM = ['Ames', 'Crawfordsville', 'MOValley']  # DJI P4 Pro V2 (20 MP)
OTHER_CAM = ['Lincoln', 'Scottsbluff']  # P4 RTK + P1 (45 MP) / M600 + X3 (12 MP)

warnings.filterwarnings('ignore')
ALPHAS = np.logspace(-2, 6, 17)
CS = np.logspace(-4, 1, 6)
EXTRA_TASKS = {'stand_count': ['Ames', 'Crawfordsville', 'MOValley'], 'dta': ['Lincoln', 'Scottsbluff'],
               'gdd': ['Lincoln', 'Scottsbluff']}

# ------------------------------------------------------------------------------- representations
# Each representation = list of (source, key, axis). axis 'sat' -> satellite TP, 'uav' -> UAV TP.
# Sources: 'hand' (run independent), 'imnet' (imnet_<L>), or a run-group name resolved via `runs`.
REPS = {
    'H-sat-rgb': [('hand', 'H_sat_rgb', 'sat')],
    'H-sat-all': [('hand', 'H_sat_all', 'sat')],
    'H-uav': [('hand', 'H_uav', 'uav')],
    'H-pred': [('xmodal', 'H_pred', 'sat')],
    'H-sat-all+pred': [('hand', 'H_sat_all', 'sat'), ('xmodal', 'H_pred', 'sat')],
    'E-imnet': [('imnet', 'E_sat', 'sat')],
    'E-satMAE': [('satmae', 'E_sat', 'sat')],
    'E-xmodal': [('xmodal', 'E_sat', 'sat')],
    'E-xmodal-dec': [('xmodal', 'E_dec', 'sat')],
    'E-xmodal+H-sat-all': [('xmodal', 'E_sat', 'sat'), ('hand', 'H_sat_all', 'sat')],
    'E-satMAE+H-sat-all': [('satmae', 'E_sat', 'sat'), ('hand', 'H_sat_all', 'sat')],
    'E-uav-real': [('xmodal', 'E_uavreal', 'uav')],
    'F-late': [('hand', 'H_sat_all', 'sat'), ('hand', 'H_uav', 'uav')],
    # RGB-only set: generated UAV / embedding combined with real satellite RGB
    'H-sat-rgb+pred': [('hand', 'H_sat_rgb', 'sat'), ('xmodal', 'H_pred', 'sat')],
    'E-xmodal+H-sat-rgb': [('xmodal', 'E_sat', 'sat'), ('hand', 'H_sat_rgb', 'sat')],
    'E-satMAE+H-sat-rgb': [('satmae', 'E_sat', 'sat'), ('hand', 'H_sat_rgb', 'sat')],
    'F-late-rgb': [('hand', 'H_sat_rgb', 'sat'), ('hand', 'H_uav', 'uav')],
    # Latent counterparts of the handcrafted reps (crossmodal encoder; 'xlat' = features/<xmodal run>__lat)
    'E-pred': [('xlat', 'E_pred', 'sat')],           # generated UAV only          (~ H-pred)
    'E-sat+pred': [('xlat', 'E_satpred', 'sat')],    # real sat + generated UAV, jointly encoded (~ H-sat+pred)
    'E-uav': [('xlat', 'E_uav', 'uav')],             # real UAV only               (~ H-uav, oracle)
    'E-uav+E-xmodal': [('xlat', 'E_uav', 'uav'), ('xmodal', 'E_sat', 'sat')],  # real UAV and real sat encoded separately, concatenated
    'E-uavMAE': [('uavmae', 'E_uav', 'uav')],        # real UAV through the UAV-only MAE (uavmae_*), control for E-uav
    'E-uavMAE-F': [('uavmaeF', 'E_uav', 'uav')],     # UAV-only MAE, per-flight normalised UAV (uavmaeF_*)
    'E-uavMAE-M': [('uavmaeM', 'E_uav', 'uav')],     # UAV-only MAE, crossmodal-matched visible-UAV masking (uavmaeM_*)
    # Section 6.4: real UAV at the flown dates + pseudo-UAV (or crossmodal embedding) at satellite-only dates
    'H-uav+pred@s': [('hand', 'H_uav', 'uav'), ('xmodal', 'H_pred', 'sonly')],
    'H-uav+H-sat-all@s': [('hand', 'H_uav', 'uav'), ('hand', 'H_sat_all', 'sonly')],
    'H-uav+sat-all+pred@s': [('hand', 'H_uav', 'uav'), ('hand', 'H_sat_all', 'sonly'), ('xmodal', 'H_pred', 'sonly')],
    'E-uav-real+E-xmodal@s': [('xmodal', 'E_uavreal', 'uav'), ('xmodal', 'E_sat', 'sonly')],
}
ORACLE = {'H-uav', 'E-uav-real', 'F-late', 'F-late-rgb', 'E-uav'}
_PAIRS = None
_CACHE = {}


def sat_tp_for(loc, t):
    global _PAIRS
    if _PAIRS is None:
        p = pd.read_csv(os.path.join(EDATA, 'tp_pairs.csv'))
        _PAIRS = {(r.location, r.uav_tp): r.sat_tp for r in p.itertuples()}
    return _PAIRS[(loc, t)]


def tp_list(tspec):
    """'t1'..'t3' aligned, 'cat' = t1..t3, 's4'..'s6' satellite-only TPs, 'cat6' = sat TP1..6,
    'cat+s46' = aligned t1..t3 + sat TP4..6."""
    if tspec.startswith('t'):
        return [('a', int(tspec[1]))]
    if tspec == 'cat':
        return [('a', 1), ('a', 2), ('a', 3)]
    if tspec.startswith('s'):
        return [('s', int(tspec[1]))]
    if tspec == 'cat6':
        return [('s', k) for k in range(1, 7)]
    if tspec == 'cat+s46':
        return [('a', 1), ('a', 2), ('a', 3), ('s', 4), ('s', 5), ('s', 6)]
    if tspec.startswith('drop'):  # aligned t1..t3 without flight k (missing-flight scenario, no substitute)
        k = int(tspec[4])
        return [('a', t) for t in (1, 2, 3) if t != k]
    raise ValueError(tspec)


class _LazyNpz(dict):
    """Reads an array from the .npz only when first requested (a run's file holds ~15 arrays, a job needs 1-3)."""

    def __init__(self, path):
        super().__init__()
        self.path = path

    def __missing__(self, k):
        with np.load(self.path) as f:
            self[k] = f[k]
        return self[k]


def load_npz(src_dir, loc):
    key = (src_dir, loc)
    if key not in _CACHE:
        _CACHE[key] = _LazyNpz(os.path.join(FEATS, src_dir, f'{loc}.npz'))
    return _CACHE[key]


def features(rep, runs, loc, tspec, subst=None):
    """Return X (n_plots, d) for `loc`. runs: {'xmodal': run_name, 'satmae': ..., 'imnet': ...}.
    subst: optional {aligned t: (source, key)} replacing real UAV at t (missing-flight scenario).
    'sub{k}' fills flight k with the crossmodal pseudo-UAV; 'hsub{k}' is the control that fills it with
    satellite-only features of the same date (H-sat-all, or the E-satMAE embedding)."""
    if tspec.startswith('sub') or tspec.startswith('hsub'):
        k = int(tspec[-1])
        if tspec.startswith('sub'):
            src = {'H-uav': ('xmodal', 'H_pred'), 'E-uav-real': ('xmodal', 'E_sat'), 'F-late': ('xmodal', 'H_pred')}[rep]
        else:
            src = {'H-uav': ('hand', 'H_sat_all'), 'E-uav-real': ('satmae', 'E_sat'), 'F-late': ('hand', 'H_sat_all')}[rep]
        subst, tspec = {k: src}, 'cat'
    blocks = []
    for kind, t in tp_list(tspec):
        for src, key, axis in REPS[rep]:
            if axis == 'sonly':
                if kind != 's':
                    continue
                axis = 'sat'
            if axis == 'uav':
                if kind == 's':
                    continue  # no UAV at satellite-only timepoints
                if subst and t in subst:  # missing flight -> pseudo-UAV from the matching satellite date
                    src, key = subst[t]
                    d = 'handcrafted' if src == 'hand' else runs[src]
                    blocks.append(load_npz(d, loc)[f'{key}_tp{sat_tp_for(loc, t)}'])
                    continue
                tp = t
            else:
                tp = sat_tp_for(loc, t) if kind == 'a' else t
            d = 'handcrafted' if src == 'hand' else runs[src]
            blocks.append(load_npz(d, loc)[f'{key}_tp{tp}'])
    return np.concatenate(blocks, 1).astype(np.float64)


_NB = {}


def neighbours(loc, k):
    """Indices (n, k) of each plot's k nearest plots in the same field (location x experiment), by (range, row) grid
    position. A range (plot length) counts 3x a row step, roughly the plots' physical aspect. Self excluded."""
    if (loc, k) not in _NB:
        lab = labels().loc[plot_uids(loc)]
        rg, rw, ex = lab['range'].values.astype(float), lab['row'].values.astype(float), lab.experiment.astype(str).values
        nb = np.zeros((len(lab), k), int)
        for e in np.unique(ex):
            i = np.flatnonzero(ex == e)
            d = (3 * (rg[i, None] - rg[None, i])) ** 2 + (rw[i, None] - rw[None, i]) ** 2
            np.fill_diagonal(d, np.inf)
            o = np.argsort(d, 1)[:, :min(k, len(i) - 1)]
            nb[i] = i[np.pad(o, ((0, 0), (0, k - o.shape[1])), mode='edge')]
        _NB[(loc, k)] = nb
    return _NB[(loc, k)]


def neighbour_norm(loc, X, k):
    """Field-neighbour normalisation: each plot's features minus the mean of its k nearest plots (unlabelled
    features only, so it is also applied at the held-out site)."""
    return X - np.nanmean(X[neighbours(loc, k)], 1)


def cross_fit(T, y, geno, alpha_groups, n=5):
    """Out-of-fold teacher predictions on the training rows (genotype-grouped, so a plot's soft target never comes
    from a model that saw its genotype)."""
    out = np.zeros(len(y))
    for tr, te in GroupKFold(min(n, len(np.unique(geno)))).split(T, y, geno):
        out[te] = ridge_nested(T[tr], y[tr], T[te], alpha_groups[tr])
    return out


def plot_uids(loc):
    return load_npz('handcrafted', loc)['plot_uid']


_LAB = None


def labels():
    global _LAB
    if _LAB is None:
        df = D.manifest()
        _LAB = df[(df.sensor == 'sat') & (df.tp == 1) & df.has_label].set_index('plot_uid')
    return _LAB


# ------------------------------------------------------------------------------------------ probes

def _clean(Xtr, Xte):
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


def ridge_path(Xtr, ytr, Xte, alphas):
    """Ridge predictions for every alpha via one SVD (inputs already standardised)."""
    ym = ytr.mean()
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    Uy = U.T @ (ytr - ym)
    XteV = Xte @ Vt.T
    return np.stack([ym + XteV @ (s / (s ** 2 + a) * Uy) for a in alphas], 1)


def ridge_nested(Xtr, ytr, Xte, groups):
    gkf = GroupKFold(min(4, len(np.unique(groups))))
    err = np.zeros(len(ALPHAS))
    for itr, iva in gkf.split(Xtr, ytr, groups):
        a, b = _clean(Xtr[itr], Xtr[iva])
        err += ((ridge_path(a, ytr[itr], b, ALPHAS) - ytr[iva, None]) ** 2).mean(0)
    best = ALPHAS[err.argmin()]
    a, b = _clean(Xtr, Xte)
    return ridge_path(a, ytr, b, [best])[:, 0]


def reduce(Xtr, Xte, k=128):
    a, b = _clean(Xtr, Xte)
    if a.shape[1] > k:
        p = PCA(k, random_state=0).fit(a)
        a, b = p.transform(a), p.transform(b)
        s = a.std(0) + 1e-8
        a, b = a / s, b / s
    return a, b


def logreg_nested(Xtr, ytr, Xte, groups):
    a, b = reduce(Xtr, Xte)
    gkf = GroupKFold(min(3, len(np.unique(groups))))
    score = np.zeros(len(CS))
    for itr, iva in gkf.split(a, ytr, groups):
        for i, C in enumerate(CS):
            m = LogisticRegression(C=C, max_iter=2000).fit(a[itr], ytr[itr])
            score[i] += (m.predict(a[iva]) == ytr[iva]).mean()
    return LogisticRegression(C=CS[score.argmax()], max_iter=2000).fit(a, ytr).predict(b)


def mlp(Xtr, ytr, Xte, task_type, seed=0):
    # clip: a few near-constant features reach |z|>20 on held-out folds and the unregularised MLP extrapolates wildly
    a, b = (np.clip(x, -5, 5) for x in reduce(Xtr, Xte))
    kw = dict(hidden_layer_sizes=(256,), alpha=1e-2, early_stopping=True, validation_fraction=0.15,
              max_iter=500, n_iter_no_change=20, random_state=seed)
    if task_type == 'reg':
        mu, sd = ytr.mean(), ytr.std() + 1e-8
        return MLPRegressor(**kw).fit(a, (ytr - mu) / sd).predict(b) * sd + mu
    return MLPClassifier(**kw).fit(a, ytr).predict(b)


def knn(Xtr, ytr, Xte, task_type, k=15):
    a, b = reduce(Xtr, Xte)
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-8
    nn = np.argsort(-(b @ a.T), 1)[:, :k]
    if task_type == 'reg':
        return ytr[nn].mean(1)
    return np.array([np.bincount(r).argmax() for r in ytr[nn].astype(int)])


def xgb(Xtr, ytr, Xte, task_type):
    import xgboost as xg
    a, b = _clean(Xtr, Xte)
    kw = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=1)
    if task_type == 'reg':
        return xg.XGBRegressor(**kw).fit(a, ytr).predict(b)
    return xg.XGBClassifier(**kw).fit(a, ytr).predict(b)


def random_fold(uids, k=5, salt=''):
    """Stable plot-level random fold from the plot id (identical for every representation, run and seed)."""
    return np.array([zlib.crc32((salt + str(u)).encode()) % k for u in uids])


def fit_predict(probe, Xtr, ytr, Xte, groups, task_type):
    if probe == 'linear':
        return ridge_nested(Xtr, ytr, Xte, groups) if task_type == 'reg' else logreg_nested(Xtr, ytr, Xte, groups)
    if probe == 'mlp':
        return mlp(Xtr, ytr, Xte, task_type)
    if probe == 'knn':
        return knn(Xtr, ytr, Xte, task_type)
    if probe == 'xgb':
        return xgb(Xtr, ytr, Xte, task_type)
    raise ValueError(probe)


# ----------------------------------------------------------------------------------------- metrics

def reg_metrics(y, p, geno=None):
    out = dict(r2=r2_score(y, p), rmse=float(np.sqrt(mean_squared_error(y, p))))
    r = pearsonr(y, p)[0] if np.std(p) > 0 else 0.0
    out.update(pearson=r, r2_pearson=r ** 2)
    if geno is not None:
        g = pd.DataFrame(dict(g=geno, y=y, p=p)).groupby('g').mean()
        out['spearman_geno'] = spearmanr(g.y, g.p)[0]
        for q in (25, 50):  # share of the true top-q% genotypes that are also in the predicted top-q% (Powadi et al. 2025)
            n = max(1, int(round(len(g) * q / 100)))
            out[f'top{q}_geno'] = len(set(g.y.nlargest(n).index) & set(g.p.nlargest(n).index)) / n
    return out


def cls_metrics(y, p):
    return dict(acc=accuracy_score(y, p), macro_f1=f1_score(y, p, average='macro'))


# ----------------------------------------------------------------------------------------- ground-truth covariates
COV_ALPHA = 3.0  # ridge shrinkage of the covariate model (genotype effects shrink toward 0 with ~3 plots each)


def _cov_design(ntr, gtr, nte, gte):
    """[standardised N rate, genotype one-hot over the training genotypes]; unseen test genotypes -> all zeros."""
    mu, sd = ntr.mean(), ntr.std() + 1e-8
    gs = np.unique(gtr)
    oh = lambda g: (np.asarray(g)[:, None] == gs[None]).astype(float)
    return np.c_[(ntr - mu) / sd, oh(gtr)], np.c_[(nte - mu) / sd, oh(gte)]


def _cov_ridge(ntr, gtr, ytr, nte, gte):
    a, b = _cov_design(ntr, gtr, nte, gte)
    am = a.mean(0)
    a, b = a - am, b - am
    w = np.linalg.solve(a.T @ a + COV_ALPHA * np.eye(a.shape[1]), a.T @ (ytr - ytr.mean()))
    return ytr.mean() + b @ w


def cov_fit(ntr, gtr, ytr, nte, gte, groups):
    """Covariate model (N rate + genotype). Returns (cross-fitted predictions for the training plots, grouped like the
    probe's inner CV so a held-out genotype/site looks the same as at test time; predictions for the test plots)."""
    oof = np.zeros(len(ytr))
    for itr, iva in GroupKFold(min(5, len(np.unique(groups)))).split(ntr, ytr, groups):
        oof[iva] = _cov_ridge(ntr[itr], gtr[itr], ytr[itr], ntr[iva], gtr[iva])
    return oof, _cov_ridge(ntr, gtr, ytr, nte, gte)


N_MAP = {'Low': 0, 'Medium': 1, 'High': 2}


def target(task, uids):
    lab = labels().loc[uids]
    if task == 'nitrogen':
        y = lab.nitrogen.map(N_MAP).values.astype(float)
    else:
        y = lab[{'yield': 'yield', 'stand_count': 'stand_count', 'dta': 'dta', 'gdd': 'gdd'}[task]].values.astype(float)
    return y, lab.genotype.values, lab.location.values


def zscore(v):
    return (v - np.nanmean(v, 0)) / (np.nanstd(v, 0) + 1e-8)


def evaluate(rep, runs, L, tspec, tasks, regimes, probes, geno_fold, tag=None, subst=None, exclude_ctx=None,
             nb=0, teacher=None, lam=0.0, cov=False):
    """All folds / regimes / tasks / probes for one (representation, held-out L, timepoint spec).
    exclude_ctx=f drops the held-out site's in-site context plots (data.context_uids at fraction f) from evaluation.
    nb=k: field-neighbour-normalised features (neighbour_norm). teacher=rep, lam: learning using privileged information:
    the training targets become (1-lam)*y + lam*teacher prediction, the teacher being a probe on `teacher` (UAV) features
    of the training plots only (cross-fitted); the student, and the evaluation, still see only `rep`. Regression only.
    cov=True (regression): ground-truth covariates (N rate + genotype) via cov_fit; out-of-fold probe and covariate predictions
    (grouped like the inner CV) are stacked with 2-weight OLS. Extra probe label 'cov_only'."""
    rows = []
    feats, tfeats = {}, {}

    def F(loc):
        if loc not in feats:
            X = features(rep, runs, loc, tspec, subst)
            feats[loc] = neighbour_norm(loc, X, nb) if nb else X
        return feats[loc]

    def TF(loc):
        if loc not in tfeats:
            tfeats[loc] = features(teacher, runs, loc, 'cat')
        return tfeats[loc]

    for task in tasks:
        ttype = 'cls' if task == 'nitrogen' else 'reg'

        def covfit(probe, Xtr, ytr, Xte, groups, ntr, gtr, nte, gte):
            if not (cov and ttype == 'reg'):
                return fit_predict(probe, Xtr, ytr, Xte, groups, ttype)
            oof, cte = cov_fit(ntr, gtr, ytr, nte, gte, groups)
            if probe == 'cov_only':
                return cte
            if regime_now[0].startswith('B'):  # cross-site: residual fit (stack weights from leave-one-training-site-out
                return fit_predict(probe, Xtr, ytr - oof, Xte, groups, ttype) + cte  # predictions are miscalibrated)
            # stacking: out-of-fold image-probe predictions (same grouping) + covariate predictions -> 2-weight OLS
            ioof = np.zeros(len(ytr))
            for itr, iva in GroupKFold(min(5, len(np.unique(groups)))).split(Xtr, ytr, groups):
                ioof[iva] = fit_predict(probe, Xtr[itr], ytr[itr], Xtr[iva], np.asarray(groups)[itr], ttype)
            w = np.linalg.lstsq(np.c_[np.ones(len(ytr)), ioof, oof], ytr, rcond=None)[0]
            return w[0] + w[1] * fit_predict(probe, Xtr, ytr, Xte, groups, ttype) + w[2] * cte
        task_locs = N_LOCS if task == 'nitrogen' else EXTRA_TASKS.get(task, LOCS)
        if L not in task_locs:
            continue
        uL = plot_uids(L)
        yL, gL, _ = target(task, uL)
        okL = ~np.isnan(yL)
        if exclude_ctx:
            okL &= ~np.isin(uL, list(D.context_uids(D.manifest(), L, exclude_ctx)))
        XL = F(L)[okL]
        yL, gL = yL[okL], gL[okL]
        nL = labels().loc[np.asarray(uL)[okL]].n_lbs.values.astype(float)
        fL = np.array([geno_fold[g] for g in gL])
        rL = random_fold(np.asarray(uL)[okL])              # random outer folds (A_r, B_zr)
        iL = random_fold(np.asarray(uL)[okL], 4, 'inner')  # random inner-CV groups (A_r)
        regime_now = [None]
        for regime in regimes:
            regime_now[0] = regime
            if regime != 'A' and task in EXTRA_TASKS:
                continue
            if regime in ('B_zc', 'B_zx') and L not in SAME_CAM:
                continue
            if regime not in ('A', 'A_r'):
                tr_locs = [l for l in task_locs if l != L and
                           (l in SAME_CAM if regime == 'B_zc' else l in OTHER_CAM if regime == 'B_zx' else True)]
                if len(tr_locs) < 2:  # location-grouped nested CV needs 2 training sites
                    continue
                Xs, ys, gs, ls, Ts, ns = [], [], [], [], [], []
                for l in tr_locs:
                    u = plot_uids(l)
                    y, g, _ = target(task, u)
                    ok = ~np.isnan(y)
                    X = F(l)[ok]
                    T = TF(l)[ok] if teacher else X[:, :0]
                    n = labels().loc[np.asarray(u)[ok]].n_lbs.values.astype(float)
                    if regime.startswith('B_z'):
                        X = zscore(X); T = zscore(T); n = zscore(n)
                        if ttype == 'reg':
                            y = y.copy(); y[ok] = zscore(y[ok])
                    Xs.append(X); ys.append(y[ok]); gs.append(g[ok]); ls.append(np.full(ok.sum(), l)); Ts.append(T); ns.append(n)
                Xo, yo, go, lo, To, no = map(np.concatenate, (Xs, ys, gs, ls, Ts, ns))
                fo = np.array([geno_fold[g] for g in go])
                XLr, yLr = (zscore(XL), (zscore(yL) if ttype == 'reg' else yL)) if regime.startswith('B_z') else (XL, yL)
                nLr = zscore(nL) if regime.startswith('B_z') else nL
                if cov:  # N response differs in sign between sites, so cross-site covariates are genotype only
                    no, nLr = np.zeros_like(no), np.zeros_like(nLr)
            for probe in probes + (['cov_only'] if cov and ttype == 'reg' else []):
                if probe == 'xgb' and not rep.startswith('H') and rep != 'F-late':
                    continue
                pooled = np.full(len(yL), np.nan)
                for k in range(5):
                    te = (rL if regime in ('A_r', 'B_zr') else fL) == k
                    if te.sum() < 3:
                        continue
                    if regime == 'A':
                        tr = ~te
                        ytr = yL[tr]
                        if teacher and lam and ttype == "reg":
                            ytr = (1 - lam) * ytr + lam * cross_fit(TF(L)[okL][tr], ytr, gL[tr], gL[tr])
                        p = covfit(probe, XL[tr], ytr, XL[te], gL[tr], nL[tr], gL[tr], nL[te], gL[te])
                        ytest = yL[te]
                    elif regime == 'A_r':
                        tr = ~te
                        p = covfit(probe, XL[tr], yL[tr], XL[te], iL[tr], nL[tr], gL[tr], nL[te], gL[te])
                        ytest = yL[te]
                    elif regime == 'P':  # one probe over all sites (raw yield), genotype-grouped folds; test = site L, fold k
                        trO = fo != k
                        p = fit_predict(probe, np.r_[Xo[trO], XL[~te]], np.r_[yo[trO], yL[~te]], XL[te],
                                        np.r_[go[trO], gL[~te]], ttype)
                        ytest = yL[te]
                    else:
                        tr = fo != k if regime not in ('B_zg', 'B_zr') else np.ones(len(fo), bool)
                        ytr = yo[tr]
                        if teacher and lam and ttype == "reg":
                            ytr = (1 - lam) * ytr + lam * cross_fit(To[tr], ytr, go[tr], lo[tr])
                        p = covfit(probe, Xo[tr], ytr, XLr[te], lo[tr], no[tr], go[tr], nLr[te], gL[te])
                        ytest = yLr[te]
                    pooled[te] = p
                    m = reg_metrics(ytest, p) if ttype == 'reg' else cls_metrics(ytest, p)
                    rows += [dict(heldout=L, regime=regime, task=task, tp=tspec, fold=k, probe=probe, metric=a, value=v)
                             for a, v in m.items()]
                ok = ~np.isnan(pooled)
                ytrue = (yL if regime in ('A', 'A_r', 'P') else yLr)[ok]
                m = reg_metrics(ytrue, pooled[ok], gL[ok]) if ttype == 'reg' else cls_metrics(ytrue, pooled[ok].astype(int))
                rows += [dict(heldout=L, regime=regime, task=task, tp=tspec, fold='pooled', probe=probe, metric=a, value=v)
                         for a, v in m.items()]
    for r in rows:
        r['rep'] = rep
        r['run'] = tag or runs.get('xmodal') or runs.get('satmae') or 'none'
    return rows


def geno_fold_map():
    return json.load(open(os.path.join(SPLITS, f'lolo_{LOCS[0]}.json')))['genotype_fold']


def run_jobs(jobs, n_jobs=30):
    """jobs: list of kwargs for evaluate(); returns tidy DataFrame."""
    gf = geno_fold_map()

    def one(j):
        os.environ['OMP_NUM_THREADS'] = '1'
        keep = {'handcrafted', *j['runs'].values()}  # bound the per-worker cache (unbounded growth OOMed the machine)
        for k in [k for k in _CACHE if k[0] not in keep]:
            del _CACHE[k]
        gc.collect()
        ctypes.CDLL('libc.so.6').malloc_trim(0)  # hand freed feature arrays back to the OS
        r = evaluate(geno_fold=gf, **{k: v for k, v in j.items() if k != 'extra'})
        for x in r:
            x.update(j.get('extra', {}))
        return r
    out = Parallel(n_jobs=n_jobs, verbose=0)(delayed(one)(j) for j in jobs)
    return pd.DataFrame([r for rr in out for r in rr])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', nargs='+', required=True)
    ap.add_argument('--xmodal', default='xmodal_s{seed}_{L}')
    ap.add_argument('--satmae', default='satmae_s{seed}_{L}')
    ap.add_argument('--locs', nargs='+', default=LOCS)
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--tps', nargs='+', default=['cat'])
    ap.add_argument('--tasks', nargs='+', default=['yield', 'nitrogen'])
    ap.add_argument('--regimes', nargs='+', default=['A', 'B_raw', 'B_z'])
    ap.add_argument('--probes', nargs='+', default=['linear'])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    jobs = []
    for rep in a.reps:
        for L in a.locs:
            for s in a.seeds:
                runs = dict(xmodal=a.xmodal.format(seed=s, L=L), satmae=a.satmae.format(seed=s, L=L), imnet=f'imnet_{L}')
                for tp in a.tps:
                    jobs.append(dict(rep=rep, runs=runs, L=L, tspec=tp, tasks=a.tasks, regimes=a.regimes,
                                     probes=a.probes, extra=dict(seed=s)))
    df = run_jobs(jobs)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, a.out), index=False)
    print(df[df.fold == 'pooled'].groupby(['rep', 'regime', 'task', 'metric']).value.mean().unstack().round(3).to_string())


if __name__ == '__main__':
    main()
