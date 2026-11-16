"""Phase 1: data audit, clean manifest, rotation-aligned tile cache, QC flags, LOLO splits.

Usage: python -m satuav.manifest  (from experiments/)
"""
import glob
import json
import os
import re
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import rasterio

from .paths import (CACHE, CSV, DATA, DATES, EDATA, LOCS, MANIFEST, N_TILES, SPLITS, TILE)

FNAME_RE = re.compile(r'^(?P<loc>[A-Za-z]+)-TP(?P<tp>\d)-(?P<exp>.+)_(?P<range>\d+)_(?P<row>\d+)\.(?P<ext>TIF|PNG)$')
DATE_LOC = {'Missouri Valley': 'MOValley'}
MIN_ROT_DEG = 1.0     # residual rotations below this are left alone
N_FOLDS = 5
FOLD_SEED = 0


def norm_exp(loc, exp):
    """Normalise experiment codes so files and CSV agree (MOValley 'Hybrids' vs CSV typo 'Hyrbrids')."""
    e = str(exp).strip().lower()
    if e in ('hyrbrids', 'hybirds'):
        e = 'hybrids'
    return e


# ----------------------------------------------------------------------------------------------- io

def read_image(sensor, path):
    """Returns (img HxWxC, valid mask HxW bool). Sat: uint16 6 bands. UAV: uint8 RGB."""
    if sensor == 'sat':
        with rasterio.open(path) as src:
            a = src.read()  # C,H,W
        img = np.transpose(a, (1, 2, 0))
        mask = (img != 0).any(2)
    else:
        u = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = u[..., [2, 1, 0]]
        mask = u[..., 3] > 0 if u.shape[2] == 4 else (img != 0).any(2)
        mask &= (img != 0).any(2)
    return img, mask


def long_axis_angle(mask):
    """Residual angle (deg, in [-45,45)) of the plot's long axis w.r.t. the image axes."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return 0.0
    (_, _), (w, h), a = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
    return float((a + 45) % 90 - 45)


def rotate(img, mask, deg):
    if abs(deg) < MIN_ROT_DEG:
        return img, mask
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(np.ceil(h * s + w * c)), int(np.ceil(h * c + w * s))
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    chans = [cv2.warpAffine(np.ascontiguousarray(img[..., i]), M, (nw, nh), flags=cv2.INTER_LINEAR)
             for i in range(img.shape[2])]
    m = cv2.warpAffine(mask.astype(np.uint8), M, (nw, nh), flags=cv2.INTER_NEAREST) > 0
    return np.stack(chans, 2), m


def to_tiles(img, mask, rot_deg, align=True):
    """Rotate so rows are axis-aligned with the long axis horizontal, crop to the plot, cut 3 squares
    along the long axis (overlapping when aspect < 3), resize each to TILE.

    align=False reproduces the paper's original cropping (no rotation / no tight crop) for the ablation.
    """
    if align:
        img, mask = rotate(img, mask, rot_deg)
        ys, xs = np.nonzero(mask)
        if len(xs):
            img, mask = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1], mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if mask.shape[0] > mask.shape[1]:
        img, mask = np.rot90(img), np.rot90(mask)
    h, w = mask.shape
    s = h
    offs = np.linspace(0, max(w - s, 0), N_TILES).round().astype(int)
    up = s < TILE
    tiles, tmasks = [], []
    for o in offs:
        t = np.ascontiguousarray(img[:, o:o + s])
        tm = np.ascontiguousarray(mask[:, o:o + s]).astype(np.uint8)
        t = np.stack([cv2.resize(t[..., i], (TILE, TILE), interpolation=cv2.INTER_LINEAR if up else cv2.INTER_AREA)
                      for i in range(t.shape[2])], 0)
        tm = cv2.resize(tm, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
        tiles.append(t)
        tmasks.append(tm)
    return np.stack(tiles), np.stack(tmasks).astype(bool)


# ------------------------------------------------------------------------------------------ scanning

def scan_files():
    rows = []
    for sensor, sub, ext in [('sat', 'Sat', 'TIF'), ('uav', 'UAV', 'PNG')]:
        for p in sorted(glob.glob(os.path.join(DATA, sub, '*', 'TP*', f'*.{ext}'))):
            m = FNAME_RE.match(os.path.basename(p))
            assert m, p
            d = m.groupdict()
            rows.append(dict(sensor=sensor, location=d['loc'], tp=int(d['tp']), exp_file=d['exp'],
                             experiment=norm_exp(d['loc'], d['exp']), range=int(d['range']), row=int(d['row']), path=p))
    return pd.DataFrame(rows)


def load_csv():
    g = pd.read_csv(CSV)
    g['experiment'] = g['experiment'].astype(str)
    # Scottsbluff: 16 rows have an empty experiment; the code is implied by the N level.
    sb = (g.location == 'Scottsbluff') & g.experiment.isin(['nan', '', 'NA'])
    g.loc[sb, 'experiment'] = g.loc[sb, 'nitrogenTreatment'].map({'Low': 'n75', 'Medium': 'n150', 'High': 'n225'})
    g['experiment'] = [norm_exp(l, e) for l, e in zip(g.location, g.experiment)]
    # 3 of those recovered rows collide with real labelled plots; keep the labelled one.
    g = (g.assign(_lab=g.genotype.notna()).sort_values('_lab', ascending=False)
          .drop_duplicates(['location', 'experiment', 'range', 'row']).sort_index().drop(columns='_lab'))
    g = g.rename(columns=dict(genotype='genotype', block='block', nitrogenTreatment='nitrogen',
                              poundsOfNitrogenPerAcre='n_lbs', yieldPerAcre='yield', totalStandCount='stand_count',
                              daysToAnthesis='dta', GDDToAnthesis='gdd'))
    g['plantingDate'] = pd.to_datetime(g['plantingDate'])
    keep = ['location', 'experiment', 'range', 'row', 'genotype', 'block', 'nitrogen', 'n_lbs', 'yield',
            'stand_count', 'dta', 'gdd', 'plantingDate', 'irrigationProvided']
    return g[keep]


def load_dates():
    d = pd.read_excel(DATES, sheet_name='Sheet1')
    d.columns = ['location', 'date', 'image', 'tp']
    d['location'] = d.location.replace(DATE_LOC)
    d['sensor'] = d.image.map({'Satellite': 'sat', 'UAV': 'uav'})
    d['tp'] = d.tp.str.replace('TP', '').astype(int)
    d['date'] = pd.to_datetime(d.date)
    return d[d.location.isin(LOCS)][['location', 'sensor', 'tp', 'date']]


def nearest_pairs(dates):
    """For each (location, UAV TP), the satellite TP with the nearest acquisition date."""
    out = []
    for loc in LOCS:
        s = dates[(dates.location == loc) & (dates.sensor == 'sat')]
        u = dates[(dates.location == loc) & (dates.sensor == 'uav')]
        for _, r in u.iterrows():
            gap = (s.date - r.date).dt.days
            j = gap.abs().values.argmin()
            out.append(dict(location=loc, uav_tp=r.tp, sat_tp=int(s.tp.values[j]), gap_days=int(gap.values[j]),
                            uav_date=r.date, sat_date=s.date.values[j],
                            paper_sat_tp=r.tp, paper_gap_days=int((s[s.tp == r.tp].date.values[0] - r.date) / np.timedelta64(1, 'D'))))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------- processing

def _angle_job(args):
    sensor, path = args
    _, m = read_image(sensor, path)
    return long_axis_angle(m)


def _tile_job(args):
    i, sensor, path, rot, n_total, align, tag = args
    img, mask = read_image(sensor, path)
    tiles, tmasks = to_tiles(img, mask, rot, align=align)
    dt = np.uint16 if sensor == 'sat' else np.uint8
    C = tiles.shape[1]
    T = np.lib.format.open_memmap(os.path.join(CACHE, f'{sensor}{tag}_tiles.npy'), mode='r+')
    M = np.lib.format.open_memmap(os.path.join(CACHE, f'{sensor}{tag}_mask.npy'), mode='r+')
    T[i] = tiles.astype(dt)
    M[i] = tmasks
    del T, M
    # Native-resolution statistics over valid pixels (used for train-only normalisation & QC)
    v = img[mask].astype(np.float64)
    st = dict(h=mask.shape[0], w=mask.shape[1], n_valid=int(mask.sum()), valid_pixel_frac=float(mask.mean()),
              tile_valid_frac=float(tmasks.mean()))
    for c in range(C):
        st[f'sum{c}'] = float(v[:, c].sum())
        st[f'sq{c}'] = float((v[:, c] ** 2).sum())
    if sensor == 'uav':
        g = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        lap = cv2.Laplacian(g, cv2.CV_32F)
        er = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        st['lap_var'] = float(lap[er].var()) if er.any() else np.nan
        st['brightness'] = float(g[mask].mean())
    return i, st


def build_cache(df, sensor, align=True, tag=''):
    sub = df[df.sensor == sensor]
    C = 6 if sensor == 'sat' else 3
    dt = np.uint16 if sensor == 'sat' else np.uint8
    n = len(sub)
    os.makedirs(CACHE, exist_ok=True)
    np.lib.format.open_memmap(os.path.join(CACHE, f'{sensor}{tag}_tiles.npy'), mode='w+', dtype=dt, shape=(n, N_TILES, C, TILE, TILE))
    np.lib.format.open_memmap(os.path.join(CACHE, f'{sensor}{tag}_mask.npy'), mode='w+', dtype=bool, shape=(n, N_TILES, TILE, TILE))
    jobs = [(int(ci), sensor, p, float(r), n, align, tag) for ci, p, r in zip(sub.cache_idx, sub.path, sub.rot_deg)]
    with Pool(30) as pool:
        stats = dict(pool.imap_unordered(_tile_job, jobs, chunksize=8))
    return pd.DataFrame.from_dict(stats, orient='index')


def qc_flags(df):
    flags = [[] for _ in range(len(df))]
    df = df.reset_index(drop=True)
    for i, r in df.iterrows():
        if r.tile_valid_frac < 0.6:
            flags[i].append('low_valid')
        if r.sensor == 'uav' and abs(r.pair_day_gap) > 7:
            flags[i].append('date_gap>7d')
    uav = df[df.sensor == 'uav']
    for (loc, tp), g in uav.groupby(['location', 'tp']):
        med, mad = g.brightness.median(), (g.brightness - g.brightness.median()).abs().median() * 1.4826
        for i in g.index[g.brightness < med - 3 * mad]:
            flags[i].append('dark/shadow')
        lv = np.log(g.lap_var + 1e-6)
        lmed, lmad = lv.median(), (lv - lv.median()).abs().median() * 1.4826
        for i in g.index[lv < lmed - 3 * lmad]:
            flags[i].append('blur')
    df['qc_flags'] = [';'.join(f) for f in flags]
    return df


def make_splits(df):
    """Genotype->fold map shared by every location and every method."""
    genos = sorted(df.genotype.dropna().unique())
    rng = np.random.RandomState(FOLD_SEED)
    perm = rng.permutation(len(genos))
    geno_fold = {genos[j]: int(k % N_FOLDS) for k, j in enumerate(perm)}
    os.makedirs(SPLITS, exist_ok=True)
    for L in LOCS:
        lab = df[(df.location == L) & df.has_label & (df.sensor == 'sat') & (df.tp == 1)]
        folds = {str(k): sorted(lab[lab.genotype.map(geno_fold) == k].plot_uid.tolist()) for k in range(N_FOLDS)}
        split = dict(heldout=L, pretrain=[l for l in LOCS if l != L], n_folds=N_FOLDS, seed=FOLD_SEED,
                     genotype_fold=geno_fold, inner_folds=folds)
        json.dump(split, open(os.path.join(SPLITS, f'lolo_{L}.json'), 'w'), indent=1)
    return geno_fold


def main():
    os.makedirs(EDATA, exist_ok=True)
    files = scan_files()
    gt = load_csv()
    dates = load_dates()
    pairs = nearest_pairs(dates)
    pairs.to_csv(os.path.join(EDATA, 'tp_pairs.csv'), index=False)

    df = files.merge(gt, on=['location', 'experiment', 'range', 'row'], how='left', indicator=True)
    df['matched'] = df._merge == 'both'
    df = df.drop(columns='_merge')
    df['plot_uid'] = df.location + '_' + df.experiment + '_' + df.range.astype(str) + '_' + df.row.astype(str)
    df = df.merge(dates.rename(columns={'date': 'acq_date'}), on=['location', 'sensor', 'tp'], how='left')
    df['days_after_planting'] = (df.acq_date - df.plantingDate).dt.days
    loc_pd = gt.groupby('location').plantingDate.median()
    miss = df.days_after_planting.isna()
    df.loc[miss, 'days_after_planting'] = (df.acq_date[miss] - df.location[miss].map(loc_pd)).dt.days

    # Pairing: UAV row -> nearest-date sat TP; sat rows -> the UAV TP they partner (if any)
    pk = pairs.set_index(['location', 'uav_tp'])
    df['paired_tp'] = np.nan
    df['pair_day_gap'] = np.nan
    u = df.sensor == 'uav'
    df.loc[u, 'paired_tp'] = [pk.loc[(l, t), 'sat_tp'] for l, t in zip(df.location[u], df.tp[u])]
    df.loc[u, 'pair_day_gap'] = [pk.loc[(l, t), 'gap_days'] for l, t in zip(df.location[u], df.tp[u])]
    s = df.sensor == 'sat'
    sat_partner = pairs.groupby(['location', 'sat_tp']).apply(lambda g: g.loc[g.gap_days.abs().idxmin()])
    for i in df.index[s]:
        key = (df.at[i, 'location'], df.at[i, 'tp'])
        if key in sat_partner.index:
            df.at[i, 'paired_tp'] = sat_partner.loc[key, 'uav_tp']
            df.at[i, 'pair_day_gap'] = -sat_partner.loc[key, 'gap_days']

    # Per-(sensor, location, TP) residual rotation from the sensor's own masks (no cross-sensor leakage)
    with Pool(30) as pool:
        df['plot_angle'] = pool.map(_angle_job, list(zip(df.sensor, df.path)), chunksize=32)
    df['rot_deg'] = df.groupby(['sensor', 'location', 'tp']).plot_angle.transform('median')

    df['cache_idx'] = df.groupby('sensor').cumcount()
    st = []
    for sensor in ['sat', 'uav']:
        sdf = df[df.sensor == sensor]
        t = build_cache(df, sensor)
        t.index = sdf.index[t.index.values]  # cache_idx is the row position within the sensor
        st.append(t)
    df = df.join(pd.concat(st))
    df['gsd_est'] = 0.30
    for (loc, tp), g in df[df.sensor == 'uav'].groupby(['location', 'tp']):
        sat = df[(df.sensor == 'sat') & (df.location == loc) & (df.tp == tp)].set_index('plot_uid').n_valid
        ratio = np.sqrt(g.n_valid.values / sat.reindex(g.plot_uid).values)
        df.loc[g.index, 'gsd_est'] = 0.30 / np.nanmedian(ratio)

    df['has_label'] = df.matched & df.genotype.notna()
    df = qc_flags(df)
    cols = ['plot_uid', 'location', 'experiment', 'range', 'row', 'genotype', 'block', 'nitrogen', 'n_lbs', 'yield',
            'stand_count', 'dta', 'gdd', 'sensor', 'tp', 'acq_date', 'days_after_planting', 'paired_tp', 'pair_day_gap',
            'path', 'h', 'w', 'gsd_est', 'valid_pixel_frac', 'has_label', 'qc_flags',
            'matched', 'exp_file', 'plot_angle', 'rot_deg', 'cache_idx', 'tile_valid_frac', 'n_valid', 'lap_var',
            'brightness', 'irrigationProvided'] + [c for c in df.columns if re.match(r'(sum|sq)\d', c)]
    df = df[cols]
    df.to_parquet(MANIFEST, index=False)
    make_splits(df)
    print(df.groupby(['sensor', 'location']).agg(n=('path', 'size'), matched=('matched', 'sum'), labelled=('has_label', 'sum')))


if __name__ == '__main__':
    main()
