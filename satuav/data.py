"""Tile access, fold-specific normalisation, pair construction, and GPU-side augmentation."""
import os

import numpy as np
import pandas as pd
import torch

from .paths import CACHE, LOCS, MANIFEST

SAT_IDX = {'sat_rgb': [0, 1, 2], 'sat_all': [0, 1, 2, 3, 4, 5]}
_MM = {}


def manifest():
    return pd.read_parquet(MANIFEST)


def mm(sensor, kind, tag=''):
    key = (sensor, kind, tag)
    if key not in _MM:
        _MM[key] = np.load(os.path.join(CACHE, f'{sensor}{tag}_{kind}.npy'), mmap_mode='r')
    return _MM[key]


def band_stats(rows, C):
    """Per-band mean/std over valid native pixels of the given manifest rows."""
    n = rows.n_valid.sum()
    mean = np.array([rows[f'sum{c}'].sum() / n for c in range(C)])
    var = np.array([rows[f'sq{c}'].sum() / n for c in range(C)]) - mean ** 2
    return mean, np.sqrt(var)


def fold_stats(df, locs):
    """Normalisation statistics from the given (training) locations only."""
    s = df[(df.sensor == 'sat') & df.location.isin(locs)]
    u = df[(df.sensor == 'uav') & df.location.isin(locs)]
    sm, ss = band_stats(s, 6 if s['sum5'].notna().all() else 3) if 'sum5' in s else band_stats(s, 3)
    um, us = band_stats(u, 3)
    return dict(sat_mean=sm.tolist(), sat_std=ss.tolist(), uav_mean=um.tolist(), uav_std=us.tolist())


def pair_table(df, locs, mismatch=False, seed=0):
    """One row per (UAV image, its nearest-date satellite image) at the given locations.
    mismatch=True pairs each UAV image with the satellite image of the same plot at a *different*
    UAV-paired timepoint (deliberately wrong date), for the correspondence ablation."""
    u = df[(df.sensor == 'uav') & df.location.isin(locs)]
    s = df[df.sensor == 'sat'].set_index(['plot_uid', 'tp']).cache_idx
    if not mismatch:
        stp = u.paired_tp.astype(int)
    else:
        rng = np.random.RandomState(seed)
        paired = df[df.sensor == 'uav'].groupby('location').paired_tp.unique()
        stp = [int(rng.choice([t for t in paired[l] if t != p])) for l, p in zip(u.location, u.paired_tp)]
    sat_idx = s.reindex(list(zip(u.plot_uid, stp))).values
    return pd.DataFrame(dict(sat=sat_idx.astype(int), uav=u.cache_idx.values, location=u.location.values,
                             plot_uid=u.plot_uid.values, uav_tp=u.tp.values, sat_tp=stp))


def context_uids(df, L, frac, seed=0):
    """In-site context plots of the held-out site L: whole genotypes in a fixed random order until >= frac of its plots.
    The order is shared across frac, so smaller context sets are subsets of larger ones."""
    p = df[(df.location == L) & (df.sensor == 'uav')].drop_duplicates('plot_uid')
    g = p.genotype.fillna(p.plot_uid)
    order = np.random.RandomState(seed).permutation(sorted(g.unique()))
    n = g.value_counts()[order].cumsum()
    chosen = n.index[:np.searchsorted(n.values, frac * len(p)) + 1]
    return set(p.plot_uid[g.isin(chosen)])


class GPUTiles:
    """Holds the tiles for a set of images in CPU RAM (int16/uint8) and serves normalised float batches on the
    GPU. (Keeping the banks on the GPU costs ~15 GB and OOMs the sat-full steps.)"""

    def __init__(self, sensor, idx, bands, stats, device='cuda', tag=''):
        T, M = mm(sensor, 'tiles', tag), mm(sensor, 'mask', tag)
        idx = np.asarray(idx)
        order = np.argsort(idx)
        inv = np.empty_like(order)
        inv[order] = np.arange(len(idx))
        # Chunked, sorted memmap reads straight into the final arrays (no full-size temporaries). 8-bit sources stay
        # uint8 (inb PNG data); 16-bit satellite reflectance is stored as int16 (torch 1.10 has no uint16).
        dt = np.uint8 if T.dtype == np.uint8 else np.int16
        bands = list(bands)
        x = np.empty((len(idx),) + (T.shape[1], len(bands)) + T.shape[3:], dt)
        m = np.empty((len(idx),) + M.shape[1:], M.dtype)
        src = idx[order]
        for s in range(0, len(src), 1024):
            sl = slice(s, s + 1024)
            x[order[sl]] = T[src[sl]][:, :, bands].astype(dt, copy=False)
            m[order[sl]] = M[src[sl]]
        self.x = torch.from_numpy(x)
        self.m = torch.from_numpy(m)
        self.device = device
        key = 'sat' if sensor == 'sat' else 'uav'
        self.mean = torch.tensor(np.array(stats[f'{key}_mean'])[bands], dtype=torch.float32, device=device).view(1, -1, 1, 1)
        self.std = torch.tensor(np.array(stats[f'{key}_std'])[bands], dtype=torch.float32, device=device).view(1, -1, 1, 1)
        self.sensor = sensor
        self.row_mean = self.row_std = None  # optional per-image stats (n, C) overriding the global ones

    def set_row_stats(self, codes, n=200, seed=0):
        """Per-group (e.g. per flight = location x date) channel mean/std of the raw valid pixels, from <= n images per
        group; afterwards get() standardises every image with its own group's statistics."""
        codes = np.asarray(codes)
        rng = np.random.RandomState(seed)
        mean = np.zeros((len(codes), self.x.shape[2]), np.float32); std = np.ones_like(mean)
        for k in np.unique(codes):
            rows = np.flatnonzero(codes == k)
            sub = rng.choice(rows, min(n, len(rows)), replace=False)
            x = self.x[sub].float()                            # (n, 3 tiles, C, H, W) raw
            m = self.m[sub].bool().unsqueeze(2).expand_as(x)
            for c in range(x.shape[2]):
                v = x[:, :, c][m[:, :, c]]
                mean[rows, c], std[rows, c] = v.mean().item(), v.std().item()
        self.row_mean = torch.from_numpy(mean).to(self.device).view(-1, mean.shape[1], 1, 1)
        self.row_std = torch.from_numpy(std).to(self.device).view(-1, mean.shape[1], 1, 1)

    def gather(self, rows, tiles):
        """CPU gather into pinned memory (safe to run in a prefetch thread)."""
        rows, tiles = rows.cpu(), tiles.cpu()
        return self.x[rows, tiles].pin_memory(), self.m[rows, tiles].pin_memory()

    def get(self, rows, tiles, pre=None):
        """rows: LongTensor of image positions, tiles: LongTensor tile id (0..2). -> (x float, valid mask) on GPU"""
        x, m = pre if pre is not None else self.gather(rows, tiles)
        x = x.to(self.device, non_blocking=True).float()
        m = m.to(self.device, non_blocking=True)
        if self.row_mean is not None:
            r = rows.to(self.device)
            x = (x - self.row_mean[r]) / self.row_std[r]
        else:
            x = (x - self.mean) / self.std
        x = x * m.unsqueeze(1)  # padded pixels -> 0 (= dataset mean)
        return x, m

    def denorm(self, x):
        return x * self.std + self.mean


def augment(xs, ms, jitter_keys=(), gen=None):
    """Identical flips / 90-degree rotations for all modalities; mild colour jitter (in normalised space)
    on jitter_keys only. xs, ms: dicts of (B,C,H,W) / (B,H,W)."""
    B = next(iter(xs.values())).shape[0]
    dev = next(iter(xs.values())).device
    hf = torch.rand(B, device=dev) < 0.5
    vf = torch.rand(B, device=dev) < 0.5
    k = torch.randint(0, 4, (B,), device=dev)
    out_x, out_m = {}, {}
    for d in xs:
        x, m = xs[d], ms[d].unsqueeze(1).float()
        x = torch.where(hf.view(-1, 1, 1, 1), x.flip(-1), x)
        m = torch.where(hf.view(-1, 1, 1, 1), m.flip(-1), m)
        x = torch.where(vf.view(-1, 1, 1, 1), x.flip(-2), x)
        m = torch.where(vf.view(-1, 1, 1, 1), m.flip(-2), m)
        xr, mr = x.clone(), m.clone()
        for r in (1, 2, 3):
            s = k == r
            if s.any():
                xr[s] = torch.rot90(x[s], r, (-2, -1))
                mr[s] = torch.rot90(m[s], r, (-2, -1))
        if d in jitter_keys:
            C = xr.shape[1]
            gain = 1 + 0.1 * (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1)            # brightness / contrast
            tint = 0.1 * (torch.rand(B, C, 1, 1, device=dev) * 2 - 1)                 # per-channel tint
            xr = (xr * gain + tint) * mr
        out_x[d], out_m[d] = xr, mr.squeeze(1) > 0.5
    return out_x, out_m
