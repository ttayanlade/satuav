"""Field-level demo: real satellite, real UAV, and UAV predicted from satellite only (all UAV tokens masked), arranged in
the field layout, for a held-out flight with cloud shadows. The prediction uses the LOLO model that never saw the location.

  python -m satuav.field_demo --loc Ames --tp 2 --run xmodalrgb_s0_{L} [--exp 4233]
  -> figs/field_demo/field_<loc>_tp<tp>.png / .pdf, results/field_demo_<loc>_tp<tp>.csv (per-plot brightness)
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from einops import rearrange

from . import data as D
from .extract import load_run
from .paths import EXP, RESULTS

CELL_H, CELL_W = 72, 24  # pixels per plot in the mosaic (3 square subtiles stacked; plot long axis vertical)


def small(img, h, w):
    """(C, H, W) float -> (h, w, C) by area averaging (H, W divisible)."""
    t = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0)
    return torch.nn.functional.adaptive_avg_pool2d(t, (h, w))[0].permute(1, 2, 0).numpy()


@torch.no_grad()
def predict_uav(model, cfg, stats, sat_idx, bs=48):
    """UAV prediction from satellite only (sat-full mode), de-normalised to 0-255. Returns (n, 3 tiles, 3, 224, 224)."""
    sd = [d for d in cfg['in_domains'] if d.startswith('sat')][0]
    sat = D.GPUTiles('sat', sat_idx, D.SAT_IDX[sd], stats, 'cuda', tag=cfg.get('tag', ''))
    um = torch.tensor(stats['uav_mean'], device='cuda').view(1, 3, 1, 1)
    us = torch.tensor(stats['uav_std'], device='cuda').view(1, 3, 1, 1)
    out = []
    for s in range(0, len(sat_idx), bs):
        r = torch.arange(s, min(len(sat_idx), s + bs), device='cuda')
        tiles = []
        for j in range(3):
            xs, ms = sat.get(r, torch.full_like(r, j))
            with torch.cuda.amp.autocast():
                x = {sd: xs, 'uav_rgb': torch.zeros(len(r), 3, 224, 224, device='cuda')}
                enc, _, info, ids_keep, ids_restore = model.encode(x, model.counts(len(r), 0, None, 'sat_full', None, 'cuda'))
                dec = model.decoder_features('uav_rgb', enc, info, ids_keep, ids_restore)
                oa = model.output_adapters['uav_rgb']
                p = oa.out_proj(dec.to(oa.out_proj.weight.dtype))
            p = rearrange(p.float(), 'b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)', nh=14, nw=14, ph=16, pw=16, c=3)
            tiles.append(((p * us + um).clamp(0, 255) * ms.unsqueeze(1)).cpu().numpy())
        out.append(np.stack(tiles, 1))
    return np.concatenate(out)


def mosaic(cells, layout, blank=1.0):
    """cells: (n, CELL_H, CELL_W, 3) in [0, 1]; layout: (n, 2) integer (grid row, grid col) -> image."""
    R, C = layout[:, 0].max() + 1, layout[:, 1].max() + 1
    img = np.full((R * CELL_H, C * CELL_W, 3), blank, np.float32)
    for (r, c), x in zip(layout, cells):
        img[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W] = x
    return img


def plot_cells(tiles, masks, lo, hi):
    """tiles (n, 3, C>=3, 224, 224) raw, masks (n, 3, 224, 224) -> (n, CELL_H, CELL_W, 3) display in [0, 1]."""
    out = []
    for t, m in zip(tiles, masks):
        x = np.concatenate([np.rot90(t[j, :3], 1, (1, 2)) for j in range(3)], 1).astype(np.float32)  # (3, 672, 224)
        mm = np.concatenate([np.rot90(m[j], 1, (0, 1)) for j in range(3)], 0).astype(np.float32)[None]
        v = small(x * mm, CELL_H, CELL_W) / np.maximum(small(mm, CELL_H, CELL_W), 1e-3)
        v = np.clip((v - lo) / (hi - lo), 0, 1)
        out.append(np.where(small(mm, CELL_H, CELL_W) > 0.5, v, 1.0))
    return np.stack(out)


def lum(tiles, masks):
    """Mean luminance of the valid pixels of each plot (raw units)."""
    w = np.array([0.299, 0.587, 0.114], np.float32).reshape(1, 1, 3, 1, 1)
    y = (tiles[:, :, :3].astype(np.float32) * w).sum(2)
    m = masks.astype(np.float32)
    return (y * m).sum((1, 2, 3)) / np.maximum(m.sum((1, 2, 3)), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--loc', default='Ames')
    ap.add_argument('--tp', type=int, default=2)
    ap.add_argument('--run', default='xmodalrgb_s0_{L}')
    ap.add_argument('--exp', default=None)
    a = ap.parse_args()
    L, t = a.loc, a.tp
    df = D.manifest()
    u = df[(df.location == L) & (df.sensor == 'uav') & (df.tp == t)]
    exp = a.exp or u.experiment.value_counts().index[0]
    u = u[u.experiment.astype(str) == str(exp)].copy()
    stab = df[(df.sensor == 'sat') & (df.location == L)].set_index(['plot_uid', 'tp']).cache_idx
    u['sat_idx'] = [int(stab.loc[(p, int(q))]) for p, q in zip(u.plot_uid, u.paired_tp)]
    rr = u['range'].astype(float).values
    rw = u['row'].astype(float).values
    layout = np.c_[(rr - rr.min()).astype(int), (rw - rw.min()).astype(int)]  # ranges along y (plot length), rows along x

    run = a.run.format(L=L)
    model, cfg, stats = load_run(run)
    pred = predict_uav(model, cfg, stats, list(u.sat_idx))
    del model
    torch.cuda.empty_cache()
    UT, UM = D.mm('uav', 'tiles'), D.mm('uav', 'mask')
    ST, SM = D.mm('sat', 'tiles'), D.mm('sat', 'mask')
    ui, si = u.cache_idx.astype(int).values, u.sat_idx.values
    uav, uavm = np.asarray(UT[ui]), np.asarray(UM[ui])
    sat, satm = np.asarray(ST[si])[:, :, :3], np.asarray(SM[si])

    # display ranges: robust percentiles over the field (per channel-agnostic), same for real and predicted UAV
    def pr(x, m):
        v = x[:, :, :3].transpose(0, 1, 3, 4, 2)[m.astype(bool)]
        return np.percentile(v, 1), np.percentile(v, 99)
    panels = [('Satellite (input)', plot_cells(sat, satm, *pr(sat, satm))),
              ('Real UAV (not used)', plot_cells(uav, uavm, *pr(uav, uavm))),
              ('Predicted UAV', plot_cells(pred, satm, *pr(pred, satm)))]

    # per-plot luminance, and the field-scale pattern (mean per range = along the field's long axis)
    d = pd.DataFrame(dict(plot_uid=u.plot_uid.values, range=rr, row=rw, sat=lum(sat, satm), uav=lum(uav, uavm),
                          pred=lum(pred, satm)))
    os.makedirs(RESULTS, exist_ok=True)
    z = d[['sat', 'uav', 'pred']].apply(lambda c: (c - c.mean()) / c.std())
    d['res_uav'], d['res_pred'] = z.uav - z.sat, z.pred - z.sat
    d.to_csv(os.path.join(RESULTS, f'field_demo_{L}_tp{t}.csv'), index=False)
    # spatial coherence of the residual: correlation of each plot's residual with the mean of its 4 grid neighbours
    pos = {(a_, b_): k for k, (a_, b_) in enumerate(layout)}
    def nb_corr(r):
        own, nbm = [], []
        for k, (a_, b_) in enumerate(layout):
            nb = [pos[q] for q in ((a_ - 1, b_), (a_ + 1, b_), (a_, b_ - 1), (a_, b_ + 1)) if q in pos]
            if nb:
                own.append(r[k]); nbm.append(np.mean(r[nb]))
        return float(np.corrcoef(own, nbm)[0, 1])
    summ = dict(location=L, tp=t, experiment=str(exp), run=run, n_plots=len(d),
                corr_uav_sat=float(d.uav.corr(d.sat)), corr_pred_sat=float(d.pred.corr(d.sat)),
                resid_sd_uav=float(d.res_uav.std()), resid_sd_pred=float(d.res_pred.std()),
                resid_neighbour_corr_uav=nb_corr(d.res_uav.values), resid_neighbour_corr_pred=nb_corr(d.res_pred.values))
    print(json.dumps(summ, indent=1))
    json.dump(summ, open(os.path.join(RESULTS, f'field_demo_{L}_tp{t}.json'), 'w'), indent=1)

    out = os.path.join(EXP, 'figs', 'field_demo')
    os.makedirs(out, exist_ok=True)
    R, C = layout[:, 0].max() + 1, layout[:, 1].max() + 1
    fig, ax = plt.subplots(1, 5, figsize=(13, 9), gridspec_kw=dict(wspace=0.08))
    for k, (title, cells) in enumerate(panels):
        ax[k].imshow(mosaic(cells, layout))
        ax[k].set_title(title, fontsize=11)
        ax[k].axis('off')
    v = np.nanmax(np.abs(d[['res_uav', 'res_pred']].values))
    for k, (col, title) in enumerate((('res_uav', 'Real UAV $-$ satellite'), ('res_pred', 'Predicted UAV $-$ satellite')), 3):
        g = np.full((R, C), np.nan)
        g[layout[:, 0], layout[:, 1]] = d[col].values
        im = ax[k].imshow(np.kron(g, np.ones((CELL_H, CELL_W))), cmap='RdBu', vmin=-v, vmax=v)
        ax[k].set_title(title, fontsize=11)
        ax[k].axis('off')
    cb = fig.colorbar(im, ax=ax[3:], fraction=0.04, pad=0.02)
    cb.set_label('Plot luminance difference (z)')
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(out, f'field_{L}_tp{t}.{ext}'), dpi=200, bbox_inches='tight')
    print('saved', os.path.join(out, f'field_{L}_tp{t}.png'))
    plt.close(fig)
    # qualitative 3-panel version (paper supplement): satellite, real UAV, predicted UAV
    fig, ax = plt.subplots(1, 3, figsize=(8, 9), gridspec_kw=dict(wspace=0.06))
    for k, (title, cells) in enumerate([('Satellite (input)', panels[0][1]), ('Real UAV', panels[1][1]),
                                        ('Predicted UAV', panels[2][1])]):
        ax[k].imshow(mosaic(cells, layout))
        ax[k].set_title(title, fontsize=11)
        ax[k].axis('off')
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(out, f'field_{L}_tp{t}_qual.{ext}'), dpi=200, bbox_inches='tight')
    print('saved', os.path.join(out, f'field_{L}_tp{t}_qual.png'))



@torch.no_grad()
def predict_uav_from(model, cfg, stats, sat_idx, uav_idx, mode, bs=32, seed=0):
    """UAV output of the crossmodal model for other inputs. mode: 'uav_full' (UAV only, all tokens visible),
    'joint_full' (satellite + UAV, all visible), 'uav_66' (UAV only, 66 random visible tokens, rest reconstructed)."""
    torch.manual_seed(seed)
    sd = [d for d in cfg['in_domains'] if d.startswith('sat')][0]
    sat = D.GPUTiles('sat', sat_idx, D.SAT_IDX[sd], stats, 'cuda', tag=cfg.get('tag', ''))
    uav = D.GPUTiles('uav', uav_idx, [0, 1, 2], stats, 'cuda', tag=cfg.get('tag', ''))
    um = torch.tensor(stats['uav_mean'], device='cuda').view(1, 3, 1, 1)
    us = torch.tensor(stats['uav_std'], device='cuda').view(1, 3, 1, 1)
    out = []
    for s in range(0, len(sat_idx), bs):
        r = torch.arange(s, min(len(sat_idx), s + bs), device='cuda')
        n = len(r)
        tiles = []
        for j in range(3):
            xs, ms = sat.get(r, torch.full_like(r, j))
            xu, mu = uav.get(r, torch.full_like(r, j))
            with torch.cuda.amp.autocast():
                if mode == 'joint_full':
                    x = {sd: xs, 'uav_rgb': xu}
                    counts = model.counts(n, 0, None, 'full', torch.ones(n, 2, dtype=torch.bool, device='cuda'), 'cuda')
                else:
                    x = {sd: torch.zeros_like(xs), 'uav_rgb': xu}
                    pres = torch.tensor([[False, True]], device='cuda').expand(n, 2)
                    counts = model.counts(n, 0, None, 'full', pres, 'cuda') if mode == 'uav_full' else \
                        torch.tensor([[0, 66]], device='cuda').expand(n, 2).contiguous()
                enc, _, info, ids_keep, ids_restore = model.encode(x, counts)
                dec = model.decoder_features('uav_rgb', enc, info, ids_keep, ids_restore)
                oa = model.output_adapters['uav_rgb']
                p = oa.out_proj(dec.to(oa.out_proj.weight.dtype))
            p = rearrange(p.float(), 'b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)', nh=14, nw=14, ph=16, pw=16, c=3)
            tiles.append(((p * us + um).clamp(0, 255) * mu.unsqueeze(1)).cpu().numpy())
        out.append(np.stack(tiles, 1))
    return np.concatenate(out)


def uav_through(loc='Ames', tp=2, run_t='xmodalrgb_s0_{L}', exp=None):
    """Compare the UAV output for satellite-only input (as in main()) with UAV-input variants; same field and stats."""
    df = D.manifest()
    u = df[(df.location == loc) & (df.sensor == 'uav') & (df.tp == tp)]
    exp = exp or u.experiment.value_counts().index[0]
    u = u[u.experiment.astype(str) == str(exp)].copy()
    stab = df[(df.sensor == 'sat') & (df.location == loc)].set_index(['plot_uid', 'tp']).cache_idx
    u['sat_idx'] = [int(stab.loc[(p, int(q))]) for p, q in zip(u.plot_uid, u.paired_tp)]
    rr, rw = u['range'].astype(float).values, u['row'].astype(float).values
    layout = np.c_[(rr - rr.min()).astype(int), (rw - rw.min()).astype(int)]
    run = run_t.format(L=loc)
    model, cfg, stats = load_run(run)
    si, ui = list(u.sat_idx), list(u.cache_idx.astype(int))
    preds = {'sat_full': predict_uav(model, cfg, stats, si)}
    for m in ('uav_full', 'joint_full', 'uav_66'):
        preds[m] = predict_uav_from(model, cfg, stats, si, ui, m)
    del model
    torch.cuda.empty_cache()
    UT, UM, ST, SM = D.mm('uav', 'tiles'), D.mm('uav', 'mask'), D.mm('sat', 'tiles'), D.mm('sat', 'mask')
    uav, uavm = np.asarray(UT[ui]), np.asarray(UM[ui])
    sat, satm = np.asarray(ST[si])[:, :, :3], np.asarray(SM[si])
    d = pd.DataFrame(dict(sat=lum(sat, satm), uav=lum(uav, uavm)))
    for m, x in preds.items():
        d[m] = lum(x, satm if m == 'sat_full' else uavm)
    z = d.apply(lambda c: (c - c.mean()) / c.std())
    pos = {(a_, b_): k for k, (a_, b_) in enumerate(layout)}
    def nb_corr(r):
        own, nbm = [], []
        for k, (a_, b_) in enumerate(layout):
            nb = [pos[q] for q in ((a_ - 1, b_), (a_ + 1, b_), (a_, b_ - 1), (a_, b_ + 1)) if q in pos]
            if nb:
                own.append(r[k]); nbm.append(np.mean(r[nb]))
        return float(np.corrcoef(own, nbm)[0, 1])
    rows = []
    for c in ['uav', 'sat_full', 'uav_full', 'joint_full', 'uav_66']:
        res = (z[c] - z.sat).values
        rows.append(dict(image=c, corr_with_sat=float(d[c].corr(d.sat)), corr_with_real_uav=float(d[c].corr(d.uav)),
                         resid_sd=float(res.std()), resid_neighbour_corr=nb_corr(res)))
    S = pd.DataFrame(rows)
    S.round(3).to_csv(os.path.join(RESULTS, f'field_demo_{loc}_tp{tp}_uavthrough.csv'), index=False)
    print(S.round(3).to_string(index=False))
    def pr(x, m):
        v = x[:, :, :3].transpose(0, 1, 3, 4, 2)[m.astype(bool)]
        return np.percentile(v, 1), np.percentile(v, 99)
    panels = [('Real UAV', uav, uavm), ('Satellite only', preds['sat_full'], satm), ('UAV only (all tokens)', preds['uav_full'], uavm),
              ('Satellite + UAV (all tokens)', preds['joint_full'], uavm), ('UAV only (66 tokens)', preds['uav_66'], uavm)]
    fig, ax = plt.subplots(1, len(panels), figsize=(14, 9), gridspec_kw=dict(wspace=0.08))
    for k, (title, x, m) in enumerate(panels):
        ax[k].imshow(mosaic(plot_cells(x, m, *pr(x, m)), layout))
        ax[k].set_title(title if k == 0 else 'Pred.: ' + title, fontsize=10)
        ax[k].axis('off')
    out = os.path.join(EXP, 'figs', 'field_demo')
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(out, f'field_{loc}_tp{tp}_uavthrough.{ext}'), dpi=200, bbox_inches='tight')
    print('saved', os.path.join(out, f'field_{loc}_tp{tp}_uavthrough.png'))

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'uavthrough':
        uav_through()
    else:
        main()
