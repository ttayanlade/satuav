"""Feature extraction: handcrafted features (real Sat / real UAV / predicted UAV) and frozen-encoder embeddings.

  python -m satuav.extract handcrafted                 # run-independent H-sat-rgb / H-sat-all / H-uav
  python -m satuav.extract run <run_name> [...]       # E-sat (+ E-dec, H-pred, E-uav-real for crossmodal runs)
  python -m satuav.extract imnet <heldout> [...]       # E-imnet (original MultiMAE weights, sat RGB input)
  python -m satuav.extract latent <run_name> [...]    # E-pred / E-satpred / E-uav (latent counterparts of H reps);
                                                      # E-uav only for UAV-only runs (uavmae_*)
Output: features/<name>/<Loc>.npz with 'plot_uid' and one array per (feature, timepoint).
"""
import argparse
import json
import os

import numpy as np
import torch
from einops import rearrange

from . import data as D
from . import model as Mdl
from .paths import FEATS, LOCS, RUNS

EPS = 1e-6


# ----------------------------------------------------------------------------------- handcrafted

def _stats(v, m):
    """v: (T,H,W) values, m: (T,H,W) bool -> [min, mean, max, std] over valid pixels."""
    x = v[m]
    if x.numel() == 0:
        return [np.nan] * 4
    return [x.min().item(), x.mean().item(), x.max().item(), x.std().item()]


def handcrafted(img, m, bands):
    """img: (T,C,H,W) reflectance-like values in [0,1]; bands: list of names matching C.
    Paper feature set: per-band min/mean/max/std + GLI, NGRDI (+ NDVI, GNDVI, SAVI, NDRE when NIR/RE exist)."""
    b = {n: img[:, i] for i, n in enumerate(bands)}
    r, g, bl = b['red'], b['green'], b['blue']
    idx = {'GLI': (2 * g - r - bl) / (2 * g + r + bl + EPS), 'NGRDI': (r - g) / (r + g + EPS)}
    if 'nir' in b:
        n = b['nir']
        idx.update(NDVI=(n - r) / (n + r + EPS), GNDVI=(n - g) / (n + g + EPS), SAVI=1.5 * (n - r) / (n + r + 0.5))
    if 'rededge' in b:
        idx['NDRE'] = (b['nir'] - b['rededge']) / (b['nir'] + b['rededge'] + EPS)
    feats, names = [], []
    for k, v in list(b.items()) + list(idx.items()):
        feats += _stats(v, m)
        names += [f'{k}_{s}' for s in ('min', 'mean', 'max', 'std')]
    return feats, names


SAT_RGB = ['red', 'green', 'blue']
SAT_ALL = ['red', 'green', 'blue', 'nir', 'rededge']  # deep blue is redundant with blue for indices


def labelled(df, loc):
    return df[(df.location == loc) & df.has_label & (df.sensor == 'sat') & (df.tp == 1)].plot_uid.tolist()


def run_handcrafted():
    df = D.manifest()
    S, SM = D.mm('sat', 'tiles'), D.mm('sat', 'mask')
    U, UM = D.mm('uav', 'tiles'), D.mm('uav', 'mask')
    out_dir = os.path.join(FEATS, 'handcrafted')
    os.makedirs(out_dir, exist_ok=True)
    for L in LOCS:
        uids = labelled(df, L)
        out = {'plot_uid': np.array(uids)}
        for sensor, tps in [('sat', range(1, 7)), ('uav', range(1, 4))]:
            sub = df[(df.sensor == sensor) & (df.location == L)].set_index(['plot_uid', 'tp']).cache_idx
            for tp in tps:
                A, B, names = [], [], None
                for u in uids:
                    ci = int(sub.loc[(u, tp)])
                    if sensor == 'sat':
                        x = torch.from_numpy(S[ci].astype(np.float32) / 1e4)
                        m = torch.from_numpy(np.array(SM[ci]))
                        fa, names_a = handcrafted(x[:, :5], m, SAT_ALL)
                        fr, names_r = handcrafted(x[:, :3], m, SAT_RGB)
                        A.append(fa); B.append(fr)
                    else:
                        x = torch.from_numpy(U[ci].astype(np.float32) / 255.)
                        m = torch.from_numpy(np.array(UM[ci]))
                        f, names_u = handcrafted(x, m, SAT_RGB)
                        A.append(f)
                if sensor == 'sat':
                    out[f'H_sat_all_tp{tp}'] = np.array(A, np.float32)
                    out[f'H_sat_rgb_tp{tp}'] = np.array(B, np.float32)
                else:
                    out[f'H_uav_tp{tp}'] = np.array(A, np.float32)
        np.savez(os.path.join(out_dir, f'{L}.npz'), **out)
        print('handcrafted', L, len(uids), flush=True)
    json.dump(dict(sat_all=names_a, sat_rgb=names_r, uav=names_u), open(os.path.join(out_dir, 'names.json'), 'w'))


# ------------------------------------------------------------------------------------ embeddings

def valid_patches(m):
    """m: (B,224,224) -> (B,196) bool, patch counted valid if > 50% valid pixels."""
    B = m.shape[0]
    return (m.float().view(B, 14, 16, 14, 16).mean((2, 4)) > 0.5).view(B, 196)


def pool(enc, ids_keep, valid_tok):
    """Global token + mean of encoder tokens at valid input patches. valid_tok: (B, N_all) bool over all domain
    patch slots (concatenated), ids_keep: (B, K) the slots that were encoded."""
    g = enc[:, -1]
    t = enc[:, :-1]
    v = torch.gather(valid_tok, 1, ids_keep).float().unsqueeze(-1)
    mean = (t * v).sum(1) / v.sum(1).clamp_min(1)
    return torch.cat([g, mean], 1)


@torch.no_grad()
def embed_location(model, cfg, stats, L, df, device='cuda', bs=96, kinds=('E_sat',)):
    sd = [d for d in cfg['in_domains'] if d.startswith('sat')][0]
    uids = labelled(df, L)
    out = {'plot_uid': np.array(uids)}
    sat_tab = df[(df.sensor == 'sat') & (df.location == L)].set_index(['plot_uid', 'tp']).cache_idx
    uav_tab = df[(df.sensor == 'uav') & (df.location == L)].set_index(['plot_uid', 'tp'])
    tag = cfg.get('tag', '')
    xmodal = 'uav_rgb' in cfg['in_domains']
    for tp in range(1, 7):
        idx = [int(sat_tab.loc[(u, tp)]) for u in uids]
        sat = D.GPUTiles('sat', idx, D.SAT_IDX[sd], stats, device, tag=tag)
        has_dec = xmodal and 'uav_rgb' in model.output_adapters
        acc = {k: [] for k in ('E_sat', 'E_dec', 'H_pred')}
        for s in range(0, len(uids), bs):
            r = torch.arange(s, min(len(uids), s + bs), device=device)
            per_tile = {k: [] for k in acc}
            pred_tiles, pred_masks = [], []
            for j in range(3):
                tj = torch.full_like(r, j)
                xs, ms = sat.get(r, tj)
                x = {sd: xs}
                if xmodal:
                    x['uav_rgb'] = torch.zeros(len(r), 3, 224, 224, device=device)
                with torch.cuda.amp.autocast():
                    outd = ['uav_rgb'] if has_dec else []
                    counts = model.counts(len(r), 0, None, 'sat_full', None, device) if xmodal else \
                        torch.full((len(r), 1), 196, dtype=torch.long, device=device)
                    enc, _, info, ids_keep, ids_restore = model.encode(x, counts)
                    vt = valid_patches(ms)
                    if xmodal:
                        vt = torch.cat([vt, torch.zeros_like(vt)], 1)
                    per_tile['E_sat'].append(pool(enc.float(), ids_keep, vt))
                    if outd:
                        dec = model.decoder_features('uav_rgb', enc, info, ids_keep, ids_restore).float()
                        pv = valid_patches(ms).float().unsqueeze(-1)
                        per_tile['E_dec'].append((dec * pv).sum(1) / pv.sum(1).clamp_min(1))
                        p = model.output_adapters['uav_rgb'].out_proj(dec.to(model.output_adapters['uav_rgb'].out_proj.weight.dtype))
                        p = rearrange(p.float(), 'b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)', nh=14, nw=14, ph=16, pw=16, c=3)
                        img = (p * torch.tensor(stats['uav_std'], device=device).view(1, 3, 1, 1) +
                               torch.tensor(stats['uav_mean'], device=device).view(1, 3, 1, 1)).clamp(0, 255) / 255.
                        pred_tiles.append(img); pred_masks.append(ms)
            acc['E_sat'].append(torch.stack(per_tile['E_sat']).mean(0).cpu().numpy())
            if pred_tiles:
                acc['E_dec'].append(torch.stack(per_tile['E_dec']).mean(0).cpu().numpy())
                P, PM = torch.stack(pred_tiles, 1), torch.stack(pred_masks, 1)
                for i in range(P.shape[0]):
                    acc['H_pred'].append(handcrafted(P[i], PM[i], SAT_RGB)[0])
        out[f'E_sat_tp{tp}'] = np.concatenate(acc['E_sat']).astype(np.float32)
        if acc['E_dec']:
            out[f'E_dec_tp{tp}'] = np.concatenate(acc['E_dec']).astype(np.float32)
            out[f'H_pred_tp{tp}'] = np.array(acc['H_pred'], np.float32)
        del sat
    # Oracle: crossmodal encoder with real UAV (+ nearest-date satellite), all tokens visible
    if xmodal:
        for tp in range(1, 4):
            ut = uav_tab.loc[[(u, tp) for u in uids]]
            sidx = [int(sat_tab.loc[(u, int(p))]) for u, p in zip(uids, ut.paired_tp)]
            sat = D.GPUTiles('sat', sidx, D.SAT_IDX[sd], stats, device, tag=tag)
            uav = D.GPUTiles('uav', list(ut.cache_idx.astype(int)), [0, 1, 2], stats, device, tag=tag)
            res = []
            for s in range(0, len(uids), bs // 2):
                r = torch.arange(s, min(len(uids), s + bs // 2), device=device)
                tiles = []
                for j in range(3):
                    tj = torch.full_like(r, j)
                    xs, ms = sat.get(r, tj)
                    xu, mu = uav.get(r, tj)
                    with torch.cuda.amp.autocast():
                        pres = torch.ones(len(r), 2, dtype=torch.bool, device=device)
                        counts = model.counts(len(r), 0, None, 'full', pres, device)
                        enc, _, _, ids_keep, _ = model.encode({sd: xs, 'uav_rgb': xu}, counts)
                    tiles.append(pool(enc.float(), ids_keep, torch.cat([valid_patches(ms), valid_patches(mu)], 1)))
                res.append(torch.stack(tiles).mean(0).cpu().numpy())
            out[f'E_uavreal_tp{tp}'] = np.concatenate(res).astype(np.float32)
    return out


@torch.no_grad()
def embed_latent_location(model, cfg, stats, L, df, device='cuda', bs=48):
    """Latent counterparts of the handcrafted reps, all from the crossmodal encoder (all tokens visible):
    E_pred (generated UAV only, via the uav_rgb input adapter), E_satpred (real sat + generated UAV jointly) at
    every satellite TP, and E_uav (real UAV only) at the 3 flown TPs. Generated UAV = sat_full decoder output (as
    for H_pred), clamped to [0, 255], renormalised and zeroed outside the satellite mask like a real UAV tile."""
    sd = [d for d in cfg['in_domains'] if d.startswith('sat')][0]
    uids = labelled(df, L)
    out = {'plot_uid': np.array(uids)}
    sat_tab = df[(df.sensor == 'sat') & (df.location == L)].set_index(['plot_uid', 'tp']).cache_idx
    uav_tab = df[(df.sensor == 'uav') & (df.location == L)].set_index(['plot_uid', 'tp'])
    tag = cfg.get('tag', '')
    um = torch.tensor(stats['uav_mean'], device=device).view(1, 3, 1, 1)
    us = torch.tensor(stats['uav_std'], device=device).view(1, 3, 1, 1)
    uav_only = torch.tensor([[False, True]], device=device)
    for tp in range(1, 7):
        sat = D.GPUTiles('sat', [int(sat_tab.loc[(u, tp)]) for u in uids], D.SAT_IDX[sd], stats, device, tag=tag)
        acc = {'E_pred': [], 'E_satpred': []}
        for s in range(0, len(uids), bs):
            r = torch.arange(s, min(len(uids), s + bs), device=device)
            tiles = {k: [] for k in acc}
            for j in range(3):
                xs, ms = sat.get(r, torch.full_like(r, j))
                vs = valid_patches(ms)
                with torch.cuda.amp.autocast():
                    x = {sd: xs, 'uav_rgb': torch.zeros(len(r), 3, 224, 224, device=device)}
                    enc, _, info, ids_keep, ids_restore = model.encode(x, model.counts(len(r), 0, None, 'sat_full', None, device))
                    dec = model.decoder_features('uav_rgb', enc, info, ids_keep, ids_restore)
                    oa = model.output_adapters['uav_rgb']
                    p = oa.out_proj(dec.to(oa.out_proj.weight.dtype))
                    p = rearrange(p.float(), 'b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)', nh=14, nw=14, ph=16, pw=16, c=3)
                    xg = (((p * us + um).clamp(0, 255) - um) / us) * ms.unsqueeze(1)
                    counts = model.counts(len(r), 0, None, 'full', uav_only.expand(len(r), 2), device)
                    enc, _, _, ids_keep, _ = model.encode({sd: torch.zeros_like(xs), 'uav_rgb': xg}, counts)
                    tiles['E_pred'].append(pool(enc.float(), ids_keep, torch.cat([torch.zeros_like(vs), vs], 1)))
                    counts = model.counts(len(r), 0, None, 'full', torch.ones(len(r), 2, dtype=torch.bool, device=device), device)
                    enc, _, _, ids_keep, _ = model.encode({sd: xs, 'uav_rgb': xg}, counts)
                    tiles['E_satpred'].append(pool(enc.float(), ids_keep, torch.cat([vs, vs], 1)))
            for k in acc:
                acc[k].append(torch.stack(tiles[k]).mean(0).cpu().numpy())
        for k in acc:
            out[f'{k}_tp{tp}'] = np.concatenate(acc[k]).astype(np.float32)
        del sat
    for tp in range(1, 4):
        ut = uav_tab.loc[[(u, tp) for u in uids]]
        uav = D.GPUTiles('uav', list(ut.cache_idx.astype(int)), [0, 1, 2], stats, device, tag=tag)
        res = []
        for s in range(0, len(uids), bs):
            r = torch.arange(s, min(len(uids), s + bs), device=device)
            tiles = []
            for j in range(3):
                xu, mu = uav.get(r, torch.full_like(r, j))
                vu = valid_patches(mu)
                with torch.cuda.amp.autocast():
                    xs = torch.zeros(len(r), len(D.SAT_IDX[sd]), 224, 224, device=device)
                    counts = model.counts(len(r), 0, None, 'full', uav_only.expand(len(r), 2), device)
                    enc, _, _, ids_keep, _ = model.encode({sd: xs, 'uav_rgb': xu}, counts)
                tiles.append(pool(enc.float(), ids_keep, torch.cat([torch.zeros_like(vu), vu], 1)))
            res.append(torch.stack(tiles).mean(0).cpu().numpy())
        out[f'E_uav_tp{tp}'] = np.concatenate(res).astype(np.float32)
        del uav
    return out


@torch.no_grad()
def embed_uav_location(model, stats, L, df, tag='', device='cuda', bs=96, flight_norm=False):
    """UAV-only MAE: E_uav (real UAV, all tokens visible) at the 3 flown TPs, same pooling as the crossmodal E_uav."""
    uids = labelled(df, L)
    out = {'plot_uid': np.array(uids)}
    uav_tab = df[(df.sensor == 'uav') & (df.location == L)].set_index(['plot_uid', 'tp'])
    for tp in range(1, 4):
        ut = uav_tab.loc[[(u, tp) for u in uids]]
        uav = D.GPUTiles('uav', list(ut.cache_idx.astype(int)), [0, 1, 2], stats, device, tag=tag)
        if flight_norm:  # one flight (L, tp): standardise with its own raw-pixel stats, as in training
            uav.set_row_stats(np.zeros(len(uids), int))
        res = []
        for s in range(0, len(uids), bs):
            r = torch.arange(s, min(len(uids), s + bs), device=device)
            tiles = []
            for j in range(3):
                xu, mu = uav.get(r, torch.full_like(r, j))
                with torch.cuda.amp.autocast():
                    counts = model.counts(len(r), 0, None, 'full', torch.ones(len(r), 1, dtype=torch.bool, device=device), device)
                    enc, _, _, ids_keep, _ = model.encode({'uav_rgb': xu}, counts)
                tiles.append(pool(enc.float(), ids_keep, valid_patches(mu)))
            res.append(torch.stack(tiles).mean(0).cpu().numpy())
        out[f'E_uav_tp{tp}'] = np.concatenate(res).astype(np.float32)
        del uav
    return out


def load_run(name, device='cuda'):
    ck = torch.load(os.path.join(RUNS, name, 'checkpoint-last.pth'), map_location='cpu')
    cfg = ck['cfg']
    m = Mdl.build(cfg, init=False)
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck['model'].items()})
    stats = json.load(open(os.path.join(RUNS, name, 'norm_stats.json')))
    return m.to(device).eval(), cfg, stats


def run_embeddings(name, model, cfg, stats, locs=LOCS):
    df = D.manifest()
    out_dir = os.path.join(FEATS, name)
    os.makedirs(out_dir, exist_ok=True)
    for L in locs:
        np.savez(os.path.join(out_dir, f'{L}.npz'), **embed_location(model, cfg, stats, L, df))
        print('embed', name, L, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['handcrafted', 'run', 'imnet', 'latent'])
    ap.add_argument('names', nargs='*')
    a = ap.parse_args()
    if a.what == 'handcrafted':
        run_handcrafted()
    elif a.what == 'latent':  # -> features/<run>__lat/
        df = D.manifest()
        for n in a.names:
            m, cfg, stats = load_run(n)
            os.makedirs(os.path.join(FEATS, f'{n}__lat'), exist_ok=True)
            uav_only = not any(d.startswith('sat') for d in cfg['in_domains'])
            for L in LOCS:
                f = embed_uav_location(m, stats, L, df, cfg.get('tag', ''), flight_norm=cfg.get('uav_norm') == 'flight') if uav_only else \
                    embed_latent_location(m, cfg, stats, L, df)
                np.savez(os.path.join(FEATS, f'{n}__lat', f'{L}.npz'), **f)
                print('latent', n, L, flush=True)
            del m
            torch.cuda.empty_cache()
    elif a.what == 'run':
        for n in a.names:
            m, cfg, stats = load_run(n)
            run_embeddings(n, m, cfg, stats)
            del m
            torch.cuda.empty_cache()
    else:
        # E-imnet per held-out location: only the normalisation statistics depend on the fold.
        df = D.manifest()
        for L in a.names or LOCS:
            m = Mdl.build_imnet('sat_rgb').cuda().eval()
            cfg = dict(in_domains=['sat_rgb'], out_domains=[], tag='')
            stats = D.fold_stats(df, [l for l in LOCS if l != L])
            run_embeddings(f'imnet_{L}', m, cfg, stats)


if __name__ == '__main__':
    main()
