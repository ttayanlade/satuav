"""Pretraining (crossmodal sat->UAV MultiMAE, or satellite-only MAE) for one LOLO fold.

Usage: python -m satuav.pretrain configs/<run>.yaml [--epochs N]

Alternatives to the UAV pixel target (fixes for crossmodal training that transfers nothing to the sat features):
  objective=distill  : sat-only MAE + predict the frozen UAV-only MAE's (uavmae_s<seed>_<heldout>) per-patch tokens and
                       pooled vector from the fully visible satellite tile
  objective=contrast : sat-only MAE + InfoNCE between the pooled sat embedding and the frozen UAV-only MAE embedding
                       of the same plot/date (other plots in the micro-batch are negatives)
  uav_target=blurflight : crossmodal MAE, but UAV standardised per flight (site x date) and reduced to 16 px patch means
  uav_norm=flight    : UAV standardised per flight (location x date, own-flight raw-pixel stats) instead of the fold's
                       global stats; extraction does the same per held-out flight (UAV-only control runs, uavmaeF_*)
  uav_vis=matched    : UAV-only MAE whose visible-token count per step is drawn from the crossmodal runs' visible-UAV
                       distribution (66 - round(66*lam_sat), lam ~ Dirichlet(alphas_xmodal), given >= 1) (uavmaeM_*)
  context_frac=f     : also pretrain on the paired images of a fraction f of the held-out site's plots (whole genotypes,
                       data.context_uids); downstream evaluation then uses only the remaining plots
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from . import data as D
from . import model as Mdl
from .paths import LOCS, RUNS

DEFAULTS = dict(in_domains=['sat_all', 'uav_rgb'], out_domains=['sat_all', 'uav_rgb'], alphas=[0.9, 0.1], K=66,
                p_satfull=0.25, epochs=50, bs=256, blr=1.5e-4, warmup=5, wd=0.05, seed=0, init='multimae',
                transductive=False, mismatch=False, masked_loss=True, pretrain_locs=None, tag='', jitter=True,
                grid_every=10, clip=3.0, objective='mae', uav_target='pixel', aux_w=1.0, tau=0.1, context_frac=0.0,
                uav_norm='global', uav_vis='fixed', alphas_xmodal=[0.9, 0.1])


def load_cfg(path):
    cfg = dict(DEFAULTS)
    cfg.update(yaml.safe_load(open(path)))
    cfg.setdefault('name', os.path.splitext(os.path.basename(path))[0])
    if cfg['pretrain_locs'] is None:
        cfg['pretrain_locs'] = [l for l in LOCS if l != cfg['heldout']]
    return cfg


def sat_domain(cfg):
    return next((d for d in cfg['in_domains'] if d.startswith('sat')), None)  # None for the UAV-only MAE


def masked_mse(pred, tgt, tok_mask, valid, use_valid=True, weight=None):
    """MSE over masked tokens (tok_mask 1 = masked), optionally restricted to valid (non-padded) pixels."""
    B, C, H, W = pred.shape
    w = F.interpolate(tok_mask.view(B, 1, 14, 14).float(), size=(H, W), mode='nearest')
    if use_valid:
        w = w * valid.unsqueeze(1).float()
    if weight is not None:
        w = w * weight.view(-1, 1, 1, 1).float()
    se = ((pred.float() - tgt) ** 2).mean(1, keepdim=True)
    return (se * w).sum() / w.sum().clamp_min(1.0)


class Pipeline:
    """Loads the GPU tile banks for a run and yields batches."""

    def __init__(self, cfg, device='cuda'):
        self.cfg = cfg
        df = D.manifest()
        self.stats = D.fold_stats(df, cfg['pretrain_locs'])
        self.sd = sat_domain(cfg)
        pairs = D.pair_table(df, cfg['pretrain_locs'], mismatch=cfg['mismatch'], seed=cfg['seed'])
        if cfg['context_frac']:  # in-site context: held-out-site plots (sat, + UAV if used), normalisation unchanged
            ctx = D.pair_table(df, [cfg['heldout']])
            pairs = pd.concat([pairs, ctx[ctx.plot_uid.isin(D.context_uids(df, cfg['heldout'], cfg['context_frac']))]],
                              ignore_index=True)
        sat_imgs = list(pairs.sat)
        n_pair = len(pairs)
        trans = []
        if cfg['transductive']:
            # Unlabelled satellite imagery of the held-out site at the UAV-partnered timepoints (no UAV, no labels)
            L = cfg['heldout']
            ptp = df[(df.sensor == 'uav') & (df.location == L)].paired_tp.unique()
            t = df[(df.sensor == 'sat') & (df.location == L) & df.tp.isin(ptp)]
            trans = list(t.cache_idx)
        self.sat_pos = {}
        uniq = sorted(set(sat_imgs) | set(trans))
        self.sat_pos = {c: i for i, c in enumerate(uniq)}
        if self.sd:
            self.sat = D.GPUTiles('sat', uniq, D.SAT_IDX[self.sd], self.stats, device, tag=cfg['tag'])
        self.has_uav = 'uav_rgb' in cfg['in_domains']
        self.xmodal = self.has_uav and self.sd is not None
        self.need_uav = self.has_uav or cfg['objective'] in ('distill', 'contrast')  # UAV as input/target or as teacher input
        if self.need_uav:
            self.uav = D.GPUTiles('uav', list(pairs.uav), [0, 1, 2], self.stats, device, tag=cfg['tag'])
            if cfg['uav_norm'] == 'flight':
                self.uav.set_row_stats(pd.factorize(pairs.location.astype(str) + '_' + pairs.uav_tp.astype(str))[0])
        if cfg['uav_target'] == 'blurflight':
            self.flight_stats(pairs, device)
        # items: (sat position, uav position or -1, tile)
        sp = np.array([self.sat_pos[c] for c in sat_imgs] + [self.sat_pos[c] for c in trans])
        up = np.r_[np.arange(n_pair), -np.ones(len(trans), int)]
        if not self.need_uav:
            up[:] = -1
        self.items = torch.from_numpy(np.stack([np.repeat(sp, 3), np.repeat(up, 3), np.tile([0, 1, 2], len(sp))], 1))
        self.n_pair, self.n_trans = n_pair, len(trans)
        self.device = device

    def prefetch(self, idx):
        it = self.items[idx]
        pre = {'it': it}
        if self.sd:
            pre['sat'] = self.sat.gather(it[:, 0], it[:, 2])
        if self.need_uav:
            pre['uav'] = self.uav.gather(it[:, 1].clamp_min(0), it[:, 2])
        return pre

    def batches(self, perm, bs):
        """Iterate batches with CPU gathering overlapped in a background thread."""
        from concurrent.futures import ThreadPoolExecutor
        n = len(perm) // bs
        with ThreadPoolExecutor(1) as ex:
            fut = ex.submit(self.prefetch, perm[:bs])
            for b in range(n):
                pre = fut.result()
                if b + 1 < n:
                    fut = ex.submit(self.prefetch, perm[(b + 1) * bs:(b + 2) * bs])
                yield self.batch(pre)

    def batch(self, pre, aug=True):
        it = pre['it'].to(self.device)
        xs, ms = {}, {}
        present = []
        if self.sd:
            xs[self.sd], ms[self.sd] = self.sat.get(it[:, 0], it[:, 2], pre['sat'])
            present.append(torch.ones(len(it), dtype=torch.bool, device=self.device))
        if self.has_uav:
            has = it[:, 1] >= 0
            xu, mu = self.uav.get(it[:, 1].clamp_min(0), it[:, 2], pre['uav'])
            if self.cfg['uav_target'] == 'blurflight':
                xu = self.flight_blur(xu, mu, it[:, 1].clamp_min(0))
            xs['uav_rgb'], ms['uav_rgb'] = xu * has.view(-1, 1, 1, 1), mu & has.view(-1, 1, 1)
            present.append(has)
        elif self.need_uav:  # teacher input only; augmented with the same flips/rotations, never jittered
            xs['uav_teacher'], ms['uav_teacher'] = self.uav.get(it[:, 1], it[:, 2], pre['uav'])
        if aug:
            xs, ms = D.augment(xs, ms, jitter_keys=('uav_rgb',) if self.cfg['jitter'] else ())
        return xs, ms, torch.stack(present, 1)


    def flight_stats(self, pairs, device, n=200):
        """Per-flight (location x UAV TP) channel mean/std of raw UAV pixels, from <= n images per flight."""
        fid = pairs.location.astype(str) + '_' + pairs.uav_tp.astype(str)
        codes, uniq = pd.factorize(fid)
        self.flight = torch.from_numpy(codes).to(device)
        fm, fs = [], []
        rng = np.random.RandomState(0)
        for k in range(len(uniq)):
            rows = np.flatnonzero(codes == k)
            rows = rng.choice(rows, min(n, len(rows)), replace=False)
            x = self.uav.x[rows].float()              # (n, 3 tiles, 3, H, W) raw
            m = self.uav.m[rows].bool().unsqueeze(2).expand_as(x)
            v = [x[:, :, c][m[:, :, c]] for c in range(3)]
            fm.append([t.mean().item() for t in v]); fs.append([t.std().item() for t in v])
        self.fmean = torch.tensor(fm, device=device).view(-1, 3, 1, 1)
        self.fstd = torch.tensor(fs, device=device).view(-1, 3, 1, 1)

    def flight_blur(self, xu, mu, rows):
        """Globally normalised UAV -> per-flight standardised, then 16 px patch means (valid pixels only)."""
        f = self.flight[rows]
        m = mu.unsqueeze(1).float()
        z = (self.uav.denorm(xu) - self.fmean[f]) / self.fstd[f] * m
        num, den = F.avg_pool2d(z, 16), F.avg_pool2d(m, 16)
        zb = F.interpolate(num / den.clamp_min(1e-6), scale_factor=16, mode='nearest')
        return zb * m


class AuxHeads(torch.nn.Module):
    """Trainable heads for the distill / contrast objectives (not saved with the encoder checkpoint)."""

    def __init__(self, objective, dim=768):
        super().__init__()
        mlp = lambda i, o: torch.nn.Sequential(torch.nn.Linear(i, 2048), torch.nn.GELU(), torch.nn.Linear(2048, o))
        if objective == 'distill':
            self.tok, self.pool = mlp(dim, dim), mlp(2 * dim, 2 * dim)
        else:
            self.ps, self.pt = mlp(2 * dim, 256), torch.nn.Linear(2 * dim, 256)


def valid_patches(m):
    B = m.shape[0]
    return (m.float().view(B, 14, 16, 14, 16).mean((2, 4)) > 0.5).view(B, 196)


def encode_full(model, d, x):
    """Single-domain model, all 196 tokens visible -> tokens in patch order (B,196,D) and global token (B,D)."""
    B = x.shape[0]
    enc, _, _, ids_keep, _ = model.encode({d: x}, torch.full((B, 1), 196, dtype=torch.long, device=x.device))
    t = torch.zeros_like(enc[:, :-1]).scatter(1, ids_keep.unsqueeze(-1).expand(-1, -1, enc.shape[-1]), enc[:, :-1])
    return t, enc[:, -1]


def pooled(t, g, v):
    v = v.float().unsqueeze(-1)
    return torch.cat([g, (t * v).sum(1) / v.sum(1).clamp_min(1)], 1)


def aux_loss(cfg, model, teacher, heads, xs, ms, ut, um, pair):
    """xs/ms: satellite tile, ut/um: paired real UAV tile (same augmentation), pair: pair id per sample."""
    sd = sat_domain(cfg)
    vs, vu = valid_patches(ms), valid_patches(um)
    with torch.no_grad():
        tt, tg = encode_full(teacher, 'uav_rgb', ut)
        tp = pooled(tt.float(), tg.float(), vu)
    st, sg = encode_full(model, sd, xs)
    sp = pooled(st.float(), sg.float(), vs)
    if cfg['objective'] == 'distill':
        w = (vs & vu).float()
        e = ((heads.tok(st.float()) - F.layer_norm(tt.float(), tt.shape[-1:])) ** 2).mean(-1)
        lt = (e * w).sum() / w.sum().clamp_min(1)
        lp = ((heads.pool(sp) - F.layer_norm(tp, tp.shape[-1:])) ** 2).mean()
        return lt + lp
    zs, zt = F.normalize(heads.ps(sp), dim=-1), F.normalize(heads.pt(tp), dim=-1)
    logits = zs @ zt.t() / cfg['tau']
    same = (pair.view(-1, 1) == pair.view(1, -1)) & ~torch.eye(len(pair), dtype=torch.bool, device=pair.device)
    logits = logits.masked_fill(same, -1e4)  # other tiles of the same plot/date are not negatives
    y = torch.arange(len(pair), device=pair.device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.t(), y))


def load_teacher(cfg, device):
    name = f'uavmae_s{cfg["seed"]}_{cfg["heldout"]}'
    ck = torch.load(os.path.join(RUNS, name, 'checkpoint-last.pth'), map_location='cpu')
    assert sorted(ck['cfg']['pretrain_locs']) == sorted(cfg['pretrain_locs']), 'teacher must share the LOLO fold'
    t = Mdl.build(ck['cfg'], init=False)
    t.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck['model'].items()})
    for p in t.parameters():
        p.requires_grad_(False)
    return t.to(device).eval()


def run_dir(cfg):
    return os.path.join(RUNS, cfg['name'])


@torch.no_grad()
def heldout_eval(model, cfg, stats, device='cuda', grid_path=None, n_grid=8):
    """Sat-full -> UAV reconstruction MSE (normalised space, valid pixels) on the held-out location.
    Used for reporting/monitoring only, never for model selection."""
    if 'uav_rgb' not in cfg['out_domains'] or cfg['uav_target'] != 'pixel':
        return {}
    df = D.manifest()
    L = cfg['heldout']
    pairs = D.pair_table(df, [L])
    sd = sat_domain(cfg)
    sat = D.GPUTiles('sat', list(pairs.sat), D.SAT_IDX[sd], stats, device, tag=cfg['tag'])
    uav = D.GPUTiles('uav', list(pairs.uav), [0, 1, 2], stats, device, tag=cfg['tag'])
    model.eval()
    tot, cnt, per_tp = 0.0, 0.0, {}
    n = len(pairs)
    for s in range(0, n, 128):
        r = torch.arange(s, min(n, s + 128), device=device)
        for j in range(3):
            tj = torch.full_like(r, j)
            xs, ms = sat.get(r, tj)
            xu, mu = uav.get(r, tj)
            with torch.cuda.amp.autocast():
                preds, *_ = model({sd: xs, 'uav_rgb': xu}, mode='sat_full', out_domains=['uav_rgb'])
            se = ((preds['uav_rgb'].float() - xu) ** 2).mean(1) * mu
            tot += se.sum().item(); cnt += mu.sum().item()
            tps = pairs.uav_tp.values[s:s + 128]
            for t in np.unique(tps):
                k = torch.from_numpy(tps == t).to(device)
                a = per_tp.setdefault(int(t), [0.0, 0.0])
                a[0] += se[k].sum().item(); a[1] += mu[k].sum().item()
    out = {'heldout_uav_mse': tot / cnt}
    out.update({f'heldout_uav_mse_tp{t}': v[0] / v[1] for t, v in per_tp.items()})
    if grid_path:
        save_grid(model, cfg, sat, uav, pairs, grid_path, n_grid)
    model.train()
    del sat, uav
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def save_grid(model, cfg, sat, uav, pairs, path, n=8):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng = np.random.RandomState(123)
    sel = np.sort(rng.choice(len(pairs), n, replace=False))
    r = torch.from_numpy(sel).to(sat.x.device)
    sd = sat_domain(cfg)
    rows = []
    for j in range(3):
        tj = torch.full_like(r, j)
        xs, _ = sat.get(r, tj)
        xu, mu = uav.get(r, tj)
        with torch.cuda.amp.autocast():
            preds, *_ = model({sd: xs, 'uav_rgb': xu}, mode='sat_full', out_domains=['uav_rgb'])
        s = sat.denorm(xs)[:, :3]
        if os.environ.get('SATUAV_DS') == 'inb':  # 8-bit PNGs: show as stored (no stretch)
            s = s.clamp(0, 255) / 255
        else:  # 16-bit reflectance: per-tile stretch for visibility
            s = (s - s.amin((2, 3), keepdim=True)) / (s.amax((2, 3), keepdim=True) - s.amin((2, 3), keepdim=True) + 1e-6)
        rows.append((s, uav.denorm(xu) / 255 * mu.unsqueeze(1), uav.denorm(preds['uav_rgb'].float()).clamp(0, 255) / 255))
    fig, ax = plt.subplots(n, 3, figsize=(9, 1.2 * n))
    for i in range(n):
        for c in range(3):
            img = torch.cat([rows[j][c][i] for j in range(3)], -1).permute(1, 2, 0).cpu().numpy()
            ax[i, c].imshow(np.clip(img, 0, 1))
            ax[i, c].set_xticks([]); ax[i, c].set_yticks([])
        ax[i, 0].set_ylabel(f'{pairs.plot_uid.values[sel[i]].split("_", 1)[1]}\nTP{pairs.uav_tp.values[sel[i]]}', fontsize=6)
    for c, t in enumerate(['Satellite (input)', 'Real UAV', 'Predicted UAV']):
        ax[0, c].set_title(t, fontsize=8)
    fig.suptitle(f'{cfg["name"]}: held-out {cfg["heldout"]}', fontsize=8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def train(cfg, device='cuda'):
    out = run_dir(cfg)
    os.makedirs(out, exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(out, 'config.yaml'), 'w'))
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    t0 = time.time()
    pipe = Pipeline(cfg, device)
    json.dump(pipe.stats, open(os.path.join(out, 'norm_stats.json'), 'w'))
    t_load = time.time() - t0
    model = Mdl.build(cfg).to(device)
    model.train()
    aux = cfg['objective'] in ('distill', 'contrast')
    teacher = load_teacher(cfg, device) if aux else None
    heads = AuxHeads(cfg['objective']).to(device) if aux else None
    nwd = model.no_weight_decay()
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim <= 1 or n in nwd or any(n.startswith(k) for k in nwd)) else decay).append(p)
    if aux:
        for p in heads.parameters():
            (no_decay if p.ndim <= 1 else decay).append(p)
    lr = cfg['blr'] * cfg['bs'] / 256
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': cfg['wd']}, {'params': no_decay, 'weight_decay': 0.}],
                            lr=lr, betas=(0.9, 0.95))
    N = len(pipe.items)
    steps_ep = N // cfg['bs']
    total = steps_ep * cfg['epochs']
    warm = steps_ep * cfg['warmup']
    sd = sat_domain(cfg)
    alphas = cfg['alphas'] if pipe.xmodal else [1.0]
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join(out, 'tb'))
    logf = open(os.path.join(out, 'log.csv'), 'w', newline='')
    lw = csv.writer(logf)
    lw.writerow(['epoch', 'loss', 'loss_sat', 'loss_uav', 'lr', 'sec', 'samples_per_sec'] + (['loss_aux'] if aux else []))
    scaler = torch.cuda.amp.GradScaler()
    step = 0
    print(f'[{cfg["name"]}] items={N} (pairs={pipe.n_pair}, trans={pipe.n_trans}) steps/ep={steps_ep} load={t_load:.0f}s', flush=True)
    for ep in range(cfg['epochs']):
        te = time.time()
        perm = torch.randperm(N)
        agg = np.zeros(3); agg_x = 0.; nb = 0
        for b, (xs, ms, present) in enumerate(pipe.batches(perm, cfg['bs'])):
            cur = lr * step / max(warm, 1) if step < warm else \
                0.5 * lr * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))
            for g in opt.param_groups:
                g['lr'] = cur
            K = cfg['K']
            if cfg['uav_vis'] == 'matched' and not pipe.xmodal:
                K = 0
                while K < 1:
                    K = cfg['K'] - int(round(np.random.dirichlet(cfg['alphas_xmodal'])[0] * cfg['K']))
            mode = 'sat_full' if (pipe.xmodal and np.random.rand() < cfg['p_satfull']) else 'random'
            # sat-full steps encode 197 tokens/sample: split into micro-batches to fit memory
            n_mb = 4 if mode == 'sat_full' else (2 if aux else 1)
            opt.zero_grad(set_to_none=True)
            loss_s_t, loss_u_t, loss_x_t = 0., 0., 0.
            for sl in torch.arange(len(present)).chunk(n_mb):
                x_mb = {k: v[sl] for k, v in xs.items() if k != 'uav_teacher'}
                m_mb = {k: v[sl] for k, v in ms.items() if k != 'uav_teacher'}
                with torch.cuda.amp.autocast():
                    preds, tmask, *_ = model(x_mb, K=K, alphas=alphas, mode=mode, present=present[sl])
                loss_s = masked_mse(preds[sd], x_mb[sd], tmask[sd], m_mb[sd], cfg['masked_loss']) \
                    if (sd in preds and mode != 'sat_full') else torch.zeros((), device=device)
                loss_u = masked_mse(preds['uav_rgb'], x_mb['uav_rgb'], tmask['uav_rgb'], m_mb['uav_rgb'], cfg['masked_loss'],
                                    weight=present[sl, -1]) if 'uav_rgb' in preds else torch.zeros((), device=device)
                loss_x = torch.zeros((), device=device)
                if aux:
                    with torch.cuda.amp.autocast():
                        loss_x = aux_loss(cfg, model, teacher, heads, x_mb[sd], m_mb[sd], xs['uav_teacher'][sl],
                                          ms['uav_teacher'][sl], pipe.items[perm[b * cfg['bs']:(b + 1) * cfg['bs']]][sl, 1].to(device))
                scaler.scale((loss_s + loss_u + cfg['aux_w'] * loss_x) / n_mb).backward()
                loss_s_t += loss_s.item() / n_mb; loss_u_t += loss_u.item() / n_mb; loss_x_t += loss_x.item() / n_mb
            if cfg['clip']:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip'])
            scaler.step(opt)
            scaler.update()
            step += 1
            agg += [loss_s_t + loss_u_t, loss_s_t, loss_u_t]; agg_x += loss_x_t; nb += 1
        torch.cuda.synchronize()
        sec = time.time() - te
        agg /= nb
        lw.writerow([ep, *agg.round(5), f'{cur:.3e}', round(sec, 1), round(steps_ep * cfg['bs'] / sec, 1)] +
                    ([round(agg_x / nb, 5)] if aux else [])); logf.flush()
        for k, v in zip(['loss', 'loss_sat', 'loss_uav'], agg):
            tb.add_scalar(f'train/{k}', v, ep)
        print(f'[{cfg["name"]}] ep {ep} loss {agg[0]:.4f} sat {agg[1]:.4f} uav {agg[2]:.4f} aux {agg_x / nb:.4f} {sec:.0f}s', flush=True)
        if pipe.xmodal and cfg['uav_target'] == 'pixel' and ((ep + 1) % cfg['grid_every'] == 0 or ep == cfg['epochs'] - 1):
            ev = heldout_eval(model, cfg, pipe.stats, device, grid_path=os.path.join(out, 'grids', f'ep{ep + 1:03d}.png'))
            tb.add_scalar('heldout/uav_mse', ev.get('heldout_uav_mse', np.nan), ep)
            with open(os.path.join(out, 'heldout.csv'), 'a') as hf:  # per-evaluation held-out sat->UAV MSE
                hf.write(f"{ep},{agg[2]:.5f},{ev.get('heldout_uav_mse', np.nan):.5f}\n")
    train_sec = time.time() - t0
    sdict = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save({'model': sdict, 'cfg': cfg}, os.path.join(out, 'checkpoint-last.pth'))
    summ = dict(name=cfg['name'], gpu_hours=train_sec / 3600, items=N, epochs=cfg['epochs'],
                final_loss=float(agg[0]), final_loss_sat=float(agg[1]), final_loss_uav=float(agg[2]))
    xmodal = pipe.xmodal
    del pipe
    torch.cuda.empty_cache()
    if xmodal and 'uav_rgb' in cfg['out_domains'] and cfg['uav_target'] == 'pixel':
        summ.update(heldout_eval(model, cfg, json.load(open(os.path.join(out, 'norm_stats.json'))), device))
    json.dump(summ, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    logf.close(); tb.close()
    print(f'[{cfg["name"]}] done {summ}', flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--name')
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.epochs:
        cfg['epochs'] = a.epochs
    if a.name:
        cfg['name'] = a.name
    train(cfg)


if __name__ == '__main__':
    main()
