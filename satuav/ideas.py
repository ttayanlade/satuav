"""Paired summaries for the old-data transfer ideas (2026-11-04).

  python -m satuav.ideas uavctl     # idea 1: E-uav (UAV via the crossmodal encoder) vs UAV-only MAE controls
  python -m satuav.ideas transfer   # ideas 2-4: LUPI soft targets, field-neighbour normalisation, sat TP1-6
Writes results/ideas_<what>.csv (one row per comparison x regime x task) and prints it.
Score: pooled R2 (yield) / accuracy (nitrogen) at the held-out site, linear probe. Delta = candidate - comparator,
paired by (site, seed); 'wins' counts sites (seed-averaged) where the candidate is higher. CI: 95% bootstrap over
sites of the seed-averaged per-site deltas (5 sites, so a rough interval).
"""
import argparse
import os

import numpy as np
import pandas as pd

from .paths import RESULTS

METRIC = {'yield': 'r2', 'nitrogen': 'acc'}


def load(name, **filt):
    d = pd.read_csv(os.path.join(RESULTS, name), low_memory=False)
    d = d[(d.fold == 'pooled') & (d.probe == 'linear')]
    d = d[d.metric == d.task.map(METRIC)]
    for k, v in filt.items():
        d = d[d[k] == v]
    return d


def paired(cand, comp, label, keys=('heldout', 'seed')):
    """cand / comp: frames with heldout, seed, regime, task, value. Returns one row per regime x task."""
    rows = []
    k = list(keys) + ['regime', 'task']
    m = cand.groupby(k).value.mean().rename('c').to_frame().join(comp.groupby(k).value.mean().rename('b'), how='inner')
    for (reg, task), g in m.groupby(level=['regime', 'task']):
        g = g.reset_index()
        site = g.groupby('heldout')[['c', 'b']].mean()
        dd = (site.c - site.b).values
        rng = np.random.RandomState(0)
        bs = [dd[rng.randint(0, len(dd), len(dd))].mean() for _ in range(5000)]
        rows.append(dict(comparison=label, regime=reg, task=task, candidate=site.c.mean(), comparator=site.b.mean(),
                         delta=dd.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                         wins=f'{(dd > 0).sum()}/{len(dd)}', n_seeds=g.seed.nunique(),
                         per_site=' '.join(f'{s[:4]}{v:+.3f}' for s, v in zip(site.index, dd))))
    return rows


def uavctl():
    lat = load('latent_rgb.csv', rep='E-uav', tp='cat')
    base = load('uavmae.csv', rep='E-uavMAE', tp='cat')
    ctl = load('uavctl.csv', tp='cat')
    F, M = ctl[ctl.rep == 'E-uavMAE-F'], ctl[ctl.rep == 'E-uavMAE-M']
    sat = load('rgb.csv', rep='E-satMAE', tp='cat')
    R = []
    R += paired(lat, base, 'E-uav[rgb] vs E-uavMAE (replicates earlier finding)')
    R += paired(lat, F, 'E-uav[rgb] vs E-uavMAE-F (per-flight-normalised UAV)')
    R += paired(lat, M, 'E-uav[rgb] vs E-uavMAE-M (crossmodal-matched masking)')
    R += paired(F, base, 'E-uavMAE-F vs E-uavMAE (effect of flight normalisation)')
    R += paired(M, base, 'E-uavMAE-M vs E-uavMAE (effect of sparse masking)')
    for nm, d in (('E-uav[rgb]', lat), ('E-uavMAE', base), ('E-uavMAE-F', F), ('E-uavMAE-M', M)):
        R += paired(d, sat, f'{nm} vs E-satMAE[rgb] (satellite comparator)')
    return pd.DataFrame(R)


def transfer():
    d = load('transfer.csv')
    d['teacher'] = d.get('teacher', pd.Series(index=d.index, dtype=object)).fillna('none')
    d['lam'] = d.get('lam', 0.0).fillna(0.0)
    d['nb'] = d.get('nb', 0).fillna(0).astype(int)
    R = []
    lupi = d[d.idea == 'lupi']
    for (rep, b), g in lupi.groupby(['rep', 'bands']):
        base = g[g.lam == 0]
        for (t, lam), c in g[g.lam > 0].groupby(['teacher', 'lam']):
            R += paired(c, base, f'LUPI {rep}[{b}] teacher={t} lam={lam} vs lam=0')
    nb = d[d.idea == 'nb']
    for (rep, b, tp), g in nb.groupby(['rep', 'bands', 'tp']):
        base = g[g.nb == 0]
        for k, c in g[g.nb > 0].groupby('nb'):
            R += paired(c, base, f'NB {rep}[{b}] {tp} k={k} vs raw')
    for (rep, b), g in nb[nb.nb.isin([0, 8])].groupby(['rep', 'bands']):
        if not (g.tp == 'cat6').any():
            continue
        for k in (0, 8):
            R += paired(g[(g.tp == 'cat6') & (g.nb == k)], g[(g.tp == 'cat') & (g.nb == 0)],
                        f'TS {rep}[{b}] cat6' + (' + nb8' if k else '') + ' vs cat raw')
    # the best of each idea against the standing comparators (H-sat-all, E-satMAE[rgb]; raw, aligned TPs)
    hsa = nb[(nb.rep == 'H-sat-all') & (nb.nb == 0) & (nb.tp == 'cat')]
    seedavg = lambda x: x.groupby(['heldout', 'regime', 'task']).value.mean().reset_index().assign(seed=0)
    for (rep, b, tp, k), c in nb[nb.nb.isin([0, 8])].groupby(['rep', 'bands', 'tp', 'nb']):
        if 'uav' not in rep.lower() and not (rep == 'H-sat-all' and tp == 'cat' and k == 0):
            R += paired(seedavg(c), seedavg(hsa), f'VS-H-sat-all {rep}[{b}] {tp} nb={k}')
    for (rep, b), g in lupi[lupi.lam > 0].groupby(['rep', 'bands']):
        for (t, lam), c in g.groupby(['teacher', 'lam']):
            R += paired(seedavg(c), seedavg(hsa[hsa.task == 'yield']), f'VS-H-sat-all LUPI {rep}[{b}] {t} lam={lam}')
    return pd.DataFrame(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['uavctl', 'transfer'])
    a = ap.parse_args()
    out = {'uavctl': uavctl, 'transfer': transfer}[a.what]()
    out.round(4).to_csv(os.path.join(RESULTS, f'ideas_{a.what}.csv'), index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80)
    print(out.drop(columns='per_site').round(3).to_string(index=False))


if __name__ == '__main__':
    main()
