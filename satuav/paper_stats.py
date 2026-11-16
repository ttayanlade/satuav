"""All yield numbers reported in the manuscript, with paired t-tests over the five held-out locations.

  python -m satuav.paper_stats            -> results/paper_*.csv (and prints every table)

Requires the probe results written by `python -m satuav.evaluate ...` (see README):
  main.csv, rgb.csv, latent.csv, latent_rgb.csv, uavmae.csv, uavctl.csv, latefuse.csv, pooledsite.csv, pooledsite_tp.csv

Scores
  within  (regime A)  : R2 of the pooled out-of-fold predictions at each held-out location, mean over locations
  cross   (regime B_z): same, probe trained on the four other locations (targets/features z-scored per location)
  allfield (regime P) : one probe over all five locations on raw yield; R2 of the predictions pooled over locations
Learned representations are averaged over seeds at each location before any comparison.

Statistics
  Delta = candidate - comparator. within / cross: mean of the five per-location differences.
  allfield: difference of the pooled R2. Significance: two-sided paired t-test on the five per-location differences
  (df = 4; for allfield, the per-location R2 of the all-field probe). * p < 0.05, ** p < 0.01.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

from . import probe as P
from .ideas import load
from .paths import LOCS, RESULTS

# label -> (results file, representation name in that file, 'bands' filter or None)
SPEC = {
    'UAV-HC': ('main.csv', 'H-uav', None),
    'UAV-MAE': ('uavmae.csv', 'E-uavMAE', None),
    'UAV-MAE-F': ('uavctl.csv', 'E-uavMAE-F', None),
    'UAV-MAE-M': ('uavctl.csv', 'E-uavMAE-M', None),
    'XM-UAV[rgb]': ('latent_rgb.csv', 'E-uav', None),
    'XM-UAV[all]': ('latent.csv', 'E-uav', None),
    'Sat-HC[rgb]': ('main.csv', 'H-sat-rgb', None),
    'Sat-HC[all]': ('main.csv', 'H-sat-all', None),
    'Sat-MAE[rgb]': ('rgb.csv', 'E-satMAE', None),
    'Sat-MAE[all]': ('main.csv', 'E-satMAE', None),
    'XM-Sat[rgb]': ('rgb.csv', 'E-xmodal', None),
    'XM-Sat[all]': ('main.csv', 'E-xmodal', None),
    'XM-Sat+UAV[rgb]': ('rgb.csv', 'E-uav-real', None),          # both modalities through the encoder jointly
    # additional comparisons used in the response to reviewers
    'XM-UAV+XM-Sat[rgb]': ('latefuse.csv', 'E-uav+E-xmodal', 'rgb'),  # separately encoded, concatenated
    'UAV-HC+Sat-HC[rgb]': ('rgb.csv', 'F-late-rgb', None),          # handcrafted feature fusion
    'GenUAV-HC[rgb]': ('rgb.csv', 'H-pred', None),                  # handcrafted features of UAV images generated from satellite
    'Sat-HC+GenUAV-HC[rgb]': ('rgb.csv', 'H-sat-rgb+pred', None),
}
# all-field (regime P) results live in pooledsite*.csv with a 'bands' column
PS_BANDS = {'UAV-HC': 'hand', 'Sat-HC[rgb]': 'hand', 'Sat-HC[all]': 'hand', 'UAV-MAE': 'uav', 'UAV-MAE-F': 'uav',
            'UAV-MAE-M': 'uav', 'XM-UAV[rgb]': 'rgb', 'XM-Sat[rgb]': 'rgb', 'Sat-MAE[rgb]': 'rgb',
            'XM-Sat+UAV[rgb]': 'rgb', 'XM-UAV+XM-Sat[rgb]': 'rgb', 'XM-UAV[all]': 'all', 'XM-Sat[all]': 'all',
            'Sat-MAE[all]': 'all'}
PS_REP = {'XM-UAV[rgb]': 'E-uav', 'XM-UAV[all]': 'E-uav', 'XM-Sat[rgb]': 'E-xmodal', 'XM-Sat[all]': 'E-xmodal',
          'XM-Sat+UAV[rgb]': 'E-uav-real', 'XM-UAV+XM-Sat[rgb]': 'E-uav+E-xmodal'}


def _get(spec, tp, metric=None):
    f, rep, bands = spec
    if metric is None:
        d = load(f, rep=rep, tp=tp)
    else:  # alternative metric (genotype ranking) stored next to R2
        d = pd.read_csv(os.path.join(RESULTS, f), low_memory=False)
        d = d[(d.fold == 'pooled') & (d.probe == 'linear') & (d.rep == rep) & (d.tp == tp) & (d.metric == metric)]
    if bands is not None and 'bands' in d:
        d = d[d.bands == bands]
    if 'seed' not in d:
        d = d.assign(seed=0)
    return d


def spec_for(lab, regime, tp='cat'):
    if regime != 'P':
        return SPEC[lab]
    rep = PS_REP.get(lab, SPEC[lab][1])
    return ('pooledsite.csv' if tp == 'cat' else 'pooledsite_tp.csv', rep, PS_BANDS[lab])


def per_site(lab, regime, tp='cat', metric=None):
    """seed-averaged score at each held-out location (Series indexed by location)."""
    d = _get(spec_for(lab, regime, tp), tp, metric)
    d = d[(d.task == 'yield') & (d.regime == regime)]
    return d.groupby(['seed', 'heldout']).value.mean().unstack().mean()


def _site_stats():
    s = {}
    for L in LOCS:
        y, _, _ = P.target('yield', P.plot_uids(L))
        y = y[~np.isnan(y)]
        s[L] = dict(n=len(y), mean=y.mean(), sst=((y - y.mean()) ** 2).sum())
    return pd.DataFrame(s).T


_ST = None


def pooled_r2(r2_site):
    """R2 of the predictions pooled over locations, from per-location R2 (each relative to its location mean)."""
    global _ST
    if _ST is None:
        _ST = _site_stats()
    st = _ST.loc[list(r2_site.index)]
    grand = (st.n * st['mean']).sum() / st.n.sum()
    tot = st.sst.sum() + (st.n * (st['mean'] - grand) ** 2).sum()
    sse = sum((1 - r2_site[L]) * _ST.loc[L, 'sst'] for L in r2_site.index)
    return 1 - sse / tot


def score(lab, regime, tp='cat'):
    s = per_site(lab, regime, tp)
    return pooled_r2(s) if regime == 'P' else s.mean()


def stars(p):
    return '**' if p < 0.01 else '*' if p < 0.05 else ''


def compare(cand, comp, regime, tp='cat'):
    C, B = per_site(cand, regime, tp), per_site(comp, regime, tp)
    dd = (C - B).dropna()
    p = stats.ttest_1samp(dd.values, 0).pvalue
    delta = pooled_r2(C) - pooled_r2(B) if regime == 'P' else dd.mean()
    return dict(candidate=cand, comparator=comp, regime=regime, tp=tp,
                cand_score=pooled_r2(C) if regime == 'P' else C.mean(),
                comp_score=pooled_r2(B) if regime == 'P' else B.mean(),
                delta=delta, p=p, sig=stars(p), wins=int((dd > 0).sum()), n=len(dd))


SETTINGS = (('allfield', 'P'), ('within', 'A'), ('cross', 'B_z'))


def main_table():
    """Table 3: absolute R2 and Delta vs the comparator in the same block."""
    rows = [('UAV-HC', None), ('UAV-MAE', 'UAV-HC'), ('XM-UAV[rgb]', 'UAV-MAE'),
            ('Sat-HC[rgb]', None), ('Sat-MAE[rgb]', 'Sat-HC[rgb]'), ('XM-Sat[rgb]', 'Sat-MAE[rgb]')]
    out = []
    for lab, comp in rows:
        r = dict(rep=lab, comparator=comp or '')
        for name, reg in SETTINGS:
            r[name] = score(lab, reg)
            if comp:
                c = compare(lab, comp, reg)
                r[f'd_{name}'], r[f'p_{name}'], r[f'sig_{name}'] = c['delta'], c['p'], c['sig']
        out.append(r)
    return pd.DataFrame(out)


def per_location_table():
    """Table 4: within-site and cross-site R2 at each held-out location."""
    labs = ['UAV-HC', 'UAV-MAE', 'XM-UAV[rgb]', 'Sat-HC[rgb]', 'Sat-MAE[rgb]', 'XM-Sat[rgb]']
    out = []
    for name, reg in (('within', 'A'), ('cross', 'B_z')):
        for lab in labs:
            s = per_site(lab, reg)
            out.append(dict(setting=name, rep=lab, **s.to_dict(), mean=s.mean()))
    return pd.DataFrame(out)


def comparisons():
    """Controls (Table 5), infrared bands (Table 6), joint input, and the fusion / generated-UAV comparisons."""
    pairs = [
        # crossmodal gain and its controls
        ('XM-UAV[rgb]', 'UAV-MAE'), ('XM-UAV[rgb]', 'UAV-MAE-F'), ('XM-UAV[rgb]', 'UAV-MAE-M'),
        ('UAV-MAE-F', 'UAV-MAE'), ('UAV-MAE-M', 'UAV-MAE'),
        # satellite imagery as an input at inference vs as a pretraining anchor
        ('XM-Sat+UAV[rgb]', 'XM-UAV[rgb]'), ('XM-UAV+XM-Sat[rgb]', 'XM-UAV[rgb]'),
        ('UAV-HC+Sat-HC[rgb]', 'UAV-HC'), ('XM-UAV[rgb]', 'UAV-HC+Sat-HC[rgb]'),
        # learned UAV representations vs satellite-only approaches
        ('XM-UAV[rgb]', 'Sat-MAE[rgb]'), ('XM-UAV[rgb]', 'Sat-HC[rgb]'),
        # UAV images generated from satellite input (previous version of the manuscript)
        ('GenUAV-HC[rgb]', 'Sat-HC[rgb]'), ('Sat-HC+GenUAV-HC[rgb]', 'Sat-HC[rgb]'),
        # infrared satellite bands
        ('Sat-HC[all]', 'Sat-HC[rgb]'), ('Sat-MAE[all]', 'Sat-MAE[rgb]'), ('XM-Sat[all]', 'XM-Sat[rgb]'),
        ('XM-UAV[all]', 'UAV-MAE'), ('XM-UAV[all]', 'XM-UAV[rgb]'),
    ]
    out = []
    for c, b in pairs:
        for name, reg in SETTINGS:
            if reg == 'P' and not (c in PS_BANDS and b in PS_BANDS):
                continue  # the all-field probe was run only for the representations in PS_BANDS
            out.append(dict(setting=name, **compare(c, b, reg)))
    return pd.DataFrame(out)


def timepoint_table():
    """Supplementary per-timepoint table."""
    out = []
    for lab in ('XM-UAV[rgb]', 'UAV-HC', 'XM-Sat[rgb]', 'Sat-HC[rgb]'):
        for name, reg in SETTINGS:
            out.append(dict(setting=name, rep=lab, **{tp: score(lab, reg, tp) for tp in ('t1', 't2', 't3', 'cat')}))
    return pd.DataFrame(out)


def ranking_table():
    """Supplementary genotype ranking table (top-25% / top-50% overlap, Spearman of genotype means)."""
    out = []
    for lab in ('XM-UAV[rgb]', 'UAV-HC', 'XM-Sat[rgb]', 'Sat-HC[rgb]'):
        for name, reg in (('allfield', 'P'), ('cross', 'B_z')):
            r = dict(setting=name, rep=lab)
            for m in ('top25_geno', 'top50_geno', 'spearman_geno'):
                spec = ('pooledsite.csv', PS_REP.get(lab, SPEC[lab][1]), PS_BANDS[lab])
                d = _get(spec, 'cat', m)
                d = d[(d.task == 'yield') & (d.regime == reg)]
                r[m] = d.groupby(['seed', 'heldout']).value.mean().unstack().mean().mean()
            out.append(r)
    return pd.DataFrame(out)


def main():
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    for name, fn in (('paper_main', main_table), ('paper_per_location', per_location_table),
                     ('paper_comparisons', comparisons), ('paper_timepoints', timepoint_table),
                     ('paper_ranking', ranking_table)):
        d = fn()
        d.round(4).to_csv(os.path.join(RESULTS, f'{name}.csv'), index=False)
        print(f'\n== {name}'), print(d.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
