"""Paper tables, figures and statistics from results/*.csv.

  python -m satuav.report all
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as tdist, wilcoxon

from .paths import FIGS, LOCS, RESULTS, RUNS, TABLES
from .probe import ORACLE
from .summary import GRID, INK, INK2, PAL, savefig

LOC_LABEL = {'Ames': 'Ames', 'Crawfordsville': 'Crawfordsville', 'Lincoln': 'Lincoln', 'MOValley': 'Missouri V.',
             'Scottsbluff': 'Scottsbluff'}
MAIN_ROWS = ['H-sat-rgb', 'H-sat-all', 'H-pred', 'H-sat-all+pred', 'E-imnet', 'E-satMAE', 'E-xmodal', 'E-xmodal-dec',
             'E-satMAE+H-sat-all', 'E-xmodal+H-sat-all', 'H-uav', 'E-uav-real', 'F-late']
BLOCKS = [  # (task, regime, metric, fold-level?, label)
    ('yield', 'A', 'r2', True, 'Yield, regime A (within held-out site), $R^2$'),
    ('yield', 'B_raw', 'r2', True, 'Yield, regime B (cross-site, raw bu/acre), $R^2$'),
    ('yield', 'B_z', 'r2', True, 'Yield, regime B (cross-site, site-standardised), $R^2$'),
    ('yield', 'B_z', 'spearman_geno', False, 'Yield, regime B, genotype-rank Spearman $\\rho$ at held-out site'),
    ('nitrogen', 'A', 'acc', True, 'Nitrogen level, regime A, accuracy'),
    ('nitrogen', 'B_z', 'acc', True, 'Nitrogen level, regime B (site-standardised), accuracy'),
]
BASE = 'H-sat-all'
BASE_DESC = {'H-sat-all': ' (handcrafted satellite RGB+NIR+RE)',
             'E-satMAE': ' (satellite-only MAE, same architecture, data and schedule)'}


def load(name):
    p = os.path.join(RESULTS, name)
    return pd.read_csv(p, dtype={'fold': str}) if os.path.exists(p) else None


def cell(df, rep, L, task, regime, metric, foldlevel, tp='cat', probe='linear'):
    q = df[(df.rep == rep) & (df.heldout == L) & (df.task == task) & (df.regime == regime) & (df.metric == metric)
           & (df.tp == tp) & (df.probe == probe)]
    q = q[q.fold != 'pooled'] if foldlevel else q[q.fold == 'pooled']
    if q.empty:
        return np.nan, np.nan
    per_seed = q.groupby('seed').value.mean()
    if foldlevel:
        f = q.groupby('fold').value.mean()  # seed-averaged fold values
        return f.mean(), f.std()
    return per_seed.mean(), per_seed.std() if len(per_seed) > 1 else np.nan


def main_table(df, rows=MAIN_ROWS, tp='cat', probe='linear', fname='main_lolo.tex', caption_extra=''):
    lines = ['\\begin{table}[ht]', '\\centering', '\\scriptsize',
             f'\\caption{{Leave-one-location-out results ({"TP1--3 concatenated" if tp == "cat" else tp}, {probe} probe). '
             'Each column is a held-out location whose UAV imagery and labels were never seen in pretraining; '
             'values are means over the 5 genotype folds (and 3 pretraining seeds for learned encoders). '
             'Best non-oracle result per column in bold; grey rows need real UAV at test time (oracles). '
             'H = handcrafted features, E = frozen encoder embeddings.' + caption_extra + '}',
             f'\\label{{tab:{fname[:-4]}}}', '\\begin{tabular}{l' + 'c' * (len(LOCS) + 1) + '}', '\\toprule',
             'Representation & ' + ' & '.join(LOC_LABEL[l] for l in LOCS) + ' & Mean \\\\']
    tidy = []
    for task, regime, metric, fl, label in BLOCKS:
        locs = [l for l in LOCS if not (task == 'nitrogen' and l == 'MOValley')]
        vals = {}
        for r in rows:
            v = [cell(df, r, L, task, regime, metric, fl, tp, probe)[0] for L in LOCS]
            v = [x if L in locs else np.nan for x, L in zip(v, LOCS)]
            if all(np.isnan(v)):
                continue
            vals[r] = v + [np.nanmean([x for x, L in zip(v, LOCS) if L in locs])]
            tidy += [dict(block=label, rep=r, heldout=c, value=x) for c, x in zip(LOCS + ['Mean'], vals[r])]
        if not vals:
            continue
        lines += ['\\midrule', f'\\multicolumn{{{len(LOCS) + 2}}}{{l}}{{\\textit{{{label}}}}} \\\\']
        M = pd.DataFrame(vals).T
        best = M.loc[[r for r in M.index if r not in ORACLE]].max()
        for r, v in M.iterrows():
            cells = []
            for c, x in zip(M.columns, v):
                s = '--' if np.isnan(x) else f'{x:.2f}'
                if r not in ORACLE and not np.isnan(x) and np.isclose(x, best[c]):
                    s = f'\\textbf{{{s}}}'
                cells.append(s)
            name = r.replace('+', '+')
            row = f'{name} & ' + ' & '.join(cells) + ' \\\\'
            if r in ORACLE:
                row = '\\rowcolor{gray!12} \\textcolor{gray}{' + name + '} & ' + ' & '.join(
                    f'\\textcolor{{gray}}{{{c}}}' for c in cells) + ' \\\\'
            lines.append(row)
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    os.makedirs(TABLES, exist_ok=True)
    open(os.path.join(TABLES, fname), 'w').write('\n'.join(lines) + '\n')
    t = pd.DataFrame(tidy)
    t.to_csv(os.path.join(RESULTS, fname.replace('.tex', '_summary.csv')), index=False)
    return t


# ------------------------------------------------------------------------------------------- stats

def fold_matrix(df, rep, task, regime, metric, tp='cat', probe='linear'):
    q = df[(df.rep == rep) & (df.task == task) & (df.regime == regime) & (df.metric == metric) & (df.tp == tp)
           & (df.probe == probe) & (df.fold != 'pooled')]
    return q.groupby(['heldout', 'fold']).value.mean()


def hier_boot(diff, n=10000, seed=0):
    """diff: Series indexed by (heldout, fold). Resample locations, then folds within each location."""
    rng = np.random.RandomState(seed)
    groups = [g.values for _, g in diff.groupby(level=0)]
    out = np.empty(n)
    for i in range(n):
        locs = rng.randint(0, len(groups), len(groups))
        out[i] = np.mean([groups[j][rng.randint(0, len(groups[j]), len(groups[j]))].mean() for j in locs])
    return np.percentile(out, [2.5, 97.5])


def paired_stats(df, reps, base=BASE, tp='cat', probe='linear', fname='paired_stats'):
    rows = []
    for task, regime, metric, fl, label in BLOCKS:
        if not fl:
            continue
        b = fold_matrix(df, base, task, regime, metric, tp, probe)
        for r in reps:
            if r == base:
                continue
            a = fold_matrix(df, r, task, regime, metric, tp, probe)
            d = (a - b).dropna()
            if len(d) < 5:
                continue
            lo, hi = hier_boot(d)
            try:
                p = wilcoxon(d.values).pvalue
            except ValueError:
                p = np.nan
            per_loc = d.groupby(level=0).mean()
            rows.append(dict(task=task, regime=regime, metric=metric, rep=r, baseline=base, mean_diff=d.mean(),
                             ci_lo=lo, ci_hi=hi, wilcoxon_p=p, n_pairs=len(d), locs_better=int((per_loc > 0).sum()),
                             n_locs=len(per_loc), cohen_dz=d.mean() / (d.std() + 1e-12)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, f'{fname}.csv'), index=False)
    lines = ['\\begin{table}[ht]', '\\centering', '\\scriptsize',
             f'\\caption{{Paired differences versus {base}{BASE_DESC.get(base, "")} across 5 held-out '
             'locations $\\times$ 5 genotype folds. CI: 95\\% hierarchical bootstrap (locations, then folds); '
             '$p$: Wilcoxon signed-rank over location$\\times$fold pairs; $d_z$: mean/sd of paired differences; '
             '\\#loc: held-out locations where the method is better.}', f'\\label{{tab:{fname}}}',
             '\\begin{tabular}{llrrrrc}', '\\toprule',
             'Task / regime & Representation & $\\Delta$ & 95\\% CI & $p$ & $d_z$ & \\#loc \\\\', '\\midrule']
    for _, r in out.iterrows():
        lines.append(f'{r.task} {r.regime.replace("_", "-")} ({r.metric.replace("_", " ")}) & {r.rep} & {r.mean_diff:+.3f} & '
                     f'[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & {r.wilcoxon_p:.3f} & {r.cohen_dz:+.2f} & {r.locs_better}/{r.n_locs} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    open(os.path.join(TABLES, f'{fname}.tex'), 'w').write('\n'.join(lines) + '\n')
    return out


# ----------------------------------------------------------------------------------------- figures

def per_location_plot(df, reps, blocks, fname='per_location', tp='cat'):
    fig, axes = plt.subplots(1, len(blocks), figsize=(3.6 * len(blocks), 3.0), sharey=False)
    axes = np.atleast_1d(axes)
    w = 0.8 / len(reps)
    for ax, (task, regime, metric, label) in zip(axes, blocks):
        locs = [l for l in LOCS if not (task == 'nitrogen' and l == 'MOValley')]
        for i, r in enumerate(reps):
            m = fold_matrix(df, r, task, regime, metric, tp)
            for j, L in enumerate(locs):
                if L not in m.index.get_level_values(0):
                    continue
                v = m.loc[L].values
                ci = tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
                x = j + (i - (len(reps) - 1) / 2) * w
                oracle = r in ORACLE
                style = dict(color=PAL[i], marker='D' if oracle else 'o', ms=5, mec=PAL[i] if oracle else 'white',
                             mew=1.2 if oracle else 1, mfc='white' if oracle else PAL[i], lw=0, zorder=3)
                ax.errorbar(x, v.mean(), yerr=ci, elinewidth=1.2, capsize=0, ecolor=PAL[i], **style,
                            label=r + (' (oracle)' if r in ORACLE else '') if j == 0 else None)
        ax.set_xticks(range(len(locs)))
        ax.set_xticklabels([LOC_LABEL[l] for l in locs], rotation=25, ha='right')
        ax.set_title(label, fontsize=8, loc='left', color=INK)
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.axhline(0, color=INK2, lw=0.6)
    axes[0].legend(frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(len(blocks) / 2 * 1.1, -0.28),
                   ncol=len(reps))
    savefig(fig, fname)


def ablation_table(df_abl, ref_df, fname='ablations'):
    """df_abl rows carry 'family'; ref_df = main results (xmodal_s0) restricted to the same locations."""
    fam_label = {
        'main': 'Main (sat\\_all, $\\alpha$=0.9/0.1, $p_{\\text{sat-full}}$=0.25, masked loss)',
        'p0': '$p_{\\text{sat-full}}=0$ ($\\alpha$=0.9/0.1, paper setting)',
        'a1-1': '$\\alpha$=1.0/1.0 (symmetric), $p$=0', 'a1.2-0.8': '$\\alpha$=1.2/0.8, $p$=0',
        'a1.5-0.5': '$\\alpha$=1.5/0.5, $p$=0', 'a0.7-0.3': '$\\alpha$=0.7/0.3, $p$=0',
        'satrgb': 'Satellite input RGB only', 'randinit': 'Random init (no MultiMAE weights)',
        'mismatch': 'Mismatched Sat--UAV dates', 'nomaskloss': 'Loss on padded pixels too',
        'origcrop': 'Original cropping (no rotation alignment)',
    }
    rows = []
    locs = sorted(df_abl.heldout.unique())
    for fam, g in [('main', ref_df[ref_df.heldout.isin(locs)])] + list(df_abl.groupby('family')):
        if fam not in fam_label:
            continue
        r = dict(family=fam, label=fam_label[fam])
        mse = []
        for L in locs:
            name = f'xmodal_s0_{L}' if fam == 'main' else f'abl-{fam}_{L}'
            p = os.path.join(RUNS, name, 'summary.json')
            if os.path.exists(p):
                mse.append(json.load(open(p)).get('heldout_uav_mse', np.nan))
        r['recon_mse'] = np.nanmean(mse) if mse else np.nan
        for rep, key in [('E-xmodal', 'E'), ('H-pred', 'Hp'), ('E-xmodal+H-sat-all', 'EH')]:
            for task, regime, metric in [('yield', 'A', 'r2'), ('yield', 'B_z', 'r2')]:
                m = fold_matrix(g[g.seed == 0] if 'seed' in g else g, rep, task, regime, metric)
                r[f'{key}_{regime}'] = m.groupby(level=0).mean().mean() if len(m) else np.nan
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RESULTS, f'{fname}_summary.csv'), index=False)  # not {fname}.csv: that is the raw probe output
    lines = ['\\begin{table}[ht]', '\\centering', '\\scriptsize',
             f'\\caption{{Ablations (held-out {", ".join(locs)}; seed 0; mean over held-out locations). '
             'Recon MSE: sat-full$\\rightarrow$UAV reconstruction error on the held-out site (normalised pixels, '
             'valid pixels). Downstream: yield $R^2$ with linear probe on TP1--3 concatenated.}',
             f'\\label{{tab:{fname}}}', '\\begin{tabular}{lccccccc}', '\\toprule',
             ' & Recon & \\multicolumn{2}{c}{E-xmodal} & \\multicolumn{2}{c}{H-pred} & \\multicolumn{2}{c}{E-xmodal+H-sat-all} \\\\',
             'Setting & MSE$\\downarrow$ & A & B-z & A & B-z & A & B-z \\\\', '\\midrule']
    for _, r in t.iterrows():
        lines.append(f'{r.label} & {r.recon_mse:.3f} & ' + ' & '.join(
            '--' if np.isnan(r[c]) else f'{r[c]:.3f}' for c in ['E_A', 'E_B_z', 'Hp_A', 'Hp_B_z', 'EH_A', 'EH_B_z']) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    open(os.path.join(TABLES, f'{fname}.tex'), 'w').write('\n'.join(lines) + '\n')
    return t


def scaling_plot(df_scale, ref_main, fname='scaling_nlocs'):
    """Yield regime A R^2 at held-out Crawfordsville vs number of pretraining locations."""
    rows = []
    for n in (1, 2, 3, 4):
        for kind, rep in [('xmodal', 'E-xmodal'), ('satmae', 'E-satMAE'), ('xmodal', 'H-pred')]:
            if n == 4:
                g = ref_main[(ref_main.seed == 0) & (ref_main.heldout == 'Crawfordsville')]
                vals = [fold_matrix(g, rep, 'yield', 'A', 'r2').mean()]
            else:
                g = df_scale[(df_scale.nloc == n) & (df_scale.kind == kind)]
                vals = [fold_matrix(gg, rep, 'yield', 'A', 'r2').mean() for _, gg in g.groupby('subset')]
            rows += [dict(n=n, rep=rep, value=v) for v in vals if not np.isnan(v)]
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RESULTS, f'{fname}.csv'), index=False)
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for i, rep in enumerate(['E-xmodal', 'E-satMAE', 'H-pred']):
        q = t[t.rep == rep]
        if q.empty:
            continue
        m = q.groupby('n').value.mean()
        ax.plot(m.index, m.values, color=PAL[i], lw=2, marker='o', ms=6, mec='white', mew=1.5, label=rep)
        ax.scatter(q.n, q.value, color=PAL[i], s=10, alpha=0.5, zorder=1)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel('Pretraining locations')
    ax.set_ylabel('Yield $R^2$ (held-out Crawfordsville, regime A)')
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=7)
    savefig(fig, fname)
    return t


def tp46_plot(df, fname='tp46_missing_flight'):
    """Left: adding satellite-only dates (TP4-6). Right: missing UAV flight, dropped vs pseudo-UAV substitute."""
    def val(rep, tp, regime='A', metric='r2'):
        m = fold_matrix(df, rep, 'yield', regime, metric, tp)
        return m.groupby(level=0).mean() if len(m) else pd.Series(dtype=float)

    left = [('H-sat-all', 'cat', 'Sat (3 dates)'), ('H-sat-all', 'cat6', 'Sat (6 dates)'),
            ('E-xmodal', 'cat', 'E-xmodal (3)'), ('E-xmodal', 'cat6', 'E-xmodal (6)'),
            ('H-uav', 'cat', 'Real UAV (3 flights)'), ('H-uav+pred@s', 'cat+s46', 'UAV + pseudo-UAV TP4-6'),
            ('H-uav+H-sat-all@s', 'cat+s46', 'UAV + Sat TP4-6'),
            ('E-uav-real+E-xmodal@s', 'cat+s46', 'E-uav-real + E-xmodal TP4-6')]
    right = []
    for k in (1, 2, 3):
        right += [('H-uav', 'cat', 'all', k), ('H-uav', f'drop{k}', 'drop', k), ('H-uav', f'sub{k}', 'pseudo', k),
                  ('H-uav', f'hsub{k}', 'satfill', k)]
    rows = []
    for rep, tp, lab in left:
        s = val(rep, tp)
        rows += [dict(panel='tp46', label=lab, heldout=L, value=v) for L, v in s.items()]
    for rep, tp, lab, k in right:
        s = val(rep, tp)
        rows += [dict(panel='missing', label=lab, k=k, heldout=L, value=v) for L, v in s.items()]
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RESULTS, f'{fname}.csv'), index=False)
    if t.empty:
        return t
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    labs = [l for _, _, l in left if l in set(t.label)]
    for i, lab in enumerate(labs):
        q = t[(t.panel == 'tp46') & (t.label == lab)]
        ax.scatter(np.full(len(q), i) + np.linspace(-0.15, 0.15, len(q)), q.value, s=12, color=INK2, alpha=0.5, zorder=2)
        ax.bar(i, q.value.mean(), width=0.62, color=PAL[0] if 'UAV' not in lab else PAL[1], alpha=0.85, zorder=1)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=35, ha='right', fontsize=7)
    ax.set_ylabel('Yield $R^2$ (regime A, mean of 5 sites)')
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax = axes[1]
    names = {'all': 'All 3 real flights', 'drop': 'Flight k missing', 'pseudo': 'Flight k → pseudo-UAV (H-pred)',
             'satfill': 'Flight k → satellite features (H-sat-all)'}
    for j, lab in enumerate(names):
        ys = [t[(t.panel == 'missing') & (t.k == k) & (t.label == lab)].value.mean() for k in (1, 2, 3)]
        ax.plot([1, 2, 3], ys, color=PAL[j], lw=2, marker='o', ms=6, mec='white', mew=1.5, label=names[lab])
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel('Missing UAV flight k')
    ax.set_ylabel('Yield $R^2$ with H-uav (regime A)')
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=7)
    savefig(fig, fname)
    return t


def recon_figure(fname='heldout_reconstructions', n=3):
    """Sat input / real UAV / predicted UAV for each held-out location (model never saw that site)."""
    import torch
    from . import data as D
    from .extract import load_run
    from .pretrain import sat_domain
    fig, axes = plt.subplots(len(LOCS) * n, 3, figsize=(8.4, 1.05 * n * len(LOCS)))
    rng = np.random.RandomState(7)
    df = D.manifest()
    for li, L in enumerate(LOCS):
        model, cfg, stats = load_run(f'xmodal_s0_{L}')
        pairs = D.pair_table(df, [L])
        sel = rng.choice(len(pairs), n, replace=False)
        sd = sat_domain(cfg)
        sat = D.GPUTiles('sat', list(pairs.sat.values[sel]), D.SAT_IDX[sd], stats)
        uav = D.GPUTiles('uav', list(pairs.uav.values[sel]), [0, 1, 2], stats)
        r = torch.arange(n, device='cuda')
        rows = []
        with torch.no_grad():
            for j in range(3):
                tj = torch.full_like(r, j)
                xs, ms = sat.get(r, tj)
                xu, mu = uav.get(r, tj)
                with torch.cuda.amp.autocast():
                    preds, *_ = model({sd: xs, 'uav_rgb': xu}, mode='sat_full', out_domains=['uav_rgb'])
                s = sat.denorm(xs)[:, :3]
                lo = torch.stack([s[i][:, ms[i]].quantile(0.01, 1) for i in range(n)]).view(n, 3, 1, 1)
                hi = torch.stack([s[i][:, ms[i]].quantile(0.99, 1) for i in range(n)]).view(n, 3, 1, 1)
                rows.append(((s - lo) / (hi - lo + 1e-6)).clamp(0, 1) * ms.unsqueeze(1))
                rows.append(uav.denorm(xu) / 255 * mu.unsqueeze(1))
                rows.append(uav.denorm(preds['uav_rgb'].float()).clamp(0, 255) / 255 * ms.unsqueeze(1))
        for i in range(n):
            for c in range(3):
                img = torch.cat([rows[3 * j + c][i] for j in range(3)], -1).permute(1, 2, 0).cpu().numpy()
                a = axes[li * n + i, c]
                a.imshow(np.clip(img, 0, 1))
                a.set_xticks([]); a.set_yticks([])
            axes[li * n + i, 0].set_ylabel(f'{LOC_LABEL[L]}\nTP{pairs.uav_tp.values[sel[i]]}', fontsize=6)
        del model
        torch.cuda.empty_cache()
    for c, tt in enumerate(['Satellite RGB (input)', 'Real UAV (not seen)', 'Predicted UAV']):
        axes[0, c].set_title(tt, fontsize=8)
    savefig(fig, fname)


COMPARISONS = [  # (method, comparator, what the difference isolates)
    ('E-xmodal', 'E-satMAE', 'crossmodal objective (same encoder, data, schedule)'),
    ('E-xmodal', 'H-sat-all', 'learned vs handcrafted satellite features'),
    ('E-xmodal+H-sat-all', 'E-satMAE+H-sat-all', 'crossmodal objective, on top of handcrafted'),
    ('H-pred', 'H-sat-all', 'handcrafted on generated UAV vs on satellite'),
    ('E-xmodal', 'F-late', 'crossmodal (sat only) vs late fusion with real UAV'),
]


RGB_COMPARISONS = [  # '[rgb]' = run-dependent rep from the RGB-only satellite runs (xmodalrgb / satmaergb)
    ('E-xmodal[rgb]', 'E-satMAE[rgb]', 'crossmodal objective, satellite RGB only'),
    ('E-xmodal[rgb]', 'H-sat-rgb', 'learned vs handcrafted, satellite RGB only'),
    ('H-pred[rgb]', 'H-sat-rgb', 'handcrafted on generated UAV vs on satellite RGB'),
    ('H-sat-rgb+pred[rgb]', 'H-sat-rgb', 'add generated-UAV features to real satellite RGB'),
    ('E-xmodal+H-sat-rgb[rgb]', 'E-satMAE+H-sat-rgb[rgb]', 'crossmodal objective on top of handcrafted, RGB only'),
    ('H-uav', 'H-sat-rgb', 'real UAV RGB vs satellite RGB (oracle gap)'),
    ('F-late-rgb', 'H-sat-rgb', 'add real UAV RGB to satellite RGB (oracle)'),
    ('H-sat-all', 'H-sat-rgb', 'band effect: NIR + red-edge, handcrafted'),
    ('E-satMAE', 'E-satMAE[rgb]', 'band effect: NIR + red-edge, satellite-only MAE'),
    ('E-xmodal', 'E-xmodal[rgb]', 'band effect: NIR + red-edge, crossmodal'),
]


def rgb_frame(main_df, rgb_df):
    """main rows (same seeds as the RGB runs) + RGB-run rows renamed '<rep>[rgb]' (hand-only reps keep their name)."""
    seeds = sorted(rgb_df[rgb_df.bands == 'rgb'].seed.unique())
    r = rgb_df.copy()
    r.loc[r.bands == 'rgb', 'rep'] = r.loc[r.bands == 'rgb', 'rep'] + '[rgb]'
    return pd.concat([main_df[main_df.seed.isin(seeds)], r], ignore_index=True)


def pooled_mean(df, rep, task, regime, metric, tp, probe):
    """Absolute score: per-location pooled metric (averaged over seeds), then mean over held-out locations."""
    q = df[(df.rep == rep) & (df.task == task) & (df.regime == regime) & (df.metric == metric) & (df.tp == tp)
           & (df.probe == probe) & (df.fold == 'pooled')]
    return q.groupby('heldout').value.mean().mean()


def relative_table(df, fname='relative', comparisons=None, out_dir=None):
    """Paired method-vs-comparator differences per task x regime x timepoint (linear) and per probe (cat)."""
    settings = [(tp, 'linear') for tp in ('t1', 't2', 't3', 'cat')] + [('cat', p) for p in ('knn', 'mlp')]
    rows = []
    for task, metric in (('yield', 'r2'), ('nitrogen', 'acc')):
        for regime in ('A', 'B_z'):
            for tp, probe in settings:
                for a, b, what in comparisons or COMPARISONS:
                    fa = fold_matrix(df, a, task, regime, metric, tp, probe)
                    fb = fold_matrix(df, b, task, regime, metric, tp, probe)
                    d = (fa - fb).dropna()
                    if len(d) < 5:
                        continue
                    lo, hi = hier_boot(d, n=2000)
                    try:
                        p = wilcoxon(d.values).pvalue
                    except ValueError:
                        p = np.nan
                    per_loc = d.groupby(level=0).mean()
                    rows.append(dict(task=task, regime=regime, tp=tp, probe=probe, method=a, comparator=b, isolates=what,
                                     method_abs=pooled_mean(df, a, task, regime, metric, tp, probe),
                                     comparator_abs=pooled_mean(df, b, task, regime, metric, tp, probe),
                                     mean_diff=d.mean(), ci_lo=lo, ci_hi=hi, wilcoxon_p=p,
                                     locs_better=int((per_loc > 0).sum()), n_locs=len(per_loc)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir or RESULTS, f'{fname}.csv'), index=False)
    lab = {'yield': 'Yield $R^2$', 'nitrogen': 'N accuracy'}
    lines = ['\\begin{table}[ht]', '\\centering', '\\scriptsize',
             '\\caption{Method versus comparator, linear probe. Absolute scores are the mean over the 5 held-out '
             'locations of the pooled per-location metric. $\\Delta$, CI (95\\% hierarchical bootstrap) and $p$ '
             '(Wilcoxon) are computed over location$\\times$genotype-fold pairs. \\#loc: held-out locations where the '
             'method is better. A: within the held-out site; B: trained on the other sites, site-standardised.}',
             f'\\label{{tab:{fname}}}', '\\begin{tabular}{lllcrrrrrc}', '\\toprule',
             'Task & Regime & Method vs comparator & Dates & Method & Comp. & $\\Delta$ & 95\\% CI & $p$ & \\#loc \\\\',
             '\\midrule']
    q = out[out.probe == 'linear']
    for (task, regime), g in q.groupby(['task', 'regime'], sort=False):
        for _, r in g.iterrows():
            tp = {'cat': 'TP1--3'}.get(r.tp, r.tp.upper().replace('T', 'TP'))
            lines.append(f'{lab[task]} & {regime.replace("_z", "")} & {r.method} vs {r.comparator} & {tp} & '
                         f'{r.method_abs:.3f} & {r.comparator_abs:.3f} & {r.mean_diff:+.3f} & '
                         f'[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & {r.wilcoxon_p:.3f} & {r.locs_better}/{r.n_locs} \\\\')
        lines.append('\\midrule')
    lines[-1] = '\\bottomrule'
    lines += ['\\end{tabular}', '\\end{table}']
    open(os.path.join(out_dir or TABLES, f'{fname}.tex'), 'w').write('\n'.join(lines) + '\n')
    return out


def loss_curves(prefix, label, fname):
    """Pretraining curves for the crossmodal runs <prefix>_s{0,1,2}_<L>: train total / UAV loss per epoch (runs/*/log.csv,
    mean and min-max over seeds) and the held-out-location UAV reconstruction MSE (sat-full, UAV fully masked; logged at
    epochs 10/20/30 only, there is no per-epoch validation split). Returns the tidy curve data."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    rows = []
    for L in LOCS:
        for s in (0, 1, 2):
            run = os.path.join(RUNS, f'{prefix}_s{s}_{L}')
            for r in pd.read_csv(os.path.join(run, 'log.csv')).itertuples():
                rows += [dict(heldout=L, seed=s, epoch=r.epoch + 1, split='train', loss=k, value=getattr(r, k))
                         for k in ('loss', 'loss_uav')]
            ev = EventAccumulator(os.path.join(run, 'tb'))
            ev.Reload()
            rows += [dict(heldout=L, seed=s, epoch=e.step + 1, split='heldout', loss='loss_uav', value=e.value)
                     for e in ev.Scalars('heldout/uav_mse')]
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, len(LOCS), figsize=(3.0 * len(LOCS), 5.2), sharex=True, sharey='row')
    for i, (k, ylab) in enumerate([('loss', 'Total loss (sat + UAV)'), ('loss_uav', 'UAV loss')]):
        for j, L in enumerate(LOCS):
            ax = axes[i, j]
            q = df[(df.heldout == L) & (df.loss == k)]
            g = q[q.split == 'train'].groupby('epoch').value
            ax.fill_between(g.min().index, g.min(), g.max(), color=PAL[i], alpha=0.2, lw=0)
            ax.plot(g.mean().index, g.mean(), color=PAL[i], lw=1.6, label='train (mean, min–max over 3 seeds)')
            h = q[q.split == 'heldout'].groupby('epoch').value
            if len(h.mean()):
                ax.errorbar(h.mean().index, h.mean(), yerr=[h.mean() - h.min(), h.max() - h.mean()], color=INK2,
                            marker='D', ms=5, mfc='white', mew=1.2, lw=1.2, ls='--', capsize=0,
                            label='held-out location UAV MSE (sat → UAV)')
            ax.grid(axis='y', color=GRID, lw=0.6)
            if i == 0:
                ax.set_title(LOC_LABEL[L], fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 1:
                ax.set_xlabel('Epoch')
        axes[i, 0].legend(frameon=False, fontsize=7, loc='upper right')
    fig.suptitle(f'Crossmodal pretraining loss, {label} (column = held-out location)', fontsize=10, color=INK, x=0.01,
                 ha='left')
    fig.tight_layout()
    savefig(fig, fname)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', nargs='+')
    a = ap.parse_args()
    main_df = load('main.csv')
    if 'main' in a.what or 'all' in a.what:
        main_table(main_df)
        for tp in ('t1', 't2', 't3'):
            main_table(main_df, tp=tp, fname=f'main_lolo_{tp}.tex')
        for probe in ('mlp', 'knn'):
            main_table(main_df, probe=probe, fname=f'main_lolo_{probe}.tex')
        paired_stats(main_df, [r for r in MAIN_ROWS if r != BASE])
        paired_stats(main_df, ['E-xmodal', 'E-xmodal-dec', 'E-xmodal+H-sat-all', 'E-satMAE+H-sat-all'], base='E-satMAE',
                     fname='paired_stats_vs_satmae')
        relative_table(main_df)
        per_location_plot(main_df, ['H-sat-all', 'E-satMAE', 'E-xmodal', 'E-xmodal+H-sat-all', 'H-uav'],
                          [('yield', 'A', 'r2', 'Yield $R^2$, within held-out site (A)'),
                           ('yield', 'B_z', 'r2', 'Yield $R^2$, cross-site, z-scored (B)'),
                           ('yield', 'B_z', 'pearson', 'Yield Pearson $r$, cross-site (B)')])
    if 'ablate' in a.what or 'all' in a.what:
        ab = load('ablations.csv')
        if ab is not None:
            ablation_table(ab, main_df)
        sc = load('scaling.csv')
        if sc is not None:
            scaling_plot(sc, main_df)
    if 'rgb' in a.what or 'all' in a.what:
        rg = load('rgb.csv')
        if rg is not None:
            relative_table(rgb_frame(main_df, rg), fname='relative_rgb', comparisons=RGB_COMPARISONS)
    if 'loss' in a.what or 'all' in a.what:
        for prefix, label, fname in [('xmodal', 'satellite all bands ↔ UAV RGB', 'loss_curves_satall'),
                                     ('xmodalrgb', 'satellite RGB ↔ UAV RGB', 'loss_curves_satrgb')]:
            loss_curves(prefix, label, fname).to_csv(os.path.join(RESULTS, f'{fname}.csv'), index=False)
    if 'tp46' in a.what or 'all' in a.what:
        t = load('tp46.csv')
        if t is not None:
            tp46_plot(t)
    if 'recon' in a.what or 'all' in a.what:
        recon_figure()


if __name__ == '__main__':
    main()
