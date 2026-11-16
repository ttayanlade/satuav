"""Phase 1 deliverables: dataset table, acquisition timeline, QC pair grids."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import CACHE, EDATA, FIGS, LOCS, MANIFEST, TABLES

PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 200, 'savefig.bbox': 'tight'})
PLATFORM = {'Ames': 'P4 Pro V2 (20 MP)', 'Crawfordsville': 'P4 Pro V2 (20 MP)', 'Lincoln': 'P4 RTK + P1 (45 MP)',
            'MOValley': 'P4 Pro V2 (20 MP)', 'Scottsbluff': 'M600 + X3 (12 MP)'}


def savefig(fig, name):
    os.makedirs(os.path.dirname(os.path.join(FIGS, name)), exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGS, f'{name}.{ext}'))
    plt.close(fig)


def dataset_table(d):
    lab = d[(d.sensor == 'sat') & (d.tp == 1)]
    rows = []
    for L in LOCS:
        g = lab[lab.location == L]
        gl = g[g.has_label]
        u = d[(d.sensor == 'uav') & (d.location == L)]
        s = d[(d.sensor == 'sat') & (d.location == L)]
        rows.append(dict(
            Location=L, Images=len(g), Labelled=len(gl), Genotypes=gl.genotype.nunique(),
            N_levels='/'.join(str(int(x)) for x in sorted(gl.n_lbs.dropna().unique()) if x > 0),
            Yield=f'{gl["yield"].mean():.0f} $\\pm$ {gl["yield"].std():.0f}',
            UAV=PLATFORM[L], UAV_GSD=f'{100 * u.gsd_est.median():.2f}',
            Sat_DAP=', '.join(str(int(x)) for x in s.groupby('tp').days_after_planting.median()),
            UAV_DAP=', '.join(str(int(x)) for x in u.groupby('tp').days_after_planting.median())))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(EDATA, 'dataset_summary.csv'), index=False)
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, 'dataset_summary.tex'), 'w') as f:
        f.write('\\begin{table}[ht]\n\\centering\\small\n\\caption{Dataset summary per location after the audit. '
                '\\textit{Images}: plot images per sensor and timepoint; \\textit{Labelled}: plots matched to a '
                'ground-truth row with a genotype. Yield in bu/acre (mean $\\pm$ sd). Satellite: Pl\\\'eiades Neo, '
                '0.30\\,m GSD, 6 bands. DAP: median days after planting of each acquisition.}\n')
        f.write('\\label{tab:dataset_summary}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lrrrllllll}\n\\toprule\n')
        f.write('Location & Images & Labelled & Genotypes & N (lbs/acre) & Yield & UAV platform & UAV GSD (cm) '
                '& Sat DAP (TP1--6) & UAV DAP (TP1--3) \\\\\n\\midrule\n')
        for r in rows:
            f.write(' & '.join(str(r[k]) for k in ['Location', 'Images', 'Labelled', 'Genotypes', 'N_levels', 'Yield',
                                                    'UAV', 'UAV_GSD', 'Sat_DAP', 'UAV_DAP']).replace('MOValley', 'Missouri Valley') + ' \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}}\n\\end{table}\n')
    return t


def timeline(d):
    pairs = pd.read_csv(os.path.join(EDATA, 'tp_pairs.csv'))
    a = d.groupby(['location', 'sensor', 'tp']).days_after_planting.median().reset_index()
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    for i, L in enumerate(LOCS):
        y = len(LOCS) - 1 - i
        s = a[(a.location == L) & (a.sensor == 'sat')]
        u = a[(a.location == L) & (a.sensor == 'uav')]
        for _, p in pairs[pairs.location == L].iterrows():
            xs = s[s.tp == p.sat_tp].days_after_planting.values[0]
            xu = u[u.tp == p.uav_tp].days_after_planting.values[0]
            ax.plot([xs, xu], [y + 0.12, y - 0.12], color=INK2, lw=1, zorder=1)
        ax.scatter(s.days_after_planting, [y + 0.12] * len(s), s=36, color=PAL[0], marker='s', zorder=3,
                   edgecolor='white', lw=1.5, label='Satellite' if i == 0 else None)
        ax.scatter(u.days_after_planting, [y - 0.12] * len(u), s=40, color=PAL[1], marker='o', zorder=3,
                   edgecolor='white', lw=1.5, label='UAV' if i == 0 else None)
        for _, r in s.iterrows():
            ax.text(r.days_after_planting, y + 0.28, f'{int(r.tp)}', ha='center', va='bottom', fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(LOCS)))
    ax.set_yticklabels([L.replace('MOValley', 'Missouri Valley') for L in LOCS[::-1]])
    ax.set_xlabel('Days after planting')
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.set_ylim(-0.6, len(LOCS) - 0.3)
    ax.legend(frameon=False, ncol=2, loc='lower right', bbox_to_anchor=(1, 1.0))
    savefig(fig, 'timeline_acquisitions')


def sat_rgb_vis(t, m):
    """t: (3 tiles, C, H, W), m: (3, H, W); one stretch per plot so tiles stay comparable."""
    rgb = t[:, [0, 1, 2]].astype(np.float32).transpose(0, 2, 3, 1)
    v = rgb[m]
    lo, hi = np.percentile(v, 1, axis=0), np.percentile(v, 99, axis=0)
    out = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1) * m[..., None]
    return np.concatenate(list(out), 1)


def qc_pairs(d, n=5, seed=0):
    S = np.load(os.path.join(CACHE, 'sat_tiles.npy'), mmap_mode='r')
    SM = np.load(os.path.join(CACHE, 'sat_mask.npy'), mmap_mode='r')
    U = np.load(os.path.join(CACHE, 'uav_tiles.npy'), mmap_mode='r')
    UM = np.load(os.path.join(CACHE, 'uav_mask.npy'), mmap_mode='r')
    rng = np.random.RandomState(seed)
    for L in LOCS:
        u = d[(d.sensor == 'uav') & (d.location == L)]
        pick = u.iloc[rng.choice(len(u), n, replace=False)]
        fig, axes = plt.subplots(n, 2, figsize=(7, 1.3 * n))
        for k, (_, r) in enumerate(pick.iterrows()):
            s = d[(d.sensor == 'sat') & (d.plot_uid == r.plot_uid) & (d.tp == int(r.paired_tp))].iloc[0]
            si = int(s.cache_idx)
            srow = sat_rgb_vis(S[si], SM[si])
            urow = np.concatenate([U[int(r.cache_idx), j].transpose(1, 2, 0) / 255. * UM[int(r.cache_idx), j][..., None]
                                   for j in range(3)], 1)
            axes[k, 0].imshow(srow)
            axes[k, 1].imshow(urow)
            axes[k, 0].set_ylabel(f'{r.plot_uid.split("_", 1)[1]}\nU-TP{r.tp}/S-TP{int(r.paired_tp)} ({int(r.pair_day_gap):+d}d)',
                                  fontsize=6)
            if r.qc_flags:
                axes[k, 1].set_title(r.qc_flags, fontsize=6, color=PAL[7])
            for a in axes[k]:
                a.set_xticks([]); a.set_yticks([])
        axes[0, 0].set_title('Satellite RGB (3 aligned tiles)', fontsize=8)
        axes[0, 1].set_title(f'UAV RGB \u2014 {L}', fontsize=8)
        savefig(fig, f'qc/pairs_{L}')


def main():
    d = pd.read_parquet(MANIFEST)
    print(dataset_table(d).to_string())
    timeline(d)
    qc_pairs(d)


if __name__ == '__main__':
    main()
