"""Builds and runs all probe job sets from whatever features exist.

  python -m satuav.evaluate main trans extra tp46 ablate scaling rgb latent
"""
import argparse
import os

import pandas as pd

from .paths import FEATS, LOCS, RESULTS
from .probe import SAME_CAM, run_jobs
from .sweep import ABL_LOCS

HAND = ['H-sat-rgb', 'H-sat-all', 'H-uav', 'F-late', 'E-imnet']
RUN_REPS = ['H-pred', 'H-sat-all+pred', 'E-satMAE', 'E-xmodal', 'E-xmodal-dec', 'E-xmodal+H-sat-all',
            'E-satMAE+H-sat-all', 'E-uav-real']
RGB_REPS = ['H-pred', 'H-sat-rgb+pred', 'E-satMAE', 'E-xmodal', 'E-xmodal-dec', 'E-xmodal+H-sat-rgb',
            'E-satMAE+H-sat-rgb', 'E-uav-real']  # run-dependent reps on the RGB-only satellite runs
LATENT_REPS = ['E-pred', 'E-sat+pred', 'E-uav']  # latent counterparts of H-pred / H-sat+pred / H-uav
TASKS = ['yield', 'nitrogen']
REGIMES = ['A', 'B_raw', 'B_z']


def have(name):
    return all(os.path.exists(os.path.join(FEATS, name, f'{L}.npz')) for L in LOCS)


def seeds_available(tmpls, locs=LOCS):
    return [s for s in (0, 1, 2) if all(have(t.format(seed=s, L=L)) for t in tmpls for L in locs)]


def jobs_for(reps, locs, seeds, tps, tasks, regimes, probes, xmodal='xmodal_s{seed}_{L}', satmae='satmae_s{seed}_{L}',
             extra=None, **kw):
    """kw: further evaluate() options (nb, teacher, lam), recorded in the output rows too."""
    J = []
    for rep in reps:
        for L in locs:
            for s in seeds:
                runs = dict(xmodal=xmodal.format(seed=s, L=L), satmae=satmae.format(seed=s, L=L), imnet=f'imnet_{L}')
                runs['xlat'] = runs['xmodal'] + '__lat'
                runs['uavmae'] = f'uavmae_s{s}_{L}__lat'
                runs['uavmaeF'] = f'uavmaeF_s{s}_{L}__lat'
                runs['uavmaeM'] = f'uavmaeM_s{s}_{L}__lat'
                for tp in tps:
                    J.append(dict(rep=rep, runs=runs, L=L, tspec=tp, tasks=tasks, regimes=regimes, probes=probes,
                                  extra=dict(seed=s, **(extra or {}), **{k: v for k, v in kw.items() if v is not None}),
                                  **kw))
    return J


def save(J, name, n_jobs=30):
    df = run_jobs(J, n_jobs)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, name), index=False)
    print(name, len(df), flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', nargs='+')
    a = ap.parse_args()
    seeds = seeds_available(['xmodal_s{seed}_{L}', 'satmae_s{seed}_{L}'])
    print('seeds with complete main runs:', seeds, flush=True)
    if 'main' in a.what:
        tps = ['t1', 't2', 't3', 'cat']
        J = jobs_for(HAND, LOCS, [0], tps, TASKS, REGIMES, ['linear'])
        J += jobs_for(HAND, LOCS, [0], ['cat'], TASKS, REGIMES, ['mlp', 'knn', 'xgb'])
        J += jobs_for(RUN_REPS, LOCS, seeds, tps, TASKS, REGIMES, ['linear'])
        J += jobs_for(RUN_REPS, LOCS, seeds, ['cat'], TASKS, REGIMES, ['mlp', 'knn', 'xgb'])
        save(J, 'main.csv')
    if 'trans' in a.what and all(have(f'xmodalT_s0_{L}') for L in LOCS):
        J = jobs_for(['H-pred', 'E-xmodal', 'E-xmodal-dec', 'E-xmodal+H-sat-all'], LOCS, [0], ['t1', 't2', 't3', 'cat'],
                     TASKS, REGIMES, ['linear'], xmodal='xmodalT_s{seed}_{L}', extra=dict(variant='LOLO-T'))
        save(J, 'trans.csv')
    if 'extra' in a.what:
        J = jobs_for(HAND, LOCS, [0], ['cat'], ['stand_count', 'dta', 'gdd'], ['A'], ['linear'])
        J += jobs_for(RUN_REPS, LOCS, seeds, ['cat'], ['stand_count', 'dta', 'gdd'], ['A'], ['linear'])
        save(J, 'extra_tasks.csv')
    if 'tp46' in a.what:
        J = jobs_for(['H-sat-all', 'H-sat-rgb', 'E-xmodal', 'E-satMAE', 'H-pred', 'E-xmodal+H-sat-all'], LOCS, seeds,
                     ['cat', 'cat6', 's4', 's5', 's6'], ['yield'], ['A', 'B_z'], ['linear'])
        J += jobs_for(['H-uav'], LOCS, seeds, ['cat', 'drop1', 'drop2', 'drop3', 'sub1', 'sub2', 'sub3', 'hsub1', 'hsub2', 'hsub3'], ['yield'],
                      ['A', 'B_z'], ['linear'])
        J += jobs_for(['E-uav-real'], LOCS, seeds, ['cat', 'drop1', 'drop2', 'drop3', 'sub1', 'sub2', 'sub3', 'hsub1', 'hsub2', 'hsub3'], ['yield'],
                      ['A', 'B_z'], ['linear'])
        J += jobs_for(['H-uav+pred@s', 'H-uav+H-sat-all@s', 'H-uav+sat-all+pred@s', 'E-uav-real+E-xmodal@s'], LOCS,
                      seeds, ['cat+s46'], ['yield'], ['A', 'B_z'], ['linear'])
        save(J, 'tp46.csv')
    if 'rgb' in a.what:
        rs = seeds_available(['xmodalrgb_s{seed}_{L}', 'satmaergb_s{seed}_{L}'])
        print('rgb seeds:', rs, flush=True)
        J = jobs_for(['F-late-rgb'], LOCS, [0], ['t1', 't2', 't3', 'cat'], TASKS, REGIMES, ['linear'])
        J += jobs_for(RGB_REPS, LOCS, rs, ['t1', 't2', 't3', 'cat'], TASKS, REGIMES, ['linear'],
                      xmodal='xmodalrgb_s{seed}_{L}', satmae='satmaergb_s{seed}_{L}', extra=dict(bands='rgb'))
        J += jobs_for(RGB_REPS, LOCS, rs, ['cat'], TASKS, REGIMES, ['knn'],
                      xmodal='xmodalrgb_s{seed}_{L}', satmae='satmaergb_s{seed}_{L}', extra=dict(bands='rgb'))
        if rs:
            save(J, 'rgb.csv')
    if 'latent' in a.what:
        for tm, sm, bands, fname in [('xmodal_s{seed}_{L}', 'satmae_s{seed}_{L}', 'all', 'latent.csv'),
                                     ('xmodalrgb_s{seed}_{L}', 'satmaergb_s{seed}_{L}', 'rgb', 'latent_rgb.csv')]:
            ls = [s for s in (0, 1, 2) if all(have(tm.format(seed=s, L=L) + '__lat') for L in LOCS)]
            print('latent', bands, 'seeds:', ls, flush=True)
            J = jobs_for(LATENT_REPS, LOCS, ls, ['t1', 't2', 't3', 'cat'], TASKS, REGIMES, ['linear'], xmodal=tm,
                         satmae=sm, extra=dict(bands=bands))
            J += jobs_for(LATENT_REPS, LOCS, ls, ['cat'], TASKS, REGIMES, ['knn'], xmodal=tm, satmae=sm,
                          extra=dict(bands=bands))
            if ls:
                save(J, fname)
    if 'seengeno' in a.what:  # cross-site with the test genotypes kept at the training sites (regime B_zg)
        hand = ['H-sat-all', 'H-sat-rgb', 'H-uav', 'F-late', 'F-late-rgb']
        lat = ['H-pred', 'E-xmodal', 'E-satMAE', 'E-uav-real'] + LATENT_REPS
        J = jobs_for(hand, LOCS, [0], ['t1', 't2', 't3', 'cat'], TASKS, ['B_zg'], ['linear'])
        J += jobs_for(lat + ['H-sat-all+pred'], LOCS, seeds, ['t1', 't2', 't3', 'cat'], TASKS, ['B_zg'], ['linear'],
                      extra=dict(bands='all'))
        J += jobs_for(lat + ['H-sat-rgb+pred'], LOCS, seeds, ['t1', 't2', 't3', 'cat'], TASKS, ['B_zg'], ['linear'],
                      xmodal='xmodalrgb_s{seed}_{L}', satmae='satmaergb_s{seed}_{L}', extra=dict(bands='rgb'))
        save(J, 'seengeno.csv', n_jobs=20)
    if 'uavmae' in a.what:  # real UAV through the UAV-only MAE (same paired UAV images as the RGB crossmodal runs)
        us = [s for s in (0, 1, 2) if all(have(f'uavmae_s{s}_{L}__lat') for L in LOCS)]
        print('uavmae seeds:', us, flush=True)
        tps = ['t1', 't2', 't3', 'cat']
        J = jobs_for(['E-uavMAE'], LOCS, us, tps, TASKS, REGIMES, ['linear'], extra=dict(bands='uav'))
        J += jobs_for(['E-uavMAE'], LOCS, us, tps, ['stand_count', 'dta', 'gdd'], ['A'], ['linear'], extra=dict(bands='uav'))
        if us:
            save(J, 'uavmae.csv', n_jobs=20)
    if 'uavctl' in a.what:  # controls for the E-uav cross-site gain (sweep uavctl)
        tps = ['t1', 't2', 't3', 'cat']
        J = []
        for rep, v in (('E-uavMAE-F', 'uavmaeF'), ('E-uavMAE-M', 'uavmaeM')):
            us = [s for s in (0, 1, 2) if all(have(f'{v}_s{s}_{L}__lat') for L in LOCS)]
            print(v, 'seeds:', us, flush=True)
            J += jobs_for([rep], LOCS, us, tps, TASKS, REGIMES, ['linear'], extra=dict(bands='uav'))
        save(J, 'uavctl.csv', n_jobs=20)
    if 'alpha' in a.what:  # XM-UAV (E-uav) under different Dirichlet alphas, reusing the all-band ablation checkpoints
        # (seed 0, ABL_LOCS): abl-a1-1 = 1:1 p0, abl-p0 = 0.9:0.1 p0 (matched p), xmodal_s0 = 0.9:0.1 p0.25 (paper)
        J = []
        for tm, al in (('abl-a1-1_{L}', '1:1,p0'), ('abl-p0_{L}', '0.9:0.1,p0'), ('xmodal_s0_{L}', '0.9:0.1,p0.25')):
            J += jobs_for(['E-uav', 'E-xmodal'], ABL_LOCS, [0], ['cat'], TASKS, ['A', 'B_z', 'P'], ['linear'], xmodal=tm,
                          satmae='satmae_s0_{L}', extra=dict(bands='all', alpha=al))
        J += jobs_for(['E-uavMAE'], ABL_LOCS, [0], ['cat'], TASKS, ['A', 'B_z', 'P'], ['linear'],
                      extra=dict(bands='uav', alpha='uav-only'))
        save(J, 'alpha.csv', n_jobs=16)
    if 'transfer' in a.what:  # ideas 2-4 (2026-11-04): LUPI soft targets, field-neighbour normalisation, sat TP1-6
        S = [0, 1, 2]
        RGB = dict(xmodal='xmodalrgb_s{seed}_{L}', satmae='satmaergb_s{seed}_{L}')
        ALL = dict(xmodal='xmodal_s{seed}_{L}', satmae='satmae_s{seed}_{L}')
        J = []
        # (2) LUPI: sat student, UAV teacher (H-uav, or E-uav from the same crossmodal run set), lam 0 / 0.5 / 1
        for reps, seeds, rs, b in ((['H-sat-all', 'H-sat-rgb'], [0], ALL, 'hand'), (['E-satMAE'], S, RGB, 'rgb'),
                                   (['E-satMAE'], S, ALL, 'all')):
            J += jobs_for(reps, LOCS, seeds, ['cat'], ['yield'], ['A', 'B_z'], ['linear'], extra=dict(bands=b, idea='lupi'),
                          teacher=None, lam=0.0, **rs)
            for t in ('H-uav', 'E-uav'):
                for lam in (0.5, 1.0):
                    J += jobs_for(reps, LOCS, seeds, ['cat'], ['yield'], ['A', 'B_z'], ['linear'],
                                  extra=dict(bands=b, idea='lupi'), teacher=t, lam=lam, **rs)
        # (3)+(4) neighbour normalisation (k=8; k=4/16 sensitivity on two reps) x aligned TPs vs sat TP1-6
        for reps, seeds, rs, b in ((['H-sat-all', 'H-sat-rgb', 'H-uav'], [0], ALL, 'hand'),
                                   (['E-satMAE', 'E-xmodal', 'E-uav', 'E-uavMAE'], S, RGB, 'rgb'),
                                   (['E-satMAE', 'E-xmodal'], S, ALL, 'all')):
            for k in (0, 8):
                for tp in ('cat', 'cat6'):
                    rr = [r for r in reps if tp == 'cat' or 'uav' not in r]
                    J += jobs_for(rr, LOCS, seeds, [tp], TASKS, ['A', 'B_z'], ['linear'], extra=dict(bands=b, idea='nb'),
                                  nb=k, **rs)
        for k in (4, 16):
            J += jobs_for(['H-sat-all'], LOCS, [0], ['cat'], TASKS, ['A', 'B_z'], ['linear'], extra=dict(bands='hand', idea='nb'), nb=k, **ALL)
            J += jobs_for(['E-satMAE'], LOCS, S, ['cat'], TASKS, ['A', 'B_z'], ['linear'], extra=dict(bands='rgb', idea='nb'), nb=k, **RGB)
        print('transfer jobs:', len(J), flush=True)
        save(J, 'transfer.csv', n_jobs=16)
    if 'random' in a.what:  # plot-level random splits instead of genotype-grouped ones: A_r (within), B_zr (cross)
        tps, R, S = ['t1', 't2', 't3', 'cat'], ['A_r', 'B_zr'], [0, 1, 2]
        RGB = dict(xmodal='xmodalrgb_s{seed}_{L}', satmae='satmaergb_s{seed}_{L}')
        J = jobs_for(['H-sat-rgb', 'H-sat-all', 'H-uav'], LOCS, [0], tps, TASKS, R, ['linear'], extra=dict(bands='hand'))
        J += jobs_for(['E-satMAE', 'E-xmodal', 'E-uav', 'E-uav-real', 'E-uav+E-xmodal'], LOCS, S, tps, TASKS, R, ['linear'],
                      extra=dict(bands='rgb'), **RGB)
        J += jobs_for(['E-satMAE', 'E-xmodal', 'E-uav'], LOCS, S, tps, TASKS, R, ['linear'], extra=dict(bands='all'))
        J += jobs_for(['E-uavMAE', 'E-uavMAE-F', 'E-uavMAE-M'], LOCS, S, tps, TASKS, R, ['linear'], extra=dict(bands='uav'))
        print('random-split jobs:', len(J), flush=True)
        save(J, 'random.csv', n_jobs=16)
    if 'cov' in a.what:  # ground-truth covariates (N rate + genotype) added to every representation, yield only
        R, S = ['A', 'A_r', 'B_zr'], [0, 1, 2]  # B_z: unseen genotypes + non-transferable N -> nothing to add
        RGB = dict(xmodal='xmodalrgb_s{seed}_{L}', satmae='satmaergb_s{seed}_{L}')
        J = jobs_for(['H-sat-rgb', 'H-sat-all', 'H-uav'], LOCS, [0], ['cat'], ['yield'], R, ['linear'], extra=dict(bands='hand'))
        J += jobs_for(['E-satMAE', 'E-xmodal', 'E-uav', 'E-uav-real', 'E-uav+E-xmodal'], LOCS, S, ['cat'], ['yield'], R,
                      ['linear'], extra=dict(bands='rgb'), **RGB)
        J += jobs_for(['E-satMAE', 'E-xmodal', 'E-uav'], LOCS, S, ['cat'], ['yield'], R, ['linear'], extra=dict(bands='all'))
        J += jobs_for(['E-uavMAE', 'E-uavMAE-F', 'E-uavMAE-M'], LOCS, S, ['cat'], ['yield'], R, ['linear'], extra=dict(bands='uav'))
        for j in J:
            j['cov'] = True
        print('covariate jobs:', len(J), flush=True)
        save(J, 'cov.csv', n_jobs=16)
    if 'pooledsite' in a.what:  # Powadi et al. 2025 scheme: one probe over all sites, genotype-grouped 5-fold (P);
        # B_z rerun for the top-25/50% genotype overlap metric
        R, S, T = ['P', 'B_z'], [0, 1, 2], (['t1', 't2', 't3'] if 'pooledsite_tp' in a.what else ['cat'])
        RGB = dict(xmodal='xmodalrgb_s{seed}_{L}', satmae='satmaergb_s{seed}_{L}')
        J = jobs_for(['H-sat-rgb', 'H-sat-all', 'H-uav'], LOCS, [0], T, TASKS, R, ['linear'], extra=dict(bands='hand'))
        J += jobs_for(['E-satMAE', 'E-xmodal', 'E-uav', 'E-uav-real', 'E-uav+E-xmodal'], LOCS, S, T, TASKS, R, ['linear'],
                      extra=dict(bands='rgb'), **RGB)
        J += jobs_for(['E-satMAE', 'E-xmodal', 'E-uav'], LOCS, S, T, TASKS, R, ['linear'], extra=dict(bands='all'))
        J += jobs_for(['E-uavMAE', 'E-uavMAE-F', 'E-uavMAE-M'], LOCS, S, T, TASKS, R, ['linear'], extra=dict(bands='uav'))
        print('pooled-site jobs:', len(J), flush=True)
        save(J, 'pooledsite_tp.csv' if 'pooledsite_tp' in a.what else 'pooledsite.csv', n_jobs=16)
    if 'fix' in a.what:  # pilots of alternative crossmodal objectives: satellite RGB embedding (E_sat) of each run
        tps = ['t1', 't2', 't3', 'cat']
        J = []
        for v in ('xdistill', 'xcontrast', 'xblur'):
            J += jobs_for(['E-xmodal'], ABL_LOCS, [0], tps, TASKS, ['A', 'B_z'], ['linear'], xmodal=v + '_s{seed}_{L}',
                          satmae='satmaergb_s{seed}_{L}', extra=dict(bands='rgb', variant=v))
        save(J, 'fix.csv', n_jobs=12)
    if 'ctx' in a.what:  # in-site context ablation; every rep scored on the plots outside the 50% context set
        tps, J = ['t1', 't2', 't3', 'cat'], []
        for v, x, m in (('none', 'xmodalrgb', 'satmaergb'), ('C25', 'xmodalrgbC25', 'satmaergbC25'),
                        ('C50', 'xmodalrgbC50', 'satmaergbC50')):
            J += jobs_for(['E-xmodal', 'E-satMAE'], ABL_LOCS, [0], tps, TASKS, ['A', 'B_z'], ['linear'],
                          xmodal=x + '_s{seed}_{L}', satmae=m + '_s{seed}_{L}', extra=dict(bands='rgb', variant=v))
        J += jobs_for(['E-uav', 'H-uav', 'H-sat-rgb'], ABL_LOCS, [0], tps, TASKS, ['A', 'B_z'], ['linear'],
                      xmodal='xmodalrgb_s{seed}_{L}', satmae='satmaergb_s{seed}_{L}', extra=dict(bands='rgb', variant='ref'))
        for j in J:
            j['exclude_ctx'] = 0.5
        save(J, 'ctx.csv', n_jobs=12)
    if 'latefuse' in a.what:  # separately encoded real UAV + real sat (crossmodal encoder), concatenated, vs joint encoding
        tps = ['t1', 't2', 't3', 'cat']
        J = jobs_for(['E-uav+E-xmodal'], LOCS, seeds, tps, TASKS, ['A', 'B_z'], ['linear'], extra=dict(bands='all'))
        J += jobs_for(['E-uav+E-xmodal'], LOCS, seeds, tps, TASKS, ['A', 'B_z'], ['linear'], xmodal='xmodalrgb_s{seed}_{L}',
                      satmae='satmaergb_s{seed}_{L}', extra=dict(bands='rgb'))
        save(J, 'latefuse.csv', n_jobs=12)
    if 'extrargb' in a.what:  # within-site extra traits, real UAV vs real satellite RGB (latent, RGB-only runs)
        tps, T = ['t1', 't2', 't3', 'cat'], ['stand_count', 'dta', 'gdd']
        J = jobs_for(['H-uav', 'H-sat-rgb'], LOCS, [0], tps, T, ['A'], ['linear'])
        J += jobs_for(['E-uav', 'E-xmodal'], LOCS, seeds, tps, T, ['A'], ['linear'], xmodal='xmodalrgb_s{seed}_{L}',
                      satmae='satmaergb_s{seed}_{L}', extra=dict(bands='rgb'))
        save(J, 'extra_tasks_rgb.csv', n_jobs=20)
    if 'samecam' in a.what:  # cross-site among the sites flown with the same UAV camera (B_zc) vs other-camera sites (B_zx)
        hand = ['H-sat-all', 'H-sat-rgb', 'H-uav', 'F-late', 'F-late-rgb']
        lat = ['H-pred', 'E-xmodal', 'E-satMAE', 'E-uav-real'] + LATENT_REPS
        tps, R = ['t1', 't2', 't3', 'cat'], ['B_zc', 'B_zx']
        J = jobs_for(hand, SAME_CAM, [0], tps, ['yield'], R, ['linear'])
        J += jobs_for(lat + ['H-sat-all+pred'], SAME_CAM, seeds, tps, ['yield'], R, ['linear'], extra=dict(bands='all'))
        J += jobs_for(lat + ['H-sat-rgb+pred'], SAME_CAM, seeds, tps, ['yield'], R, ['linear'],
                      xmodal='xmodalrgb_s{seed}_{L}', satmae='satmaergb_s{seed}_{L}', extra=dict(bands='rgb'))
        save(J, 'samecam.csv', n_jobs=20)
    if 'ablate' in a.what:
        fams = ['p0', 'a1-1', 'a1.2-0.8', 'a1.5-0.5', 'a0.7-0.3', 'satrgb', 'randinit', 'mismatch', 'nomaskloss',
                'origcrop']
        J = []
        for fam in fams:
            locs = [L for L in ABL_LOCS if have(f'abl-{fam}_{L}')]
            if locs:
                J += jobs_for(['E-xmodal', 'H-pred', 'E-xmodal+H-sat-all'], locs, [0], ['cat'], ['yield', 'nitrogen'],
                              ['A', 'B_z'], ['linear'], xmodal=f'abl-{fam}_{{L}}', extra=dict(family=fam))
        if J:
            save(J, 'ablations.csv')
    if 'scaling' in a.what:
        J = []
        for n in (1, 2, 3):
            for i in (0, 1):
                xm, sm = f'abl-nloc{n}-{i}_Crawfordsville', f'abl-nloc{n}-{i}-satmae_Crawfordsville'
                if have(xm):
                    J += jobs_for(['E-xmodal', 'H-pred'], ['Crawfordsville'], [0], ['cat'], ['yield'], ['A', 'B_z'],
                                  ['linear'], xmodal=xm, extra=dict(nloc=n, subset=i, kind='xmodal'))
                if have(sm):
                    J += jobs_for(['E-satMAE'], ['Crawfordsville'], [0], ['cat'], ['yield'], ['A', 'B_z'], ['linear'],
                                  satmae=sm, extra=dict(nloc=n, subset=i, kind='satmae'))
        if J:
            save(J, 'scaling.csv')


if __name__ == '__main__':
    main()
