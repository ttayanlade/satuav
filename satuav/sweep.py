"""Resumable sweep: writes configs/<run>.yaml, pretrains, extracts features. Skips finished steps.

  python -m satuav.sweep main|seeds|trans|ablate|rgb|uavmae|uavctl|fix|ctx [--dry]
"""
import argparse
import os
import subprocess
import sys

import yaml

from .paths import EXP, FEATS, LOCS, RUNS

EPOCHS = 30
SAT_ONLY = dict(in_domains=['sat_all'], out_domains=['sat_all'])
RGB_XMODAL = dict(in_domains=['sat_rgb', 'uav_rgb'], out_domains=['sat_rgb', 'uav_rgb'])
RGB_SAT_ONLY = dict(in_domains=['sat_rgb'], out_domains=['sat_rgb'])
UAV_ONLY = dict(in_domains=['uav_rgb'], out_domains=['uav_rgb'])
ABL_LOCS = ['Crawfordsville', 'Lincoln', 'Scottsbluff']  # representative folds for the costlier ablations


def runs(group):
    R = []
    if group == 'main':
        for L in LOCS:
            R.append((f'xmodal_s0_{L}', dict(heldout=L)))
            R.append((f'satmae_s0_{L}', dict(heldout=L, **SAT_ONLY)))
    elif group == 'trans':
        for L in LOCS:
            R.append((f'xmodalT_s0_{L}', dict(heldout=L, transductive=True)))
    elif group == 'seeds':
        for s in (1, 2):
            for L in LOCS:
                R.append((f'xmodal_s{s}_{L}', dict(heldout=L, seed=s)))
                R.append((f'satmae_s{s}_{L}', dict(heldout=L, seed=s, **SAT_ONLY)))
    elif group == 'rgb':
        # RGB-only satellite input: is the crossmodal gain an artefact of NIR/red-edge? (seed 0 first)
        # xmodalrgb_s0_{Crawfordsville,Lincoln,Scottsbluff} are symlinks to the identical abl-satrgb_* runs
        for s in (0, 1, 2):
            for L in LOCS:
                R.append((f'xmodalrgb_s{s}_{L}', dict(heldout=L, seed=s, **RGB_XMODAL)))
                R.append((f'satmaergb_s{s}_{L}', dict(heldout=L, seed=s, **RGB_SAT_ONLY)))
    elif group == 'uavmae':
        # UAV-only MAE on the same paired UAV images: is E-uav's gain over H-uav from crossmodal training?
        for s in (0, 1, 2):
            for L in LOCS:
                R.append((f'uavmae_s{s}_{L}', dict(heldout=L, seed=s, **UAV_ONLY)))
    elif group == 'uavctl':
        # controls for E-uav (UAV through the crossmodal encoder) beating E-uavMAE cross-site: is it the sat anchor,
        # or (F) flight colour offsets the UAV-only MAE keeps, or (M) the much sparser visible-UAV masking?
        for s in (0, 1, 2):
            for L in LOCS:
                R.append((f'uavmaeF_s{s}_{L}', dict(heldout=L, seed=s, uav_norm='flight', **UAV_ONLY)))
                R.append((f'uavmaeM_s{s}_{L}', dict(heldout=L, seed=s, uav_vis='matched', **UAV_ONLY)))
    elif group == 'fix':
        # pilots of alternative crossmodal objectives (seed 0, RGB satellite, ablation sites); see pretrain docstring
        for L in ABL_LOCS:
            R.append((f'xdistill_s0_{L}', dict(heldout=L, objective='distill', **RGB_SAT_ONLY)))
            R.append((f'xcontrast_s0_{L}', dict(heldout=L, objective='contrast', **RGB_SAT_ONLY)))
            R.append((f'xblur_s0_{L}', dict(heldout=L, uav_target='blurflight', **RGB_XMODAL)))
    elif group == 'ctx':
        # in-site context: a fraction of the held-out site's plots (whole genotypes) joins pretraining; crossmodal
        # gets their sat + UAV, satMAE their sat only. Evaluated on the plots outside the largest context set.
        for p in (25, 50):
            for L in ABL_LOCS:
                R.append((f'xmodalrgbC{p}_s0_{L}', dict(heldout=L, context_frac=p / 100, **RGB_XMODAL)))
                R.append((f'satmaergbC{p}_s0_{L}', dict(heldout=L, context_frac=p / 100, **RGB_SAT_ONLY)))
    elif group == 'ablate':
        for L in ABL_LOCS:
            R.append((f'abl-p0_{L}', dict(heldout=L, p_satfull=0.0)))
            R.append((f'abl-a1-1_{L}', dict(heldout=L, alphas=[1.0, 1.0], p_satfull=0.0)))
            R.append((f'abl-satrgb_{L}', dict(heldout=L, in_domains=['sat_rgb', 'uav_rgb'], out_domains=['sat_rgb', 'uav_rgb'])))
            R.append((f'abl-randinit_{L}', dict(heldout=L, init='random')))
            R.append((f'abl-mismatch_{L}', dict(heldout=L, mismatch=True)))
            R.append((f'abl-nomaskloss_{L}', dict(heldout=L, masked_loss=False)))
            R.append((f'abl-origcrop_{L}', dict(heldout=L, tag='_orig')))
        for L in ABL_LOCS:
            for a in ([1.2, 0.8], [1.5, 0.5], [0.7, 0.3]):
                R.append((f'abl-a{a[0]}-{a[1]}_{L}', dict(heldout=L, alphas=a, p_satfull=0.0)))
        # data scaling: number of pretraining locations for held-out Crawfordsville (main run = 4 locations)
        others = ['Ames', 'Lincoln', 'MOValley', 'Scottsbluff']
        subsets = {1: [['Ames'], ['Scottsbluff']], 2: [['Ames', 'Scottsbluff'], ['Lincoln', 'MOValley']],
                   3: [['Ames', 'Lincoln', 'Scottsbluff'], ['Ames', 'MOValley', 'Scottsbluff']]}
        for n, ss in subsets.items():
            for i, s in enumerate(ss):
                R.append((f'abl-nloc{n}-{i}_Crawfordsville', dict(heldout='Crawfordsville', pretrain_locs=s)))
                R.append((f'abl-nloc{n}-{i}-satmae_Crawfordsville', dict(heldout='Crawfordsville', pretrain_locs=s, **SAT_ONLY)))
    return R


def done_features(name):
    return all(os.path.exists(os.path.join(FEATS, name, f'{L}.npz')) for L in LOCS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('groups', nargs='+')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    os.makedirs(os.path.join(EXP, 'configs'), exist_ok=True)
    py = sys.executable
    for g in a.groups:
        for name, cfg in runs(g):
            cfg = dict(cfg, epochs=EPOCHS, name=name)
            path = os.path.join(EXP, 'configs', f'{name}.yaml')
            yaml.safe_dump(cfg, open(path, 'w'))
            ck = os.path.join(RUNS, name, 'checkpoint-last.pth')
            todo = [] if os.path.exists(ck) else ['pretrain']
            if name.startswith('uavmae'):  # (also uavmaeF_/uavmaeM_)  # UAV-only: just the latent E_uav features
                if not all(os.path.exists(os.path.join(FEATS, f'{name}__lat', f'{L}.npz')) for L in LOCS):
                    todo.append('latent')
            elif not done_features(name):
                todo.append('extract')
            print(f'{name}: {todo or "done"}', flush=True)
            if a.dry:
                continue
            if 'pretrain' in todo:
                subprocess.run([py, '-m', 'satuav.pretrain', path], cwd=EXP, check=True)
            if 'extract' in todo:
                subprocess.run([py, '-m', 'satuav.extract', 'run', name], cwd=EXP, check=True)
            if 'latent' in todo:
                subprocess.run([py, '-m', 'satuav.extract', 'latent', name], cwd=EXP, check=True)


if __name__ == '__main__':
    main()
