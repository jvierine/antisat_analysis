from astropy.time import Time, TimeDelta
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import h5py
from pprint import pprint
import pandas as pd

import sys

import odlab
import sorts
import pyorb
import pyant

HERE = Path(__file__).parent.resolve()
OUTPUT = HERE / 'output' / 'russian_asat'
print(f'Using {OUTPUT} as output')

out_pth = OUTPUT / 'orbit_determination_v2'

results_data_file = out_pth / 'iod_results.pickle'

with open(results_data_file, 'rb') as fh:
    iod_results = pickle.load(fh)

pprint(list(iod_results.keys()))

diffs = {}
levels = ['mcmc', 'lsq', 'start']
for key in levels:
    diffs[key] = [
        abs(x - ref)
        for x, ref in zip(iod_results[key], iod_results['tle'])
        if x is not None
    ]
ids = [ind for ind, x in zip(iod_results['id'], iod_results['tle']) if x is not None]
for key in levels:
    diffs[key] = np.stack(diffs[key])
    diffs[key][:, 2:] = np.degrees(diffs[key][:, 2:])

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
axes[0].hist(np.log10(diffs['lsq'][:, 0]*1e-3))
axes[1].hist(diffs['lsq'][:, 1])
axes[2].hist(diffs['lsq'][:, 2])
axes[3].hist(diffs['lsq'][:, 3])

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
axes[0].hist(np.log10(diffs['mcmc'][:, 0]*1e-3))
axes[1].hist(diffs['mcmc'][:, 1])
axes[2].hist(diffs['mcmc'][:, 2])
axes[3].hist(diffs['mcmc'][:, 3])

dr = [
    np.array([
        x[rdi] 
        for x in iod_results['lsq_mean_dr']
        if x is not None
    ])
    for rdi in [0, 1]
]
dv = [
    np.array([
        x[rdi] 
        for x in iod_results['lsq_mean_dv']
        if x is not None
    ])
    for rdi in [0, 1]
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].hist(dr[0]*1e-3)
axes[0, 1].hist(dv[0]*1e-3)
axes[1, 0].hist(dr[1]*1e-3)
axes[1, 1].hist(dv[1]*1e-3)

dt = [
    np.abs((iod_results['dt'][x][0] - iod_results['dt'][x][1]).sec/3600.0)
    for x in ids
]
norad = [
    iod_results['norad'][x]
    for x in ids
]
d = {
    'norad': norad,
    'dr_uhf': dr[0]*1e-3,
    'dv_uhf': dv[0]*1e-3,
    'dr_esr': dr[1]*1e-3,
    'dv_esr': dv[1]*1e-3,
    'da_lsq': diffs['lsq'][:, 0]*1e-3,
    'de_lsq': diffs['lsq'][:, 1],
    'di_lsq': diffs['lsq'][:, 2],
    'draan_lsq': diffs['lsq'][:, 3],
    'da_mcmc': diffs['mcmc'][:, 0]*1e-3,
    'de_mcmc': diffs['mcmc'][:, 1],
    'di_mcmc': diffs['mcmc'][:, 2],
    'draan_mcmc': diffs['mcmc'][:, 3],
    'dt': dt,
}
df = pd.DataFrame(data=d)

print(df.style.to_latex())
