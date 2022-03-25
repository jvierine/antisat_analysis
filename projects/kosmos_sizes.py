from astropy.time import Time, TimeDelta
import h5py
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from pprint import pprint
# plt.style.use('dark_background')

import sorts
import pyorb
import pyant

HERE = Path(__file__).parent.resolve()
OUTPUT = HERE / 'output' / 'russian_asat'
print(f'Using {OUTPUT} as output')

tot_pth = OUTPUT / 'all_kosmos_stat'
tot_pth.mkdir(exist_ok=True)

with open(OUTPUT / 'paths.pickle', 'rb') as fh:
    paths = pickle.load(fh)


category_files = {}
dates = {}
all_dates = []
for radar in paths['data_paths']:
    category_files[radar] = []
    dates[radar] = []
    for ind in range(len(paths['data_paths'][radar])):
        date = paths['data_paths'][radar][ind].stem.replace('.', '-')
        dates[radar].append(date)
        all_dates.append(date)
        out_categories = OUTPUT / f'{date}_{radar}_categories.npy'
        category_files[radar].append(out_categories)
pprint(category_files)

all_dates = np.sort(np.unique(all_dates))


class Categories:
    uncorrelated_other = 0
    correlated_other = 1
    uncorrelated_kosmos = 2
    correlated_kosmos = 3


def kosmos_select(select):
    return np.logical_or(
        select == Categories.uncorrelated_kosmos,
        select == Categories.correlated_kosmos,
    )


def load_data(file):
    with h5py.File(file, "r") as h:
        v = h["v"][()]
        t = h["t"][()]
        r = h["r"][()]    
        snr = h["snr"][()]
        dur = h["dur"][()]
        diam = h["diams"][()]

    return t, r, v, snr, dur, diam


radar = 'uhf'
ind = 0

select = np.load(category_files[radar][ind])
data_file = paths['data_paths'][radar][ind]
date = dates[radar][ind]
rcs_path = OUTPUT / f'{date.replace("-", ".")}_{radar}_rcs'

t, r, v, snr, dur, diam = load_data(data_file)
times = Time(t, format='unix', scale='utc')

collected_res = rcs_path / 'collected_results.pickle'
print(collected_res)
with open(collected_res, 'rb') as fh:
    results = pickle.load(fh)

kosmos_cat = kosmos_select(select)

size_dist = None
kosmos_dist = None
meas_inds = []
for ind in range(len(results['t_unix_peak'])):
    t0 = results['t_unix_peak'][ind]
    if np.isnan(t0):
        meas_inds.append(0)
        continue
    dt = np.abs(t0 - times.unix)
    meas_ind = np.argmin(dt)
    meas_inds.append(meas_ind)

    if size_dist is None:
        size_dist = np.zeros_like(results['estimated_diam_prob'][ind])
        kosmos_dist = np.zeros_like(results['estimated_diam_prob'][ind])
        size_bins = np.copy(results['estimated_diam_bins'][ind])

    if kosmos_cat[meas_ind]:
        kosmos_dist += results['estimated_diam_prob'][ind]
    size_dist += results['estimated_diam_prob'][ind]
meas_inds = np.array(meas_inds)

kosmos_diams = results['estimated_diam'][kosmos_cat[meas_inds]]


fig, ax = plt.subplots(figsize=(12, 8))
ax.hist(
    results['estimated_diam'],
    bins=size_bins, 
    label='Total',
    color='b',
)
ax.hist(
    kosmos_diams,
    bins=size_bins, 
    label='KOSMOS-1408',
    color='r',
)

ax.set_xlabel('Diameter at peak SNR [log10(cm)]')
ax.set_ylabel('Frequency (from distance function peak) [1]')
ax.set_title('Size distribution minimum distance function')
ax.legend()
fig.savefig(rcs_path / 'kosmos_diam_peak_dist.png')
plt.close(fig)


fig, ax = plt.subplots(figsize=(12, 8))
ax.bar(
    (size_bins[:-1] + size_bins[1:])*0.5,
    size_dist, 
    align='center', 
    width=np.diff(size_bins),
    label='Total',
    color='b',
)
ax.bar(
    (size_bins[:-1] + size_bins[1:])*0.5,
    kosmos_dist, 
    align='center', 
    width=np.diff(size_bins),
    label='KOSMOS-1408',
    color='r',
)

ax.set_xlabel('Diameter at peak SNR [log10(cm)]')
ax.set_ylabel('Frequency (from matching function probability) [1]')
ax.set_title('Size distribution from distance function distribution')
ax.legend()
fig.savefig(rcs_path / 'kosmos_diam_prob_dist.png')
plt.close(fig)
