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


def correlated_select(select):
    return np.logical_or(
        select == Categories.correlated_other,
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
filter_limit = 0.5

select = np.load(category_files[radar][ind])
data_file = paths['data_paths'][radar][ind]
date = dates[radar][ind]
rcs_path = OUTPUT / f'{date.replace("-", ".")}_{radar}_rcs'
rcs_plot_path = OUTPUT / f'{date.replace("-", ".")}_{radar}_rcs_plots'
rcs_plot_path.mkdir(exist_ok=True)


t, r, v, snr, dur, diam = load_data(data_file)
times = Time(t, format='unix', scale='utc')

collected_res = rcs_path / 'collected_results.pickle'
print(collected_res)
with open(collected_res, 'rb') as fh:
    results = pickle.load(fh)

keep_inds = results['match'] <= filter_limit

kosmos_cat = kosmos_select(select)
correlated_cat = correlated_select(select)

size_dist = None
kosmos_dist = None
correlated_dist = None
uncorrelated_dist = None
predicted_dist = None
bin_mids = None
size_means = []
kosmos_means = []
correlated_means = []
meas_inds = []
for ind in range(len(results['t_unix_peak'])):
    t0 = results['t_unix_peak'][ind]
    if np.isnan(t0):
        meas_inds.append(0)
        continue
    dt = np.abs(t0 - times.unix)
    meas_ind = np.argmin(dt)
    meas_inds.append(meas_ind)
    if not keep_inds[ind]:
        continue

    if size_dist is None:
        size_dist = np.zeros_like(results['estimated_diam_prob'][ind])
        kosmos_dist = np.zeros_like(results['estimated_diam_prob'][ind])
        correlated_dist = np.zeros_like(results['estimated_diam_prob'][ind])
        uncorrelated_dist = np.zeros_like(results['estimated_diam_prob'][ind])
        predicted_dist = np.zeros_like(results['estimated_diam_prob'][ind])
        size_bins = np.copy(results['estimated_diam_bins'][ind])
        bin_mids = (size_bins[:-1] + size_bins[1:])*0.5

    mean_d = np.average(
        bin_mids, 
        weights=results['estimated_diam_prob'][ind],
    )
    if not np.isnan(results['predicted_diam'][ind]):
        predicted_dist += results['estimated_diam_prob'][ind]
    if kosmos_cat[meas_ind]:
        kosmos_means.append(mean_d)
        kosmos_dist += results['estimated_diam_prob'][ind]
    if correlated_cat[meas_ind]:
        correlated_means.append(mean_d)
        correlated_dist += results['estimated_diam_prob'][ind]
    else:
        uncorrelated_dist += results['estimated_diam_prob'][ind]

    size_means.append(mean_d)
    size_dist += results['estimated_diam_prob'][ind]
meas_inds = np.array(meas_inds)

kosmos_diams = results['estimated_diam'][kosmos_cat[meas_inds]]

have_predictions = np.logical_not(np.isnan(results['predicted_diam']))
keep_stats = np.logical_and(have_predictions, keep_inds)
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, sharex=True)
axes[0, 0].hist(
    np.log10(results['predicted_diam'][keep_stats]*1e2),
    bins=size_bins, 
    color='b',
)
axes[0, 0].set_title('Predicted')
axes[1, 0].bar(
    bin_mids,
    predicted_dist, 
    align='center', 
    width=np.diff(size_bins),
    color='b',
)
axes[1, 0].set_title('Estimated dist')
axes[0, 1].hist(
    np.log10(results['estimated_diam'][keep_stats]*1e2),
    bins=size_bins, 
    color='b',
)
axes[0, 1].set_title('Estimated peak')
axes[0, 1].hist(
    np.log10(results['estimated_diam'][keep_stats]*1e2),
    bins=size_bins, 
    color='b',
)
axes[1, 1].set_title('Difference (-under/+over)')
axes[1, 1].hist(
    np.log10(results['estimated_diam'][keep_stats]*1e2) - np.log10(results['predicted_diam'][keep_stats]*1e2),
    color='b',
)
for ax in axes[1, :]:
    ax.set_xlabel('Diameter at peak SNR [log10(cm)]')
for ax in axes[:, 0]:
    ax.set_ylabel('Frequency [1]')

fig.suptitle('Size distribution estimated versus predicted')
fig.savefig(rcs_plot_path / 'predicted_vs_estimated_diam.png')
plt.close(fig)
exit()

fig, ax = plt.subplots(figsize=(12, 8))
ax.hist(
    np.log10(results['estimated_diam'][keep_inds]*1e2),
    bins=size_bins, 
    label='Total',
    color='b',
)
ax.hist(
    np.log10(kosmos_diams*1e2),
    bins=size_bins, 
    label='KOSMOS-1408',
    color='r',
)

ax.set_xlabel('Diameter at peak SNR [log10(cm)]')
ax.set_ylabel('Frequency (from distance function peak) [1]')
ax.set_title('Size distribution minimum distance function')
ax.legend()
fig.savefig(rcs_plot_path / 'kosmos_diam_peak_dist.png')
plt.close(fig)

not_kosmos_picks = np.logical_and(np.logical_not(kosmos_cat)[meas_inds], keep_inds)
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].hist(
    np.log10(results['estimated_diam'][not_kosmos_picks]*1e2),
    bins=size_bins, 
    color='b',
)
axes[0].set_title('Total')
axes[1].hist(
    np.log10(kosmos_diams*1e2),
    bins=size_bins, 
    color='r',
)
axes[1].set_title('KOSMOS-1408')
axes[1].set_xlabel('Diameter at peak SNR [log10(cm)]')
axes[0].set_ylabel('Frequency [1]')
axes[1].set_ylabel('Frequency [1]')
fig.suptitle('Size distribution minimum distance function')
fig.savefig(rcs_plot_path / 'kosmos_diam_peak_dist_sep.png')
plt.close(fig)


fig, ax = plt.subplots(figsize=(12, 8))
ax.hist(
    np.log10(results['estimated_diam'][keep_inds]*1e2),
    bins=size_bins, 
    label='Total',
    color='b',
)
correlated_picks = np.logical_and(correlated_cat[meas_inds], keep_inds)
ax.hist(
    np.log10(results['estimated_diam'][correlated_picks]*1e2),
    bins=size_bins, 
    label='Correlated',
    color='r',
)

ax.set_xlabel('Diameter at peak SNR [log10(cm)]')
ax.set_ylabel('Frequency (from distance function peak) [1]')
ax.set_title('Size distribution minimum distance function')
ax.legend()
fig.savefig(rcs_plot_path / 'correlated_diam_peak_dist.png')
plt.close(fig)


fig, ax = plt.subplots(figsize=(12, 8))
ax.hist(
    np.log10(results['estimated_diam'][keep_inds]*1e2),
    bins=size_bins, 
    label='Total',
    color='b',
)
uncorrelated_picks = np.logical_and(np.logical_not(correlated_cat[meas_inds]), keep_inds)
ax.hist(
    np.log10(results['estimated_diam'][uncorrelated_picks]*1e2),
    bins=size_bins, 
    label='Uncorrelated',
    color='r',
)

ax.set_xlabel('Diameter at peak SNR [log10(cm)]')
ax.set_ylabel('Frequency (from distance function peak) [1]')
ax.set_title('Size distribution minimum distance function')
ax.legend()
fig.savefig(rcs_plot_path / 'uncorrelated_diam_peak_dist.png')
plt.close(fig)


fig, ax = plt.subplots(figsize=(12, 8))
ax.bar(
    bin_mids,
    size_dist, 
    align='center', 
    width=np.diff(size_bins),
    label='Total',
    color='b',
)
ax.bar(
    bin_mids,
    kosmos_dist, 
    align='center', 
    width=np.diff(size_bins),
    label='KOSMOS-1408',
    color='r',
)

ax.set_xlabel('Diameter at peak SNR [log10(cm)]')
ax.set_ylabel('Frequency (from distance function probability) [1]')
ax.set_title('Size distribution from distance function distribution')
ax.legend()
fig.savefig(rcs_plot_path / 'kosmos_diam_prob_dist.png')
plt.close(fig)


fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True, sharex=True)
axes[0].bar(
    bin_mids,
    size_dist - kosmos_dist, 
    align='center', 
    width=np.diff(size_bins),
    color='b',
)
axes[0].set_title('Other')
axes[1].bar(
    bin_mids,
    kosmos_dist, 
    align='center', 
    width=np.diff(size_bins),
    color='r',
)
axes[1].set_title('KOSMOS-1408')

axes[1].set_xlabel('Diameter at peak SNR [log10(cm)]')
axes[0].set_ylabel('Frequency [1]')
axes[1].set_ylabel('Frequency [1]')
fig.suptitle('Size distribution from distance function distribution')
fig.savefig(rcs_plot_path / 'kosmos_diam_prob_dist_sep.png')
plt.close(fig)


fig, ax = plt.subplots(figsize=(12, 8))
ax.bar(
    bin_mids,
    size_dist, 
    align='center', 
    width=np.diff(size_bins),
    label='Total',
    color='b',
)
ax.bar(
    bin_mids,
    correlated_dist, 
    align='center', 
    width=np.diff(size_bins),
    label='Correlated',
    color='r',
)

ax.set_xlabel('Diameter at peak SNR [log10(cm)]')
ax.set_ylabel('Frequency (from distance function probability) [1]')
ax.set_title('Size distribution from distance function distribution')
ax.legend()
fig.savefig(rcs_plot_path / 'correlated_diam_prob_dist.png')
plt.close(fig)


fig, ax = plt.subplots(figsize=(12, 8))
ax.bar(
    bin_mids,
    size_dist, 
    align='center', 
    width=np.diff(size_bins),
    label='Total',
    color='b',
)
ax.bar(
    bin_mids,
    uncorrelated_dist, 
    align='center', 
    width=np.diff(size_bins),
    label='Uncorrelated',
    color='r',
)

ax.set_xlabel('Diameter at peak SNR [log10(cm)]')
ax.set_ylabel('Frequency (from distance function probability) [1]')
ax.set_title('Size distribution from distance function distribution')
ax.legend()
fig.savefig(rcs_plot_path / 'uncorrelated_diam_prob_dist.png')
plt.close(fig)


fig, ax = plt.subplots(figsize=(12, 8))
ax.hist(
    size_means,
    bins=size_bins, 
    label='Total',
    color='b',
)
ax.hist(
    kosmos_means,
    bins=size_bins, 
    label='KOSMOS-1408',
    color='r',
)

ax.set_xlabel('Diameter at peak SNR [log10(cm)]')
ax.set_ylabel('Frequency (from mean distance function probability) [1]')
ax.set_title('Size distribution from mean distance function distribution')
ax.legend()
fig.savefig(rcs_plot_path / 'kosmos_diam_mean_dist.png')
plt.close(fig)


fig, ax = plt.subplots(figsize=(12, 8))
ax.hist(
    size_means,
    bins=size_bins, 
    label='Total',
    color='b',
)
ax.hist(
    correlated_means,
    bins=size_bins, 
    label='Correlated',
    color='r',
)

ax.set_xlabel('Diameter at peak SNR [log10(cm)]')
ax.set_ylabel('Frequency (from mean distance function probability) [1]')
ax.set_title('Size distribution from mean distance function distribution')
ax.legend()
fig.savefig(rcs_plot_path / 'correlated_diam_mean_dist.png')
plt.close(fig)


fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(
    results['match'][keep_inds],
    np.log10(results['estimated_diam'][keep_inds]*1e2),
    color='b',
    label='Kept',
)
ax.scatter(
    results['match'][np.logical_not(keep_inds)],
    np.log10(results['estimated_diam'][np.logical_not(keep_inds)]*1e2),
    color='r',
    label='Rejected',
)


ax.set_xlabel('Distance function')
ax.set_ylabel('Diameter at peak SNR [log10(cm)]')
ax.set_title('Distance function versus estimated diameter')
ax.legend()
fig.savefig(rcs_plot_path / 'dist_vs_diam.png')
plt.close(fig)
