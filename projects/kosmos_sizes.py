from astropy.time import Time, TimeDelta
import h5py
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

from pprint import pprint

import sorts
import pyant

import sys

'''

python projects/kosmos_sizes.py 
python projects/kosmos_sizes.py trunc

'''

if len(sys.argv) > 1:
    arg = sys.argv[1]
else:
    arg = ''

min_gain_filter = 25.0  # dB
filter_limit = 0.010

TRUNC_ANALYSIS = False
if arg.lower().strip() == 'trunc':
    TRUNC_ANALYSIS = True

HERE = Path(__file__).parent.resolve()
OUTPUT = HERE / 'output' / 'russian_asat'
print(f'Using {OUTPUT} as output')

tot_pth = OUTPUT / 'all_kosmos_stat'
tot_pth.mkdir(exist_ok=True)

with open(OUTPUT / 'paths.pickle', 'rb') as fh:
    paths = pickle.load(fh)

tx = sorts.radars.eiscat_uhf.tx[0]
d_separatrix = tx.beam.wavelength/(np.pi*np.sqrt(3.0))
tx.beam.sph_point(
    azimuth=90, 
    elevation=75,
)

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

select = np.load(category_files[radar][ind])
data_file = paths['data_paths'][radar][ind]
date = dates[radar][ind]

if TRUNC_ANALYSIS:
    rcs_path = OUTPUT / f'{date.replace("-", ".")}_{radar}_rcs_trunc_gain'
    rcs_plot_path = OUTPUT / f'{date.replace("-", ".")}_{radar}_rcs_trunc_gain_plots'
else:
    rcs_path = OUTPUT / f'{date.replace("-", ".")}_{radar}_rcs'
    rcs_plot_path = OUTPUT / f'{date.replace("-", ".")}_{radar}_rcs_plots'

rcs_plot_path.mkdir(exist_ok=True)


t, r, v, snr, dur, diam = load_data(data_file)
times = Time(t, format='unix', scale='utc')

collected_res = rcs_path / 'collected_results.pickle'
print(collected_res)
with open(collected_res, 'rb') as fh:
    results = pickle.load(fh)

keep_inds = np.logical_and.reduce([
    results['match'] <= filter_limit,
    results['estimated_gain'] >= min_gain_filter,
])


kosmos_cat = kosmos_select(select)
correlated_cat = correlated_select(select)

size_dist = None
kosmos_dist = None
correlated_dist = None
uncorrelated_dist = None
predicted_dist = None
bin_mids = None
kosmos_dist_peaks = []
size_dist_peaks = []
size_means = []
kosmos_means = []
correlated_means = []
meas_inds = []
k_vecs = []

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

    max_snr_ind = results['snr_max_ind'][ind]
    k_vecs.append(results['estimated_path'][ind][:, max_snr_ind])

    diam_dist = results['estimated_diam_prob'][ind]

    if size_dist is None:
        size_dist = np.zeros_like(diam_dist)
        kosmos_dist = np.zeros_like(diam_dist)
        correlated_dist = np.zeros_like(diam_dist)
        uncorrelated_dist = np.zeros_like(diam_dist)
        predicted_dist = np.zeros_like(diam_dist)
        size_bins = np.copy(results['estimated_diam_bins'][ind])
        bin_mids = (size_bins[:-1] + size_bins[1:])*0.5

    mean_d = np.average(
        bin_mids, 
        weights=diam_dist,
    )

    if not np.isnan(results['predicted_diam'][ind]):
        predicted_dist += diam_dist
    if kosmos_cat[meas_ind]:
        kosmos_dist_peaks.append(bin_mids[np.argmax(diam_dist)])
        kosmos_means.append(mean_d)
        kosmos_dist += diam_dist
    if correlated_cat[meas_ind]:
        correlated_means.append(mean_d)
        correlated_dist += diam_dist
    else:
        uncorrelated_dist += diam_dist
    size_dist_peaks.append(bin_mids[np.argmax(diam_dist)])
    size_means.append(mean_d)
    size_dist += diam_dist
meas_inds = np.array(meas_inds)
kosmos_dist_peaks = np.array(kosmos_dist_peaks)
size_dist_peaks = np.array(size_dist_peaks)
k_vecs = np.array(k_vecs).T

kosmos_diams = results['estimated_diam'][kosmos_cat[meas_inds]]

have_predictions = np.logical_not(np.isnan(results['predicted_diam']))
keep_stats = np.logical_and(have_predictions, keep_inds)


fig, ax = plt.subplots(figsize=(12, 8))
ax2 = ax.twinx()

zenith_ang = pyant.coordinates.vector_angle(tx.beam.pointing, k_vecs, radians=False)

beam0 = tx.beam.copy()

beam0.sph_point(
    azimuth=0, 
    elevation=90,
)
theta = np.linspace(90, 85, num=1000)
_kv = np.zeros((3, len(theta)))
_kv[1, :] = theta
_kv[2, :] = 1
_kv = pyant.coordinates.sph_to_cart(_kv, radians=False)
S = beam0.gain(_kv)

p1, = ax2.plot(90 - theta, np.log10(S)*10.0, color='r')
ax.hist(zenith_ang, bins=10)

ax2.yaxis.label.set_color(p1.get_color())
ax2.tick_params(axis='y', colors=p1.get_color())
ax.set_title('Estimated peak SNR location')
ax.set_xlabel('Off-axis angle [deg]')
ax.set_ylabel('Frequency')
ax2.set_ylabel('Gain [dB]')
fig.savefig(rcs_plot_path / 'estimated_offaxis_angles.png')
plt.close(fig)

side_lobe = 0.85
fig, ax = plt.subplots(figsize=(12, 8))
pyant.plotting.gain_heatmap(tx.beam, min_elevation=85.0, ax=ax)
ax.plot(
    k_vecs[0, zenith_ang < side_lobe], 
    k_vecs[1, zenith_ang < side_lobe], 
    '.k', 
    label=f'Main lobe (N={np.sum(zenith_ang < side_lobe)})'
)
ax.plot(
    k_vecs[0, zenith_ang >= side_lobe], 
    k_vecs[1, zenith_ang >= side_lobe], 
    'xk', 
    label=f'Side lobes (N={np.sum(zenith_ang >= side_lobe)})'
)
ax.legend()
ax.set_title('Estimated peak SNR location')
ax.set_xlabel('kx [1]')
ax.set_ylabel('ky [1]')
fig.savefig(rcs_plot_path / 'estimated_k_vecs.png')
plt.close(fig)


logd_separatrix = np.log10(d_separatrix*1e2)
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
_d = np.log10(results['boresight_diam']*1e2)
_rcs = sorts.signals.hard_target_rcs(
    tx.beam.wavelength, 
    results['boresight_diam'],
)
_rcs = np.log10(_rcs*1e4)
axes[0].scatter(
    _d[_d < logd_separatrix],
    _rcs[_d < logd_separatrix],
    color='b',
    label=f'Rayleigh (N={np.sum(_d < logd_separatrix)})',
)
axes[0].scatter(
    _d[_d >= logd_separatrix],
    _rcs[_d >= logd_separatrix],
    color='r',
    label=f'Optical (N={np.sum(_d >= logd_separatrix)})',
)
axes[0].legend()
axes[0].set_title('Minimum diameter')
_d = np.log10(results['estimated_diam'][keep_inds]*1e2)
_rcs = sorts.signals.hard_target_rcs(
    tx.beam.wavelength, 
    results['estimated_diam'][keep_inds],
)
_rcs = np.log10(_rcs*1e4)
axes[1].scatter(
    _d[_d < logd_separatrix],
    _rcs[_d < logd_separatrix],
    color='b',
    label=f'Rayleigh (N={np.sum(_d < logd_separatrix)})',
)
axes[1].scatter(
    _d[_d >= logd_separatrix],
    _rcs[_d >= logd_separatrix],
    color='r',
    label=f'Optical (N={np.sum(_d >= logd_separatrix)})',
)
axes[1].legend()
axes[1].set_title('Estimated diameter')
axes[1].set_xlabel('Diameter at peak SNR [log10(cm)]')
axes[0].set_ylabel('RCS at peak SNR [log10(cm^2)]')
axes[1].set_ylabel('RCS at peak SNR [log10(cm^2)]')
fig.suptitle('Scattering function region comparison')
fig.savefig(rcs_plot_path / 'scattering_estimated_vs_boresight_diam.png')
plt.close(fig)


fig, axes = plt.subplots(1, 3, figsize=(10, 5))
axes[0].hist(
    np.log10(results['predicted_diam'][keep_stats]*1e2),
    bins=int(np.sqrt(np.sum(keep_stats))), 
    color='b',
)
axes[0].set_title('Predicted from TLE')
axes[1].hist(
    np.log10(results['estimated_diam'][keep_stats]*1e2),
    bins=int(np.sqrt(np.sum(keep_stats))), 
    color='b',
)
axes[1].set_title('Estimated')
axes[2].set_title('Fraction [-lower/+higher estimate]')
axes[2].hist(
    np.log10(results['estimated_diam'][keep_stats]/results['predicted_diam'][keep_stats]),
    bins=int(np.sqrt(np.sum(keep_stats))),
    color='b',
)
axes[0].set_xlabel('Diameter [log10(cm)]')
axes[1].set_xlabel('Diameter [log10(cm)]')
axes[2].set_xlabel('Diameter fraction [log10(estimated/predicted)]')
for ax in axes:
    ax.set_ylabel('Frequency [1]')

fig.suptitle('Size distribution estimated versus predicted')
fig.set_tight_layout(True)
fig.savefig(rcs_plot_path / 'predicted_vs_estimated_diam.png')
plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].hist(
    np.log10(results['boresight_diam']*1e2),
    bins=size_bins, 
    color='b',
)
axes[0].set_title('Estimated')
axes[1].hist(
    np.log10(results['event_boresight_diam']*1e2),
    bins=size_bins, 
    color='r',
)
axes[1].set_title('Events file')
axes[1].set_xlabel('Diameter at peak SNR [log10(cm)]')
axes[0].set_ylabel('Frequency [1]')
axes[1].set_ylabel('Frequency [1]')
fig.suptitle('Minimum possible diameter distribution')
fig.savefig(rcs_plot_path / 'events_diam_vs_boresight_diam.png')
plt.close(fig)


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


fig, ax = plt.subplots(figsize=(12, 8))
ax.hist(
    size_dist_peaks,
    bins=size_bins, 
    label='Total',
    color='b',
)
ax.hist(
    kosmos_dist_peaks,
    bins=size_bins, 
    label='KOSMOS-1408',
    color='r',
)

ax.set_xlabel('Diameter at peak SNR [log10(cm)]')
ax.set_ylabel('Frequency (from distance function probability peak) [1]')
ax.set_title('Size distribution from peak of probability distribution')
ax.legend()
fig.savefig(rcs_plot_path / 'kosmos_diam_prob_dist_peak_dist.png')
plt.close(fig)


not_kosmos_picks = np.logical_and(np.logical_not(kosmos_cat)[meas_inds], keep_inds)
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].hist(
    np.log10(results['estimated_diam'][not_kosmos_picks]*1e2),
    bins=size_bins, 
    color='b',
)
axes[0].set_title('Background')
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
axes[0].set_title('Background')
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


fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex='all')
ax = axes[0]
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

ax.axvline(filter_limit, color='r')
ax.set_xlabel('Distance function')
ax.set_ylabel('Diameter at peak SNR [log10(cm)]')
ax.set_title('Distance function versus estimated diameter')
ax.legend()

ax = axes[1]
ax.scatter(
    results['match'][keep_inds],
    results['estimated_gain'][keep_inds],
    color='b',
)
ax.scatter(
    results['match'][np.logical_not(keep_inds)],
    results['estimated_gain'][np.logical_not(keep_inds)],
    color='r',
)

ax.axvline(filter_limit, color='r')
ax.axhline(min_gain_filter, color='r')
ax.set_xlabel('Distance function')
ax.set_ylabel('Gain at peak SNR [dB]')

fig.savefig(rcs_plot_path / 'dist_vs_diam.png')
plt.close(fig)
