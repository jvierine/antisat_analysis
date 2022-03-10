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


with open(OUTPUT / 'paths.pickle', 'rb') as fh:
    paths = pickle.load(fh)


category_files = {}
dates = {}
for radar in paths['data_paths']:
    category_files[radar] = []
    dates[radar] = []
    for ind in range(len(paths['data_paths'][radar])):
        date = paths['data_paths'][radar][ind].stem.replace('.', '-')
        dates[radar].append(date)
        out_categories = OUTPUT / f'{date}_{radar}_categories.npy'
        category_files[radar].append(out_categories)
pprint(category_files)


save_paths = {}
for radar in dates:
    save_paths[radar] = []
    for date in dates[radar]:
        save_path = OUTPUT / f'{date}_{radar}_kosmos_stat'
        save_path.mkdir(exist_ok=True)
        save_paths[radar].append(save_path)


radar_title = {
    'esr': [
        'EISCAT Svalbard Radar 32m',
        'EISCAT Svalbard Radar 32m',
        'EISCAT Svalbard Radar 42m',
        'EISCAT Svalbard Radar 42m',
    ],
    'uhf': ['EISCAT UHF']*3,
}


def load_data(file):
    with h5py.File(file, "r") as h:
        v = h["v"][()]
        t = h["t"][()]
        r = h["r"][()]    
        snr = h["snr"][()]
        dur = h["dur"][()]
        diam = h["diams"][()]

    return t, r, v, snr, dur, diam


def plot_statistics(
            data, select, radar, 
            ind, outp, 
            include_total=False, size=(10, 8),
        ):

    t, r, v, snr, dur, diam = data
    snrdb = 10*np.log10(snr)
    
    bins = int(np.round(np.sqrt(len(r))))

    fig, axes = plt.subplots(2, 2, figsize=size, sharey='all')
    
    col = 'r' if include_total else 'b'
    
    bin_edges = np.linspace(np.nanmin(r), np.nanmax(r), num=bins + 1, endpoint=True)
    if include_total:
        axes[0, 0].hist(r, bins=bin_edges, color='b', label='Total')
    axes[0, 0].hist(r[select], bins=bin_edges, color=col, label='KOSMOS-1408')
    axes[0, 0].set_xlabel("Range [km]")
    axes[0, 0].set_ylabel("Detections")
    if include_total: 
        axes[0, 0].legend()

    bin_edges = np.linspace(np.nanmin(v), np.nanmax(v), num=bins + 1, endpoint=True)
    if include_total:
        axes[0, 1].hist(v, bins=bin_edges, color='b')
    axes[0, 1].hist(v[select], bins=bin_edges, color=col)
    axes[0, 1].set_xlabel("Range-rate [km]")
    axes[0, 1].set_ylabel("Detections")

    bin_edges = np.linspace(np.nanmin(snrdb), np.nanmax(snrdb), num=bins + 1, endpoint=True)
    if include_total:
        axes[1, 0].hist(snrdb, bins=bin_edges, color='b')
    axes[1, 0].hist(snrdb[select], bins=bin_edges, color=col)
    axes[1, 0].set_xlabel("SNR [dB]")
    axes[1, 0].set_ylabel("Detections")
    
    ok_vals = np.logical_or(np.isinf(diam), np.isnan(diam))
    ok_vals = np.logical_not(ok_vals)
    ok_vals_select = np.logical_and(select, ok_vals)
    _logdiam = np.log10(diam[ok_vals])
    
    bin_edges = np.linspace(np.nanmin(_logdiam), np.nanmax(_logdiam), num=bins + 1, endpoint=True)
    if include_total:
        axes[1, 1].hist(_logdiam, bins=bin_edges, color='b')
    axes[1, 1].hist(np.log10(diam[ok_vals_select]), bins=bin_edges, color=col)
    axes[1, 1].set_xlabel("Minimum diameter [log10(cm)]")
    axes[1, 1].set_ylabel("Detections")
    
    fig.suptitle(f'Detections {radar_title[radar][ind]} {dates[radar][ind]}')
    if include_total:
        fig.savefig(outp / f'{date}_{radar}_stat.png')
    else:
        fig.savefig(outp / f'{date}_{radar}_kosmos_stat.png')
    plt.close(fig)


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


for radar in save_paths:
    for ind in range(len(save_paths[radar])):
        select = np.load(category_files[radar][ind])
        data_file = paths['data_paths'][radar][ind]

        plot_statistics(
            load_data(data_file), 
            kosmos_select(select), 
            radar, 
            ind, 
            save_paths[radar][ind],
            include_total=False,
        )
        plot_statistics(
            load_data(data_file), 
            kosmos_select(select), 
            radar, 
            ind, 
            save_paths[radar][ind],
            include_total=True,
        )
