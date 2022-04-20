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

fig_format = 'eps'
# fig_format = 'png'

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
            data, select, fname, title,
            include_total=False, size=(10, 8),
            background=False,
        ):

    t, r, v, snr, dur, diam = data
    snrdb = 10*np.log10(snr)
    
    bins = int(np.round(np.sqrt(len(r))))

    fig, axes = plt.subplots(2, 2, figsize=size, sharey='all')
    
    col = 'r' if include_total else 'b'
    
    bin_edges = np.linspace(np.nanmin(r), np.min([np.nanmax(r), 2800]), num=bins + 1, endpoint=True)
    if include_total:
        if background:
            axes[0, 0].hist(r[np.logical_not(select)], bins=bin_edges, color='b', label='Background')
        else:
            axes[0, 0].hist(r, bins=bin_edges, color='b', label='Total')
    axes[0, 0].hist(r[select], bins=bin_edges, color=col, label='KOSMOS-1408')
    axes[0, 0].set_xlabel("Range [km]")
    axes[0, 0].set_ylabel("Detections")
    if include_total: 
        axes[0, 0].legend()

    bin_edges = np.linspace(np.max([np.nanmin(v), -2.5]), np.min([np.nanmax(v), 2.5]), num=bins + 1, endpoint=True)
    if include_total:
        if background:
            axes[0, 1].hist(v[np.logical_not(select)], bins=bin_edges, color='b')
        else:
            axes[0, 1].hist(v, bins=bin_edges, color='b')
    axes[0, 1].hist(v[select], bins=bin_edges, color=col)
    axes[0, 1].set_xlabel("Range-rate [km]")
    axes[0, 1].set_ylabel("Detections")

    bin_edges = np.linspace(np.nanmin(snrdb), np.nanmax(snrdb), num=bins + 1, endpoint=True)
    if include_total:
        if background:
            axes[1, 0].hist(snrdb[np.logical_not(select)], bins=bin_edges, color='b')
        else:
            axes[1, 0].hist(snrdb, bins=bin_edges, color='b')
    axes[1, 0].hist(snrdb[select], bins=bin_edges, color=col)
    axes[1, 0].set_xlabel("SNR [dB]")
    axes[1, 0].set_ylabel("Detections")
    
    ok_vals = np.logical_or(np.isinf(diam), np.isnan(diam))
    ok_vals = np.logical_not(ok_vals)
    ok_vals_select = np.logical_and(select, ok_vals)
    ok_vals_not_select = np.logical_and(np.logical_not(select), ok_vals)
    _logdiam = np.log10(diam[ok_vals])
    
    bin_edges = np.linspace(np.nanmin(_logdiam), np.nanmax(_logdiam), num=bins + 1, endpoint=True)
    if include_total:
        if background:
            axes[1, 1].hist(np.log10(diam[ok_vals_not_select]), bins=bin_edges, color='b')
        else:
            axes[1, 1].hist(_logdiam, bins=bin_edges, color='b')
    axes[1, 1].hist(np.log10(diam[ok_vals_select]), bins=bin_edges, color=col)
    axes[1, 1].set_xlabel("Minimum diameter [log10(cm)]")
    axes[1, 1].set_ylabel("Detections")
    
    fig.suptitle(title)
    fig.savefig(fname)
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


data_type = {
    'uhf': [
        ['90.0 Az 75.0 El EISCAT UHF 2021-11-[23,25,29]', (0, 1, 2)],
    ],
    'esr': [
        ['90.0 Az 75.0 El ESR-32m 2021-11-[19]', (0,)],
        ['96.4 Az 82.1 El ESR-32m 2021-11-[23]', (1,)],
        ['185.5 Az 82.1 El ESR-42m 2021-11-[25,29]', (2, 3)],
    ],
}

pprint(data_type)


def escape(x):
    return x.strip()\
        .replace(' ', '_')\
        .replace('-', '_')\
        .replace('[', '')\
        .replace(']', '')\
        .replace(',', '-')


tot_data = {}
for radar in data_type:
    tot_data[radar] = []
    for ind in range(len(data_type[radar])):
        tot_data[radar].append([[], [list() for x in range(6)]])

for radar in save_paths:
    for ind in range(len(save_paths[radar])):
        for typi, inds in enumerate(data_type[radar]):
            if ind in inds[1]:
                list_ptr = typi
                break
        select = np.load(category_files[radar][ind])
        tot_data[radar][list_ptr][0].append(select)
        data_file = paths['data_paths'][radar][ind]
        data = load_data(data_file)
        for x in range(6):
            tot_data[radar][list_ptr][1][x].append(data[x])

        for include_total in [True, False]:
            kosm = 'kosmos_' if include_total else ''

            plot_statistics(
                data, 
                kosmos_select(select), 
                save_paths[radar][ind] / f'{dates[radar][ind]}_{radar}_{kosm}stat.{fig_format}',
                f'Detections {radar_title[radar][ind]} {dates[radar][ind]}',
                include_total=include_total,
            )
        plot_statistics(
            data, 
            kosmos_select(select), 
            save_paths[radar][ind] / f'{dates[radar][ind]}_{radar}_kosmos_bg_stat.{fig_format}',
            f'Detections {radar_title[radar][ind]} {dates[radar][ind]}',
            include_total=True,
            background=True,
        )

tot_size_dist = {
    'kosmos': [],
    'background': [],
}

for radar in data_type:
    for ind in range(len(data_type[radar])):
        for x in range(6):
            tot_data[radar][ind][1][x] = np.concatenate(tot_data[radar][ind][1][x])
        type_data = tot_data[radar][ind][1]
        type_select = np.concatenate(tot_data[radar][ind][0])

        t, r, v, snr, dur, diam = type_data
        select = kosmos_select(type_select)
        tot_size_dist['kosmos'].append(diam[select])
        tot_size_dist['background'].append(diam[np.logical_not(select)])

        for include_total in [True, False]:
            kosm = 'kosmos_' if include_total else ''
            name = data_type[radar][ind][0]
            plot_statistics(
                type_data, 
                select, 
                tot_pth / f'{escape(name)}_{kosm}stat.{fig_format}',
                data_type[radar][ind][0],
                include_total=include_total,
            )
        plot_statistics(
            type_data, 
            select, 
            tot_pth / f'{escape(name)}_kosmos_bg_stat.{fig_format}',
            data_type[radar][ind][0],
            include_total=True,
            background=True,
        )

tot_size_dist['kosmos'] = np.concatenate(tot_size_dist['kosmos'])
tot_size_dist['background'] = np.concatenate(tot_size_dist['background'])

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for axi, key in enumerate(['background', 'kosmos']):
    ok_vals = np.logical_or(np.isinf(tot_size_dist[key]), np.isnan(tot_size_dist[key]))
    ok_vals = np.logical_not(ok_vals)
    _logdiam = np.log10(tot_size_dist[key][ok_vals])
    bins = int(np.round(np.sqrt(len(_logdiam))))
    bin_edges = np.linspace(np.nanmin(_logdiam), np.nanmax(_logdiam), num=bins + 1, endpoint=True)
    axes[axi].hist(_logdiam, bins=bin_edges, color='b')

axes[1].set_xlabel("Minimum diameter [log10(cm)]")
axes[0].set_ylabel("Detections")
axes[1].set_ylabel("Detections")
axes[0].set_title('All background detections')
axes[1].set_title('All KOSMOS-1408 detections')

fig.savefig(tot_pth / f'kosmos_min_diam_stat.{fig_format}')
plt.close(fig)
