#!/usr/bin/env python

'''
Example execution 

RUN:

PLOT:

./beampark_elevation_doppler.py plot cache/**/az*.h5 -o ./plots/
./beampark_elevation_doppler.py plot -d /home/danielk/data/spade/beamparks/uhf/2021.11.23.h5 -o ./plots/ -- ./cache/eiscat_uhf/az90.0_el75.0.h5

'''
import sys
import datetime
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import Time, TimeDelta
import matplotlib.dates as mdates
import h5py


HERE = Path(__file__).parent


def load_data(file):
    with h5py.File(file, "r") as h:
        v = h["v"][()]
        t = h["t"][()]
        r = h["r"][()]    
        snr = h["snr"][()]
        dur = h["dur"][()]
        diam = h["diams"][()]

    return t, r, v, snr, dur, diam


def set_mdates(ax):
    locator = mdates.AutoDateLocator(minticks=12, maxticks=18)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')


def count_hist(t, start_time, end_time, bin_size = 1800.0):

    h0 = Time(datetime.datetime(
        start_time.datetime.year, 
        start_time.datetime.month, 
        start_time.datetime.day, 
        start_time.datetime.hour, 
        0, 
        0,
    )).unix
    h1 = Time(datetime.datetime(
        end_time.datetime.year, 
        end_time.datetime.month, 
        end_time.datetime.day, 
        end_time.datetime.hour, 
        0, 
        0,
    ))
    h1 += TimeDelta(3600, format='sec')
    h1 = h1.unix

    h0_missing = (np.min(t) - h0)/bin_size
    if h0_missing > 1:
        h0 += int(h0_missing)*bin_size
        h0_missing -= int(h0_missing)

    h1_missing = (h1 - np.max(t))/bin_size
    if h1_missing > 1:
        h1 -= int(h1_missing)*bin_size
        h1_missing -= int(h1_missing)

    bins = np.arange(h0, h1+bin_size, bin_size)
    weights = np.ones(bins.size - 1)
    weights[0] = 1 - h0_missing
    weights[-1] = 1 - h1_missing

    hist, bin_edges = np.histogram(t, bins=bins)

    hist = hist/weights
    bin_centers = Time((bin_edges[:-1] + bin_edges[1:])*0.5, format='unix', scale='utc')
    bin_centers = bin_centers.datetime

    return bin_centers, hist


def save_fig(name, save_path, fig, radar_escaped, fig_name):
    fout = f'{radar_escaped}_{name}_{fig_name}.png'
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path / fout)
        plt.close(fig)


def time_name_formatting(t, radar):
    radar_escaped = radar.lower().replace(' ', '_')
    t_dt = Time(t, format='unix', scale='utc').datetime
    start_time = Time(np.min(t), format='unix', scale='utc')
    end_time = Time(np.max(t), format='unix', scale='utc')
    fig_name = str(start_time.datetime.date())

    return radar_escaped, t_dt, start_time, end_time, fig_name


def plot_statistics(data, radar, save_path=None, size=(10, 8)):

    t, r, v, snr, dur, diam = data
    radar_escaped, t_dt, start_time, end_time, fig_name = time_name_formatting(t, radar)

    filt = np.logical_and(v > -3, v < 3)

    bins = int(np.round(np.sqrt(len(r))))

    fig, axes = plt.subplots(2, 2, figsize=size, sharey='all')
    
    axes[0, 0].hist(r[filt], bins=bins)
    axes[0, 0].set_xlabel("Range [km]")
    axes[0, 0].set_ylabel("Detections")

    axes[0, 1].hist(v[filt], bins=bins)
    axes[0, 1].set_xlabel("Range-rate [km]")
    axes[0, 1].set_ylabel("Detections")

    axes[1, 0].hist(10*np.log10(snr[filt]), bins=bins)
    axes[1, 0].set_xlabel("SNR [dB]")
    axes[1, 0].set_ylabel("Detections")

    _diam = diam[filt]
    sel = np.logical_or(np.isinf(_diam), np.isnan(_diam))
    sel = np.logical_not(sel)
    _diam = _diam[sel]

    axes[1, 1].hist(np.log10(_diam), bins=bins)
    axes[1, 1].set_xlabel("Diameter [log10(cm)]")
    axes[1, 1].set_ylabel("Detections")

    fig.suptitle(f'Observation statistics at {radar} {start_time.iso}')
    save_fig('stat', save_path, fig, radar_escaped, fig_name)


def plot_detections(data, radar, save_path=None, size=(10, 8)):

    t, r, v, snr, dur, diam = data
    radar_escaped, t_dt, start_time, end_time, fig_name = time_name_formatting(t, radar)

    bin_size = 1800.0
    bin_centers, hist = count_hist(t, start_time, end_time, bin_size=bin_size)

    fig, axes = plt.subplots(3, 1, sharex='all', figsize=size)

    axes[0].bar(bin_centers, hist, width=datetime.timedelta(seconds=bin_size))
    axes[0].set_ylabel("Detections [per 30 min]")

    axes[1].scatter(t_dt, r, 4)
    axes[1].set_ylabel('Range [km]')
    axes[1].set_ylim([200, 2000])

    axes[2].scatter(t_dt, v, 4)
    axes[2].set_ylabel('Range-rate [km/s]')
    axes[2].set_xlabel('Time')
    axes[2].set_ylim([-3, 3])

    set_mdates(axes[2])

    fig.suptitle(f'Observations at {radar} {start_time.iso}')
    save_fig('obs', save_path, fig, radar_escaped, fig_name)

    fig, ax = plt.subplots(figsize=size)
    ax.bar(bin_centers, hist, width=datetime.timedelta(seconds=bin_size))
    ax.set_ylabel("Detections [per 30 min]")
    ax.set_xlabel('Time')
    set_mdates(ax)
    fig.suptitle(f'Observations at {radar} {start_time.iso}')
    save_fig('counts', save_path, fig, radar_escaped, fig_name)

    fig, axes = plt.subplots(2, 1, sharex='all', figsize=size)
    axes[0].scatter(t_dt, r, 4)
    axes[0].set_ylabel('Range [km]')
    axes[0].set_ylim([200, 2000])

    axes[1].scatter(t_dt, v, 4)
    axes[1].set_ylabel('Range-rate [km/s]')
    axes[1].set_xlabel('Time')
    axes[1].set_ylim([-3, 3])

    set_mdates(axes[1])
    fig.suptitle(f'Observations at {radar} {start_time.iso}')
    save_fig('obs_vs_t', save_path, fig, radar_escaped, fig_name)

    fig, ax = plt.subplots(figsize=size)
    ax.scatter(r, v, 4)
    ax.set_xlabel('Range [km]')
    ax.set_ylabel('Range-rate [km/s]')
    ax.set_xlim([200, 2000])
    ax.set_ylim([-3, 3])
    fig.suptitle(f'Observations at {radar} {start_time.iso}')
    save_fig('range_range_rate', save_path, fig, radar_escaped, fig_name)


def main():

    parser = argparse.ArgumentParser(
        description='Plot the observation data for a beampark',
    )
    parser.add_argument('-px', '--plot-width', default=10, type=int, help='Plot width in inches')
    parser.add_argument('-py', '--plot-height', default=6, type=int, help='Plot height in inches')
    parser.add_argument('-d', '--dark', default=False, action='store_true', help='Dark-mode plots')
    parser.add_argument('-o', '--output', type=str, default=None, help='Directory to save plots in')
    parser.add_argument('-r', '--radar', type=str, default='', help='Name of the radar')
    parser.add_argument(
        '-l', '--locale-dates', 
        default=False, 
        action='store_true',
        help='Do not override the locale',
    )
    parser.add_argument('data', type=str, nargs='+', help='Path(s) to measurement data to plot')

    args = parser.parse_args()

    if args.dark:
        plt.style.use('dark_background')
    else:
        plt.style.use('tableau-colorblind10')

    if not args.locale_dates:
        import locale
        locale.setlocale(locale.LC_TIME, 'en_US.utf8')

    for file in args.data:
        file = Path(file)
        print(f'Plotting {file}...')

        if args.output is not None:
            output_path = Path(args.output).resolve()
        else:
            output_path = Path('.').resolve()

        data = load_data(file)
        plot_statistics(
            data, 
            radar = args.radar, 
            save_path=output_path, 
            size=(args.plot_width, args.plot_height),
        )
        plot_detections(
            data, 
            radar = args.radar, 
            save_path=output_path, 
            size=(args.plot_width, args.plot_height),
        )


if __name__ == '__main__':
    main()
