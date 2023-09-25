#!/usr/bin/env python

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import h5py
from astropy.time import Time

import sorts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pick starlinks')
    parser.add_argument('measurements', type=str, help='Input correlation data')
    parser.add_argument('correlations', type=str, help='Input correlation data')
    parser.add_argument('tle', type=str, help='Input tle file')
    parser.add_argument('tle_select', type=str, help='Input subpop tle file')
    parser.add_argument('selected_correlations', type=str, help='Input correlation results')
    parser.add_argument(
        '--output', type=str, default='', help='Output folder for plots and results'
    )
    parser.add_argument('--name', type=str, default='', help='Legend string')

    parser.add_argument(
        '-f', '--format',
        default='png',
        help='Plot format',
    )
    args = parser.parse_args()

    if len(args.output) > 0:
        out_path = pathlib.Path(args.output).resolve()
        out_path.mkdir(exist_ok=True)
    else:
        out_path = None

    measurements_pth = pathlib.Path(args.measurements).resolve()
    correlations_pth = pathlib.Path(args.correlations).resolve()
    tle_pth = pathlib.Path(args.tle).resolve()
    sub_tle_pth = pathlib.Path(args.tle_select).resolve()
    selected_correlations_pth = pathlib.Path(args.selected_correlations).resolve()

    print('Loading monostatic measurements')
    with h5py.File(str(measurements_pth), 'r') as h_det:
        r = h_det['r'][()]
        t = h_det['t'][()]  # Unix seconds
        v = h_det['v'][()]

        inds_sort = np.argsort(t)

        epoch = Time(t.min(), format='unix')
        t = t[inds_sort] - t.min()
        r = r[inds_sort]
        v = v[inds_sort]

    with h5py.File(correlations_pth, 'r') as ds:
        indecies = ds['matched_object_index'][()]
        metric = ds['matched_object_metric'][()]
        name = ds.attrs['radar_name']

    print(f"Loading {tle_pth}")
    pop = sorts.population.tle_catalog(tle_pth, cartesian=False)
    pop.unique(target_epoch=None)

    print(f"Loading {sub_tle_pth}")
    sub_pop = sorts.population.tle_catalog(sub_tle_pth, cartesian=False)
    sub_pop.unique(target_epoch=None)

    select = np.load(selected_correlations_pth)
    not_select = np.logical_not(select)

    select_oids = pop["oid"][indecies[0]]
    in_sub = np.full(select_oids.shape, False, dtype=bool)
    for ind, oid in enumerate(select_oids):
        in_sub[ind] = oid in sub_pop["oid"]

    select_sub = np.logical_and(select, in_sub)
    select_nsub = np.logical_and(select, np.logical_not(in_sub))

    title_str = f'{name.upper().replace("_", " ")} - {epoch.datetime.date()}'

    fig, axes = plt.subplots(2, 1, figsize=(15, 15))

    ax = axes[0]
    ax.plot(t[select_nsub]/3600.0, r[select_nsub], '.r', label='Correlated - other')
    ax.plot(t[not_select]/3600.0, r[not_select], '.b', label='Uncorrelated')
    ax.plot(t[select_sub]/3600.0, r[select_sub], 'xg', label=f'Correlated - {args.name}')
    ax.legend()
    ax.set_ylabel('Range [km]')
    ax.set_xlabel('Time [h]')
    ax.set_title(title_str)

    ax = axes[1]
    ax.plot(t[select_nsub]/3600.0, v[select_nsub], '.r', label='Correlated - other')
    ax.plot(t[not_select]/3600.0, v[not_select], '.b', label='Uncorrelated')
    ax.plot(t[select_sub]/3600.0, v[select_sub], 'xg', label=f'Correlated - {args.name}')

    ax.set_ylabel('Range-rate [km/s]')
    ax.set_xlabel('Time [h]')

    if out_path is not None:
        fig.savefig(out_path / f'{name}_rv_t_correlations_subpopulation.{args.format}')
        plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.plot(r[select_nsub], v[select_nsub], '.r', label='Correlated - other')
    ax.plot(r[not_select], v[not_select], '.b', label='Uncorrelated')
    ax.plot(r[select_sub], v[select_sub], 'xg', label=f'Correlated - {args.name}')
    ax.legend()
    ax.set_xlabel('Range [km]')
    ax.set_ylabel('Range-rate [km/s]')
    ax.set_title(title_str)

    if out_path is not None:
        fig.savefig(out_path / f'{name}_rv_correlations_subpopulation.{args.format}')
        plt.close(fig)

    if out_path is None:
        plt.show()
