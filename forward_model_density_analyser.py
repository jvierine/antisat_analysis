#!/usr/bin/env python

import sys
import pathlib
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sio
from matplotlib import cm


from forward_model_density_mapper import generate_population, get_config

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError('Not enough input arguments, need a config file...')
    cfg = pathlib.Path(sys.argv[1])

    config = get_config(cfg)

    base_path = pathlib.Path(config.get('Cache', 'output_folder'))
    data_path = base_path / 'simulation' / 'master' / 'collected_results.npy'

    pop = generate_population(config)

    data = np.load(data_path)

    bins = 30
    cols = ['oid', 't', 'range', 'range_rate', 'snr']

    # ['oid', 'm', 'd', 'A', 'C_R', 'C_D'],
    # ['a', 'e', 'i', 'raan', 'aop', 'mu0']

    fig, ax = plt.subplots(1, 1)
    ax.hist(np.log10(pop['d']))
    ax.set_xlabel('Diameter [log10(m)]')
    ax.set_ylabel('Frequency [1]')

    x = data['range']*1e-3
    y = data['range_rate']*1e-3
    z = pop['i'][data['oid']]
    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), num=100),
        np.linspace(y.min(), y.max(), num=100),
    )

    i_vec = np.linspace(
        config.getfloat('Population', 'i_min_deg'),
        config.getfloat('Population', 'i_max_deg'),
        num=config.getint('Population', 'i_samples'),
    )

    fig, ax = plt.subplots(1, 1)

    ax.scatter(x, y, 4, z, alpha=0.1)
    ax.set_xlabel('Range [km]')
    ax.set_ylabel('Range rate [km/s]')

    plt.show()
    exit()
    
    points = np.vstack((x, y)).T
    grid_z = sio.griddata(points, z, (grid_x, grid_y), method='linear')

    fig, ax = plt.subplots(1, 1)
    pmesh = ax.pcolormesh(grid_x, grid_y, grid_z)
    ax.set_xlabel('Range [km]')
    ax.set_ylabel('Range rate [km/s]')
    fig.colorbar(pmesh, ax=ax)

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].hist2d(pop['i'][data['oid']], data['t']/3600.0, bins=bins)
    axes[0, 0].set_xlabel('Inclination [deg]')
    axes[0, 0].set_ylabel('Time [h]')

    axes[0, 1].hist2d(pop['i'][data['oid']], data['range']*1e-3, bins=bins)
    axes[0, 1].set_xlabel('Inclination [deg]')
    axes[0, 1].set_ylabel('Range [km]')

    axes[1, 0].hist2d(pop['i'][data['oid']], data['range_rate']*1e-3, bins=bins)
    axes[1, 0].set_xlabel('Inclination [deg]')
    axes[1, 0].set_ylabel('Range rate [km/s]')

    axes[1, 1].hist2d(pop['i'][data['oid']], 10*np.log(data['snr']), bins=bins)
    axes[1, 1].set_xlabel('Inclination [deg]')
    axes[1, 1].set_ylabel('SNR [dB]')

    # fig, axes = plt.subplots(2, 2)
    # axes[0, 0].hist(data['t']/3600.0)
    # axes[0, 0].set_xlabel('Time [h]')
    # axes[0, 0].set_ylabel('Frequency [1]')

    # axes[0, 1].plot(data['t']/3600.0, data['range']*1e-3, '.')
    # axes[0, 1].set_xlabel('Time [h]')
    # axes[0, 1].set_ylabel('Range [km]')

    # axes[1, 0].plot(data['t']/3600.0, data['range_rate']*1e-3, '.')
    # axes[1, 0].set_xlabel('Time [h]')
    # axes[1, 0].set_ylabel('Range rate [km/s]')

    # axes[1, 1].plot(data['t']/3600.0, 10*np.log(data['snr']), '.')
    # axes[1, 1].set_xlabel('Time [h]')
    # axes[1, 1].set_ylabel('SNR [dB]')

    plt.show()
