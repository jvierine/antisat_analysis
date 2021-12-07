
import argparse
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import h5py
from astropy.time import Time

def plot_measurement_data(observed_data, simulated_data, axes=None):
    '''Plot the observed and simulated population object measurment parameters.
    '''
    t = observed_data['t']
    r = observed_data['r']
    v = observed_data['v']
    r_ref = simulated_data['r_ref']
    v_ref = simulated_data['v_ref']

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 15))
    else:
        fig = None

    axes[0].plot(t - t[0], r*1e-3, label='measurement')
    axes[0].plot(t - t[0], r_ref*1e-3, label='simulation')
    axes[0].set_ylabel('Range [km]')
    axes[0].set_xlabel('Time [s]')

    axes[1].plot(t - t[0], v*1e-3, label='measurement')
    axes[1].plot(t - t[0], v_ref*1e-3, label='simulation')
    axes[1].set_ylabel('Velocity [km/s]')
    axes[1].set_xlabel('Time [s]')

    axes[1].legend()

    return fig, axes


def plot_correlation_residuals(observed_data, simulated_data, axes=None):
    '''Plot the correlation residuals between the measurement and simulated population object.
    '''
    t = observed_data['t']
    r = observed_data['r']
    v = observed_data['v']
    r_ref = simulated_data['r_ref']
    v_ref = simulated_data['v_ref']

    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    else:
        fig = None
    
    axes[0, 0].hist((r_ref - r)*1e-3)
    axes[0, 0].set_xlabel('Range residuals [km]')

    axes[0, 1].hist((v_ref - v)*1e-3)
    axes[0, 1].set_xlabel('Velocity residuals [km/s]')
    
    axes[1, 0].plot(t - t[0], (r_ref - r)*1e-3)
    axes[1, 0].set_ylabel('Range residuals [km]')
    axes[1, 0].set_xlabel('Time [s]')

    axes[1, 1].plot(t - t[0], (v_ref - v)*1e-3)
    axes[1, 1].set_ylabel('Velocity residuals [km/s]')
    axes[1, 1].set_xlabel('Time [s]')

    return fig, axes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyse beampark correlation for a beampark')
    parser.add_argument('input', type=str, help='Input correlation data')

    args = parser.parse_args()

    input_pth = pathlib.Path(args.input).resolve()

    with open(input_pth, 'rb') as fh:
        indecies, metric, cdat = pickle.load(fh)

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.plot(metric['dr'], metric['dv'], '.b')
    ax.set_xlabel('Range residuals [m]')
    ax.set_ylabel('Range-rate residuals [m/s]')

    # fig, axes = plt.subplots(2, 3, figsize=(15, 15))
    # plot_measurement_data(dat, cdat[0][0], axes=axes[:, 0],)
    # plot_correlation_residuals(dat, cdat[0][0], axes=axes[:, 1:])

    plt.show()
