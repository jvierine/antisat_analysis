#!/usr/bin/env python

'''
Correlating data with TLE catalog
===================================

'''
import argparse
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import h5py
from astropy.time import Time

import sorts

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

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


def vector_diff_metric(t, r, v, r_ref, v_ref, **kwargs):
    '''Return a vector of measurnment error scaled residuals for range and range rate.
    '''
    r_std = kwargs.get('r_std')
    v_std = kwargs.get('v_std')

    dr = (r_ref - r)/r_std
    dv = (v_ref - v)/v_std
    
    ret = np.empty((len(t),), dtype=[('dr', np.float64), ('dv', np.float64)])
    ret['dr'] = dr
    ret['dv'] = dv

    return ret


def sorting_function(metric):
    '''How to sort the matches using a weight between range and range rate.
    '''
    P = np.abs(metric['dr']) + 100*np.abs(metric['dv'])
    return np.argsort(P, axis=0)


def run():

    if comm is not None:
        print('MPI detected')
        if comm.size > 1:
            print('Using MPI...')

    parser = argparse.ArgumentParser(description='Calculate doppler-inclination correlation for a beampark')
    parser.add_argument('radar', type=str, help='The observing radar system')
    parser.add_argument('catalog', type=str, help='TLE catalog path')
    parser.add_argument('input', type=str, help='Observation data location')
    parser.add_argument('output', type=str, help='Results output location')
    parser.add_argument('-o', '--override', action='store_true', help='Override output location if it exists')

    args = parser.parse_args()

    radar = getattr(sorts.radars, args.radar)

    tle_pth = pathlib.Path(args.catalog).resolve()
    output_pth = pathlib.Path(args.output).resolve()
    input_pth = pathlib.Path(args.input).resolve()
    
    if output_pth.is_file() and not args.override:
        dat = {}
        print('Using cached data, not reading measurements file')
    else:
        # Each entry in the input `measurements` list must be a dictionary that contains the following fields:
        #   * 't': [numpy.ndarray] Times relative epoch in seconds
        #   * 'r': [numpy.ndarray] Two-way ranges in meters
        #   * 'v': [numpy.ndarray] Two-way range-rates in meters per second
        #   * 'epoch': [astropy.Time] epoch for measurements
        #   * 'tx': [sorts.TX] Pointer to the TX station
        #   * 'rx': [sorts.RX] Pointer to the RX station
        print('Loading monostatic measurements')
        with h5py.File(str(input_pth), 'r') as h_det:
            r = h_det['r'][()]*1e3  # km -> m, one way
            t = h_det['t'][()]  # Unix seconds
            v = h_det['v'][()]*1e3  # km/s -> m/s
            v = -v  # Inverted definition of range rate, one way

            inds = np.argsort(t)
            inds = inds[:12] #FOR DEBUG
            t = t[inds]
            r = r[inds]
            v = v[inds]

            t = Time(t, format='unix', scale='utc')
            epoch = t[0]
            t = (t - epoch).sec

            # ########################
            # 
            # This is the interface 
            # layer between the input
            # data type and the 
            # correlator
            #
            # #######################
            dat = {
                'r': r*2,
                'v': v*2,
                'r_std': r*2*0.01,
                'v_std': v*2*0.01,
                't': t,
                'epoch': epoch,
                'tx': radar.tx[0],
                'rx': radar.rx[0],
                'measurement_num': len(t),
            }

    print('Loading TLE population')
    pop = sorts.population.tle_catalog(tle_pth, cartesian=False)

    # correlate requires output in ECEF 
    pop.out_frame = 'ITRS'

    print(f'Population size: {len(pop)}')
    print('Correlating data and population')

    if output_pth.is_file() and not args.override:
        print('Loading correlation data from cache')
        with open(output_pth, 'rb') as fh:
            indecies, metric, cdat = pickle.load(fh)
    else:
        if comm is not None:
            comm.barrier()

        indecies, metric, cdat = sorts.correlate(
            measurements = [dat],
            population = pop,
            n_closest = 1,
            meta_variables=[
                'r_std',
                'v_std',
            ],
            metric=vector_diff_metric, 
            sorting_function=sorting_function,
            metric_dtype=[('dr', np.float64), ('dv', np.float64)],
            metric_reduce=None,
        )

        if comm is None or comm.rank == 0:
            with open(output_pth, 'wb') as fh:
                 pickle.dump([indecies, metric, cdat], fh)

    if comm is None or comm.rank == 0:
        print('Individual measurement match metric:')
        for mind, (ind, dst) in enumerate(zip(indecies.T, metric.T)):
            print(f'measurement = {mind}')
            for res in range(len(ind)):
                print(f'-- result rank = {res} | object ind = {ind[res]} | metric = {dst[res]} | obj = {pop["oid"][ind[res]]}')

        # fig, axes = plt.subplots(2, 3, figsize=(15, 15))
        # plot_measurement_data(dat, cdat[0][0], axes=axes[:, 0],)
        # plot_correlation_residuals(dat, cdat[0][0], axes=axes[:, 1:])
        # plt.show()


if __name__ == '__main__':
    run()