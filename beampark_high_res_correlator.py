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

'''
Examples:

python beampark_correlator.py eiscat_uhf ~/data/spade/beamparks/uhf/2015.10.22/space-track.tles ~/data/spade/beamparks/uhf/2015.10.22/h5 ~/data/spade/beamparks/uhf/2015.10.22/correlation.pickle
mpirun -n 6 ./beampark_correlator.py eiscat_uhf ~/data/spade/beamparks/uhf/2015.10.22/space-track.tles ~/data/spade/beamparks/uhf/2015.10.22/h5 ~/data/spade/beamparks/uhf/2015.10.22/correlation.pickle -o
'''

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None


def vector_diff_metric(t, r, v, r_ref, v_ref, **kwargs):
    '''Return a vector of residuals for range and range rate.
    '''
    ret = np.empty((len(t),), dtype=[('dr', np.float64), ('dv', np.float64)])
    ret['dr'] = r_ref - r
    ret['dv'] = v_ref - v
    return ret


def vector_diff_metric_std_normalized(t, r, v, r_ref, v_ref, **kwargs):
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

    # TODO: 
    # Make this able to switch between events.txt format (1 point / object) and high res mode (multiple points / object)
 
    parser = argparse.ArgumentParser(description='Calculate TLE catalog correlation for a beampark')
    parser.add_argument('radar', type=str, help='The observing radar system')
    parser.add_argument('catalog', type=str, help='TLE catalog path')
    parser.add_argument('input', type=str, help='Observation data folder')
    parser.add_argument('output', type=str, help='Results output folder')
    parser.add_argument('-o', '--override', action='store_true', help='Override output location if it exists')
    parser.add_argument('--std', action='store_true', help='Use measurement errors')

    args = parser.parse_args()

    if args.std:
        metric = vector_diff_metric_std_normalized
    else:
        metric = vector_diff_metric

    radar = getattr(sorts.radars, args.radar)

    tle_pth = pathlib.Path(args.catalog).resolve()
    output_pth = pathlib.Path(args.output).resolve()
    input_pth = pathlib.Path(args.input).resolve()
    
    if args.std:
        meta_vars = [
            'r_std',
            'v_std',
        ]
    else:
        meta_vars = []


    print('Loading TLE population')
    pop = sorts.population.tle_catalog(tle_pth, cartesian=False)
    print(f'Population size: {len(pop)}')

    # correlate requires output in ECEF 
    pop.out_frame = 'ITRS'

    input_files = list(input_pth.glob('**/*.h5'))

    for fid, file in enumerate(input_files):

        output = output_pth / f'input{fid}.pickle'

        if output.is_file() and not args.override:
            print('Using cached data, not reading measurements file')
            continue
        else:

            # Each entry in the input `measurements` list must be a dictionary that contains the following fields:
            #   * 't': [numpy.ndarray] Times relative epoch in seconds
            #   * 'r': [numpy.ndarray] Two-way ranges in meters
            #   * 'v': [numpy.ndarray] Two-way range-rates in meters per second
            #   * 'epoch': [astropy.Time] epoch for measurements
            #   * 'tx': [sorts.TX] Pointer to the TX station
            #   * 'rx': [sorts.RX] Pointer to the RX station
            print(f'Loading monostatic measurements {file}')
            with h5py.File(str(file), 'r') as h_det:
                r = h_det['r'][()]*1e3  # km -> m, one way
                t = h_det['t'][()]  # Unix seconds
                v = h_det['v'][()]*1e3  # km/s -> m/s

                inds = np.argsort(t)
                # inds = inds[:12] #FOR DEBUG
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
                    't': t,
                    'epoch': epoch,
                    'tx': radar.tx[0],
                    'rx': radar.rx[0],
                    'measurement_num': len(t),
                }

                if args.std:
                    dat['r_std'] = h_det['r_std'][()]*1e3
                    dat['v_std'] = h_det['v_std'][()]*1e3

                measurements = [dat]

        print('Correlating data and population')

        if output.is_file() and not args.override:
            print('Loading correlation data from cache')
            with open(output, 'rb') as fh:
                indecies, metric, cdat = pickle.load(fh)
        else:
            if comm is not None:
                comm.barrier()

            indecies, metric, cdat = sorts.correlate(
                measurements = measurements,
                population = pop,
                n_closest = 1,
                meta_variables=meta_vars,
                metric=metric, 
                sorting_function=sorting_function,
                metric_dtype=[('dr', np.float64), ('dv', np.float64)],
                metric_reduce=lambda x, y: x + y,
            )

            if comm is None or comm.rank == 0:
                with open(output, 'wb') as fh:
                     pickle.dump([indecies, metric, cdat], fh)

        if comm is None or comm.rank == 0:
            print('Individual measurement match metric:')
            for mind, (ind, dst) in enumerate(zip(indecies.T, metric.T)):
                print(f'measurement = {mind}')
                for res in range(len(ind)):
                    print(f'-- result rank = {res} | object ind = {ind[res]} | metric = {dst[res]} | obj = {pop["oid"][ind[res]]}')


if __name__ == '__main__':
    run()