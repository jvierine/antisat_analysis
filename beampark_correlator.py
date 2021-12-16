#!/usr/bin/env python

'''
Correlating data with TLE catalog
===================================

'''
import argparse
import pathlib
import pickle

import numpy as np
import h5py
from astropy.time import Time

import sorts

'''
Examples:

python beampark_correlator.py eiscat_uhf ~/data/spade/beamparks/uhf/2015.10.22/space-track.tles ~/data/spade/beamparks/uhf/2015.10.22/h5 ~/data/spade/beamparks/uhf/2015.10.22/correlation.pickle
mpirun -n 6 ./beampark_correlator.py eiscat_uhf ~/data/spade/beamparks/uhf/2015.10.22/space-track.tles ~/data/spade/beamparks/uhf/2015.10.22/h5 ~/data/spade/beamparks/uhf/2015.10.22/correlation.pickle -o
mpirun -n 6 ./beampark_correlator.py eiscat_esr ~/data/spade/beamparks/esr/2021.11.23/{space-track.tles,leo.h5,correlation.h5} -o
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


def save_correlation_data(output_pth, indecies, metric, cdat, meta=None):
    print(f'Saving correlation data to {output_pth}')
    with h5py.File(output_pth, 'w') as ds:

        match_index = np.arange(indecies.shape[0])
        observation_index = np.arange(indecies.shape[1])

        # Create global attributes for dataset
        if meta is not None:
            for key in meta:
                ds.attrs[key] = meta[key]

        ds['mch_ind'] = match_index
        ds_mch_ind = ds['mch_ind']
        ds_mch_ind.make_scale('match_index')
        ds_mch_ind.attrs['long_name'] = 'nth matching index'

        ds['obs_ind'] = observation_index
        ds_obs_ind = ds['obs_ind']
        ds_obs_ind.make_scale('observation_index')
        ds_obs_ind.attrs['long_name'] = 'radar observation index'

        def _create_ia_var(base, name, long_name, data, scales, units=None):
            base[name] = data.copy()
            var = base[name]
            for ind in range(len(scales)):
                var.dims[ind].attach_scale(scales[ind])
            var.attrs['long_name'] = long_name
            if units is not None:
                var.attrs['units'] = units

        scales = [ds_mch_ind, ds_obs_ind]
        _create_ia_var(ds, 'match_oid', 'Matched object id in the used population', indecies, scales)
        _create_ia_var(ds, 'match_metric', 'Matching metric values for the object', metric, scales)

        # we only had one input measurnment dict so input data index is 0
        inp_dat_index = 0

        c_grp = ds.create_group("correlation_data")
        for mind in match_index:
            m_grp = c_grp.create_group(f'match_number_{mind}')
            for oind in observation_index:
                o_grp = m_grp.create_group(f'observation_number_{oind}')
                _create_ia_var(o_grp, 'r', 'simulated range', cdat[mind][oind][inp_dat_index]['r_ref'], [ds_obs_ind], units='m')
                _create_ia_var(o_grp, 'v', 'simulated range rate', cdat[mind][oind][inp_dat_index]['v_ref'], [ds_obs_ind], units='m/s')
                _create_ia_var(o_grp, 'match', 'calculated metric', cdat[mind][oind][inp_dat_index]['match'], [ds_obs_ind])


def run():

    if comm is not None:
        print('MPI detected')
        if comm.size > 1:
            print('Using MPI...')
 
    parser = argparse.ArgumentParser(description='Calculate TLE catalog correlation for a beampark')
    parser.add_argument('radar', type=str, help='The observing radar system')
    parser.add_argument('catalog', type=str, help='TLE catalog path')
    parser.add_argument('input', type=str, help='Observation data folder/file')
    parser.add_argument('output', type=str, help='Results output location')
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

    measurements = []
    if output_pth.is_file() and not args.override:
        print('Using cached data, not reading measurements file')
    else:
        if input_pth.is_file():
            input_files = [input_pth]
        else:
            input_files = input_pth.glob('**/*.h5')
        
        for in_file in input_files:

            # Each entry in the input `measurements` list must be a dictionary that contains the following fields:
            #   * 't': [numpy.ndarray] Times relative epoch in seconds
            #   * 'r': [numpy.ndarray] Two-way ranges in meters
            #   * 'v': [numpy.ndarray] Two-way range-rates in meters per second
            #   * 'epoch': [astropy.Time] epoch for measurements
            #   * 'tx': [sorts.TX] Pointer to the TX station
            #   * 'rx': [sorts.RX] Pointer to the RX station
            print('Loading monostatic measurements')
            with h5py.File(str(in_file), 'r') as h_det:
                r = h_det['r'][()]*1e3  # km -> m, one way
                t = h_det['t'][()]  # Unix seconds
                v = h_det['v'][()]*1e3  # km/s -> m/s

                inds = np.argsort(t)
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
                    'r': r*2,  # two way
                    'v': v*2,  # two way
                    't': t,
                    'epoch': epoch,
                    'tx': radar.tx[0],
                    'rx': radar.rx[0],
                    'measurement_num': len(t),
                }

                if args.std:
                    dat['r_std'] = h_det['r_std'][()]*1e3
                    dat['v_std'] = h_det['v_std'][()]*1e3

                measurements.append(dat)

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

        MPI = comm is not None and comm.size > 1

        indecies, metric, cdat = sorts.correlate(
            measurements = measurements,
            population = pop,
            n_closest = 1,
            meta_variables=meta_vars,
            metric=metric, 
            sorting_function=sorting_function,
            metric_dtype=[('dr', np.float64), ('dv', np.float64)],
            metric_reduce=None,
            MPI = MPI,
        )

        if comm is None or comm.rank == 0:
            save_correlation_data(
                output_pth, 
                indecies, 
                metric, 
                cdat, 
                meta = dict(
                    radar_name = args.radar,
                    tx_lat = radar.tx[0].lat,
                    tx_lon = radar.tx[0].lon,
                    tx_alt = radar.tx[0].alt,
                    rx_lat = radar.rx[0].lat,
                    rx_lon = radar.rx[0].lon,
                    rx_alt = radar.rx[0].alt,

                ),
            )

    if comm is None or comm.rank == 0:
        print('Individual measurement match metric:')
        for mind, (ind, dst) in enumerate(zip(indecies.T, metric.T)):
            print(f'measurement = {mind}')
            for res in range(len(ind)):
                print(f'-- result rank = {res} | object ind = {ind[res]} | metric = {dst[res]} | obj = {pop["oid"][ind[res]]}')


if __name__ == '__main__':
    run()
