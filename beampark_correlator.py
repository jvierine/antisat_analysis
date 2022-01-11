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

python beampark_correlator.py eiscat_uhf \
    ~/data/spade/beamparks/uhf/2015.10.22/space-track.tles \
    ~/data/spade/beamparks/uhf/2015.10.22/leo.h5 \
    ~/data/spade/beamparks/uhf/2015.10.22/correlation.pickle

mpirun -n 6 \
    ./beampark_correlator.py eiscat_uhf \
    ~/data/spade/beamparks/uhf/2015.10.22/space-track.tles \
    ~/data/spade/beamparks/uhf/2015.10.22/leo.h5 \
    ~/data/spade/beamparks/uhf/2015.10.22/correlation.pickle -c

mpirun -n 6 ./beampark_correlator.py eiscat_esr ~/data/spade/beamparks/esr/2021.11.23/{space-track.tles,leo.h5,correlation.h5} -c
'''

# dtype for residuals
res_t = np.dtype([('dr', np.float64), ('dv', np.float64), ('metric', np.float64), ('jitter_index', np.float64)])
t_jitter = np.linspace(-5, 5, num=11)

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None


def default_propagation_handling(obj, t, t_measurement_indices, measurements):
    states = obj.get_state(t)
    return states


def jitter_propagation_handling(obj, t, t_measurement_indices, measurements):
    t_get = t[:, None] + t_jitter[None, :]

    ret_shape = t_get.shape
    t_get.shape = (t_get.size, )

    states = obj.get_state(t_get)
    states.shape = (6, ) + ret_shape

    return states


def vector_diff_metric(t, r, v, r_ref, v_ref, **kwargs):
    '''Return a vector of residuals for range and range rate.
    '''
    r_std = kwargs.get('r_std', None)
    v_std = kwargs.get('v_std', None)

    index_tuple = (slice(None), ) + tuple(None for x in range(len(r_ref.shape) - 1))

    base_shape = r_ref.shape

    ret = np.empty(base_shape, dtype=res_t)
    ret['dr'] = r_ref - r[index_tuple]
    ret['dv'] = v_ref - v[index_tuple]
    if r_std is not None:
        ret['dr'] /= r_std[index_tuple]
        ret['dv'] /= v_std[index_tuple]

    ret['metric'] = np.hypot(
        ret['dr']/kwargs['dr_scale'],
        ret['dv']/kwargs['dv_scale'],
    )

    # Reduce jitter if it exists
    if len(base_shape) > 1:
        ret.shape = (base_shape[0], np.prod(base_shape[1:]))
        inds = np.argmin(ret['metric'], axis=1)
        ret = ret[np.arange(base_shape[0]), inds]
        ret['jitter_index'] = inds
    else:
        ret['jitter_index'] = np.nan

    return ret


def save_correlation_data(output_pth, indices, metric, correlation_data, meta=None):
    print(f'Saving correlation data to {output_pth}')
    with h5py.File(output_pth, 'w') as ds:

        match_index = np.arange(indices.shape[0])
        observation_index = np.arange(indices.shape[1])

        # Create global attributes for dataset
        if meta is not None:
            for key in meta:
                ds.attrs[key] = meta[key]

        ds['object_index'] = np.array(list(correlation_data.keys()))
        ds_obj_ind = ds['object_index']
        ds_obj_ind.make_scale('object_index')
        ds_obj_ind.attrs['long_name'] = 'Object index in the used population'

        cartesian_pos = ds.create_dataset(
            "cartesian_pos_axis",
            data=[s.encode() for s in ["x", "y", "z"]],
        )
        cartesian_pos.make_scale('cartesian_pos_axis')
        cartesian_pos.attrs['long_name'] = 'Cartesian position axis names'
        # FIX this

        cartesian_vel = ds.create_dataset(
            "cartesian_vel_axis",
            data=[s.encode() for s in ["vx", "vy", "vz"]],
        )
        cartesian_vel.make_scale('cartesian_vel_axis')
        cartesian_vel.attrs['long_name'] = 'Cartesian velocity axis names'

        ds['maching_rank'] = match_index
        ds_mch_ind = ds['maching_rank']
        ds_mch_ind.make_scale('maching_rank')
        ds_mch_ind.attrs['long_name'] = 'Matching rank numbering from best as lowest rank to worst at hightest rank'

        ds['observation_index'] = observation_index
        ds_obs_ind = ds['observation_index']
        ds_obs_ind.make_scale('observation_index')
        ds_obs_ind.attrs['long_name'] = 'Observation index in the input radar data'

        def _create_ia_var(base, name, long_name, data, scales, units=None):
            base[name] = data.copy()
            var = base[name]
            for ind in range(len(scales)):
                var.dims[ind].attach_scale(scales[ind])
            var.attrs['long_name'] = long_name
            if units is not None:
                var.attrs['units'] = units

        scales = [ds_mch_ind, ds_obs_ind]

        _create_ia_var(ds, 'matched_object_index', 'Index of the correlated object', indices, scales)
        _create_ia_var(ds, 'matched_object_metric', 'Correlation metric for the correlated object', metric, scales)

        # We currently only supply one dat dict to the correlator
        measurement_set_index = 0

        def stacker(x, key):
            return np.stack([val[measurement_set_index][key] for _, val in x.items()], axis=0)

        scales = [ds_obj_ind, ds_obs_ind]
        _create_ia_var(ds, 'simulated_range', 'Simulated range', stacker(correlation_data, 'r_ref'), scales, units='m')
        _create_ia_var(ds, 'simulated_range_rate', 'Simulated range rate', stacker(correlation_data, 'v_ref'), scales, units='m/s')
        _create_ia_var(ds, 'simulated_correlation_metric', 'Calculated metric for the simulated ITRS state', stacker(correlation_data, 'match'), scales)
        _create_ia_var(
            ds, 
            'simulated_position', 
            'Simulated ITRS positions', 
            np.stack([val[measurement_set_index]['states'][:3, ...].T for _, val in correlation_data.items()], axis=0),
            scales + [cartesian_pos],
            units='m',
        )
        _create_ia_var(
            ds, 
            'simulated_velocity', 
            'Simulated ITRS velocities', 
            np.stack([val[measurement_set_index]['states'][3:, ...].T for _, val in correlation_data.items()], axis=0),
            scales + [cartesian_vel],
            units='m/s',
        )


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
    parser.add_argument('-c', '--clobber', action='store_true', help='Override output location if it exists')
    parser.add_argument('--std', action='store_true', help='Use measurement errors')
    parser.add_argument('--jitter', action='store_true', help='Use time jitter')
    parser.add_argument('--range-rate-scaling', default=0.2, type=float, help='Scaling used on range rate in the sorting function of the correlator')
    parser.add_argument('--range-scaling', default=1.0, type=float, help='Scaling used on range in the sorting function of the correlator')

    args = parser.parse_args()

    s_dr = args.range_rate_scaling
    s_r = args.range_scaling

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

    if args.jitter:
        propagation_handling = jitter_propagation_handling
    else:
        propagation_handling = default_propagation_handling

    meta_vars.append('dr_scale')
    meta_vars.append('dv_scale')

    measurements = []
    if output_pth.is_file() and not args.clobber:
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
                    'dr_scale': s_r,
                    'dv_scale': s_dr,
                }

                if args.std:
                    dat['r_std'] = h_det['r_std'][()]*1e3
                    dat['v_std'] = h_det['v_std'][()]*1e3

                measurements.append(dat)

    print('Loading TLE population')
    pop = sorts.population.tle_catalog(tle_pth, cartesian=False)
    pop.unique()

    # correlate requires output in ECEF 
    pop.out_frame = 'ITRS'

    print(f'Population size: {len(pop)}')
    print('Correlating data and population')

    if output_pth.is_file() and not args.clobber:
        print('Loading correlation data from cache')
        with open(output_pth, 'rb') as fh:
            indices, metric, cdat = pickle.load(fh)
    else:
        if comm is not None:
            comm.barrier()

        MPI = comm is not None and comm.size > 1

        indices, metric, correlation_data = sorts.correlate(
            measurements = measurements,
            population = pop,
            n_closest = 1,
            meta_variables = meta_vars,
            metric = vector_diff_metric, 
            sorting_function = lambda x: np.argsort(x['metric'], axis=0),
            metric_dtype = res_t,
            metric_reduce = None,
            scalar_metric = False,
            propagation_handling = propagation_handling,
            MPI = MPI,
        )

        if comm is None or comm.rank == 0:
            save_correlation_data(
                output_pth, 
                indices, 
                metric, 
                correlation_data, 
                meta = dict(
                    radar_name = args.radar,
                    tx_lat = radar.tx[0].lat,
                    tx_lon = radar.tx[0].lon,
                    tx_alt = radar.tx[0].alt,
                    rx_lat = radar.rx[0].lat,
                    rx_lon = radar.rx[0].lon,
                    rx_alt = radar.rx[0].alt,
                    range_scaling = s_r,
                    range_rate_scaling = s_dr,
                ),
            )

    if comm is None or comm.rank == 0:
        print('Individual measurement match metric:')
        for mind, (ind, dst) in enumerate(zip(indices.T, metric.T)):
            print(f'measurement = {mind}')
            for res in range(len(ind)):
                print(f'-- result rank = {res} | object ind = {ind[res]} | metric = {dst[res]} | obj = {pop["oid"][ind[res]]}')


if __name__ == '__main__':
    run()
