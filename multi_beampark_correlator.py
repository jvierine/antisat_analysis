#!/usr/bin/env python

import argparse
import pathlib
import pickle

import matplotlib.pyplot as plt
from tabulate import tabulate

import numpy as np
import h5py
from astropy.time import Time

import sorts

'''
Example execution

python multi_beampark_correlator.py \
    ~/data/spade/beamparks/uhf/2021.11.23/space-track.tles \
    -o ~/data/spade/beamparks/uhf/2021.11.23/dual_correlations.npy \
    ~/data/spade/beamparks/{uhf,esr}/2021.11.23/correlation.h5
'''


def metric_merge(metric, dr_scale, dv_scale):
    return np.sqrt((metric['dr']/dr_scale)**2 + (metric['dv']/dv_scale)**2)


def get_matches(data, bp, index):
    return data[bp['beampark'][index]]['metric'][bp['measurement_id'][index]]


def main(input_args=None):
    parser = argparse.ArgumentParser(description='Analyse beampark correlation for a beampark')
    parser.add_argument('catalog', type=str, help='Catalog to which data was correlated')
    parser.add_argument('inputs', type=str, nargs='+', help='Input correlation data files')
    parser.add_argument('-o', '--output', type=str, default='', help='Output OID save location')
    parser.add_argument('--target-epoch', type=str, default=None, help='When filtering unique TLEs use this target epoch [ISO]')

    # This assumes correlation was done using the exact same TLE population so that
    #  object indecies match up in each correlation file

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    tle_pth = pathlib.Path(args.catalog).resolve()
    input_pths = [pathlib.Path(x).resolve() for x in args.inputs]

    if len(input_pths) < 2:
        raise ValueError('Need at least 2 correlation files')

    match_data = {}
    objects = {}

    scale_x = 1.0
    scale_y = 0.2

    for ind, input_pth in enumerate(input_pths):
        print(f'Loading: {input_pth}')
        with h5py.File(input_pth, 'r') as ds:
            indecies = ds['matched_object_index'][()]
            metric = ds['matched_object_metric'][()]
            name = ds.attrs['radar_name']

        x = metric['dr']*1e-3
        y = metric['dv']*1e-3

        elip_dst = np.sqrt((x/scale_x)**2 + (y/scale_y)**2)
        
        select = np.logical_and(
            elip_dst[0,:] < 1.0,
            np.logical_not(np.isnan(elip_dst[0,:])),
        )

        # Pick only best matches in row 0
        match_data[ind] = {
            'match': indecies[0, :], 
            'metric': metric[0, :],
            'correlated': select,
        }

        for oid in np.unique(match_data[ind]['match']):
            indecies = np.where(match_data[ind]['match'] == oid)[0]

            if oid not in objects:
                objects[oid] = {
                    'beampark': [ind], 
                    'measurement_id': [indecies],
                    'num': 0,
                    'total_num': 1,
                }
            else:
                objects[oid]['beampark'].append(ind)
                objects[oid]['measurement_id'].append(indecies)
                objects[oid]['total_num'] += 1

            if np.any(match_data[ind]['correlated'][indecies]):
                objects[oid]['num'] += 1
    

    possible_multi = 0
    multi_match_oids = []
    multi_match_mids = []
    for oid, bp in objects.items():
        # skip all objects only seen in one beampark
        if bp['num'] < 2:
            continue

        mids = [np.nan]*len(input_pths)
        for bpid, mid in zip(bp['beampark'], bp['measurement_id']):
            # Assume only once in each beampark, this will have to be updated 
            # if the same object is detected twice in one experiment
            mids[bpid] = mid[0]
        multi_match_mids.append(mids)

        multi_match_oids.append(oid)

        merged_data = np.empty((bp['num'],), dtype=np.float64)

        for index in range(bp['num']):
            match = get_matches(match_data, bp, index)
            _m = metric_merge(match, scale_x*1e3, scale_y*1e3)
            _m = _m[np.logical_not(np.isnan(_m))]
            if _m.size > 0:
                _m = np.min(_m)
            else:
                _m = np.nan

            merged_data[index] = _m
            ID = bp['beampark'][index]
            print(f'Catalog-index (oid) - {oid:<6}: Beampark-{ID} -> residuals=({match["dr"][0]: .3e} m, {match["dv"][0]: .3e} m/s) -> combined={_m:.3e}')

        possible_multi += 1

    multi_match_mids = np.array(multi_match_mids)

    print(f'Possible multi-beampark observed objects: {possible_multi}')

    print('Loading TLE population')
    pop = sorts.population.tle_catalog(tle_pth, cartesian=False)
    pop.out_frame = 'ITRS'

    if args.target_epoch is not None:
        args.target_epoch = Time(args.target_epoch, format='iso', scale='utc').mjd
    pop.unique(target_epoch=args.target_epoch)

    print(f'Population size: {len(pop)}')

    get_fields = ['oid', 'mjd0', 'line1', 'line2']

    _dtype = [(f'mid{ind}', np.int64) for ind in range(len(input_pths))]
    _dtype += [('cid', np.int64)]
    for key in get_fields:
        _dtype += [(key, pop.dtypes[pop.fields.index(key)])]

    _data = pop.data[multi_match_oids][get_fields]
    _data_m = np.empty((len(_data),), dtype=_dtype)
    _data_m['cid'] = multi_match_oids
    for ind in range(len(input_pths)):
        _data_m[f'mid{ind}'] = multi_match_mids[:, ind]
    for key in get_fields:
        _data_m[key] = _data[key]

    if len(args.output) > 0:
        np.save(args.output, _data_m)

    print(tabulate(_data_m, headers=[f'MID-{ind}' for ind in range(len(input_pths))] + ['OID', 'NORAD-ID', 'mjd0', 'line1', 'line2']))


if __name__ == '__main__':
    main()
