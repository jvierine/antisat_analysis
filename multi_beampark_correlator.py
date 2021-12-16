
import argparse
import pathlib
import pickle

import matplotlib.pyplot as plt
#!/usr/bin/env python

import numpy as np
import h5py
from astropy.time import Time

import sorts

'''
Example execution

python multi_beampark_correlator.py ~/data/spade/beamparks/uhf/2021.11.23/space-track.tles ~/data/spade/beamparks/{uhf,esr}/2021.11.23/correlation.h5
'''


def metric_merge(metric):
    return np.sqrt(metric['dr']**2 + 40*metric['dv']**2)


def get_matches(data, bp, index):
    return data[bp['beampark'][index]]['metric'][bp['measurement_id'][index]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyse beampark correlation for a beampark')
    parser.add_argument('catalog', type=str, help='Catalog to which data was correlated')
    parser.add_argument('inputs', type=str, nargs='+', help='Input correlation data files')

    # This assumes correlation was done using the exact same TLE population so that
    #  object indecies match up in each correlation file

    args = parser.parse_args()

    tle_pth = pathlib.Path(args.catalog).resolve()
    input_pths = [pathlib.Path(x).resolve() for x in args.inputs]

    if len(input_pths) < 2:
        raise ValueError('Need at least 2 correlation files')

    match_data = {}
    objects = {}

    for ind, input_pth in enumerate(input_pths):
        print(f'Loading: {input_pth}')
        with h5py.File(input_pth, 'r') as ds:
            indecies = ds['match_oid'][()]
            metric = ds['match_metric'][()]
            name = ds.attrs['radar_name']

        # Pick only best matches in row 0
        match_data[ind] = {
            'match': indecies[0, :], 
            'metric': metric[0, :],
        }

        for oid in np.unique(match_data[ind]['match']):
            indecies = np.where(match_data[ind]['match'] == oid)

            if oid not in objects:
                objects[oid] = {
                    'beampark': [ind], 
                    'measurement_id': [indecies],
                    'num': 1,
                }
            else:
                objects[oid]['beampark'].append(ind)
                objects[oid]['measurement_id'].append(indecies)
                objects[oid]['num'] += 1
    
    possible_multi = 0
    for oid, bp in objects.items():
        # skip all objects only seen in one beampark
        if bp['num'] < 2:
            continue

        merged_data = np.empty((bp['num'],), dtype=np.float64)

        for index in range(bp['num']):
            match = get_matches(match_data, bp, index)
            _m = metric_merge(match)
            _m = _m[np.logical_not(np.isnan(_m))]
            if _m.size > 0:
                _m = np.min(_m)
            else:
                _m = np.nan

            merged_data[index] = _m
            ID = bp['beampark'][index]
            print(f'OID - {oid}: Beampark-{ID} -> match={match}')

        print(f'OID - {oid}: {merged_data}')
        possible_multi += 1

    print(f'Possible multi-beampark observed objects: {possible_multi}')

    # print('Loading TLE population')
    # pop = sorts.population.tle_catalog(tle_pth, cartesian=False)
    # pop.out_frame = 'ITRS'
    # print(f'Population size: {len(pop)}')

    # check orbit improvement