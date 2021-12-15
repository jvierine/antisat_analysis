
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

python multi_beampark_correlator.py ~/data/spade/beamparks/uhf/2021.11.23/space-track.tles ~/data/spade/beamparks/{uhf,esr}/2021.11.23/correlation.pickle
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyse beampark correlation for a beampark')
    parser.add_argument('catalog', type=str, help='Catalog to which data was correlated')
    parser.add_argument('inputs', type=str, nargs='+', help='Input correlation data files')

    # This assumes correlation was done using the exact same TLE population so that
    #  object indecies match up in each correlation file

    args = parser.parse_args()

    tle_pth = pathlib.Path(args.catalog).resolve()
    input_pths = [pathlib.Path(x).resolve() for x in args.inputs]

    # if len(input_pths) < 2:
    #     raise ValueError('Need at least 2 correlation files')

    # print('Loading TLE population')
    # pop = sorts.population.tle_catalog(tle_pth, cartesian=False)
    # pop.out_frame = 'ITRS'
    # print(f'Population size: {len(pop)}')

    match_data = {}

    for ind, input_pth in enumerate(input_pths):
        print(f'Loading: {input_pth}')
        with open(input_pth, 'rb') as fh:
            indecies, metric, _ = pickle.load(fh)

        # Pick only best matches in row 0
        match_data[ind] = {
            'match': indecies[0,:], 
            'metric': metric[0,:],
        }


    # print('Individual measurement match metric:')
    # for mind, (ind, dst) in enumerate(zip(indecies.T, metric.T)):
    #     print(f'measurement = {mind}')
    #     for res in range(len(ind)):
    #         obj = pop.get_object(ind[res])

