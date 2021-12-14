
import argparse
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import h5py
from astropy.time import Time

import sorts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyse beampark correlation for a beampark')
    parser.add_argument('input', type=str, help='Input correlation data')

    args = parser.parse_args()

    tle_pth = pathlib.Path(args.catalog).resolve()
    input_pth = pathlib.Path(args.input).resolve()

    print('Loading TLE population')
    pop = sorts.population.tle_catalog(tle_pth, cartesian=False)
    pop.out_frame = 'ITRS'
    print(f'Population size: {len(pop)}')

    with open(input_pth, 'rb') as fh:
        indecies, metric, cdat = pickle.load(fh)

    print('Individual measurement match metric:')
    for mind, (ind, dst) in enumerate(zip(indecies.T, metric.T)):
        print(f'measurement = {mind}')
        for res in range(len(ind)):
            obj = pop.get_object(ind[res])

