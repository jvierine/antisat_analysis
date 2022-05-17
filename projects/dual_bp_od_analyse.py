from astropy.time import Time, TimeDelta
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import h5py
from pprint import pprint

import sys

import odlab
import sorts
import pyorb
import pyant

HERE = Path(__file__).parent.resolve()
OUTPUT = HERE / 'output' / 'russian_asat'
print(f'Using {OUTPUT} as output')

out_pth = OUTPUT / 'orbit_determination'

results_data_file = out_pth / 'iod_results.pickle'

with open(results_data_file, 'rb') as fh:
    iod_results = pickle.load(fh)

pprint(iod_results)
