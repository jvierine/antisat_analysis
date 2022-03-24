#!/usr/bin/env python

from pathlib import Path
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sorts
import pyorb
import orekit

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    class COMM_WORLD:
        rank = 0
        size = 1
    comm = COMM_WORLD()

target_t = 3600.0*24*7*12
snapshot_interval = 3600.0*24

settings = dict(
    in_frame='TEME',
    out_frame='TEME',
    solar_activity_strength='WEAK',
    integrator='DormandPrince853',
    min_step=0.001,
    max_step=240.0,
    position_tolerance=20.0,
    earth_gravity='HolmesFeatherstone',
    gravity_order=(10, 10),
    solarsystem_perturbers=['Moon', 'Sun'],
    drag_force=True,
    atmosphere='DTM2000',
    radiation_pressure=True,
    solar_activity='Marshall',
    constants_source='WGS84',
)

HERE = Path(__file__).parent.resolve()
OUTPUT = HERE / 'output' / 'russian_asat'
print(f'Using {OUTPUT} as output')

tle_file = OUTPUT / 'kosmos-1408-2021-11-15.tle'
orekit_data = OUTPUT / 'orekit-data-master.zip'
cache_file = OUTPUT / 'kosmos-1408_orekit_core_propagated.npy'
plot_folder = OUTPUT / 'kosmos-1408_orekit_propagated'
plot_folder.mkdir(exist_ok=True)


if not orekit_data.is_file():
    sorts.propagator.Orekit.download_quickstart_data(
        orekit_data, verbose=True
    )

pop = sorts.population.tle_catalog(tle_file, kepler=True)
pop.unique()

print(f'TLEs = {len(pop)}')

tle_obj = pop.get_object(0)
print(tle_obj)

tle_obj.out_frame = 'TEME'
state0 = tle_obj.get_state(0).flatten()
mjd0 = tle_obj.epoch.mjd

m = 1.0
d = 1.0
A = np.pi*(d*0.5)**2
t_vec = np.arange(0, target_t, snapshot_interval, dtype=np.float64)

prop = sorts.propagator.Orekit(
    orekit_data=orekit_data,
    settings=settings,
)

states = prop.propagate(
    t_vec, state0, 
    epoch=mjd0, A=A, m=m, C_R=1.0, C_D=1.0,
)

orb = pyorb.Orbit(
    M0=pyorb.M_earth, m=m, 
    num=1, epoch=mjd0, 
    degrees=True,
    cartesian=states,
)

plt.plot(t_vec/(3600.0*24), orb.Omega)
plt.show()