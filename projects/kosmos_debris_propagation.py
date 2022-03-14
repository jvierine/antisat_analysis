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

fragments = 1000
target_t = 3600.0*24*7*12
snapshot_interval = 3600.0*24
log10d_mu = -1.1
log10d_std = 0.1
delta_v_std = 200.0  # m/s
clobber = False
np.random.seed(289784)

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
cache_file = OUTPUT / 'kosmos-1408_orekit_propagated.npy'
anim_file = OUTPUT / 'kosmos-1408_orekit_propagated.avi'
plot_folder = OUTPUT / 'kosmos-1408_orekit_propagated'
plot_folder.mkdir(exist_ok=True)

def run_orekit(
                target_t, 
                states0, 
                mjd0, 
                A,
                worker_inds,
                prop,
                pbar,
            ):

    out_states = np.empty(
        (6, len(worker_inds), len(target_t)), 
        dtype=np.float64,
    )
    cnt = 0
    for ind in worker_inds:
        try:
            _st = prop.propagate(
                target_t, states0[:, ind], 
                epoch=mjd0, A=A[ind], C_R=1.0, C_D=1.0,
            )
            out_states[:, cnt, :] = _st
        except orekit.JavaError:
            out_states[:, cnt, :] = np.nan
        cnt += 1
        pbar.update(1)

    return out_states


def get_future_orb_dist():

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

    # Random sample size distribution
    d = 10.0**(np.random.randn(fragments)*log10d_std + log10d_mu)
    A = np.pi*(d*0.5)**2
    states0 = np.empty((6, fragments), dtype=np.float64)
    states0[:3, :] = state0[:3, None]
    states0[3:, :] = state0[3:, None] + np.random.randn(3, fragments)*delta_v_std
    t_vec = np.arange(0, target_t, snapshot_interval, dtype=np.float64)
    states = np.empty((6, fragments, len(t_vec)), dtype=np.float64)

    inds = list(range(fragments))
    step = int(fragments/comm.size + 1)
    all_inds = [
        np.array(inds[i:(i + step)], dtype=np.int64) 
        for i in range(0, len(inds), step)
    ]

    assert all_inds[-1][-1] == inds[-1]

    prop = sorts.propagator.Orekit(
        orekit_data=orekit_data,
        settings=settings,
    )

    pbar = tqdm(total=len(all_inds[comm.rank]), position=comm.rank)
    worker_inds = all_inds[comm.rank]
    
    out_states = run_orekit(
        t_vec, 
        states0, 
        mjd0, 
        A,
        worker_inds,
        prop,
        pbar,
    )
    states[:, worker_inds, :] = out_states
    if comm.size > 1:
        if comm.rank == 0:
            for thr_id in range(comm.size):
                if thr_id != 0:
                    states[:, all_inds[thr_id], :] = comm.recv(
                        source=thr_id, 
                        tag=thr_id,
                    )
        else:
            comm.send(out_states, dest=0, tag=comm.rank)
            exit()
    
    pbar.close()

    return states


if cache_file.is_file() and not clobber:
    states = np.load(cache_file)
else:
    states = get_future_orb_dist()
    if comm.rank == 0:
        np.save(cache_file, states)

t_vec = np.arange(0, target_t, snapshot_interval, dtype=np.float64)/(3600.0*24)

for ind in range(states.shape[2]):
    orbs = pyorb.Orbit(M0=pyorb.M_earth, degrees=True, num=fragments)
    orbs.cartesian = states[:, :, ind]
    
    fig, ax = sorts.plotting.kepler_scatter(
        orbs.kepler.T, 
        axis_labels='earth-orbit',
        title=f'KOSMOS-1408 DEB proxy-distribution t={t_vec[ind]:.1f} days',
        limits=[
            (1, 1.3),
            (0, 0.2),
            (60, 100),
            (0, 360),
            (0, 360),
            (0, 360),
        ],
    )
    fig.savefig(plot_folder / f't_int{ind}.png', bbox_inches='tight')
    plt.close(fig)

try:
    subprocess.check_call(
        f'ffmpeg -start_number 0 -r 6 -i t_int%d.png -c:v ffv1 {str(anim_file.resolve())}', 
        cwd=str(plot_folder.resolve()),
        shell=True,
    )
except subprocess.CalledProcessError as e:
    print(e)
    print('Could not create movie from animation frames... probably ffmpeg is missing')
