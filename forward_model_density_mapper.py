import sys
import configparser
import logging
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import Time, TimeDelta
import scipy.optimize as sio
import h5py

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    class COMM_WORLD:
        rank = 0
        size = 1
    comm = COMM_WORLD()

import sorts
import pyorb

logger = logging.getLogger(__file__)

DEFAULT_CONFIG = {
    'General': {
        'epoch': "2000-01-01T00:00:00Z",
    }
}


def process_orbit(
            state,
            parameters,
            epoch,
            t,
            propagator,
            propagator_options,
            radar,
            profiler,
            obs_epoch,
            num,
            cores=0,
            custom_scheduler_getter=None,
        ):
    space_orbit = sorts.SpaceObject(
        propagator,
        propagator_options = propagator_options,
        x = state[0],
        y = state[1],
        z = state[2],
        vx = state[3],
        vy = state[4],
        vz = state[5],
        epoch = epoch,
        parameters = parameters,
    )
    space_orbit.in_frame = 'TEME'
    space_orbit.out_frame = 'ITRS'
    space_orbit.orbit.type = 'mean'
    space_orbit.orbit.degrees = True

    odatas = []

    if cores > 1:
        pool = mp.Pool(cores)
        reses = []

        pbar = tqdm(total=num, desc='Orbit mean anomaly sampling')
        for ai, anom in enumerate(np.linspace(0, 360.0, num=num)):
            if custom_scheduler_getter is not None:
                _kwargs = dict(custom_scheduler=custom_scheduler_getter())
            else:
                _kwargs = {}

            reses.append(pool.apply_async(
                _process_orbit_job, 
                args=(
                    anom, 
                    space_orbit.copy(),
                    radar.copy(),
                    t,
                    obs_epoch,
                ), 
                kwds=_kwargs,
            ))
        pool_status = np.full((num, ), False)
        while not np.all(pool_status):
            for fid, res in enumerate(reses):
                if pool_status[fid]:
                    continue
                time.sleep(0.001)
                _ready = res.ready()
                if not pool_status[fid] and _ready:
                    pool_status[fid] = _ready
                    pbar.update()

        for fid, res in enumerate(reses):
            odata = res.get()
            odatas.append(odata)

        pool.close()
        pool.join()
        pbar.close()

    else:
        if custom_scheduler_getter is not None:
            scheduler = custom_scheduler_getter()
        else:
            scheduler = TrackingScheduler(
                radar = radar, 
                profiler = profiler, 
                logger = logger,
            )
        for anom in tqdm(np.linspace(0, 360.0, num=num)):
            space_orbit.orbit.anom = anom

            states_orb = space_orbit.get_state(t)
            interpolator = sorts.interpolation.Legendre8(states_orb, t)
            if isinstance(scheduler, TrackingScheduler):
                scheduler.set_tracker(t, states_orb)
                passes = scheduler.passes
            else:
                passes = scheduler.radar.find_passes(t, states_orb)

            odatas += [scheduler.observe_passes(
                passes, 
                space_object = space_orbit, 
                epoch = obs_epoch, 
                interpolator = interpolator,
                snr_limit = False,
            )]

    return odatas


def get_config(path):

    logger.info(f'Loading config: {path}')

    config = configparser.ConfigParser(
        interpolation=None, 
        allow_no_value=True,
    )

    config.read_dict(DEFAULT_CONFIG)
    config.read([path])

    return config


def forward_model(scheduler, a, inc, Omega, diameter, num):

    nu = np.linspace(0, np.pi*2, num=num)

    results = {
        't': None,
        'range': None,
        'range_rate': None,
        'snr': None,
    }
    return results


def main():

    if len(sys.argv) < 2:
        raise ValueError('Not enough input arguments, need a config file...')
    cfg = pathlib.Path(sys.argv[1])

    config = get_config(cfg)

    radar = getattr(sorts.radars, config.get('General', 'radar'))

    epoch = Time(config.get('General', 'epoch'), format='isot')

    logger.info(f'Using epoch: {epoch}')


if __name__ == '__main__':
    main()
