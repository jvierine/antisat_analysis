#!/usr/bin/env python

import sys
import configparser
import logging
import pathlib
import argparse

import numpy as np
from astropy.time import Time, TimeDelta
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

logger = logging.getLogger(__file__)

DEFAULT_CONFIG = {
    'General': {
        'epoch': "2000-01-01T00:00:00Z",
        'start_time_hours': 0.0,
        'end_time_hours': 48.0,
        'seed': 123786577,
        'azimuth_deg': 90,
        'elevation_deg': 75,
        'dwell_sec': 0.1,
        'time_slice_separation_sec': 0.1,
        'propagator': 'SGP4',
        'propagation_time_step_sec': 20.0,
    },
    'Cache': {
        'output_folder': None,
    },
    'Population': {
        'density_g_per_cm3': 0.2,
        'a_min_m': 6671000.0,
        'a_max_m': 9371000.0,
        'i_min_deg': 80.0,
        'i_max_deg': 90.0,
        'Omega_min_deg': 0.0,
        'Omega_max_deg': 360.0,
        'a_samples': 100,
        'i_samples': 3,
        'Omega_samples': 50,
        'orbit_samples': 200,
        'lognorm_diam_mean': -1.0,
        'lognorm_diam_std': 0.5,
    },
}


class ObservedScanning(
            sorts.scheduler.StaticList,
            sorts.scheduler.ObservedParameters,
        ):
    pass


def get_scheduler(radar, config):
    scan = sorts.radar.scans.Beampark(
        azimuth=config.getfloat('General', 'azimuth_deg'), 
        elevation=config.getfloat('General', 'elevation_deg'),
        dwell=config.getfloat('General', 'dwell_sec'),
    )

    scanner = sorts.controller.Scanner(
        radar,
        scan,
        t = np.arange(
            config.getfloat('General', 'start_time_hours')*3600.0, 
            config.getfloat('General', 'end_time_hours')*3600.0, 
            config.getfloat('General', 'time_slice_separation_sec'),
        ),
        r = np.array([500e3]),
        t_slice = scan.dwell(),
    )

    return ObservedScanning(
        radar = radar, 
        controllers = [scanner], 
    )


class ForwardModel(sorts.Simulation):
    def __init__(self, population, config, *args, **kwargs):
        self.population = population
        self.config = config
        self.epoch = Time(config.get('General', 'epoch'), format='isot')
        self.inds = list(range(len(population)))

        # self.inds = self.inds[:12]

        super().__init__(*args, **kwargs)

        if self.logger is not None:
            self.logger.always(f'Population of size {len(population)} objects loaded.')

        self.cols = ['t', 'range', 'range_rate', 'snr']
        self._dtype = [('oid', np.int64)] + [(key, np.float64) for key in self.cols]
        self.collected_data = np.empty((0,), dtype=self._dtype)

        self.steps['simulate'] = self.simulate
        self.steps['collect'] = self.collect

    def add_data(self, index, data):
        new_data = np.empty((len(data['t']),), dtype=self._dtype)
        new_data['oid'] = index
        for key in self.cols:
            new_data[key] = data[key]
        self.collected_data = np.append(self.collected_data, new_data)

    def save_collected(self, **kwargs):
        logger.info(f'Saving collected data of size {self.collected_data.size}')
        np.save(self.get_path('collected_results.npy'), self.collected_data)

    @sorts.MPI_action(action='barrier')
    @sorts.iterable_step(iterable='inds', MPI=True, log=True, reduce=lambda x, y: None)
    @sorts.cached_step(caches='pickle')
    def simulate(self, index, item, **kwargs):
        obj = self.population.get_object(item)

        t = np.arange(
            self.config.getfloat('General', 'start_time_hours')*3600.0, 
            self.config.getfloat('General', 'end_time_hours')*3600.0, 
            self.config.getfloat('General', 'propagation_time_step_sec'),
        )

        state = obj.get_state(t)
        interpolator = sorts.interpolation.Legendre8(state, t)

        passes = self.scheduler.radar.find_passes(
            t, 
            state, 
            cache_data = False,
        )

        data = self.scheduler.observe_passes(
            passes, 
            space_object = obj, 
            epoch = self.epoch, 
            interpolator = interpolator,
            snr_limit = True,
        )

        return data

    @sorts.MPI_single_process(process_id = 0)
    @sorts.pre_post_actions(post='save_collected')
    @sorts.iterable_cache(steps='simulate', caches='pickle', MPI=False, log=True, reduce=lambda x, y: None)
    def collect(self, index, item, **kwargs):
        data = item[0][0]
        if data is not None and len(data) > 0:
            for pass_data in data:
                if pass_data is None:
                    continue
                self.add_data(index, pass_data)


def generate_population(config):

    if config.get('General', 'propagator') == 'SGP4':
        propagator = sorts.propagator.SGP4

    options = dict(
        settings = dict(
            in_frame='TEME',
            out_frame='ITRS',
        ),
    )

    cache_file = pathlib.Path(config.get('Cache', 'output_folder')) / 'population.h5'

    if cache_file.is_file():
        logger.info(f'Using cached population: {cache_file}')
        pop = sorts.Population.load( 
            cache_file, 
            propagator, 
            propagator_options = options,
        )
        return pop

    rho = config.getfloat('Population', 'density_g_per_cm3')

    pop = sorts.Population(
        fields = [
            'oid', 'a', 'e', 
            'i', 'raan', 'aop', 
            'mu0', 'mjd0', 'm', 
            'd', 'A', 'C_R', 'C_D',
        ],
        space_object_fields = ['oid', 'm', 'd', 'A', 'C_R', 'C_D'],
        state_fields = ['a', 'e', 'i', 'raan', 'aop', 'mu0'],
        epoch_field = {'field': 'mjd0', 'format': 'mjd', 'scale': 'utc'},
        propagator = propagator,
        propagator_options = options,
    )

    a, inc, Omega, nu = np.meshgrid(
        np.linspace(
            config.getfloat('Population', 'a_min_m'),
            config.getfloat('Population', 'a_max_m'),
            num=config.getint('Population', 'a_samples'),
        ),
        np.linspace(
            config.getfloat('Population', 'i_min_deg'),
            config.getfloat('Population', 'i_max_deg'),
            num=config.getint('Population', 'i_samples'),
        ),
        np.linspace(
            config.getfloat('Population', 'Omega_min_deg'),
            config.getfloat('Population', 'Omega_max_deg'),
            num=config.getint('Population', 'Omega_samples'),
        ),
        np.linspace(0, 360.0, num=config.getint('Population', 'orbit_samples')),
        indexing='ij',
    )

    size = a.size

    pop.allocate(size)

    d_mu = config.getfloat('Population', 'lognorm_diam_mean')
    d_std = config.getfloat('Population', 'lognorm_diam_std')

    # For consistency
    np.random.seed(config.getint('General', 'seed'))

    # * 0: oid - Object ID
    # * 1: a - Semi-major axis in m
    # * 2: e - Eccentricity 
    # * 3: i - Inclination in degrees
    # * 4: raan - Right Ascension of ascending node in degrees
    # * 5: aop - Argument of perihelion in degrees
    # * 6: mu0 - Mean anoamly in degrees
    # * 7: mjd0 - Epoch of object given in Modified Julian Days
    pop['oid'] = np.arange(size)
    pop['a'] = np.reshape(a, (size,))
    pop['e'] = 0.0
    pop['i'] = np.reshape(inc, (size,))
    pop['raan'] = np.reshape(Omega, (size,))
    pop['aop'] = 0.0
    pop['mu0'] = np.reshape(nu, (size,))
    pop['mjd0'] = Time(config.get('General', 'epoch'), format='isot').mjd
    pop['d'] = 10.0**(np.random.randn(size)*d_std + d_mu)
    pop['A'] = np.pi*(pop['d']*0.5)**2
    pop['m'] = 4.0/3.0*np.pi*(pop['d']*0.5)**3*rho
    pop['C_R'] = 0.1
    pop['C_D'] = 2.7

    if comm.rank == 0:
        pop.save(cache_file)

    return pop


def get_config(path):

    logger.info(f'Loading config: {path}')

    config = configparser.ConfigParser(
        interpolation=None, 
        allow_no_value=True,
    )

    config.read_dict(DEFAULT_CONFIG)

    if path is not None:
        config.read([path])

    return config


def main():

    handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    parser = argparse.ArgumentParser(description='Forward model to detmerine observation probabilitiy')
    parser.add_argument('config', type=pathlib.Path, help='Path to config')
    parser.add_argument('-c', '--clean', action='store_true', help='Remove simulation data before running')

    args = parser.parse_args()

    if not args.config.is_file():
        config = get_config(None)

        logger.info(f'NO CONFIG DETECTED AT {args.config}')
        logger.info(f'Writing default config to "{args.config}"...')

        if comm.rank == 0:
            with open(args.config, 'w') as fh:
                config.write(fh)

        logger.info('Exiting')
        return
    else:
        config = get_config(args.config)

    radar = getattr(sorts.radars, config.get('General', 'radar'))
    logger.info('Using radar: {}'.format(config.get('General', 'radar')))

    radar.rx = radar.rx[:1]

    epoch = Time(config.get('General', 'epoch'), format='isot')

    logger.info(f'Using epoch: {epoch}')

    cache_loc = pathlib.Path(config.get('Cache', 'output_folder'))
    if comm.rank == 0:
        cache_loc.mkdir(exist_ok=True)

    logger.info(f'Using cache: {cache_loc}')

    root = cache_loc / 'simulation'
    if comm.rank == 0:
        root.mkdir(exist_ok=True)

    if args.clean and comm.rank == 0:
        pop_file = cache_loc / 'population.h5'
        if pop_file.is_file():
            logger.info(f'Cleaning population cache {pop_file}...')
            pop_file.unlink()

    if comm.size > 1:
        comm.barrier()

    scheduler = get_scheduler(radar, config)

    population = generate_population(config)

    sim = ForwardModel(population, config, scheduler, root)

    sim.profiler.start('total')

    if args.clean:
        logger.info(f'Cleaning branch {sim.branch_name}...')
        sim.delete(sim.branch_name)

    sim.run()

    sim.profiler.stop('total')
    print(sim.profiler.fmt(normalize='total'))


if __name__ == '__main__':
    main()
