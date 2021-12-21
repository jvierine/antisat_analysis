#!/usr/bin/env python

import sys
import configparser
import logging
import pathlib
import argparse


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import scipy.interpolate as sio
from matplotlib import cm
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
        'radar': None, 
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
        'a_min_m': 6700000.0,
        'a_max_m': 6900000.0,
        'a_samples': 20,
        'e_min': 0.0,
        'e_max': 0.2,
        'e_samples': 10,
        'i_min_deg': 80.0,
        'i_max_deg': 85.0,
        'i_samples': 10,
        'aop_min_deg': 0.0,
        'aop_max_deg': 360.0,
        'aop_samples': 20,
        'raan_min_deg': 91.7105,
        'raan_max_deg': 91.7105,
        'raan_samples': 1,
        'orbit_samples': 100,
        'diam': 0.1,
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

        super().__init__(*args, **kwargs)

        if self.logger is not None:
            self.logger.always(f'Population of size {len(population)} objects loaded.')

        self.cols = ['t', 'range', 'range_rate', 'snr']
        self._dtype = [('oid', np.int64)] + [(key, np.float64) for key in self.cols]
        self.collected_data = np.empty((0,), dtype=self._dtype)

        self.steps['simulate'] = self.simulate
        self.steps['collect'] = self.collect

    def extract_data(self, index, data):
        all_datas = []
        pass_data = data[0][0]
        if pass_data is not None and len(pass_data) > 0:
            for ps in pass_data:
                if ps is None:
                    continue
                all_datas.append(ps)

        new_data = np.empty((len(all_datas),), dtype=self._dtype)
        new_data['oid'] = index
        for ind, pd in enumerate(all_datas):

            max_snr_ind = np.argmax(pd['snr'])
            for key in self.cols:
                new_data[key][ind] = pd[key][max_snr_ind]
        return new_data

    def save_collected(self, **kwargs):
        logger.info(f'Saving collected data of size {self.collected_data.size}')
        np.save(self.get_path('collected_results.npy'), self.collected_data)

    @sorts.MPI_action(action='barrier')
    @sorts.iterable_step(iterable='inds', MPI=True, log=True, reduce=lambda x, y: None)
    @sorts.cached_step(caches='npy')
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
            snr_limit = False,
        )

        peak_data = self.extract_data(index, data)

        return peak_data

    @sorts.MPI_action(action='barrier')
    @sorts.MPI_single_process(process_id = 0)
    @sorts.pre_post_actions(post='save_collected')
    @sorts.iterable_cache(steps='simulate', caches='npy', MPI=False, log=True, reduce=lambda x, y: None)
    def collect(self, index, item, **kwargs):
        self.collected_data = np.append(self.collected_data, item)


def get_sample_vectors(config):
    return (
        np.linspace(
            config.getfloat('Population', 'a_min_m'),
            config.getfloat('Population', 'a_max_m'),
            num=config.getint('Population', 'a_samples'),
        ),
        np.linspace(
            config.getfloat('Population', 'e_min'),
            config.getfloat('Population', 'e_max'),
            num=config.getint('Population', 'e_samples'),
        ),
        np.linspace(
            config.getfloat('Population', 'i_min_deg'),
            config.getfloat('Population', 'i_max_deg'),
            num=config.getint('Population', 'i_samples'),
        ),
        np.linspace(
            config.getfloat('Population', 'aop_min_deg'),
            config.getfloat('Population', 'aop_max_deg'),
            num=config.getint('Population', 'aop_samples'),
        ),
        np.linspace(
            config.getfloat('Population', 'raan_min_deg'),
            config.getfloat('Population', 'raan_max_deg'),
            num=config.getint('Population', 'raan_samples'),
        ),
        np.linspace(0, 360.0, num=config.getint('Population', 'orbit_samples')),
    )


def get_sample_grid(config):
    return np.meshgrid(
        *get_sample_vectors(config),
        indexing='ij',
    )


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

    a, e, inc, aop, raan, nu = get_sample_grid(config)

    size = a.size

    pop.allocate(size)

    d = config.getfloat('Population', 'diam')

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
    del a
    pop['e'] = np.reshape(e, (size,))
    del e
    pop['i'] = np.reshape(inc, (size,))
    del inc
    pop['raan'] = np.reshape(raan, (size,))
    del raan
    pop['aop'] = np.reshape(aop, (size,))
    del aop
    pop['mu0'] = np.reshape(nu, (size,))
    del nu
    pop['mjd0'] = Time(config.get('General', 'epoch'), format='isot').mjd
    pop['d'] = d
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


def distribution_gui(data, pop, config):

    fig = plt.figure()
    axes = [
        fig.add_subplot(2, 2, 1),
        fig.add_subplot(2, 2, 2),
        fig.add_subplot(2, 2, 3),
        fig.add_subplot(2, 2, 4),
    ]
    plt.subplots_adjust(bottom=0.25)

    l1, = axes[0].plot([], [], '.b')
    axes[0].set_xlabel('Range [km]')
    axes[0].set_ylabel('Range-rate [km/s]')
    axes[0].set_xlim([data['range'].min()*1e-3, data['range'].max()*1e-3])
    axes[0].set_ylim([data['range_rate'].min()*1e-3, data['range_rate'].max()*1e-3])
    l2, = axes[1].plot([], [], '.b')
    axes[1].set_xlabel('Time [h]')
    axes[1].set_ylabel('SNR [dB]')
    axes[1].set_xlim([data['t'].min()/3600.0, data['t'].max()/3600.0])
    axes[1].set_ylim(np.log10(np.array([data['snr'].min(), data['snr'].max()]))*10)
    l3, = axes[2].plot([], [], '.b')
    axes[2].set_xlabel('Time [h]')
    axes[2].set_ylabel('Range [km]')
    axes[2].set_xlim([data['t'].min()/3600.0, data['t'].max()/3600.0])
    axes[2].set_ylim([data['range'].min()*1e-3, data['range'].max()*1e-3])
    l4, = axes[3].plot([], [], '.b')
    axes[3].set_xlabel('Time [h]')
    axes[3].set_ylabel('Range-rate [km/s]')
    axes[3].set_xlim([data['t'].min()/3600.0, data['t'].max()/3600.0])
    axes[3].set_ylim([data['range_rate'].min()*1e-3, data['range_rate'].max()*1e-3])

    axcolor = 'lightgoldenrodyellow'
    ax_a = [0.25, 0.05, 0.2, 0.03]
    ax_e = [0.25, 0.1, 0.2, 0.03]
    ax_i = [0.25, 0.15, 0.2, 0.03]

    ax_omega = [0.6, 0.05, 0.2, 0.03]
    ax_Omega = [0.6, 0.1, 0.2, 0.03]

    slider_axis = [ax_a, ax_e, ax_i, ax_omega, ax_Omega]
    slider_names = ['a [AU]', 'e [1]', 'i [deg]', 'aop [deg]', 'raan [deg]']
    data_names = ['a', 'e', 'i', 'aop', 'raan']

    sample_vectors = get_sample_vectors(config)

    sliders = []
    for ax_size, sname, x, key in zip(slider_axis, slider_names, sample_vectors, data_names):
        if x.min() == x.max():
            continue

        sax = plt.axes(ax_size, facecolor=axcolor)
        slide = Slider(sax, sname, x.min(), x.max(), valinit=x.min(), valstep=x)
        sliders.append((slide, key))

    def update(val, draw=True):

        select = np.full(data['range'].shape, True, dtype=bool)
        for slide, key in sliders:
            select = np.logical_and(
                select,
                pop[key][data['oid']] == slide.val,
            )

        l1.set_xdata(data['range'][select]*1e-3)
        l1.set_ydata(data['range_rate'][select]*1e-3)

        l2.set_xdata(data['t'][select]/3600.0)
        l2.set_ydata(np.log10(data['snr'][select])*10)

        l3.set_xdata(data['t'][select]/3600.0)
        l3.set_ydata(data['range'][select]*1e-3)

        l4.set_xdata(data['t'][select]/3600.0)
        l4.set_ydata(data['range_rate'][select]*1e-3)

        if draw:
            fig.canvas.draw_idle()

    for slide, key in sliders:
        slide.on_changed(update)

    update(None, draw=False)

    plt.show()

    return fig, axes


def analysis(config, pop):
    base_path = pathlib.Path(config.get('Cache', 'output_folder'))
    data_path = base_path / 'simulation' / 'master' / 'collected_results.npy'

    data = np.load(data_path)

    print(data.shape)

    bins = 30
    # cols = ['oid', 't', 'range', 'range_rate', 'snr']

    # ['oid', 'm', 'd', 'A', 'C_R', 'C_D'],
    # ['a', 'e', 'i', 'raan', 'aop', 'mu0']

    fig, ax = plt.subplots(1, 1)
    ax.hist(np.log10(pop['d']))
    ax.set_xlabel('Diameter [log10(m)]')
    ax.set_ylabel('Frequency [1]')

    gui_fig, gui_axes = distribution_gui(data, pop, config)

    fig, ax = plt.subplots(1, 1)
    ax.hist2d(data['range']*1e-3, data['range_rate']*1e-3, bins=bins)
    ax.set_xlabel('Range [km]')
    ax.set_ylabel('Range rate [km/s]')

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].hist2d(pop['i'][data['oid']], data['t']/3600.0, bins=bins)
    axes[0, 0].set_xlabel('Inclination [deg]')
    axes[0, 0].set_ylabel('Time [h]')

    axes[0, 1].hist2d(pop['i'][data['oid']], data['range']*1e-3, bins=bins)
    axes[0, 1].set_xlabel('Inclination [deg]')
    axes[0, 1].set_ylabel('Range [km]')

    axes[1, 0].hist2d(pop['i'][data['oid']], data['range_rate']*1e-3, bins=bins)
    axes[1, 0].set_xlabel('Inclination [deg]')
    axes[1, 0].set_ylabel('Range rate [km/s]')

    axes[1, 1].hist2d(pop['i'][data['oid']], 10*np.log(data['snr']), bins=bins)
    axes[1, 1].set_xlabel('Inclination [deg]')
    axes[1, 1].set_ylabel('SNR [dB]')

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].hist(data['t']/3600.0)
    axes[0, 0].set_xlabel('Time [h]')
    axes[0, 0].set_ylabel('Frequency [1]')

    axes[0, 1].hist2d(data['t']/3600.0, data['range']*1e-3, bins=bins)
    axes[0, 1].set_xlabel('Time [h]')
    axes[0, 1].set_ylabel('Range [km]')

    axes[1, 0].hist2d(data['t']/3600.0, data['range_rate']*1e-3, bins=bins)
    axes[1, 0].set_xlabel('Time [h]')
    axes[1, 0].set_ylabel('Range rate [km/s]')

    axes[1, 1].hist2d(data['t']/3600.0, 10*np.log(data['snr']), bins=bins)
    axes[1, 1].set_xlabel('Time [h]')
    axes[1, 1].set_ylabel('SNR [dB]')

    plt.show()


def main():

    handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    parser = argparse.ArgumentParser(description='Forward model to detmerine observation probabilitiy')
    parser.add_argument('config', type=pathlib.Path, help='Path to config')
    parser.add_argument(
        'action',
        choices=['run', 'collect', 'analyse'],
        default='run',
        const='run',
        nargs='?',
        help='Execute action',
    )
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
        exit()
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

    if args.clean and comm.rank == 0 and args.action == 'run':
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

    if args.clean and args.action == 'run':
        logger.info(f'Cleaning branch {sim.branch_name}...')
        sim.delete(sim.branch_name)

    if args.action == 'run':
        sim.run(step='simulate')
    elif args.action == 'collect':
        sim.run(step='collect')
    elif args.action == 'analyse':
        analysis(config, population)

    sim.profiler.stop('total')
    print(sim.profiler.fmt(normalize='total'))


if __name__ == '__main__':
    main()
