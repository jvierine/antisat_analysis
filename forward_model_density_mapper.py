#!/usr/bin/env python

import sys
import configparser
import logging
import pathlib
import argparse
import json


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import scipy.interpolate as sio
import scipy.optimize as sop
import scipy.stats as st
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
        'TEME_TO_TLE_minimize_start_samples': 20, 
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
    'Measurement paths': {},
    'Measurements': {
        'epoch': None,
        't_min_h': 0,
        't_max_h': 24.0,
        't_samples': 48,
        'range_min_m': 300e3,
        'range_max_m': 900e3,
        'range_samples': 30,
        'range_rate_min_m_per_s': -6e3,
        'range_rate_max_m_per_s': 6e3,
        'range_rate_samples': 20,
        'snr_min_db': 0,
        'snr_max_db': 24.0,
        'snr_samples': 48,
    },
}


def get_midpoints(vec):
    return 0.5*(vec[:-1] + vec[1:])


def get_edges(vec, delta=None):
    # Assume equal grid
    if delta is None:
        delta = vec[1] - vec[0]
    return np.append(vec - delta, [vec[-1] + delta])


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
        _tle_samps = self.config.getint('General', 'TEME_TO_TLE_minimize_start_samples')
        obj.propagator.set(TEME_TO_TLE_minimize_start_samples=_tle_samps)

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


def get_measurnment_vectors(config):
    return (
        np.linspace(
            config.getfloat('Measurements', 't_min_h')*3600.0,
            config.getfloat('Measurements', 't_max_h')*3600.0,
            num=config.getint('Measurements', 't_samples'),
        ),
        np.linspace(
            config.getfloat('Measurements', 'range_min_m'),
            config.getfloat('Measurements', 'range_max_m'),
            num=config.getint('Measurements', 'range_samples'),
        ),
        np.linspace(
            config.getfloat('Measurements', 'range_rate_min_m_per_s'),
            config.getfloat('Measurements', 'range_rate_max_m_per_s'),
            num=config.getint('Measurements', 'range_rate_samples'),
        ),
        np.linspace(
            config.getfloat('Measurements', 'snr_min_db'),
            config.getfloat('Measurements', 'snr_max_db'),
            num=config.getint('Measurements', 'snr_samples'),
        ),
    )


def get_measurnment_grid(config):
    return np.meshgrid(
        *get_measurnment_vectors(config),
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

    l2, = axes[1].plot([], [], '.b')
    axes[1].set_xlabel('Time [h]')
    axes[1].set_ylabel('SNR [dB]')

    l3, = axes[2].plot([], [], '.b')
    axes[2].set_xlabel('Time [h]')
    axes[2].set_ylabel('Range [km]')

    l4, = axes[3].plot([], [], '.b')
    axes[3].set_xlabel('Time [h]')
    axes[3].set_ylabel('Range-rate [km/s]')

    axes[1].set_xlim([data['t'].min()/3600.0, data['t'].max()/3600.0])
    axes[2].set_xlim([data['t'].min()/3600.0, data['t'].max()/3600.0])
    axes[3].set_xlim([data['t'].min()/3600.0, data['t'].max()/3600.0])

    axes[0].set_xlim([data['range'].min()*1e-3, data['range'].max()*1e-3])
    axes[0].set_ylim([data['range_rate'].min()*1e-3, data['range_rate'].max()*1e-3])
    axes[1].set_ylim(np.log10(np.array([data['snr'].min(), data['snr'].max()]))*10)
    axes[2].set_ylim([data['range'].min()*1e-3, data['range'].max()*1e-3])
    axes[3].set_ylim([data['range_rate'].min()*1e-3, data['range_rate'].max()*1e-3])

    axcolor = 'lightgoldenrodyellow'
    ax_a = [0.25, 0.05, 0.2, 0.03]
    ax_e = [0.25, 0.1, 0.2, 0.03]
    ax_i = [0.25, 0.15, 0.2, 0.03]

    ax_omega = [0.6, 0.05, 0.2, 0.03]
    ax_Omega = [0.6, 0.1, 0.2, 0.03]
    ax_snr = [0.6, 0.15, 0.2, 0.03]

    slider_axis = [ax_a, ax_e, ax_i, ax_omega, ax_Omega]
    slider_names = ['a [AU]', 'e [1]', 'i [deg]', 'aop [deg]', 'raan [deg]']
    data_names = ['a', 'e', 'i', 'aop', 'raan']

    sample_vectors = get_sample_vectors(config)[:-1]  # remove nu

    sliders = []
    for ax_size, sname, x, key in zip(slider_axis, slider_names, sample_vectors, data_names):
        if x.min() == x.max():
            continue

        sax = plt.axes(ax_size, facecolor=axcolor)
        slide = Slider(sax, sname, x.min(), x.max(), valinit=x.min(), valstep=x)
        sliders.append((slide, key))

    snr_db = 10*np.log10(data['snr'])

    s_ax_snr = plt.axes(ax_snr, facecolor=axcolor)
    slide_snr = Slider(s_ax_snr, 'Min SNR [dB]', snr_db.min(), snr_db.max(), valinit=snr_db.min(), valstep=1.0)

    def update(val, draw=True, lim_update=False):

        select = np.full(data['range'].shape, True, dtype=bool)
        for slide, key in sliders:
            select = np.logical_and(
                select,
                pop[key][data['oid']] == slide.val,
            )

        select = np.logical_and(
            select,
            snr_db >= slide_snr.val,
        )

        # print(pop.data[np.unique(data['oid'][select])])

        l1.set_xdata(data['range'][select]*1e-3)
        l1.set_ydata(data['range_rate'][select]*1e-3)

        l2.set_xdata(data['t'][select]/3600.0)
        l2.set_ydata(np.log10(data['snr'][select])*10)

        l3.set_xdata(data['t'][select]/3600.0)
        l3.set_ydata(data['range'][select]*1e-3)

        l4.set_xdata(data['t'][select]/3600.0)
        l4.set_ydata(data['range_rate'][select]*1e-3)

        if lim_update:
            axes[0].set_xlim([data['range'][select].min()*1e-3, data['range'][select].max()*1e-3])
            axes[0].set_ylim([data['range_rate'][select].min()*1e-3, data['range_rate'][select].max()*1e-3])
            axes[1].set_ylim(np.log10(np.array([data['snr'][select].min(), data['snr'][select].max()]))*10)
            axes[2].set_ylim([data['range'][select].min()*1e-3, data['range'][select].max()*1e-3])
            axes[3].set_ylim([data['range_rate'][select].min()*1e-3, data['range_rate'][select].max()*1e-3])

        if draw:
            fig.canvas.draw_idle()

    for slide, key in sliders:
        slide.on_changed(update)
    slide_snr.on_changed(update)

    update(None, draw=False)

    plt.show()

    return fig, axes


def snr_gui(data, pop, config, radar):

    fig = plt.figure()
    axes = [
        fig.add_subplot(2, 1, 1),
        fig.add_subplot(2, 2, 3),
        fig.add_subplot(2, 2, 4),
    ]
    plt.subplots_adjust(bottom=0.12)

    xedges = np.linspace(data['t'].min()/3600.0, data['t'].max()/3600.0, 100)
    yedges = np.linspace(-100, 250, 100)
    xedges_p = (xedges[:-1] + xedges[1:])*0.5
    yedges_p = (yedges[:-1] + yedges[1:])*0.5

    H, _, _ = np.histogram2d(data['t']/3600.0, 10.0*np.log10(data['snr']), bins=[xedges, yedges])

    X, Y = np.meshgrid(xedges_p, yedges_p)

    # l1, = axes[0].plot([], [], '.b')
    pmesh = axes[0].pcolormesh(X, Y, H.T)
    # cb = plt.colorbar(pmesh)
    axes[0].set_xlabel('Time [h]')
    axes[0].set_ylabel('SNR [dB]')

    l2, = axes[1].plot([], [], '-')
    axes[1].set_xlabel('SNR [dB]')
    axes[1].set_ylabel('Frequency [1]')

    l3a, = axes[2].plot([], [], '-b')
    l3b, = axes[2].plot([], [], '-r')
    axes[2].set_xlabel('Diameter [log10(m)]')
    axes[2].set_ylabel('Probabilitiy [1]')

    axes[0].set_xlim([xedges[0], xedges[-1]])
    axes[0].set_ylim([yedges[0], yedges[-1]])
    axes[2].set_xlim([-2, 1])

    axcolor = 'lightgoldenrodyellow'
    pos_ax_mu = [0.25, 0.05, 0.2, 0.03]
    ax_mu = plt.axes(pos_ax_mu, facecolor=axcolor)
    pos_ax_sig = [0.6, 0.05, 0.2, 0.03]
    ax_sig = plt.axes(pos_ax_sig, facecolor=axcolor)

    slide_mu = Slider(ax_mu, 'Mean', -2.0, 0.0, valinit=-1.0, valstep=0.1)
    slide_std = Slider(ax_sig, 'Std', 0.01, 1, valinit=0.1, valstep=0.01)

    log10_d = np.linspace(-5, 2, 200)

    objs = np.unique(data['oid'])
    oid_to_data = np.empty(data['snr'].shape, dtype=np.int64)
    for ind in range(len(data['oid'])):
        oid_to_data[ind] = np.argwhere(objs == data['oid'][ind])

    # Just random sample once
    xi_samps = np.random.randn(len(objs))

    def update(val, draw=True):

        log10_d_P = st.norm.pdf(log10_d, scale=slide_std.val, loc=slide_mu.val)
        sample_d = 10.0**(xi_samps*slide_std.val + slide_mu.val)
        samp_F, samp_X = np.histogram(np.log10(sample_d), bins=100, density=True)
        samp_X = (samp_X[:-1] + samp_X[1:])*0.5

        scaling = np.empty_like(sample_d)
        for ind in range(len(sample_d)):
            scaling[ind] = sorts.signals.hard_target_snr_scaling(
                pop['d'][objs][ind], 
                sample_d[ind], 
                radar.tx[0].wavelength,
            )
        new_snr = 10.0*np.log10(data['snr']*scaling[oid_to_data])

        snr_F, snr_X = np.histogram(new_snr, bins=100)
        snr_X = (snr_X[:-1] + snr_X[1:])*0.5

        # l1.set_xdata(data['t']/3600.0)
        # l1.set_ydata(new_snr)
        H, _, _ = np.histogram2d(data['t']/3600.0, new_snr, bins=[xedges, yedges])
        pmesh.update({'array': H.T.ravel()})

        l2.set_xdata(snr_X)
        l2.set_ydata(snr_F)

        l3a.set_xdata(samp_X)
        l3a.set_ydata(samp_F)

        l3b.set_xdata(log10_d)
        l3b.set_ydata(log10_d_P)

        axes[1].set_xlim([snr_X.min(), snr_X.max()])
        axes[1].set_ylim([snr_F.min(), snr_F.max()])

        axes[2].set_xlim([np.append(samp_X, log10_d).min(), np.append(samp_X, log10_d).max()])
        axes[2].set_ylim([np.append(samp_F, log10_d_P).min(), np.append(samp_F, log10_d_P).max()])

        if draw:
            fig.canvas.draw_idle()

    slide_mu.on_changed(update)
    slide_std.on_changed(update)

    update(None, draw=False)

    plt.show()

    return fig, axes


def explore(config, pop, radar):
    base_path = pathlib.Path(config.get('Cache', 'output_folder'))
    data_path = base_path / 'simulation' / 'master' / 'collected_results.npy'

    data = np.load(data_path)

    print(f'Data shape = {data.shape}')

    bins = 30

    fig, ax = plt.subplots(1, 1)
    ax.hist(np.log10(pop['d']))
    ax.set_xlabel('Diameter [log10(m)]')
    ax.set_ylabel('Frequency [1]')

    gui_fig_sn, gui_axes_sn = snr_gui(data, pop, config, radar)

    gui_fig, gui_axes = distribution_gui(data, pop, config)

    a, e, inc, aop, raan, nu = get_sample_vectors(config)

    # sample_vectors = get_sample_vectors(config)

    fig, ax = plt.subplots(1, 1)
    ax.hist2d(data['range']*1e-3, data['range_rate']*1e-3, bins=bins)
    ax.set_xlabel('Range [km]')
    ax.set_ylabel('Range rate [km/s]')

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].hist2d(pop['i'][data['oid']], data['t']/3600.0, bins=[len(inc), bins])
    axes[0, 0].set_xlabel('Inclination [deg]')
    axes[0, 0].set_ylabel('Time [h]')

    axes[0, 1].hist2d(pop['i'][data['oid']], data['range']*1e-3, bins=[len(inc), bins])
    axes[0, 1].set_xlabel('Inclination [deg]')
    axes[0, 1].set_ylabel('Range [km]')

    axes[1, 0].hist2d(pop['i'][data['oid']], data['range_rate']*1e-3, bins=[len(inc), bins])
    axes[1, 0].set_xlabel('Inclination [deg]')
    axes[1, 0].set_ylabel('Range rate [km/s]')

    axes[1, 1].hist2d(pop['i'][data['oid']], 10*np.log(data['snr']), bins=[len(inc), bins])
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


def invert(config, population, radar):
    data_files = config.items("Measurement paths")
    data_files = [pathlib.Path(x) for key, x in data_files]
    assert len(data_files) > 0, 'No measurement data paths in configuration'

    # Load forward model data
    base_path = pathlib.Path(config.get('Cache', 'output_folder'))
    model_path = base_path / 'simulation' / 'master' / 'collected_results.npy'

    plots_folder = base_path / 'density_estimation'
    plots_folder.mkdir(exist_ok=True)

    model_data = np.load(model_path)

    objs = np.unique(model_data['oid'])
    oid_to_data = np.empty(model_data['snr'].shape, dtype=np.int64)
    for ind in range(len(model_data['oid'])):
        oid_to_data[ind] = np.argwhere(objs == model_data['oid'][ind])

    log10d_mu = config.getfloat('Measurements', 'log10d_mu')
    log10d_std = config.getfloat('Measurements', 'log10d_std')

    # Just random sample once
    xi_samps = np.random.randn(len(objs))
    sample_d = 10.0**(xi_samps*log10d_std + log10d_mu)

    scaling = np.empty_like(sample_d)
    for ind in range(len(sample_d)):
        scaling[ind] = sorts.signals.hard_target_snr_scaling(
            population['d'][objs][ind], 
            sample_d[ind], 
            radar.tx[0].wavelength,
        )
    model_data['snr'] = 10.0*np.log10(model_data['snr']*scaling[oid_to_data])

    min_snr_db = config.getfloat('Measurements', 'min_snr_db')
    keep = model_data['snr'] > min_snr_db
    model_data = model_data[keep]

    print(f'Total model data shape ({len(keep) - np.sum(keep)} low SNR removed): {model_data.shape}')

    t_samp, r_samp, v_samp, snr_samp = get_measurnment_vectors(config)
    a, e, inc, aop, raan, nu = get_sample_vectors(config)

    # Load measurement data
    names = ['t', 'r', 'v', 'snr']
    model_names = ['t', 'range', 'range_rate', 'snr']
    orb_names = ['a', 'e', 'i', 'raan', 'aop']

    data_epoch = Time(config.get('Measurements', 'epoch'), format='isot')
    print(f'Using measurement epoch: {data_epoch}')

    sim_epoch = Time(config.get('General', 'epoch'), format='isot')

    sidereal_day = 86164.0905

    depoch = -np.mod((data_epoch - sim_epoch).sec/sidereal_day, 1)

    print(f'Offsetting simulation epoch by {depoch} sidereal days to match observations')
    sim_epoch_offset = depoch*sidereal_day

    model_data['t'] += sim_epoch_offset

    meas_data = []
    for data_pth in data_files:
        # Load data and append to common array
        with h5py.File(str(data_pth), 'r') as h_det:
            tmp_data = np.empty((len(h_det['t']), 4), dtype=np.float64)

            for ind, key in enumerate(names):
                tmp_data[:, ind] = h_det[key][()]

            # km -> m
            tmp_data[:, 1:3] *= 1e3

            # Shift to epoch relative time
            tmp_data[:, 0] -= data_epoch.unix

            # We use dB for SNR
            tmp_data[:, 3] = 10.0*np.log10(tmp_data[:, 3])

            meas_data.append(tmp_data)

    meas_data = np.concatenate(meas_data, axis=0)

    # Filter out nans
    not_nan = np.sum(np.isnan(meas_data), axis=1) == 0
    print(f'Removing {meas_data.shape[0] - np.sum(not_nan)} NaN measurements')
    meas_data = meas_data[not_nan, :]

    print(f'Total measurement data shape: {meas_data.shape}')

    meas_samps = [t_samp, r_samp, v_samp, snr_samp]

    # Create discretization of measurements
    measurement_density, _ = np.histogramdd(
        meas_data,
        bins=meas_samps,
    )

    # Create discretization of model
    model_density, _ = np.histogramdd(
        np.stack([model_data[key] for key in model_names], axis=1),
        bins=meas_samps,
    )

    orb_edges = []
    orb_samps = []
    orb_keys = []
    for key, samps in zip(orb_names, [a, e, inc, aop, raan]):
        if len(samps) > 1:
            print(f'Using sample vector in "{key}" for theory matrix')
            orb_edges.append(get_edges(samps))
            orb_samps.append(samps)
            orb_keys.append(key)

    model_cols = [model_data[key] for key in model_names]
    pop_cols = [population[key][model_data['oid']] for key in orb_keys]
    complete_data_cols = model_cols + pop_cols

    complete_data_array = np.stack(complete_data_cols, axis=1)

    print(f'Model and population joint data array shape: {complete_data_array.shape}')

    del complete_data_cols
    del model_cols
    del pop_cols

    R, _ = np.histogramdd(
        complete_data_array,
        bins=meas_samps + orb_edges,
    )

    print(f'Theory map    shape: {R.shape} = {R.size} elements')

    orb_sizes = tuple(len(x) for x in orb_samps)
    meas_sizes = tuple(len(x) - 1 for x in meas_samps)
    total_orb_samps = np.prod([len(x) for x in orb_samps])
    total_meas_samps = np.prod([len(x) - 1 for x in meas_samps])

    intermed_size = (total_meas_samps, ) + orb_sizes
    
    R = np.reshape(R, intermed_size)
    R = np.reshape(R, (total_meas_samps, total_orb_samps))

    print(f'Theory matrix shape: {R.shape} = {R.size} elements')

    # Normalize to orbit probability by deviding by mean anomaly sampling
    R /= float(config.getint('Population', 'orbit_samples'))

    radar_name = config.get('General', 'radar')
        
    for dati, ptr in enumerate([measurement_density, model_density]):
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        _dens = np.sum(ptr, axis=(2, 3))
        _X, _Y = np.meshgrid(get_midpoints(t_samp), get_midpoints(r_samp), indexing='ij')

        axes[0, 0].pcolormesh(_X/3600.0, _Y*1e-3, _dens)
        axes[0, 0].set_xlabel('Time [h]')
        axes[0, 0].set_ylabel('Range [km]')

        _dens = np.sum(ptr, axis=(1, 3))
        _X, _Y = np.meshgrid(get_midpoints(t_samp), get_midpoints(v_samp), indexing='ij')

        axes[0, 1].pcolormesh(_X/3600.0, _Y*1e-3, _dens)
        axes[0, 1].set_xlabel('Time [h]')
        axes[0, 1].set_ylabel('Range rate [km/s]')

        _dens = np.sum(ptr, axis=(0, 3))
        _X, _Y = np.meshgrid(get_midpoints(r_samp), get_midpoints(v_samp), indexing='ij')

        axes[1, 0].pcolormesh(_X*1e-3, _Y*1e-3, _dens)
        axes[1, 0].set_xlabel('Range [km]')
        axes[1, 0].set_ylabel('Range rate [km/s]')

        _dens = np.sum(ptr, axis=(0, 2))
        _X, _Y = np.meshgrid(get_midpoints(r_samp), get_midpoints(snr_samp), indexing='ij')

        axes[1, 1].pcolormesh(_X*1e-3, _Y, _dens)
        axes[1, 1].set_xlabel('Range [km]')
        axes[1, 1].set_ylabel('SNR [dB]')

        if dati == 0:
            fig.suptitle(f'{radar_name}: Measurement density marginals')
            fig.savefig(plots_folder / 'measurement_density_marginals.png')
        elif dati == 1:
            fig.suptitle(f'{radar_name}: Model density marginals')
            fig.savefig(plots_folder / 'model_density_marginals.png')

    measurement_density = np.reshape(measurement_density, (total_meas_samps, ))

    print('Performing theory metrix inversion ...')
    print(f'R rank: {np.linalg.matrix_rank(R)} ')

    print('Removing cols that do not contribute rank (does not produce measurements) ...')
    print(f'R rank: {np.linalg.matrix_rank(R)} == {R.shape} ')
    keep_cols = np.sum(R, axis=0) > 0

    print(f'{R.shape[1] - np.sum(keep_cols)} cols removed')
    R = R[:, keep_cols]
    print(f'R rank: {np.linalg.matrix_rank(R)} == {R.shape} ')

    # # Manual inversion?
    # X = R.T @ R
    # print('X = R^T R')
    # print(f'X condition: {np.linalg.cond(X)} < {1/sys.float_info.epsilon}')
    # print(f'X rank: {np.linalg.matrix_rank(X)} == {X.shape} ')

    # # fig, ax = plt.subplots(1, 1)
    # # ax.imshow(R)
    # # plt.show()

    # # is it right to use Moore-Penrose? or should we try solve()? or .inv()?
    # X = np.linalg.pinv(X)
    # print('Xi = inv(X)')
    # R = X @ R.T
    # print('Ri = Xi R^T')
    # # Resue memory to save ram, hence use R instead of Ri

    print('Estimating density ...')

    # _varrho = R @ measurement_density

    _varrho = sop.lsq_linear(
        R, 
        measurement_density, 
        bounds = (
            np.zeros(R.shape[1], dtype=np.float64),
            np.full(R.shape[1], np.inf, dtype=np.float64),
        ),
    )
    print(_varrho)

    varrho = np.full((total_orb_samps, ), np.nan, dtype=np.float64)
    varrho[keep_cols] = _varrho.x

    print('Density estimation done')
    varrho = np.reshape(varrho, orb_sizes)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    _X, _Y = np.meshgrid(a, inc, indexing='ij')

    mesh = ax.pcolormesh(_X*1e-3, _Y, varrho)
    ax.set_xlabel('Semi-major-axis [km]')
    ax.set_ylabel('Inclination [deg]')
    ax.set_title('Estimated orbit number density')
    fig.colorbar(mesh)

    fig.savefig(plots_folder / 'population_density_estimation.png')

    plt.show()


def main(input_args=None):

    handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    parser = argparse.ArgumentParser(description='Forward model to detmerine observation probabilitiy')
    parser.add_argument('config', type=pathlib.Path, help='Path to config')
    parser.add_argument(
        'action',
        choices=['run', 'collect', 'explore', 'invert'],
        default='run',
        const='run',
        nargs='?',
        help='Execute action',
    )
    parser.add_argument('-c', '--clean', action='store_true', help='Remove simulation data before running')
    
    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

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

    # For consistency
    np.random.seed(config.getint('General', 'seed'))

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
    elif args.action == 'explore':
        explore(config, population, radar)
    elif args.action == 'invert':
        invert(config, population, radar)

    sim.profiler.stop('total')
    print(sim.profiler.fmt(normalize='total'))


if __name__ == '__main__':
    main()
