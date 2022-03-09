#!/usr/bin/env python

'''
Example execution 

RUN:

PLOT:

./beampark_elevation_doppler.py plot cache/**/az*.h5 -o ./plots/
./beampark_elevation_doppler.py plot -d /home/danielk/data/spade/beamparks/uhf/2021.11.23.h5 -o ./plots/ -- ./cache/eiscat_uhf/az90.0_el75.0.h5

'''

import sys
import datetime
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import Time, TimeDelta
import scipy.optimize as sio
import h5py
from tqdm import tqdm

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

HERE = Path(__file__).parent

Re_km = 6371


def ecef_close(ecef1, ecef2, atol_pos=1e2, atol_vel=1e-1):
    return np.allclose(ecef1[:3], ecef2[:3], atol=atol_pos) and \
           np.allclose(ecef1[3:], ecef2[3:], atol=atol_vel)


def get_off_axis_angle(station, states, ecef_k_vector):
    """
    one dimensional angle between on-axis position and target
    """

    dstate = states[:3, :] - station.ecef[:, None]

    k_dot_p = np.sum(dstate*ecef_k_vector[:, None], axis=0)

    p_dot_p = np.sqrt(np.sum(dstate**2.0, axis=0))
    angle = np.arccos(k_dot_p/p_dot_p)
    return angle


def get_range_and_range_rate(station, states):
    """
    return the range and range rate of a radar measurement
    in units of meters and meters per second.

     states is an array 6xN with N state vectors
     dims 0,1,2 = x,y,z  (m)
          3,4,5 = vx,vy,vz (m/s)
    """
    k = np.zeros([3, states.shape[1]], dtype=np.float64)

    k = states[:3, :] - station.ecef[:, None]

    # ranges (distance from radar to satellite)
    ranges = np.sqrt(np.sum(k**2.0, axis=0))

    k_unit = k/ranges

    # k dot v where k in unit normalized
    range_rates = np.sum(k_unit*states[3:, :], axis=0)

    return ranges, range_rates


def plot_inclination_results(anom, Omega, angle, ranges, range_rates, samples, orb_hits=None):
    fig, ax = plt.subplots()
    pc = ax.pcolormesh(
        anom.reshape(samples),
        Omega.reshape(samples),
        angle.reshape(samples),
        shading='gouraud',
    )
    fig.colorbar(pc, ax=ax)
    if orb_hits is not None:
        for x in orb_hits:
            ax.plot([np.mod(x[0] + 360, 360)], [np.mod(x[1] + 360, 360)], 'or')

    figs = [fig]
    axes = [ax]

    fig, ax = plt.subplots(1, 2)
    pc = ax[0].pcolormesh(
        anom.reshape(samples),
        Omega.reshape(samples),
        ranges.reshape(samples),
        shading='gouraud',
    )
    fig.colorbar(pc, ax=ax[0])
    pc = ax[1].pcolormesh(
        anom.reshape(samples),
        Omega.reshape(samples),
        range_rates.reshape(samples),
        shading='gouraud',
    )
    fig.colorbar(pc, ax=ax[1])

    figs += [fig]
    axes += [ax]
    return figs, axes


def compute_ecef_states(x, epoch, semi_major_axis, inclination):
    """
    x:  2-tuple of orbit elements mean anomaly and longitude of ascending node
    epoch: time for transformation between inertial and Earth-fixed systems
    semi_major_axis, inclination: the other orbital elements

    Returns 6-element state vector(s) [px, py, pz, vx, vy, vz]
    with units m for position and m/s for velocity

    Factored out as common to the two functions below
    """
    orbit = pyorb.Orbit(
        M0 = pyorb.M_earth,
        degrees = True,
        a = semi_major_axis,
        e = 0,
        i = inclination,
        omega = 0,
        Omega = x[1],
        anom = x[0],
        num = 1,
        type = 'mean',
    )
    return sorts.frames.convert(epoch, orbit.cartesian, in_frame='GCRS', out_frame='ITRS')


def optim_off_axis_angle(x, epoch, st, k_ecef, semi_major_axis, inclination):
    ecef = compute_ecef_states(x, epoch, semi_major_axis, inclination)
    return np.degrees(get_off_axis_angle(st, ecef, k_ecef))


def orb_to_range_and_range_rate(x, epoch, st, semi_major_axis, inclination):
    ecef = compute_ecef_states(x, epoch, semi_major_axis, inclination)
    return get_range_and_range_rate(st, ecef)


def cache_path(station, radar_name='generic', topdir='cache', azim=None, elev=None, degrees=True):
    """
    Create Path of file for r_rdot cache, to make files findable

    Arguments:
    - station: An instance of sorts.radar.tx_rx.Tx or Rx.
        The station is used for geographic location, and the first beam found
        is used for pointing (but see azim and elev keywords, below)
    Keyword arguments:
    - radar_name: string.  Default: 'generic'
        The name used for the cache directory.
    - topdir: string or Path. Default: 'cache'
        Path for cache dir.
    - azim: scalar, default: None
        If given, will be used to set beam pointing
    - elev: scalar, default: None
        If given, will be used to set beam pointing
    - degrees: boolean, default: True
        If True, azim and elev are given in degrees

    scheme: <topdir>/<radar_name>/az<value>_el<value>.h5
        where <value> is in degrees with one decimal and no leading zeroes except '0.x'

    example:  cache/eiscat_esr/az181.6_el81.6.h5
    """

    if azim is None:
        azim = station.beam.azimuth
    if elev is None:
        elev = station.beam.elevation

    if not degrees:
        azim = np.degrees(azim)
        elev = np.degrees(elev)

    fname = f'az{azim:.1f}_el{elev:.1f}.h5'
    return Path(topdir) / radar_name / fname


def build_cache(station, radar_name='generic', cache_dir=HERE / 'cache', azim=None, elev=None,
                degrees=True, clobber=False, dry_run=False):
    """
    Create a cache file of r_rdot computations for a given radar and pointing direction.

    Arguments:
    - station: An instance of sorts.radar.tx_rx.Tx or Rx.
        The station is used for geographic location, and the first beam found
        is used for pointing (but see azim and elev keywords, below)
    Keyword arguments:
    - radar_name: string.  Default: 'generic'
        The Path or name used for the first sublevel of the cache directory.
    - cache_dir: string or Path. Default: 'cache'
        The Path or name used for the toplevel cache directory.
    - azim: scalar, default: None
        If given, will be used to set beam pointing
    - elev: scalar, default: None
        If given, will be used to set beam pointing
    - degrees: boolean, default: True
        If True, azim and elev are given in degrees
    - clobber: boolean, default: False
        If True, existing cache files will be overwritten
    - dry_run: boolean, default: False
        If True, nothing will be written to file

    Structure of generated file:

    Dimensions:
      incl:
        long name: orbit plane inclination
        units: degrees
        values: from 50 to 130 in increments of 0.5

      sema:
        long name: semi-major axis
        units: metres
        values: between 300 and 3000 km above Earth radius

    Datasets:
      nu_a(incl, sma):
        long name: mean anomaly for ascending transit
        units: degrees
      nu_d(incl, sma):
        long name: mean anomaly for descending transit
        units: degrees
      Om_a(incl, sma):
        long name: longitude of ascending node for ascending transit
        units: degrees
      Om_d(incl, sma):
        long name: longitude of ascending node for descending transit
        units: degrees
      theta_a(incl, sma):
        long name: minimum off-axis angle for ascending transit
        units: degrees
      theta_d(incl, sma):
        long name: minimum off-axis angle for descending transit
        units: degrees
      r_a(incl, sma):
        long name: range at minimum off-axis angle for ascending transit
        units: metres
      r_d(incl, sma):
        long name: range at minimum off-axis angle for descending transit
        units: metres
      rdot_a(incl, sma):
        long name: range rate at minimum off-axis angle for ascending transit
        units: metres/second
      rdot_d(incl, sma):
        long name: range rate at minimum off-axis angle for descending transit
        units: metres/second
    """

    _2pi = 2 * np.pi

    cpath = cache_path(station, radar_name=radar_name,
                    topdir=cache_dir, azim=azim, elev=elev, degrees=degrees)

    if cpath.exists() and not clobber and not dry_run:
        raise FileExistsError('Cache file {cpath.name} exists in cache')

    if azim is None:
        azim = station.beam.azimuth
    if elev is None:
        elev = station.beam.elevation

    station.beam.sph_point(azim, elev, radians=False)
    k_ecef = station.pointing_ecef

    # Epoch of observation
    epoch = Time("2000-01-01T00:10:00Z", format='isot')

    # Values of dimension axis for orbit semimajor axis
    sema = 1000 * (Re_km + np.linspace(300, 3000, 20))

    # Values of dimension axis for orbit plane inclination
    incl = np.r_[69:111.1:0.5]

    if comm.rank == 0 and dry_run:
        incl = np.r_[92:100.1:0.5]
        print("\n\n  *** DEBUG LIMITED INCLINATIONS ***\n\n")

    if comm.rank == 0 and not dry_run:
        print('Rank-0: Setting up cache file')
        with h5py.File(cpath, 'w') as ds:

            # Create global attributes for dataset
            ds.attrs['radar_name'] = radar_name
            ds.attrs['epoch'] = epoch.isot

            # Create scalar variables for dataset constants
            def _create_scalar_var(name, long_name, units, value):
                ds.create_dataset(name, shape=(), data=value)
                var = ds[name]
                var.attrs['long_name'] = long_name
                var.attrs['units'] = units

            _create_scalar_var('site_lat', 'Geodetic latitude of site', 'degrees', station.lat)
            _create_scalar_var('site_lon', 'Geodetic longitude of site', 'degrees', station.lon)
            _create_scalar_var('site_alt', 'Altitude above MSL of site', 'metres', station.alt)
            _create_scalar_var('azimuth',  'Azimuth of pointing direction', 'degrees', azim)
            _create_scalar_var('elevation','Elevation of pointing direction', 'degrees', elev)

            # Create dimension axis for orbit plane inclination
            ds['incl'] = incl.copy()
            ds_incl = ds['incl']
            ds_incl.make_scale('inclination')
            ds_incl.attrs['long_name'] = 'orbit plane inclination'
            ds_incl.attrs['units'] = 'degrees'
            ds_incl.attrs['valid_range'] = 0.0, 180.0

            # Create dimension axis for orbit semimajor axis
            ds['sema'] = sema.copy()
            ds_sema = ds['sema']
            ds_sema.make_scale('semimajor_axis')
            ds_sema.attrs['long_name'] = 'orbit semimajor axis'
            ds_sema.attrs['units'] = 'metres'
            ds_sema.attrs['valid_range'] = 1e0, 1e12

            def _create_ia_var(name, long_name, units, valid_range):
                ds[name] = np.zeros((len(ds_incl), len(ds_sema)))
                var = ds[name]
                var.dims[0].attach_scale(ds_incl)
                var.dims[1].attach_scale(ds_sema)
                var.attrs['long_name'] = long_name
                var.attrs['units'] = units
                var.attrs['valid_range'] = valid_range

            _create_ia_var('nu_a', 'mean anomaly for ascending transit', 'degrees', (0.0, 360.0))
            _create_ia_var('nu_d', 'mean anomaly for descending transit', 'degrees', (0.0, 360.0))
            _create_ia_var('Om_a', 'longitude of ascending node for ascending transit', 'degrees', (0.0, 360.0))
            _create_ia_var('Om_d', 'longitude of ascending node for descending transit', 'degrees', (0.0, 360.0))
            _create_ia_var('theta_a', 'minimum off-axis angle for ascending transit', 'degrees', (0.0, 360.0))
            _create_ia_var('theta_d', 'minimum off-axis angle for descending transit', 'degrees', (0.0, 360.0))
            _create_ia_var('r_a', 'range at minimum off-axis angle for ascending transit', 'metres', (1e0, 1e12))
            _create_ia_var('r_d', 'range at minimum off-axis angle for descending transit', 'metres', (1e0, 1e12))
            _create_ia_var('rdot_a', 'range rate at minimum off-axis angle for ascending transit', 'metres/second', (1e0, 1e12))
            _create_ia_var('rdot_d', 'range rate at minimum off-axis angle for descending transit', 'metres/second', (1e0, 1e12))

    # First some invariants
    samples = 101, 103  # for mean anomaly and longitude of ascending node, respectively
    # Now start looking for orbits that intersect the pointing of the beam
    anom, Omega = np.meshgrid(
        np.linspace(0, 360, num=samples[0]),
        np.linspace(0, 360, num=samples[1]),
        indexing='ij',
    )
    anom.shape = -1             # unravel
    Omega.shape = -1

    orbit = pyorb.Orbit(
        M0 = pyorb.M_earth,
        degrees = True,
        a = sema[0],
        e = 0,
        i = 0,
        omega = 0,
        Omega = Omega,
        anom = anom,
        num = anom.size,
        type = 'mean',
    )

    pbars = []
    iters = []
    for rank in range(comm.size):
        iters.append(list(range(rank, len(incl), comm.size)))
        pbars.append(tqdm(total=len(iters[-1])*len(sema), position=rank))
    pbar = pbars[comm.rank]

    var_names = [
        'nu_a', 'nu_d', 'Om_a',
        'Om_d', 'theta_a', 'theta_d',
        'r_a', 'r_d', 'rdot_a',
        'rdot_d',
    ]
    RAM_ds = {key: np.empty((len(incl), len(sema)), dtype=incl.dtype) for key in var_names}

    for ii in iters[comm.rank]:
        inci = incl[ii]
        orbit.i = inci

        for kk, a in enumerate(sema):
            orbit.a = a

            states = orbit.cartesian
            ecef = sorts.frames.convert(epoch, states, in_frame='GCRS', out_frame='ITRS')
            angle = np.degrees(get_off_axis_angle(station, ecef, k_ecef))

            ranges, range_rates = get_range_and_range_rate(station, ecef)
            inds = np.argsort(angle)
            x_start = [anom[inds[0]], Omega[inds[0]]]
            # Find first minimum
            xhat1 = sio.minimize(
                lambda x: optim_off_axis_angle(x, epoch, station, k_ecef, a, inci),
                x_start,
                method='Nelder-Mead',
            )

            if xhat1.fun > 1.0:
                # If the best we could do was more than 1 degrees off axis,
                # then there's no point in looking for a second solution
                xhat2 = xhat1
                r1, rdot1 = orb_to_range_and_range_rate(xhat1.x, epoch, station, a, inci)
                r2, rdot2 = r1, rdot1
            else:
                # Clever trick here.
                # Find the index of the smallest angle which satisfies the inequality
                other_ind = np.argmax(np.abs(anom[inds] - anom[inds[0]]) > 5.0) #
                # other_ind = np.argmax(anom[inds] > anom[inds[0]] + 5.0)       # Nope, bad idea
                x_start2 = [anom[inds[other_ind]], Omega[inds[other_ind]]]
                xhat2 = sio.minimize(
                    lambda x: optim_off_axis_angle(x, epoch, station, k_ecef, a, inci),
                    x_start2,
                    method='Nelder-Mead',
                )
                # now we have two solutions, figure out if they are distinct,
                # which one is ascending and descending, and their ranges and range rates,
                # then put all of those (and the off-axis angles) into the file

                # ascending node is the one with positive z velocity in ECEF
                ecef1 = compute_ecef_states(xhat1.x, epoch, a, inci)
                ecef2 = compute_ecef_states(xhat2.x, epoch, a, inci)

                if not ecef_close(ecef1, ecef2):
                    assert np.sign(ecef1.item(5) * ecef2.item(5)) < 0, \
                        "Should be one ascending and one descending solution!"

                # make sure xhat1 is the ascending solution
                if np.sign(ecef1.item(5)) < 0:
                    xhat1, xhat2 = xhat2, xhat1
                    ecef1, ecef2 = ecef2, ecef1

                r1, rdot1 = get_range_and_range_rate(station, ecef1)
                r2, rdot2 = get_range_and_range_rate(station, ecef2)

            # Store data
            RAM_ds['nu_a'][ii,kk] = xhat1.x[0]
            RAM_ds['nu_d'][ii,kk] = xhat2.x[0]
            RAM_ds['Om_a'][ii,kk] = xhat1.x[1]
            RAM_ds['Om_d'][ii,kk] = xhat2.x[1]
            RAM_ds['theta_a'][ii,kk] = xhat1.fun
            RAM_ds['theta_d'][ii,kk] = xhat2.fun
            RAM_ds['r_a'][ii,kk] = r1[0]
            RAM_ds['r_d'][ii,kk] = r2[0]
            RAM_ds['rdot_a'][ii,kk] = rdot1[0]
            RAM_ds['rdot_d'][ii,kk] = rdot2[0]

            # Iter done
            pbar.update(1)

    for pb in pbars:
        pb.close()

    def tagger(data_id, key_id):
        return data_id*100 + key_id

    if comm.size > 1:
        if comm.rank == 0:
            print('Rank-0: Receiving all results')

            for T in range(1, comm.size):
                for ID in range(T, len(incl), comm.size):
                    for ki, key in enumerate(var_names):
                        RAM_ds[key][ID,:] = comm.recv(source=T, tag=tagger(ID, ki))
                    print(f'received packet {ID} from PID{T}')
        else:
            print(f'Rank-{comm.rank}: Sending results')

            for ID in range(comm.rank, len(incl), comm.size):
                for ki, key in enumerate(var_names):
                    comm.send(RAM_ds[key][ID,:], dest=0, tag=tagger(ID, ki))
        print(f'Rank-{comm.rank}: Data distributing done')

    if comm.rank == 0 and not dry_run:
        with h5py.File(cpath, 'a') as ds:
            print(f'Rank-{comm.rank}: Saving to cache file')
            for key in var_names:
                ds[key][:, :] = RAM_ds[key]
        print(f'Rank-{comm.rank}: Cache file done')


def plot_from_cachefile(cachefilename, incl=None, linestyles=None, symb=None, cmap=None, legend=True):

    good_limit = 1.0            # how much off-axis pointing is okay (degrees)

    if linestyles is None:
        lst = {'ok_a' : '-',
               'ok_d' : '--',
               'nok_a': ':',
               'nok_d': ':'}
    else:
        lst = linestyles.copy()

    if symb is not None:
        for k in lst:
            lst[k] = symb + lst[k]

    if cmap is None:
        cmap = plt.get_cmap('tab10')

    if incl is None:
        incl = np.arange(70, 110.1, 5)

    with h5py.File(cachefilename, 'r') as ds:

        dsincl = ds['incl'][:]
        sma = ds['sema'][:]
        height = sma - Re_km*1e3       # height over average Earth radius

        lh = []
        lab = []
        for ii, inci in enumerate(incl):
            ix = np.where(dsincl == inci)[0][0]

            theta_a = ds['theta_a'][ix,:]
            r_a = ds['r_a'][ix,:]
            rdot_a = ds['rdot_a'][ix,:]

            theta_d = ds['theta_d'][ix,:]
            r_d = ds['r_d'][ix,:]
            rdot_d = ds['rdot_d'][ix,:]

            ok_a = np.where(theta_a <= 1.0)
            nok_a = np.where(theta_a >= 1.0)
            ok_d = np.where(theta_d <= 1.0)
            nok_d = np.where(theta_d >= 1.0)

            hh = plt.plot(r_a[ok_a],  rdot_a[ok_a],  lst['ok_a'],
                          r_d[ok_d],  rdot_d[ok_d],  lst['ok_d'],
                          r_a[nok_a], rdot_a[nok_a], lst['nok_a'],
                          r_d[nok_d], rdot_d[nok_d], lst['nok_d'], color=cmap(ii))
            lh.append(hh[0])
            lab.append(f'incl = {inci}')

        lat = ds['site_lat'][()]
        slat = f'{lat:.1f}N' if lat >= 0 else f'{-lat:.1f}S'
        lon = ds['site_lon'][()]
        slon = f'{lon:.1f}E' if lon >= 0 else f'{-lon:.1f}W'
        az = ds['azimuth'][()]
        el = ds['elevation'][()]

        # Use the 'units' attribute of the dataset to label plots
        plt.title(f'range rate vs range at {slat} {slon} az {az:.1f} el {el:.1f}')
        plt.xlabel(f"range [{ds['r_a'].attrs['units']}]")
        plt.ylabel(f"range rate [{ds['rdot_a'].attrs['units']}]")

        if legend:
            plt.legend(lh, lab)
        else:
            return lh, lab


def unix2datestr(x):
    d0 = np.floor(x/60.0)
    frac = x-d0*60.0
    stri = datetime.datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:')
    return "%s%02.2f" % (stri, frac)


def plot_with_measurement_data(
            data_fname, 
            cache_fname, 
            site='unknown', 
            date='', 
            out_path=None, 
            out_name='',
        ):

    with h5py.File(data_fname, "r") as h:
        v = h["v"][()]*1e3
        rgs = h["r"][()]*1e3
    
    pincs = np.array([70, 74, 82, 87, 90, 95, 99, 102])

    fig = plt.figure(figsize=(10, 8))

    # info on i and a
    H, x, y = np.histogram2d(v, rgs, range=[[-3e3, 3e3], [0, 3000e3]], bins=(100, 100))
    
    plt.subplot(121)
    plot_from_cachefile(cache_fname, incl=pincs)

    cdops = np.copy(v)
    cdops[cdops > 4] = 4e3
    cdops[cdops < -4] = -4e3

    plt.scatter(rgs, v, s=1, c="black", lw=0, cmap="Spectral")
    plt.title("Detections")
    plt.ylim([-2.5e3, 2.5e3])

    plt.subplot(122)
    plot_from_cachefile(cache_fname, incl=pincs, legend=False)

    plt.pcolormesh(y, x, H)
    plt.title("Histogram")
    plt.ylabel("")

    plt.colorbar()
    fig.suptitle(f'Observations at {site} {date}')
    plt.tight_layout()
    
    if out_path is None:
        plt.show()
    else:
        fig.savefig(out_path / ('r_dr_meas_' + out_name))
        plt.close(fig)
    

def draw_annotated_maps(cachefilename, station, ii, kk, ah=None):

    with h5py.File(cachefilename, 'r') as ds:

        incl = ds['incl'][:]
        sema = ds['sema'][:]

        azim = ds['azimuth'][()]
        elev = ds['elevation'][()]

        Om_a = ds['Om_a'][ii][kk]
        Om_d = ds['Om_d'][ii][kk]

        nu_a = ds['nu_a'][ii][kk]
        nu_d = ds['nu_d'][ii][kk]

        r_a = ds['r_a'][ii][kk]
        r_d = ds['r_d'][ii][kk]

        rdot_a = ds['rdot_a'][ii][kk]
        rdot_d = ds['rdot_d'][ii][kk]

        epoch = Time(ds.attrs['epoch'], format='isot')

    inci = incl[ii]
    a = sema[kk]

    print(f"inclination: {inci}, semimajor axis {a:.0f}m (altitude {a/1000 - Re_km:.1f} km)")
    print(f"ascending  range {r_a/1000:.1f} km, range rate {rdot_a/1000:.3f} km/s")
    print(f"descending range {r_d/1000:.1f} km, range rate {rdot_d/1000:.3f} km/s")

    # Epoch of observation

    shape = 101, 103
    nux = np.linspace(0, 360, shape[0])
    Omx = np.linspace(0, 360, shape[1])

    station.beam.sph_point(azim, elev, radians=False)
    k_ecef = station.pointing_ecef

    anom, Omega = np.meshgrid(nux, Omx, indexing='ij')
    anom.shape = -1             # unravel
    Omega.shape = -1

    orbit = pyorb.Orbit(
        M0 = pyorb.M_earth,
        degrees = True,
        a = sema[0],
        e = 0,
        i = 0,
        omega = 0,
        Omega = Omega,
        anom = anom,
        num = anom.size,
        type = 'mean',
    )

    orbit.i = inci
    orbit.a = a

    # maps
    states = orbit.cartesian
    ecef = sorts.frames.convert(epoch, states, in_frame='GCRS', out_frame='ITRS')
    angle = np.degrees(get_off_axis_angle(station, ecef, k_ecef))
    r, rdot = get_range_and_range_rate(station, ecef)

    if ah is None:
        ah = [plt.gca()]
    try:
        ah.shape = -1
        n_plots = len(ah)
    except TypeError:
        ah = [ah]
        n_plots = 1

    if n_plots >= 1:
        angle.shape = shape
        imh = ah[0].pcolormesh(nux, Omx, angle.T);
        ah[0].text(nu_a, Om_a, 'A', color='red')
        ah[0].text(nu_d, Om_d, 'D', color='red')
        ah[0].plot(nu_a, Om_a, 'r+', nu_d, Om_d, 'rx')
        plt.colorbar(imh, ax=ah[0], fraction=0.04, pad=0.01)
        ah[0].set_xlabel('mean anomaly [$\\nu$]')
        ah[0].set_ylabel('longitude of ascending node [$\Omega$]')
        ah[0].set_title('Off-axis angle [deg]')

    if n_plots >= 2:
        r.shape = shape
        rmin = min(r_a, r_d)
        imh = ah[1].pcolormesh(nux, Omx, r.T-rmin, vmax=4000);
        ah[1].text(nu_a, Om_a, 'A', color='red')
        ah[1].text(nu_d, Om_d, 'D', color='red')
        ah[1].plot(nu_a, Om_a, 'r+', nu_d, Om_d, 'rx')
        plt.colorbar(imh, ax=ah[1], fraction=0.04, pad=0.01)
        ah[1].set_xlabel('mean anomaly [$\\nu$]')
        ah[1].set_ylabel('longitude of ascending node [$\Omega$]')
        ah[1].set_title('Range [m]')

    if n_plots >= 3:
        rdot.shape = shape
        imh = ah[2].pcolormesh(nux, Omx, rdot.T);
        ah[2].text(nu_a, Om_a, 'A', color='red')
        ah[2].text(nu_d, Om_d, 'D', color='red')
        ah[2].plot(nu_a, Om_a, 'r+', nu_d, Om_d, 'rx')
        plt.colorbar(imh, ax=ah[2], fraction=0.04, pad=0.01)
        ah[2].set_xlabel('mean anomaly [$\\nu$]')
        ah[2].set_ylabel('longitude of ascending node [$\Omega$]')
        ah[2].set_title('Range rate [m/s]')

    if n_plots >= 4:
        vz = ecef[5].reshape(shape)
        imh = ah[3].pcolormesh(nux, Omx, vz.T, cmap=plt.cm.Accent, vmin=-400, vmax=400);
        ah[3].text(nu_a, Om_a, 'A', color='red')
        ah[3].text(nu_d, Om_d, 'D', color='red')
        ah[3].plot(nu_a, Om_a, 'r+', nu_d, Om_d, 'rx')
        plt.colorbar(imh, ax=ah[3], fraction=0.04, pad=0.01)
        ah[3].set_xlabel('mean anomaly [$\\nu$]')
        ah[3].set_ylabel('longitude of ascending node [$\Omega$]')
        ah[3].set_title('z-component of velocity [m/s]')

    fh = ah[0].get_figure()
    fh.suptitle(f"inc {inci} sm axis {a:.0f}m (alt{a/1000 - Re_km:.1f} km)")
    #
    #            f"ascending  range {r_a/1000:.1f} km, range rate {rdot_a/1000:.3f} km/s\n" + \
    #            f"descending range {r_d/1000:.1f} km, range rate {rdot_d/1000:.3f} km/s\n")


def plot_Om_nu(cachefilename):
    with h5py.File(cachefilename, 'r') as ds:

        incl = ds['incl'][:]
        sema = ds['sema'][:]

        azim = ds['azimuth'][()]
        elev = ds['elevation'][()]

        Om_a = ds['Om_a'][...]
        Om_d = ds['Om_d'][...]

        nu_a = ds['nu_a'][...]
        nu_d = ds['nu_d'][...]

        theta_a = ds['theta_a'][...]
        theta_d = ds['theta_d'][...]

        radar_name = ds.attrs['radar_name']

    plot_title = f'{radar_name} az {azim} el {elev}'

    fh, ah = plt.subplots(2,2, sharex='all', sharey='all')

    ax = ah[0,0]
    imh = ax.pcolormesh(sema, incl, Om_a)
    ax.set_xlabel('Sem.mj.ax [m]')
    ax.set_ylabel('Inclination [deg]')
    ax.set_title('Longitude asc.node [deg] asc. sol')
    plt.colorbar(imh, ax=ax, fraction=0.04, pad=0.01)
    ax.contour(sema, incl, theta_a, [1, 2, 5, 10], cmap='inferno_r')

    ax = ah[1,0]
    imh = ax.pcolormesh(sema, incl, Om_d)
    ax.set_xlabel('Sem.mj.ax [m]')
    ax.set_ylabel('Inclination [deg]')
    ax.set_title('Longitude asc.node [deg] dsc. sol')
    plt.colorbar(imh, ax=ax, fraction=0.04, pad=0.01)
    ax.contour(sema, incl, theta_d, [1, 2, 5, 10], cmap='inferno_r')

    ax = ah[0,1]
    imh = ax.pcolormesh(sema, incl, nu_a)
    ax.set_xlabel('Sem.mj.ax [m]')
    ax.set_ylabel('Inclination [deg]')
    ax.set_title('Mean anomaly [deg] asc. sol')
    plt.colorbar(imh, ax=ax, fraction=0.04, pad=0.01)
    ax.contour(sema, incl, theta_a, [1, 2, 5, 10], cmap='inferno_r')

    ax = ah[1,1]
    imh = ax.pcolormesh(sema, incl, nu_d)
    ax.set_xlabel('Sem.mj.ax [m]')
    ax.set_ylabel('Inclination [deg]')
    ax.set_title('Mean anomaly [deg] asc. sol')
    plt.colorbar(imh, ax=ax, fraction=0.04, pad=0.01)
    ax.contour(sema, incl, theta_d, [1, 2, 5, 10], cmap='inferno_r')

    fh.suptitle(plot_title)


def build_maps(station, ii, kk, azim=None, elev=None):

    if azim is None:
        azim = station.beam.azimuth
    if elev is None:
        elev = station.beam.elevation

    station.beam.sph_point(azim, elev, radians=False)
    k_ecef = station.pointing_ecef

    # Epoch of observation
    epoch = Time("2000-01-01T00:10:00Z", format='isot')

    # Values of dimension axis for orbit plane inclination
    if np.issubdtype(type(ii), int):
        incl = np.r_[69:111.1:0.5]
        inci = incl[ii]
    else:
        inci = ii

    # Values of dimension axis for orbit semimajor axis
    if np.issubdtype(type(kk), int):
        sema = 1000 * (Re_km + np.linspace(300, 3000, 20))
        a = sema[kk]
    else:
        a = kk

    # First some invariants
    samples = 101, 103  # for mean anomaly and longitude of ascending node, respectively
    # Now start looking for orbits that intersect the pointing of the beam
    anom, Omega = np.meshgrid(
        np.linspace(0, 360, num=samples[0]),
        np.linspace(0, 360, num=samples[1]),
        indexing='ij',
    )
    anom.shape = -1             # unravel
    Omega.shape = -1

    orbit = pyorb.Orbit(
        M0 = pyorb.M_earth,
        degrees = True,
        a = sema[0],
        e = 0,
        i = 0,
        omega = 0,
        Omega = Omega,
        anom = anom,
        num = anom.size,
        type = 'mean',
    )

    orbit.i = inci
    orbit.a = a

    states = orbit.cartesian
    ecef = sorts.frames.convert(epoch, states, in_frame='GCRS', out_frame='ITRS')
    angle = np.degrees(get_off_axis_angle(station, ecef, k_ecef))
    r, rdot = get_range_and_range_rate(station, ecef)

    print(f"inclination: {inci}, semimajor axis {a:.0f}m (altitude {a/1000 - Re_km:.1f} km)")

    return states, ecef, angle, r, rdot


def do_create_demoplots():
    figsize = 10,7
    # UHF east-pointing demo plot
    cachefilename = 'cache/eiscat_uhf/az90.0_el75.0.h5'
    fig = plt.figure(figsize=figsize)
    plot_from_cachefile(cachefilename, incl=[69, 70, 75, 80, 85, 90, 95, 100, 105, 110])
    plt.savefig('new_cache_figure_uhf_east.png')
    plot_Om_nu(cachefilename)
    plt.savefig('Om_nu_figure_uhf_east.png')

    cachefilename = 'cache/eiscat_uhf/az90.0_el45.0.h5'
    fig = plt.figure(figsize=figsize)
    plot_from_cachefile(cachefilename, incl=[69.5, 70, 75, 80, 85, 90, 95, 100, 105, 110])
    plt.savefig('new_cache_figure_uhf_east_low.png')
    plot_Om_nu(cachefilename)
    plt.savefig('Om_nu_figure_uhf_east_low.png')

    # ESR field-aligned demo plot
    cachefilename = 'cache/eiscat_esr/az185.5_el82.1.h5'
    fig = plt.figure(figsize=figsize)
    plot_from_cachefile(cachefilename, incl=[77, 78, 80, 85, 90, 95, 100, 102, 104])
    plt.savefig('new_cache_figure_esr_field_aligned.png')
    plot_Om_nu(cachefilename)
    plt.savefig('Om_nu_figure_esr_field_aligned.png')

    cachefilename = 'cache/eiscat_esr/az90.0_el75.0.h5'
    fig = plt.figure(figsize=figsize)
    plot_from_cachefile(cachefilename, incl=[78.5, 80, 85, 90, 95, 101])
    plt.savefig('new_cache_figure_esr_east.png')
    plot_Om_nu(cachefilename)
    plt.savefig('Om_nu_figure_esr_east.png')

    cachefilename = 'cache/eiscat_esr/az0.0_el35.0.h5'
    fig = plt.figure(figsize=figsize)
    plot_from_cachefile(cachefilename, incl=[78.5, 80, 85, 90, 95, 101])
    plt.savefig('new_cache_figure_esr_north.png')
    plot_Om_nu(cachefilename)
    plt.savefig('Om_nu_figure_esr_north.png')


def main(input_args=None):

    parser = argparse.ArgumentParser(description='Calculate doppler-inclination correlation for a beampark')

    subparsers = parser.add_subparsers(dest='command')
    parser_run = subparsers.add_parser('run', help='Run calculation and build cache')

    parser_run.add_argument('radar', type=str, help='The observing radar system')
    parser_run.add_argument('azimuth', type=float, help='Azimuth of beampark')
    parser_run.add_argument('elevation', type=float, help='Elevation of beampark')
    parser_run.add_argument('-c', '--clobber', action='store_true', help='Remove cache data before running')
    parser_run.add_argument('-d', '--dry-run', action='store_true', help='Do not actually write to file')

    parser_plot = subparsers.add_parser('plot', help='Plot results from cache')
    parser_plot.add_argument('cache', type=str, nargs='+', help='Path(s) to cache file to plot')
    parser_plot.add_argument('-o', '--output', type=str, default=None, help='Directory to save plots in')
    parser_plot.add_argument(
        '-d', '--data', 
        type=str, 
        default=None, 
        nargs='+', 
        help='Paths to measurement data to plot alongside cache results',
    )

    parser_demo = subparsers.add_parser('demo', help='Do predefined demo plotting')

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    if args.command == 'run':

        radar = getattr(sorts.radars, args.radar)
        st = radar.tx[0]

        epoch = Time("2000-01-01T00:10:00Z", format='isot')

        build_cache(st, radar_name=args.radar, azim=args.azimuth, elev=args.elevation,
                        degrees=True, clobber=args.clobber, dry_run=args.dry_run)
    elif args.command == 'plot':

        for cache in args.cache:
            print(f'Plotting {cache}...')
            with h5py.File(cache, 'r') as ds:
                radar_name = ds.attrs['radar_name']

            fig = plt.figure()
            plot_from_cachefile(cache)
            fig_name = Path(cache).stem
            fout = f'{radar_name}_cache_{fig_name}.png'
            if args.output is not None:
                fout = str(Path(args.output).resolve() / fout)
            fig.savefig(fout)
            plt.close(fig)
        if args.data is not None:
            for cache, data in zip(args.cache, args.data):
                print(f'Plotting {cache} + {data}...')
                
                with h5py.File(cache, 'r') as ds:
                    radar_name = ds.attrs['radar_name']

                with h5py.File(data, 'r') as ds:
                    date = Time(np.min(ds["t"]), format='unix', scale='utc')
                
                if args.output is None:
                    out_path = Path('.')
                else:
                    out_path = Path(args.output).resolve()
                fout = f'{radar_name}_cache_{Path(cache).stem}.png'

                plot_with_measurement_data(
                    Path(data).resolve(), 
                    Path(cache).resolve(), 
                    velmin=-2.3, 
                    velmax=2.3, 
                    site=radar_name, 
                    date=date, 
                    out_path=out_path, 
                    out_name=fout,
                )

    elif args.command == 'demo':
        do_create_demoplots()


if __name__ == '__other_main__':
    import sys

    if len(sys.argv) > 1:
        cachefilename = sys.argv[1]
    else:
        cachefilename = 'cache/eiscat_esr/az185.5_el82.1.h5'

    ii = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    kk = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    cpath = Path(cachefilename)
    radar_name = cpath.parts[cpath.parts.index('cache') + 1]
    radar = getattr(sorts.radar.instances, radar_name)
    station = radar.tx[0]

    with h5py.File(cachefilename, 'r') as ds:

        incl = ds['incl'][:]
        sema = ds['sema'][:]

        azim = ds['azimuth'][()]
        elev = ds['elevation'][()]

        Om_a = ds['Om_a'][ii][kk]
        Om_d = ds['Om_d'][ii][kk]

        nu_a = ds['nu_a'][ii][kk]
        nu_d = ds['nu_d'][ii][kk]

        r_a = ds['r_a'][ii][kk]
        r_d = ds['r_d'][ii][kk]

        rdot_a = ds['rdot_a'][ii][kk]
        rdot_d = ds['rdot_d'][ii][kk]

    inci = incl[ii]
    a = sema[kk]

    print(f"inclination: {inci}, semimajor axis {a:.0f}m (altitude {a/1000 - Re_km:.1f} km)")
    print(f"ascending  range {r_a/1000:.1f} km, range rate {rdot_a/1000:.3f} km/s")
    print(f"descending range {r_d/1000:.1f} km, range rate {rdot_d/1000:.3f} km/s")

    # Epoch of observation
    epoch = Time("2000-01-01T00:10:00Z", format='isot')

    shape = 101, 103
    nux = np.linspace(0, 360, shape[0])
    Omx = np.linspace(0, 360, shape[1])

    station.beam.sph_point(azim, elev, radians=False)
    k_ecef = station.pointing_ecef

    anom, Omega = np.meshgrid(nux, Omx, indexing='ij')
    anom.shape = -1             # unravel
    Omega.shape = -1

    orbit = pyorb.Orbit(
        M0 = pyorb.M_earth,
        degrees = True,
        a = sema[0],
        e = 0,
        i = 0,
        omega = 0,
        Omega = Omega,
        anom = anom,
        num = anom.size,
        type = 'mean',
    )

    orbit.i = inci
    orbit.a = a

    # maps
    states = orbit.cartesian
    ecef = sorts.frames.convert(epoch, states, in_frame='GCRS', out_frame='ITRS')
    angle = np.degrees(get_off_axis_angle(station, ecef, k_ecef))
    r, rdot = get_range_and_range_rate(station, ecef)

if __name__ == '__main__':
    main()

