import sys
import argparse
from pathlib import Path

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


def optim_off_axis_angle(x, epoch, st, k_ecef, semi_major_axis, inclination):
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
    states = orbit.cartesian
    ecef = sorts.frames.convert(epoch, states, in_frame='GCRS', out_frame='ITRS')
    angle = np.degrees(get_off_axis_angle(st, ecef, k_ecef))

    return angle


def orb_to_range_and_range_rate(x, epoch, st, semi_major_axis, inclination):
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
    states = orbit.cartesian
    ecef = sorts.frames.convert(epoch, states, in_frame='GCRS', out_frame='ITRS')

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


def build_cache(station, radar_name='generic', cache_dir='cache', azim=None, elev=None,
                degrees=True, clobber=False):
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

    if cpath.exists() and not clobber:
        raise FileExistsError('Cache file {cpath.name} exists in cache')

    if azim is None:
        azim = station.beam.azimuth
    if elev is None:
        elev = station.beam.elevation

    with h5py.File(cpath, 'w') as ds:

        # Create global attributes for dataset
        ds.attrs['radar_name'] = radar_name

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
        ds['incl'] = np.r_[50:130.1:0.5]
        incl = ds['incl']
        incl.make_scale('inclination')
        incl.attrs['long_name'] = 'orbit plane inclination'
        incl.attrs['units'] = 'degrees'
        incl.attrs['valid_range'] = 0.0, 180.0

        # Create dimension axis for orbit semimajor axis
        ds['sema'] = 1000 * (6371 + np.linspace(300, 3000, 20))
        sema = ds['sema']
        sema.make_scale('semimajor_axis')
        sema.attrs['long_name'] = 'orbit semimajor axis'
        sema.attrs['units'] = 'metres'
        sema.attrs['valid_range'] = 1e0, 1e12

        def _create_ia_var(name, long_name, units, valid_range):
            ds[name] = np.zeros((len(incl), len(sema)))
            var = ds[name]
            var.dims[0].attach_scale(incl)
            var.dims[1].attach_scale(sema)
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
        epoch = Time("2000-01-01T00:10:00Z", format='isot')
        samples = 101, 103  # for mean anomaly and longitude of ascending node, respectively
        # Now start looking for orbits that intersect the pointing of the beam
        anom, Omega = np.meshgrid(
            np.linspace(0, 360, num=samples[0]),
            np.linspace(0, 360, num=samples[1]),
            indexing='ij',
        )
        anom.shape = -1             # unravel
        Omega.shape = -1

        k_ecef = station.pointing_ecef
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
        for ii, inci in enumerate(incl):
            orbit.i = inci

            print(f"inclination: {inci}")
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

                # Clever trick here.
                # Find the index of the smallest angle which satisfies the inequality
                other_ind = np.argmax(np.abs(anom[inds] - anom[inds[0]]) > 5.0)
                x_start2 = [anom[inds[other_ind]], Omega[inds[other_ind]]]
                xhat2 = sio.minimize(
                    lambda x: optim_off_axis_angle(x, epoch, station, k_ecef, a, inci),
                    x_start2,
                    method='Nelder-Mead',
                )

                # now we have two solutions, figure out if they are distinct,
                # which one is ascending and descending, and their ranges and range rates,
                # then put all of those (and the off-axis angles) into the file

                # ascending node is the one with smallest mean anomaly, put that in xhat1

                if xhat1.x[0] > xhat2.x[0]:
                    xhat1, xhat2 = xhat2, xhat1

                r1, rdot1 = orb_to_range_and_range_rate(xhat1.x, epoch, station, a, inci)
                r2, rdot2 = orb_to_range_and_range_rate(xhat2.x, epoch, station, a, inci)

                # Store data
                ds['nu_a'][ii,kk] = xhat1.x[0]
                ds['nu_d'][ii,kk] = xhat2.x[0]
                ds['Om_a'][ii,kk] = xhat1.x[0]
                ds['Om_d'][ii,kk] = xhat2.x[0]
                ds['theta_a'][ii,kk] = xhat1.fun
                ds['theta_d'][ii,kk] = xhat2.fun
                ds['r_a'][ii,kk] = r1[0]
                ds['r_d'][ii,kk] = r2[0]
                ds['rdot_a'][ii,kk] = rdot1[0]
                ds['rdot_d'][ii,kk] = rdot2[0]

                print(".", end="")
                sys.stdout.flush()
            print("")


def plot_from_cachefile(cachefilename, incl=None):

    good_limit = 1.0            # how much off-axis pointing is okay (degrees)
    cmap = plt.get_cmap('tab10')

    if incl is None:
        incl = np.arange(70, 110.1, 5)

    with h5py.File(cachefilename, 'r') as ds:

        dsincl = ds['incl'][:]
        sma = ds['sema'][:]
        height = sma - 6371e3       # height over average Earth radius

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

            hh = plt.plot(r_a[ok_a], rdot_a[ok_a], '-',
                          r_d[ok_d], rdot_d[ok_d], '--',
                          r_a[nok_a], rdot_a[nok_a], ':',
                          r_d[nok_d], rdot_d[nok_d], ':', color=cmap(ii))
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
        plt.legend(lh, lab)



def do_create_demoplots():

    # UHF east-pointing demo plot
    cachefilename = 'cache/eiscat_uhf/az90.0_el75.0.h5'
    fig = plt.figure()
    plot_from_cachefile(cachefilename, incl=[69, 70, 75, 80, 85, 90, 95, 100, 105, 110])
    plt.savefig('new_cache_figure_uhf_east.png')

    cachefilename = 'cache/eiscat_uhf/az90.0_el45.0.h5'
    fig = plt.figure()
    plot_from_cachefile(cachefilename, incl=[69.5, 70, 75, 80, 85, 90, 95, 100, 105, 110])
    plt.savefig('new_cache_figure_uhf_east_low.png')

    # ESR field-aligned demo plot
    cachefilename = 'cache/eiscat_esr/az185.5_el82.1.h5'
    fig = plt.figure()
    plot_from_cachefile(cachefilename, incl=[77, 78, 80, 85, 90, 95, 100, 102, 104])
    plt.savefig('new_cache_figure_esr_field_aligned.png')

    cachefilename = 'cache/eiscat_esr/az90.0_el75.0.h5'
    fig = plt.figure()
    plot_from_cachefile(cachefilename, incl=[78.5, 80, 85, 90, 95, 101])
    plt.savefig('new_cache_figure_esr_east.png')

    cachefilename = 'cache/eiscat_esr/az0.0_el35.0.h5'
    fig = plt.figure()
    plot_from_cachefile(cachefilename, incl=[78.5, 80, 85, 90, 95, 101])
    plt.savefig('new_cache_figure_esr_north.png')





def old_main():

    raise RuntimeError('Code left for illustration only, use build_cache() to create new hdf5 cache files')

    parser = argparse.ArgumentParser(description='Calculate doppler-inclination correlation for a beampark')
    parser.add_argument('radar', type=str, help='The observing radar system')
    parser.add_argument('cache', type=str, help='Cache location')
    parser.add_argument('azimuth', type=float, help='Azimuth of beampark')
    parser.add_argument('elevation', type=float, help='Elevation of beampark')
    parser.add_argument('inclinations', type=float, help='inclinations', nargs='+')

    args = parser.parse_args()

    radar = getattr(sorts.radars, args.radar)

    epoch = Time("2000-01-01T00:10:00Z", format='isot')

    samples = (100, 100)
    a_samples = 20
    optim = True
    R_earth = 6371e3

    semi_major_axis = R_earth + np.linspace(300e3, 3000e3, num=a_samples)

    st = radar.tx[0]
    st.beam.sph_point(args.azimuth, args.elevation, radians=False)
    
    hf = h5py.File(args.cache, 'a')

    for inclination in args.inclinations:
        all_hits = [[], []]
        all_obs = [[], []]
        inclination = np.round(inclination, decimals=3)
        dataset_key = f'{inclination}'
        
        if dataset_key + f'_node{0}_obs' in hf:
            continue

        k_ecef = st.pointing_ecef

        anom, Omega = np.meshgrid(
            np.linspace(0, 360, num=samples[0]),
            np.linspace(0, 360, num=samples[1]),
            indexing='ij',
        )
        anom = anom.reshape(anom.size)
        Omega = Omega.reshape(Omega.size)

        for a in semi_major_axis:

            orbit = pyorb.Orbit(
                M0 = pyorb.M_earth, 
                degrees = True,
                a = a, 
                e = 0, 
                i = 0, 
                omega = 0,
                Omega = Omega,
                anom = anom,
                num = anom.size,
                type = 'mean',
            )

            print(orbit)

            orbit.i = inclination
            states = orbit.cartesian
            ecef = sorts.frames.convert(epoch, states, in_frame='GCRS', out_frame='ITRS')
            angle = np.degrees(get_off_axis_angle(st, ecef, k_ecef))

            ranges, range_rates = get_range_and_range_rate(st, ecef)

            inds = np.argsort(angle)

            x_start = [anom[inds[0]], Omega[inds[0]]]
            orb_hits = []
            xhats = [] 
            if optim:
                xhat1 = sio.minimize(
                    lambda x: optim_off_axis_angle(x, epoch, st, k_ecef, a, inclination),
                    x_start,
                    method='Nelder-Mead',
                )
                print(x_start)
                print(xhat1)
                xhats += [xhat1]
                all_hits[0] += [xhat1.x]
                all_obs[0] += [list(orb_to_range_and_range_rate(xhat1.x, epoch, st, a, inclination))]
                orb_hits += [xhat1.x]
            else:
                all_hits[0] += [x_start]
                all_obs[0] += [list(orb_to_range_and_range_rate(x_start, epoch, st, a, inclination))]
                orb_hits += [x_start]

            other_ind = np.argmax(np.abs(anom[inds] - anom[inds[0]]) > 5.0)
            
            if other_ind.size > 0:
                x_start2 = [anom[inds[other_ind]], Omega[inds[other_ind]]]
                if optim:
                    xhat2 = sio.minimize(
                        lambda x: optim_off_axis_angle(x, epoch, st, k_ecef, a, inclination),
                        x_start2,
                        method='Nelder-Mead',
                    )
                    print(x_start2)
                    print(xhat2)
                    xhats += [xhat2]
                    xhats = [xhat1, xhat2]
                    all_hits[1] += [xhat2.x]
                    all_obs[1] += [list(orb_to_range_and_range_rate(xhat2.x, epoch, st, a, inclination))]
                    orb_hits += [xhat2.x]
                else:
                    all_hits[1] += [x_start2]
                    all_obs[1] += [list(orb_to_range_and_range_rate(x_start2, epoch, st, a, inclination))]
                    orb_hits += [x_start2]

                if all_obs[1][-1][1] > all_obs[0][-1][1]:
                    _x = all_obs[1][-1]
                    all_obs[1][-1] = all_obs[0][-1]
                    all_obs[0][-1] = _x

                    _x = all_hits[1][-1] 
                    all_hits[1][-1] = all_hits[0][-1]
                    all_hits[0][-1] = _x

        for node in range(2):
            all_obs[node] = np.squeeze(np.array(all_obs[node])).T
            all_hits[node] = np.squeeze(np.array(all_hits[node])).T
            hf.create_dataset(dataset_key + f'_node{node}_obs', data=all_obs[node])
            hf.create_dataset(dataset_key + f'_node{node}_hit', data=all_hits[node])

            # figs, axes = plot_inclination_results(
            #     anom, 
            #     Omega, 
            #     angle, 
            #     ranges, 
            #     range_rates, 
            #     samples, 
            #     orb_hits=orb_hits,
            # )
            # plt.show()

    fig = plt.figure()
    axes = [
        fig.add_subplot(3, 1, 1),
        [
            fig.add_subplot(3, 2, 3),
            fig.add_subplot(3, 2, 4),
        ],
        [
            fig.add_subplot(3, 2, 5),
            fig.add_subplot(3, 2, 6),
        ],
    ]
    nodes = {
        0: 'Asc',
        1: 'Dec',
    }
    for inci, inclination in enumerate(args.inclinations):
        all_hits = [[], []]
        all_obs = [[], []]
        inclination = np.round(inclination, decimals=3)
        dataset_key = f'{inclination}'

        for node, col in zip(range(2), ['b', 'r']):
            all_obs[node] = hf[dataset_key + f'_node{node}_obs'][()]
            all_hits[node] = hf[dataset_key + f'_node{node}_hit'][()]

            axes[0].plot(
                all_obs[node][0, :], 
                all_obs[node][1, :], 
                '-' + col, 
                label=f'{nodes[node]}. @ {inclination:.2f} deg incl.',
            )
            axes[1][node].plot(
                all_obs[node][0, :], 
                all_hits[node][0, :], 
                '-' + col, 
                label=f'{nodes[node]}. @ {inclination:.2f} deg incl.',
            )
            axes[2][node].plot(
                all_obs[node][0, :], 
                all_hits[node][1, :], 
                '-' + col, 
                label=f'{nodes[node]}. @ {inclination:.2f} deg incl.',
            )
    axes[0].legend()
    for node in range(2):
        axes[1][node].legend()
        axes[2][node].legend()
    axes[0].set_xlabel('Range [m]')
    axes[0].set_ylabel('Doppler [m/s]')
    axes[1][0].set_ylabel('Mean anomaly [deg]')
    axes[2][0].set_ylabel('Longitude of the ascending node [deg]')
    axes[2][0].set_xlabel('Range [m]')
    axes[2][1].set_xlabel('Range [m]')
    plt.show()

    hf.close()


if __name__ == '__main__':
    do_create_demoplots()
