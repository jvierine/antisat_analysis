from astropy.time import Time, TimeDelta
import h5py
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import scipy.optimize as sciopt
import scipy.interpolate as sciint
from pprint import pprint
from tqdm import tqdm
# plt.style.use('dark_background')

import sorts
import pyorb
import pyant

HERE = Path(__file__).parent.resolve()
OUTPUT = HERE / 'output' / 'russian_asat'
print(f'Using {OUTPUT} as output')

tot_pth = OUTPUT / 'all_kosmos_a_i'
tot_pth.mkdir(exist_ok=True)

with open(OUTPUT / 'paths.pickle', 'rb') as fh:
    paths = pickle.load(fh)


zero_res = 10000
orb_samp = 300


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


category_files = {}
dates = {}
all_dates = []
for radar in paths['data_paths']:
    category_files[radar] = []
    dates[radar] = []
    for ind in range(len(paths['data_paths'][radar])):
        date = paths['data_paths'][radar][ind].stem.replace('.', '-')
        dates[radar].append(date)
        all_dates.append(date)
        out_categories = OUTPUT / f'{date}_{radar}_categories.npy'
        category_files[radar].append(out_categories)
pprint(category_files)

all_dates = np.sort(np.unique(all_dates))

save_paths = {}
for radar in dates:
    save_paths[radar] = []
    for date in dates[radar]:
        save_path = OUTPUT / f'{date}_{radar}_a_i'
        save_path.mkdir(exist_ok=True)
        save_paths[radar].append(save_path)


class Categories:
    uncorrelated_other = 0
    correlated_other = 1
    uncorrelated_kosmos = 2
    correlated_kosmos = 3


def kosmos_select(select):
    return np.logical_or(
        select == Categories.uncorrelated_kosmos,
        select == Categories.correlated_kosmos,
    )


def load_data(file):
    with h5py.File(file, "r") as h:
        v = h["v"][()]
        t = h["t"][()]
        r = h["r"][()]    
        snr = h["snr"][()]
        dur = h["dur"][()]
        diam = h["diams"][()]

    return t, r, v, snr, dur, diam


data_type = {
    'uhf': [
        ['90.0 Az 75.0 El EISCAT UHF 2021-11-[23,25,29]', (0, 1, 2)],
    ],
    'esr': [
        ['90.0 Az 75.0 El ESR-32m 2021-11-[19]', (0,)],
        ['96.4 Az 82.1 El ESR-32m 2021-11-[23]', (1,)],
        ['185.5 Az 82.1 El ESR-42m 2021-11-[25,29]', (2, 3)],
    ],
}
radar_info = {
    'uhf': [
        [(90., 75., 1.), sorts.radars.eiscat_uhf.tx[0]],
        [(90., 75., 1.), sorts.radars.eiscat_uhf.tx[0]],
        [(90., 75., 1.), sorts.radars.eiscat_uhf.tx[0]],
    ],
    'esr': [
        [(90., 75., 1.), sorts.radars.eiscat_esr.tx[0]],
        [(96.4, 82.1, 1.), sorts.radars.eiscat_esr.tx[0]],
        [(185.5, 82.1, 1.), sorts.radars.eiscat_esr.tx[1]],
        [(185.5, 82.1, 1.), sorts.radars.eiscat_esr.tx[1]],
    ],
}
pprint(data_type)


def escape(x):
    return x.strip()\
        .replace(' ', '_')\
        .replace('-', '_')\
        .replace('[', '')\
        .replace(']', '')\
        .replace(',', '-')


prop = sorts.propagator.Kepler(
    settings=dict(
        in_frame='TEME',
        out_frame='ITRS',
    ),
)


def theta_orbit(r_teme, epoch, theta):

    orb = pyorb.Orbit(
        M0=pyorb.M_earth, m=0, 
        num=1, epoch=epoch, 
        degrees=True,
        direct_update=False,
        auto_update=True,
    )
    orb.a = np.linalg.norm(r_teme)
    orb.e = 0
    orb.omega = 0

    # angle of the velocity vector in the perpendicular plane of the position
    # is the free parameter

    # find the velocity of a circular orbit
    orb.i = 0
    orb.Omega = 0
    orb.anom = 0
    v_norm = orb.speed[0]

    x_hat = np.array([1, 0, 0])  # Z-axis unit vector
    z_hat = np.array([0, 0, 1])  # Z-axis unit vector
    # formula for creating the velocity vectors
    b3 = r_teme/np.linalg.norm(r_teme)  # r unit vector
    b1 = np.cross(b3, z_hat)  # Az unit vector
    b1 = b1/np.linalg.norm(b1)
    if np.linalg.norm(b1) < 1e-12:
        b1 = np.cross(b3, x_hat)  # Az unit vector

    b2 = np.cross(b3, b1)  # El unit vector
    v1 = np.cos(theta)
    v2 = np.sin(theta)
    inp_vec = v1[None, :]*b1[:, None] + v2[None, :]*b2[:, None]
    v_temes = v_norm*inp_vec

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(v_temes[0, :], v_temes[1, :], v_temes[2, :], '.')
    # plt.show()

    orb.allocate(len(theta))
    orb.update(
        x = r_teme[0],
        y = r_teme[1],
        z = r_teme[2],
        vx = v_temes[0, :],
        vy = v_temes[1, :],
        vz = v_temes[2, :],
    )
    return orb


orb_data = {}
for radar in paths['data_paths']:
    orb_data[radar] = []

    for ind in range(len(paths['data_paths'][radar])):

        tx = radar_info[radar][ind][1]
        sph_point = np.array(radar_info[radar][ind][0])

        local_point = pyant.coordinates.sph_to_cart(sph_point, radians=False)
        ecef_point = sorts.frames.enu_to_ecef(tx.lat, tx.lon, tx.alt, local_point, radians=False)
        ecef_st = tx.ecef

        select = np.load(category_files[radar][ind])
        data_file = paths['data_paths'][radar][ind]
        t, r, v, snr, dur, diam = load_data(data_file)
        times = Time(t, format='unix', scale='utc')

        ai = {
            'a': np.zeros_like(t),
            'i': np.zeros_like(t),
            'raan': np.zeros_like(t),
        }
        
        for evi in range(len(t)):
            r_obj = r[evi]*1e3
            v_obj = v[evi]*1e3
            r_ecef = ecef_st + r_obj*ecef_point
            epoch = times[evi]

            r_teme = sorts.frames.convert(
                epoch, 
                np.hstack([r_ecef, np.ones_like(r_ecef)]), 
                in_frame='ITRS', 
                out_frame='TEME',
            )
            r_teme = r_teme[:3]

            theta = np.linspace(0, np.pi*2, orb_samp)
            orb = theta_orbit(r_teme, epoch, theta)

            states = sorts.frames.convert(
                epoch, 
                orb.cartesian, 
                in_frame='TEME', 
                out_frame='ITRS',
            )

            del orb

            sim_range, sim_doppler = get_range_and_range_rate(tx, states)

            del states

            assert v_obj >= sim_doppler.min() and v_obj <= sim_doppler.max(),\
                f'v_obj = {v_obj}, lims = {sim_doppler.min()}, {sim_doppler.max()}'

            doppler_diff_interp = sciint.interp1d(theta, sim_doppler - v_obj, kind='linear')
            theta_samp = np.linspace(0, np.pi*2, zero_res)
            vals = doppler_diff_interp(theta_samp)
            signs = np.sign(vals)
            sign_diffs = np.abs(np.diff(signs))
            root_inds = np.argwhere(sign_diffs > 1).flatten()
            roots = theta_samp[root_inds]

            print('Roots: ', np.degrees(roots))

            det_orbs = theta_orbit(r_teme, epoch, roots)
            det_orbs.calculate_kepler()
            for orb in det_orbs:
                print(orb)

            exit()

            # Generates a nice plot
            # fig, axes = plt.subplots(3, 2)
            # axes[0, 0].plot(orb.i, orb.Omega, '-')
            # axes[0, 0].set_xlabel('Inclination [deg]')
            # axes[0, 0].set_ylabel('Longitude of the ascending node [deg]')
            # axes[1, 0].plot(orb.i, sim_range*1e-3, '-')
            # axes[1, 0].set_xlabel('Inclination [deg]')
            # axes[1, 0].set_ylabel('Range [km/s]')
            # axes[2, 0].plot(orb.i, sim_doppler*1e-3, '-')
            # axes[2, 0].set_xlabel('Inclination [deg]')
            # axes[2, 0].set_ylabel('Doppler [km/s]')

            # axes[0, 1].plot(np.degrees(theta), orb.Omega, '-')
            # axes[0, 1].set_xlabel('Theta [deg]')
            # axes[0, 1].set_ylabel('Longitude of the ascending node [deg]')
            # axes[1, 1].plot(np.degrees(theta), orb.i, '-')
            # axes[1, 1].set_xlabel('Theta [deg]')
            # axes[1, 1].set_ylabel('Inclination [deg]')
            # axes[2, 1].plot(np.degrees(theta), sim_doppler*1e-3, '-')
            # axes[2, 1].set_xlabel('Theta [deg]')
            # axes[2, 1].set_ylabel('Doppler [km/s]')
            # fig.suptitle('Theta-parametrized orbit')
            # plt.show()

        orb_data[radar].append(ai)


# for radar in save_paths:
#     for ind in range(len(save_paths[radar])):
#         for typi, inds in enumerate(data_type[radar]):
#             if ind in inds[1]:
#                 list_ptr = typi
#                 break

#         for include_total in [True, False]:
#             kosm = 'kosmos_' if include_total else ''

#             plot_statistics(
#                 data, 
#                 kosmos_select(select), 
#                 save_paths[radar][ind] / f'{dates[radar][ind]}_{radar}_{kosm}stat.png',
#                 f'Detections {radar_title[radar][ind]} {dates[radar][ind]}',
#                 include_total=include_total,
#             )

# for radar in data_type:
#     for ind in range(len(data_type[radar])):
#         for x in range(6):
#             tot_data[radar][ind][1][x] = np.concatenate(tot_data[radar][ind][1][x])
#         type_data = tot_data[radar][ind][1]
#         type_select = np.concatenate(tot_data[radar][ind][0])

#         for include_total in [True, False]:
#             kosm = 'kosmos_' if include_total else ''
#             name = data_type[radar][ind][0]
#             plot_statistics(
#                 type_data, 
#                 kosmos_select(type_select), 
#                 tot_pth / f'{escape(name)}_{kosm}stat.png',
#                 data_type[radar][ind][0],
#                 include_total=include_total,
#             )
