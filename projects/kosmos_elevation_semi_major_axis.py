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


kosmos_prediction_file = OUTPUT / 'kosmos-1408_orekit_core_propagated.npz'
kosmos_prediction = np.load(kosmos_prediction_file)
print(kosmos_prediction)
kosmos_Omega = sciint.interp1d(kosmos_prediction['t'], kosmos_prediction['Omega'])

zero_res = 10000
orb_samp = 300
clobber = False


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

radar_title = {
    'esr': [
        'EISCAT Svalbard Radar 32m',
        'EISCAT Svalbard Radar 32m',
        'EISCAT Svalbard Radar 42m',
        'EISCAT Svalbard Radar 42m',
    ],
    'uhf': ['EISCAT UHF']*3,
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
pprint(radar_info)


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

    x_hat = np.array([1, 0, 0], dtype=np.float64)  # Z-axis unit vector
    z_hat = np.array([0, 0, 1], dtype=np.float64)  # Z-axis unit vector
    # formula for creating the velocity vectors
    b3 = r_teme/np.linalg.norm(r_teme)  # r unit vector
    b3 = b3.flatten()

    b1 = np.cross(b3, z_hat)  # Az unit vector
    if np.linalg.norm(b1) < 1e-12:
        b1 = np.cross(b3, x_hat)  # Az unit vector
    b1 = b1/np.linalg.norm(b1)

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
orb_sols = {}
for radar in paths['data_paths']:
    orb_data[radar] = []
    orb_sols[radar] = []

    for ind in tqdm(range(len(paths['data_paths'][radar])), position=1):

        cache_path = save_paths[radar][ind] / 'kep_elements.pickle'
        if cache_path.is_file() and not clobber:
            with open(cache_path, 'rb') as fh:
                kepler_elems, matching_solutions = pickle.load(fh)
            orb_data[radar].append(kepler_elems)
            orb_sols[radar].append(matching_solutions)
            continue

        tx = radar_info[radar][ind][1]
        sph_point = np.array(radar_info[radar][ind][0])

        local_point = pyant.coordinates.sph_to_cart(sph_point, radians=False)
        ecef_point = sorts.frames.enu_to_ecef(tx.lat, tx.lon, tx.alt, local_point, radians=False)
        ecef_st = tx.ecef

        data_file = paths['data_paths'][radar][ind]
        t, r, v, snr, dur, diam = load_data(data_file)
        times = Time(t, format='unix', scale='utc')

        kosmos_dt = times - Time(kosmos_prediction['epoch_unix'], format='unix', scale='utc')
        predict_Omega = kosmos_Omega(kosmos_dt.sec)

        kepler_elems = np.full((6, 2, len(t)), np.nan, dtype=np.float64)
        matching_solutions = np.full((len(t), ), np.nan, dtype=np.int64)
        
        for evi in tqdm(range(len(t)), position=0):
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

            del orb

            sim_range, sim_doppler = get_range_and_range_rate(tx, states)

            del states

            if v_obj < sim_doppler.min() or v_obj > sim_doppler.max():
                # print(f'v_obj = {v_obj}, lims = {sim_doppler.min()}, {sim_doppler.max()}')
                continue

            doppler_diff_interp = sciint.interp1d(theta, sim_doppler - v_obj, kind='linear')
            theta_samp = np.linspace(0, np.pi*2, zero_res)
            vals = doppler_diff_interp(theta_samp)
            signs = np.sign(vals)
            sign_diffs = np.abs(np.diff(signs))
            root_inds = np.argwhere(sign_diffs > 1).flatten()
            roots = theta_samp[root_inds]

            # print(f'Theta-roots: {np.degrees(roots)} deg')

            det_orbs = theta_orbit(r_teme, epoch, roots)
            det_orbs.calculate_kepler()

            if len(roots) > 1:
                matching_solutions[evi] = np.argmin(np.abs(det_orbs.Omega - predict_Omega[evi]))
            else:
                matching_solutions[evi] = 0

            for ri in range(len(roots)):
                kepler_elems[:, ri, evi] = det_orbs.kepler[:, ri]
        with open(cache_path, 'wb') as fh:
            pickle.dump([kepler_elems, matching_solutions], fh)
        orb_data[radar].append(kepler_elems)
        orb_sols[radar].append(matching_solutions)

for radar in orb_sols:
    for ind in range(len(orb_sols[radar])):
        nan_inds = np.logical_or(
            np.isnan(orb_sols[radar][ind]), 
            orb_sols[radar][ind] < 0
        )
        orb_sols[radar][ind][nan_inds] = 0


def plot_single_kepler(keps, solution_select, select):

    bins = int(np.round(np.sqrt(keps.shape[2])))
    
    fig, axes = plt.subplots(1, 3, sharey='all')

    use_keps = np.empty((keps.shape[0], keps.shape[2]), dtype=keps.dtype)
    for ind, sol in enumerate(solution_select):
        use_keps[:, ind] = keps[:, sol, ind]
    selects = np.logical_not(np.any(np.isnan(use_keps), axis=0))
    print(f'Fraction not calculated {1 - np.sum(selects)/len(selects)}')

    selects = np.logical_and(selects, select)
    axes[0].hist(use_keps[0, selects]*1e-3, bins=bins, label='KOSMOS-1408')
    axes[1].hist(use_keps[2, selects], bins=bins)
    axes[2].hist(use_keps[4, selects], bins=bins)
    axes[0].legend()
    axes[0].set_xlabel('Semi-major-axis [km]')
    axes[0].set_ylabel('Frequency')
    axes[1].set_xlabel('Inclination [deg]')
    axes[2].set_xlabel('RAAN [deg]')
    
    return fig, axes


def plot_kepler(keps, select=None):

    bins = int(np.round(np.sqrt(keps.shape[2])))
    
    fig, axes = plt.subplots(2, 3, sharey='all', sharex='col')

    selects = np.logical_not(np.any(np.isnan(keps[:, 0, :]), axis=0))

    print(f'Fraction not calculated {1 - np.sum(selects)/len(selects)}')

    n1, bins1, patches = axes[0, 0].hist(keps[0, 0, selects]*1e-3, bins=bins)
    axes[0, 0].set_ylabel('Frequency')
    n2, bins2, patches = axes[0, 1].hist(keps[2, 0, selects], bins=bins)
    n3, bins3, patches = axes[0, 2].hist(keps[4, 0, selects], bins=bins)
    if select is not None:
        selects = np.logical_and(selects, select)
        axes[0, 0].hist(keps[0, 0, selects]*1e-3, bins=bins1)
        axes[0, 1].hist(keps[2, 0, selects], bins=bins2)
        axes[0, 2].hist(keps[4, 0, selects], bins=bins3)

    selects = np.logical_not(np.any(np.isnan(keps[:, 1, :]), axis=0))

    n1, bins1, patches = axes[1, 0].hist(keps[0, 1, selects]*1e-3, bins=bins, label='Total')
    axes[1, 0].set_xlabel('Semi-major-axis [km]')
    axes[1, 0].set_ylabel('Frequency')
    n2, bins2, patches = axes[1, 1].hist(keps[2, 1, selects], bins=bins)
    axes[1, 1].set_xlabel('Inclination [deg]')
    n3, bins3, patches = axes[1, 2].hist(keps[4, 1, selects], bins=bins)
    axes[1, 2].set_xlabel('RAAN [deg]')
    if select is not None:
        selects = np.logical_and(selects, select)
        axes[1, 0].hist(keps[0, 1, selects]*1e-3, bins=bins1, label='KOSMOS-1408')
        axes[1, 1].hist(keps[2, 1, selects], bins=bins2)
        axes[1, 2].hist(keps[4, 1, selects], bins=bins3)
        axes[1, 0].legend()
    
    return fig, axes


all_kepler_elems = []
all_selects = []
all_solutions = []
all_select_solutions = []
for radar in save_paths:
    for ind in range(len(save_paths[radar])):
        select = np.load(category_files[radar][ind])
        all_selects.append(select)

        kepler_elems = orb_data[radar][ind]
        select_solutions = orb_sols[radar][ind]
        all_kepler_elems.append(kepler_elems)
        all_select_solutions.append(select_solutions)
        fig, axes = plot_kepler(kepler_elems)
        fig.suptitle(f'{radar_title[radar][ind]} circular-orbits')
        fig.savefig(save_paths[radar][ind] / f'{radar_title[radar][ind].replace(" ", "_")}_kep_elements.png')
        plt.close(fig)

        fig, axes = plot_kepler(kepler_elems, select=kosmos_select(select))
        fig.suptitle(f'{radar_title[radar][ind]} circular-orbits')
        fig.savefig(save_paths[radar][ind] / f'{radar_title[radar][ind].replace(" ", "_")}_kosmos_kep_elements.png')
        plt.close(fig)

        fig, axes = plot_single_kepler(kepler_elems, select_solutions, kosmos_select(select))
        fig.suptitle(f'{radar_title[radar][ind]} disambiguated circular-orbits')
        fig.savefig(save_paths[radar][ind] / f'{radar_title[radar][ind].replace(" ", "_")}_kosmos_kep_elements_disambiguated.png')
        plt.close(fig)


all_kepler_elems = np.concatenate(all_kepler_elems, axis=2)
all_selects = np.concatenate(all_selects, axis=0)
all_select_solutions = np.concatenate(all_select_solutions, axis=0)

fig, axes = plot_kepler(all_kepler_elems)
fig.suptitle('EISCAT campaigns 2021-11-[19-29] circular-orbits')
fig.savefig(tot_pth / 'all_kep_elements.png')
plt.close(fig)

fig, axes = plot_kepler(all_kepler_elems, select=kosmos_select(all_selects))
fig.suptitle('EISCAT campaigns 2021-11-[19-29] circular-orbits')
fig.savefig(tot_pth / 'all_kosmos_kep_elements.png')
plt.close(fig)

fig, axes = plot_single_kepler(all_kepler_elems, all_select_solutions, kosmos_select(all_selects))
fig.suptitle('EISCAT campaigns 2021-11-[19-29] disambiguated circular-orbits')
fig.savefig(tot_pth / 'all_kosmos_kep_elements_disambiguated.png')
plt.close(fig)
