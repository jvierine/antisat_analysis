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

import sorts
import pyorb
import pyant


HERE = Path(__file__).parent.resolve()
OUTPUT = HERE / 'output' / 'russian_asat'
print(f'Using {OUTPUT} as output')

ref_output = OUTPUT / 'all_kosmos_a_i'
ref_output.mkdir(exist_ok=True)


orb_samp = 1000
zero_res = 10000

radar = sorts.radars.eiscat_uhf
tx = radar.tx[0]

sph_point = np.array((90., 75., 1.))
epoch = Time('J2000')

r_obj = 500e3
delta_v_std = 0.1e3  # m/s
np.random.seed(87243)


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


local_point = pyant.coordinates.sph_to_cart(sph_point, radians=False)
ecef_point = sorts.frames.enu_to_ecef(tx.lat, tx.lon, tx.alt, local_point, radians=False)
ecef_st = tx.ecef

teme_point = sorts.frames.convert(
    epoch, 
    np.hstack([ecef_point, np.ones_like(ecef_point)]), 
    in_frame='ITRS', 
    out_frame='TEME',
)
teme_point = teme_point[:3]
teme_st = sorts.frames.convert(
    epoch, 
    np.hstack([ecef_st, np.ones_like(ecef_st)]), 
    in_frame='ITRS', 
    out_frame='TEME',
)
teme_st = teme_st[:3]

r_ecef = ecef_st + r_obj*ecef_point
r_teme = sorts.frames.convert(
    epoch, 
    np.hstack([r_ecef, np.ones_like(r_ecef)]), 
    in_frame='ITRS', 
    out_frame='TEME',
)
r_teme = r_teme[:3]

base_orb = theta_orbit(r_teme, epoch, np.array([90.0]))

orb_0 = pyorb.Orbit(
    M0=pyorb.M_earth, m=0, 
    num=orb_samp, epoch=epoch, 
    degrees=True,
    direct_update=True,
    auto_update=True,
)

orb_0.kepler = np.tile(base_orb.kepler, (1, orb_samp))
orb_0.v = orb_0.v + np.random.randn(3, orb_samp)*delta_v_std

orb_0.omega = np.random.rand(len(orb_0))*360.0
anoms = np.zeros_like(orb_0.anom)
angs = np.zeros_like(orb_0.anom)
nu_samp = np.linspace(0, 360, 2000)
for ind, orb in tqdm(enumerate(orb_0), total=orb_samp):
    _orbs = orb.copy()
    _orbs.allocate(len(nu_samp))
    _orbs._kep[:, :] = orb._kep[:, 0][:, None]
    _orbs._kep[5, :] = nu_samp
    _orbs.calculate_cartesian()
    ang = pyant.coordinates.vector_angle(
        teme_point, 
        _orbs.cartesian[:3, :] - teme_st[:, None],
    )
    anoms[ind] = nu_samp[np.argmin(ang)]
    angs[ind] = np.min(ang)

orb_0.anom = anoms

states = sorts.frames.convert(
    epoch, 
    orb_0.cartesian, 
    in_frame='TEME', 
    out_frame='ITRS',
)


fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
sorts.plotting.grid_earth(ax)
ax.plot(states[0, :], states[1, :], states[2, :], ".b")
ax.plot(
    [ecef_st[0], ecef_st[0] + ecef_point[0]*1000e3], 
    [ecef_st[1], ecef_st[1] + ecef_point[1]*1000e3], 
    [ecef_st[2], ecef_st[2] + ecef_point[2]*1000e3], 
    "-g",
)
ax.set_title('ITRS distribution')
fig.savefig(ref_output / 'ITRS_distribution.png')
# plt.show()
plt.close(fig)

ranges, range_rates = get_range_and_range_rate(tx, states)

kepler_elems = np.full((6, orb_samp), np.nan, dtype=np.float64)

for evi in tqdm(range(orb_samp)):
    r_obj = ranges[evi]
    v_obj = range_rates[evi]
    r_ecef = ecef_st + r_obj*ecef_point

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
    if len(roots) == 0:
        continue

    det_orbs = theta_orbit(r_teme, epoch, roots)
    det_orbs.calculate_kepler()

    sol = np.argmin(np.abs(det_orbs.Omega - orb_0.Omega[evi]))

    kepler_elems[:, evi] = det_orbs.kepler[:, sol]


true_col = 'r'

fig, axes = plt.subplots(2, 4, sharey='all', sharex='col', figsize=(12, 7))

# axes[0, 0].axvline(kepler_elems[0, 0]/sorts.constants.R_earth, label='Estimated')
axes[0, 0].hist(kepler_elems[0, :]/sorts.constants.R_earth, label='Estimated')
axes[0, 1].axvline(kepler_elems[1, 0])
axes[0, 2].hist(kepler_elems[2, :])
axes[0, 3].hist(kepler_elems[4, :])
axes[0, 0].legend()
axes[0, 0].set_xlabel('Semi-major-axis [R_earth]')
axes[0, 2].set_xlabel('Inclination [deg]')
axes[0, 3].set_xlabel('RAAN [deg]')
axes[0, 0].set_ylabel('Frequency')

axes[1, 0].hist(orb_0.kepler[0, :]/sorts.constants.R_earth, label='True', color=true_col)
axes[1, 1].hist(orb_0.kepler[1, :], color=true_col)
axes[1, 2].hist(orb_0.kepler[2, :], color=true_col)
axes[1, 3].hist(orb_0.kepler[4, :], color=true_col)
axes[1, 0].legend()

axes[1, 0].set_xlabel('Semi-major-axis [R_earth]')
axes[1, 1].set_xlabel('Eccentricity [1]')
axes[1, 2].set_xlabel('Inclination [deg]')
axes[1, 3].set_xlabel('RAAN [deg]')
axes[1, 0].set_ylabel('Frequency')

fig.suptitle('Eccentric orbits with circular assumption')
fig.savefig(ref_output / 'eccentric_effect_test.png')

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(orb_0.period, orb_0.apoapsis, '.b', label='Apogee')
ax.plot(orb_0.period, orb_0.periapsis, '.r', label='Perigee')
ax.legend()

fig.suptitle('Input distribution gabbard diagram')
fig.savefig(ref_output / 'eccentric_effect_gabbard.png')

plt.show()
