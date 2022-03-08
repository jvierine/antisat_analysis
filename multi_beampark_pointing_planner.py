from pathlib import Path

import numpy as np
import scipy.optimize as sciopt
import scipy.signal as scisig
from astropy.time import TimeDelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sorts
import pyorb
import pyant
'''

STEPS:

get the kosmos TLEs

./space_track_download.py 24h 2021-11-15 \
    /home/danielk/git/antisat_analysis/projects/output/russian_asat/kosmos-1408-2021-11-15.tle \
    -k space-track -n "COSMOS 1408"

configure below

run this

'''

base_path = Path(__file__).parent
tle_file = base_path/'projects'/'output'/'russian_asat'/'kosmos-1408-2021-11-15.tle'

radars = [
    sorts.radars.eiscat_uhf,
    sorts.radars.eiscat_esr,
]
total_systems = len(radars)
names = [
    'EISCAT UHF',
    'EISCAT ESR',
]
fixed_pointing = 0
azimuth = 90.0
elevation = 75.0
elevation_limits = [
    70.0,
]
off_axis_lim = 8.0
station_inds = [
    (0, 0),
    (0, 0),
]
t_max = 3*24*3600.0
dt = 1.0

### SETTINGS END ###

t_samp = np.arange(0, t_max, dt)
print(f'Sampling {len(t_samp)} times epoch + t={0} h -> {t_max/3600.0} h')
# Set which states to find pass pointing for
for radar, (txi, rxi) in zip(radars, station_inds):
    radar.tx = [radar.tx[txi]]
    radar.rx = [radar.rx[rxi]]

# set static pointing
radar_fixed = radars[fixed_pointing]
radar_fixed.tx[0].beam.sph_point(azimuth, elevation)
print(f'Setting pointing to: \n\
    azimuth = {radar_fixed.tx[0].beam.azimuth} deg\n\
    elevation = {radar_fixed.tx[0].beam.elevation} deg')

pop = sorts.population.tle_catalog(tle_file, kepler=True)
pop.unique()

print(f'TLEs = {len(pop)}')

tle_obj = pop.get_object(0)
tle_obj.out_frame = 'ITRS'
print(tle_obj)

ecef_point = radar_fixed.tx[0].pointing_ecef
ecef_st = radar_fixed.tx[0].ecef

ecef_states = tle_obj.get_state(t_samp)
ecef_states = ecef_states[:3, :] - ecef_st[:, None]

off_axis_ang = pyant.coordinates.vector_angle(
    ecef_point, 
    ecef_states, 
    radians=False,
)

minima_ind, = scisig.argrelextrema(off_axis_ang, np.less)
minima_ind = minima_ind[off_axis_ang[minima_ind] < off_axis_lim]

best_ind = np.argmin(off_axis_ang)
best_ang = off_axis_ang[best_ind]
best_t = t_samp[best_ind]
print(f'\nind={best_ind}, t pass = {best_t}, off axis angle = {best_ang} deg')

fig, ax = plt.subplots()
ax.plot(t_samp/3600.0, off_axis_ang, '-b')
ax.plot(t_samp[minima_ind]/3600.0, off_axis_ang[minima_ind], 'or')

for mind in range(len(minima_ind)):
    t_pass = t_samp[minima_ind[mind]]

    ECEF_state = tle_obj.get_state(t_pass)
    tle_obj.out_frame = 'TEME'
    TEME_state = tle_obj.get_state(t_pass)

    print('ECEF: ', ECEF_state)
    print('TEME: ', TEME_state)

    orb = pyorb.Orbit(
        M0=pyorb.M_earth, m=0, 
        num=1, 
        degrees=True,
        type='mean',
    )
    orb.cartesian = TEME_state

    obj = sorts.SpaceObject(
        propagator = sorts.propagator.Kepler,
        propagator_options = {
            'settings': {
                'in_frame': 'TEME',
                'out_frame': 'ITRS',
            }
        },
        state = orb,
        epoch = tle_obj.epoch + TimeDelta(t_pass, format='sec'), 
        parameters={
            'd': 1,
            'm': 0,
        }
    )

    print(obj)
    print(f'in_frame ={obj.in_frame}')
    print(f'out_frame={obj.out_frame}')

    period = orb.period[0]
    print(f'Orbital period {period/3600.0} h')

    t_search = np.arange(0, period, dt, dtype=np.float64)
    t_search_h = t_search/3600.0
    ecef_states = obj.get_state(t_search)

    data = {
        'az': [None]*total_systems,
        'el': [None]*total_systems,
    }

    for ri, radar in enumerate(radars):
        enu = radar.tx[0].enu(ecef_states)
        azel = pyant.coordinates.cart_to_sph(enu, radians=False)
        data['az'][ri] = np.mod(azel[0, :] + 360.0, 360)
        data['el'][ri] = azel[1, :]

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    sorts.plotting.grid_earth(ax)
    for ri, radar in enumerate(radars):
        ax.plot(radar.tx[0].ecef[0], radar.tx[0].ecef[1], radar.tx[0].ecef[2], 'or')
    ax.plot(
        [ecef_st[0], ecef_st[0] + ecef_point[0]*1000e3], 
        [ecef_st[1], ecef_st[1] + ecef_point[1]*1000e3], 
        [ecef_st[2], ecef_st[2] + ecef_point[2]*1000e3], 
        '-g',
    )
    ax.plot(ecef_states[0,:], ecef_states[1,:], ecef_states[2,:], '-b')
    ax.set_title('Orbit')


    fig = plt.figure(constrained_layout=True)
    fig.suptitle('Multi-beampark passes')

    subfigs = fig.subfigures(nrows=total_systems, ncols=1)
    for row, subfig in enumerate(subfigs):

        addon = ' [FIXED]' if row == 0 else ''
        subfig.suptitle(f'{names[row]}{addon}')

        axes = subfig.subplots(nrows=1, ncols=2)
        axes[0].plot(t_search_h, data['az'][row])
        axes[0].set_xlabel('Detection epoch + time [h]')
        axes[0].set_ylabel('Azimuth [deg]')
        axes[1].plot(t_search_h, data['el'][row])
        axes[1].set_xlabel('Detection epoch + time [h]')
        axes[1].set_ylabel('Elevation [deg]')

        if row == 0:
            axes[0].axhline(azimuth, color='r')
            axes[1].axhline(elevation, color='r')
        else:
            axes[1].axhline(elevation_limits[row - 1], color='g')

plt.show()