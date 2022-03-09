from pathlib import Path

import numpy as np
import scipy.optimize as sciopt
import scipy.signal as scisig
from astropy.time import TimeDelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

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

output_folder = base_path/'projects'/'output'/'russian_asat'/'dual_beampark_planner'
output_folder.mkdir(exist_ok=True)

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
station_inds = [
    (0, 0),
    (0, 0),
]
dt = 1.0
target_t = 15*3600.0

rotation_offset_limits = (0., 24*3600.0)
rotation_offset_num = 24*60
rotation_offset = TimeDelta(
    np.linspace(
        rotation_offset_limits[0],
        rotation_offset_limits[1],
        rotation_offset_num
    ), 
    format='sec',
)

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
print(tle_obj)
print(tle_obj.epoch.isot)

epoch0 = tle_obj.epoch

ecef_point = radar_fixed.tx[0].pointing_ecef
ecef_st = radar_fixed.tx[0].ecef

tle_obj.out_frame = 'TEME'
teme_state0 = tle_obj.get_state([0])

orb0 = pyorb.Orbit(
    M0=pyorb.M_earth, m=0, 
    num=1, 
    degrees=True,
    type='mean',
)
orb0.cartesian = teme_state0[:, :1]
period = orb0.period[0]

t_samp = np.arange(0, period, dt)
print(f'Sampling {len(t_samp)} times epoch + t={0/3600.0} h -> {period/3600.0} h')

teme_states = tle_obj.get_state(t_samp)
tle_obj.out_frame = 'ITRS'

best_ind = np.empty((rotation_offset_num, ), dtype=np.int64)
best_ang = np.empty((rotation_offset_num, ), dtype=np.float64)
best_t = np.empty((rotation_offset_num, ), dtype=np.float64)

def get_offaxis(epc):
    ecef_states = sorts.frames.convert(
        epc + TimeDelta(t_samp, format='sec'), 
        teme_states, 
        in_frame = 'TEME', 
        out_frame = 'ITRS',
    )
    ecef_states = ecef_states[:3, :] - ecef_st[:, None]

    off_axis_ang = pyant.coordinates.vector_angle(
        ecef_point, 
        ecef_states, 
        radians=False,
    )
    return off_axis_ang

### SETTINGS END ###
for ind in tqdm(range(rotation_offset_num)):
    epoch = epoch0 + rotation_offset[ind]

    off_axis_ang = get_offaxis(epoch)

    best_ind[ind] = np.argmin(off_axis_ang)
    best_ang[ind] = off_axis_ang[best_ind[ind]]
    best_t[ind] = t_samp[best_ind[ind]]

fig, ax = plt.subplots()
ax.plot(rotation_offset.sec/3600.0, best_ang, '-b')
ax.set_xlabel('Rotation offset [h]')
ax.set_ylabel('Best off-axis angle [deg]')
fig.savefig(output_folder / 'rotation_offset.png')

best_offset = np.argmin(best_ang)
offset = rotation_offset.sec[best_offset]

minima_ind, = scisig.argrelextrema(best_ang, np.less)

for mind in range(len(minima_ind)):
    t_pass = t_samp[best_ind[minima_ind[mind]]]
    t_offset = rotation_offset.sec[minima_ind[mind]]
    ep_off = TimeDelta(t_pass + t_offset, format='sec')

    print(f'\n\
        ind={best_ind[minima_ind[mind]]}, \n\
        t pass = {best_t[minima_ind[mind]]}, \n\
        offset = {rotation_offset.sec[minima_ind[mind]]} s, \n\
        off axis angle = {best_ang[minima_ind[mind]]} deg')

    epoch = epoch0 + rotation_offset[minima_ind[mind]]
    off_axis_ang = get_offaxis(epoch)

    fig, ax = plt.subplots()
    ax.plot(t_samp/3600.0, off_axis_ang, '-b')
    ax.plot(t_samp[best_ind[minima_ind[mind]]]/3600.0, off_axis_ang[best_ind[minima_ind[mind]]], 'or')
    fig.savefig(output_folder / f'minima{mind}_off_axis_ang.png')

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
        epoch = epoch0 + ep_off, 
        parameters={
            'd': 1,
            'm': 0,
        }
    )

    print(obj)
    print(f'in_frame ={obj.in_frame}')
    print(f'out_frame={obj.out_frame}')
    print(f'Orbital period {period/3600.0} h')

    t_search = np.arange(-period*1.5, period*1.5, dt, dtype=np.float64)
    t_search_h = t_search/3600.0
    ecef_states = obj.get_state(t_search)

    data = {
        'az': [None]*total_systems,
        'el': [None]*total_systems,
        'el_maxima': [None]*total_systems,
    }

    for ri, radar in enumerate(radars):
        enu = radar.tx[0].enu(ecef_states)
        azel = pyant.coordinates.cart_to_sph(enu, radians=False)
        data['az'][ri] = np.mod(azel[0, :] + 360.0, 360)
        data['el'][ri] = azel[1, :]

        data['el_maxima'][ri], = scisig.argrelextrema(azel[1, :], np.greater)

        print(f'Radar {ri}')
        print('Elevation maxima:')
        print(f'Az = {data["az"][ri][data["el_maxima"][ri]]} deg')
        print(f'El = {data["el"][ri][data["el_maxima"][ri]]} deg')
        print(f't = {t_search_h[data["el_maxima"][ri]]*60} min')

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
    ax.plot(ecef_states[0,-1], ecef_states[1,-1], ecef_states[2,-1], 'xb')
    ax.set_title('Orbit')
    fig.savefig(output_folder / f'minima{mind}_orbit.png')


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

    fig.savefig(output_folder / f'minima{mind}_results.png')

plt.show()