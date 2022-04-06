from astropy.time import Time, TimeDelta
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import h5py

from lamberthub import izzo2015

import odlab
import sorts
import pyorb
import pyant

HERE = Path(__file__).parent.resolve()
OUTPUT = HERE / 'output' / 'russian_asat'
print(f'Using {OUTPUT} as output')

out_pth = OUTPUT / 'orbit_determination'
out_pth.mkdir(exist_ok=True)

with open(OUTPUT / 'paths.pickle', 'rb') as fh:
    paths = pickle.load(fh)


def solve_lambert(pos1, pos2, dt):
    mu = pyorb.M_earth*pyorb.G
    vel1, vel2 = izzo2015(mu, pos1, pos2, dt)
    orb = pyorb.Orbit(
        M0=pyorb.M_earth, 
        m=0, 
        num=1, 
        radians=False,
    )
    orb.cartesian = np.hstack([pos1, pos2])
    return orb


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


def load_data(file):
    with h5py.File(file, "r") as h:
        v = h["v"][()]*1e3
        t = h["t"][()]
        r = h["r"][()]*1e3
        snr = h["snr"][()]

    inds = np.argsort(t)

    t = t[inds]
    r = r[inds]
    v = v[inds]
    snr = snr[inds]

    times = Time(t, format='unix', scale='utc')
    epoch = times[0]
    t = (times - epoch).sec

    return dict(
        epoch=epoch,
        times=times,
        t=t,
        r=r,
        v=v,
        snr=snr,
    )


dates = {}
all_dates = []
for radar in paths['data_paths']:
    dates[radar] = []
    for ind in range(len(paths['data_paths'][radar])):
        date = paths['data_paths'][radar][ind].stem.replace('.', '-')
        dates[radar].append(date)
        all_dates.append(date)
all_dates = np.sort(np.unique(all_dates))

tle_file = paths['tle_catalogs'][1]
corr1_file = paths['correlations']['uhf'][0]
corr2_file = paths['correlations']['esr'][1]
dual_corr_file = out_pth / f'{str(all_dates[1])}_dual_correlations.npy'


if not dual_corr_file.is_file():
    cmd = f'python multi_beampark_correlator.py {tle_file} -o {dual_corr_file} {corr1_file} {corr2_file}'
    print('FILE NOT FOUND, RUN: \n\n')
    print(cmd)
    exit()

dual_corrs = np.load(dual_corr_file)


txs = [
    sorts.radars.eiscat_uhf.tx[0],
    sorts.radars.eiscat_esr.tx[0],
]

radar_info = {
    'uhf': [
        (90., 75., 1.),
        (90., 75., 1.),
        (90., 75., 1.),
    ],
    'esr': [
        (90., 75., 1.),
        (96.4, 82.1, 1.),
        (185.5, 82.1, 1.),
        (185.5, 82.1, 1.),
    ],
}

prop = sorts.propagator.SGP4(
    settings=dict(
        out_frame='ITRS',
        tle_input=True,
    ),
)

data_ids = {'uhf': 0, 'esr': 1}
dual_ind = 1
radar_lst = ['uhf', 'esr']


mids = [dual_corrs[f'mid{ind}'][dual_ind] for ind in range(len(radar_lst))]
params = sorts.propagator.SGP4.get_TLE_parameters(
    dual_corrs['line1'][dual_ind],
    dual_corrs['line2'][dual_ind],
)
epoch0 = Time(params['jdsatepoch'], params['jdsatepochF'], format='jd')
datas = []

for radar in radar_lst:
    data_file = paths['data_paths'][radar][data_ids[radar]]
    data = load_data(data_file)
    datas.append(data)

tv = [dat['times'][mid] - epoch0 for mid, dat in zip(mids, datas)]
tv = np.array([x.sec for x in tv])

states = prop.propagate(
    tv, [dual_corrs['line1'][dual_ind], dual_corrs['line2'][dual_ind]],
)

prop.set(out_frame='TEME')
true_prior = prop.propagate(
    np.min(tv), [dual_corrs['line1'][dual_ind], dual_corrs['line2'][dual_ind]],
)
prop.set(out_frame='ITRS')

sim_r = []
meas_r = []
sim_v = []
meas_v = []
meas_times = []
for ind, (tx, dat, mid) in enumerate(zip(txs, datas, mids)):
    r_s, v_s = get_range_and_range_rate(tx, states[:, ind].reshape(6, 1))
    sim_r.append(r_s[0])
    sim_v.append(v_s[0])
    meas_r.append(dat['r'][mid])
    meas_v.append(dat['v'][mid])
    meas_times.append(dat['times'][mid])

sim_r = np.array(sim_r)
meas_r = np.array(meas_r)
sim_v = np.array(sim_v)
meas_v = np.array(meas_v)

print('r-err [m]: ', meas_r - sim_r)
print('v-err [m]: ', meas_v - sim_v)

pos_vecs = []

for ind, radar in enumerate(radar_lst):
    sph_point = np.array(radar_info[radar][data_ids[radar]])
    local_point = pyant.coordinates.sph_to_cart(sph_point, radians=False)
    ecef_point = sorts.frames.enu_to_ecef(
        txs[ind].lat, txs[ind].lon, txs[ind].alt, 
        local_point, 
        radians=False,
    )
    ecef_st = txs[ind].ecef

    r_ecef = ecef_st + meas_r[ind]*ecef_point

    r_teme = sorts.frames.convert(
        meas_times[ind], 
        np.hstack([r_ecef, np.ones_like(r_ecef)]), 
        in_frame='ITRS', 
        out_frame='TEME',
    )
    r_teme = r_teme[:3]
    pos_vecs.append(r_teme)

dt = tv[1] - tv[0]
print('times: ', [dat['times'][mid].iso for mid, dat in zip(mids, datas)])
print('dt: ', dt)

if dt < 0:
    orb = solve_lambert(pos_vecs[1], pos_vecs[0], -dt)
else:
    orb = solve_lambert(pos_vecs[0], pos_vecs[1], dt)

print('inital lambert guess: ')
print(orb)

odlab.profiler.enable('odlab')

# od_prop = sorts.propagator.Kepler(
#     settings=dict(
#         in_frame='TEME',
#         out_frame='ITRS',
#     )
# )
od_prop = sorts.propagator.SGP4(
    settings=dict(
        in_frame='TEME',
        out_frame='ITRS',
    )
)

source_data = []

r_err = 5.0e2
v_err = 1.0e2
dates = [odlab.times.mjd2npdt(_t.mjd) for _t in meas_times]

for ind, tx in enumerate(txs):
    radar_data = np.empty((1,), dtype=odlab.sources.RadarTracklet.dtype)
    radar_data['date'][0] = dates[ind]
    radar_data['r'][0] = meas_r[ind]
    radar_data['v'][0] = meas_r[ind]
    radar_data['r_sd'][0] = r_err
    radar_data['v_sd'][0] = v_err

    source_data.append({
        'data': radar_data,
        'meta': dict(
            tx_ecef = tx.ecef,
            rx_ecef = tx.ecef,
        ),
        'index': 1,
    })

paths = odlab.SourcePath.from_list(source_data, 'ram')

sources = odlab.SourceCollection(paths = paths)
sources.details()

mean_prior, B, tle_epoch = od_prop.get_mean_elements(
    dual_corrs['line1'][dual_ind], 
    dual_corrs['line2'][dual_ind], 
    radians=False,
)


# state_variables = ['x', 'y', 'z', 'vx', 'vy', 'vz']
state_variables = ['a', 'e', 'inc', 'apo', 'raan', 'mu0']
variables = state_variables
dtype = [(name, 'float64') for name in variables]
state0_named = np.empty((1,), dtype=dtype)
for ind, name in enumerate(variables):
    # state0_named[name] = orb.cartesian[ind, 0]
    # state0_named[name] = true_prior[ind]
    state0_named[name] = mean_prior[ind]


input_data_state = {
    'sources': sources,
    'Models': [odlab.RadarPair]*len(sources),
    # 'date0': np.min(dates),
    'date0': odlab.times.mjd2npdt(tle_epoch.mjd),
    'params': dict(
        B = B,
        M0 = pyorb.M_earth,
        SGP4_mean_elements = True,
    ),
}

post = odlab.OptimizeLeastSquares(
    data = input_data_state,
    variables = variables,
    state_variables = state_variables,
    start = state0_named,
    prior = None,
    propagator = od_prop,
    method = 'Nelder-Mead',
    options = dict(
        maxiter = 10000,
        disp = False,
        xatol = 1e-3,
    ),
)


post.run()

print(post.results)

# res_cart = np.array([post.results.MAP[key][0] for key in state_variables])
res_mean = np.array([post.results.MAP[key][0] for key in state_variables])

od_prop.set(out_frame='TEME')
_state0 = od_prop.propagate(
    0, res_mean, epoch=tle_epoch, B=B, SGP4_mean_elements=True,
)

result_orb = pyorb.Orbit(
    M0=pyorb.M_earth, 
    m=0, 
    num=1, 
    radians=False,
    cartesian = _state0.reshape(6, 1),
    # cartesian = res_cart.reshape(6, 1),
)
print(result_orb)

deltas = [0.1]*3 + [0.05]*3
Sigma_orb = post.linear_MAP_covariance(post.results.MAP, deltas)
print('Linearized MAP covariance:')
print(Sigma_orb)

print(odlab.profiler)
