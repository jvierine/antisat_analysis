from astropy.time import Time, TimeDelta
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import h5py
from pprint import pprint

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    class COMM_WORLD:
        rank = 0
        size = 1
    comm = COMM_WORLD()

import sys

top_dir = Path(__file__).parents[1].resolve()
sys.path.append(str(top_dir))

from convert_spade_events_to_h5 import get_spade_data
from convert_spade_events_to_h5 import read_spade, date2unix
from convert_spade_events_to_h5 import MIN_SNR
from convert_spade_events_to_h5 import read_extended_event_data, load_spade_extended

from lamberthub import izzo2015

import odlab
import sorts
import pyorb
import pyant

clobber = False

err_t_prop = 10*3600.0
err_t_step = 10.0

steps = 10000
step_arr = np.ones((6,), dtype=np.float64)*0.1

err_t_samp = np.arange(0, err_t_prop, err_t_step, dtype=np.float64)

t_jitter = np.linspace(-5, 5, num=11)

HERE = Path(__file__).parent.resolve()
OUTPUT = HERE / 'output' / 'russian_asat'
print(f'Using {OUTPUT} as output')

out_pth = OUTPUT / 'orbit_determination'
out_pth.mkdir(exist_ok=True)

results_data_file = out_pth / 'iod_results.pickle'

with open(OUTPUT / 'paths.pickle', 'rb') as fh:
    paths = pickle.load(fh)


def multi_vector_angle(a, b, radians=False):
    a_norm = np.linalg.norm(a, axis=0)
    b_norm = np.linalg.norm(b, axis=0)
    proj = np.sum(a*b, axis=0)/(a_norm*b_norm)
    proj[proj > 1.0] = 1.0
    proj[proj < -1.0] = -1.0

    theta = np.arccos(proj)
    if not radians:
        theta = np.degrees(theta)

    return theta


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


def get_range_and_range_rate(station, states):
    """
    return the range and range rate of a radar measurement
    in units of meters and meters per second.

     states is an array 6xN with N state vectors
     dims 0,1,2 = x,y,z  (m)
          3,4,5 = vx,vy,vz (m/s)
    """
    k = np.zeros([3, states.shape[1]], dtype=np.float64)

    k = station.ecef[:, None] - states[:3, :]

    # ranges (distance from radar to satellite)
    ranges = np.sqrt(np.sum(k**2.0, axis=0))

    k_unit = k/ranges

    # k dot v where k in unit normalized
    range_rates = -np.sum(k_unit*states[3:, :], axis=0)

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


high_res_paths = {
    'esr': [
        Path('/home/danielk/data/spade/beamparks/esr/2021.11.23/LEO_32m_20211123_LONGRESULTS'),
    ], 
    'uhf': [
        Path('/home/danielk/data/spade/beamparks/uhf/2021.11.23/LEO_UHF_20211123_LONGRESULTS'),
    ],
}
high_res_files = {}
for radar in high_res_paths:
    for input_pth in high_res_paths[radar]:
        _high_dat = {
            'summaries': list(input_pth.rglob('events.txt')),
            'files': list(input_pth.rglob('*.hlist')),
        }
        if radar in high_res_files:
            high_res_files[radar]['summaries'] += _high_dat['summaries']
            high_res_files[radar]['files'] += _high_dat['files']
        else:
            high_res_files[radar] = _high_dat

if not dual_corr_file.is_file():
    # cmd = f'python multi_beampark_correlator.py {tle_file} -o {dual_corr_file} {corr1_file} {corr2_file}'
    print('FILE NOT FOUND, RUN: ')
    print('''
python multi_beampark_correlator.py \
    /home/danielk/git/antisat_analysis/projects/output/russian_asat/2021_11_23_spacetrack.tle \
    -o /home/danielk/git/antisat_analysis/projects/output/russian_asat/orbit_determination/2021-11-23_dual_correlations.npy \
    /home/danielk/git/antisat_analysis/projects/output/russian_asat/2021.11.23_uhf_correlation.pickle \
    /home/danielk/git/antisat_analysis/projects/output/russian_asat/2021.11.23_uhf_correlation_plots/eiscat_uhf_selected_correlations.npy \
    /home/danielk/git/antisat_analysis/projects/output/russian_asat/2021.11.23_esr_correlation.pickle \
    /home/danielk/git/antisat_analysis/projects/output/russian_asat/2021.11.23_esr_correlation_plots/eiscat_esr_selected_correlations.npy
        ''')
    exit()
else:
    print(f'{dual_corr_file} found')
    print('loading...')

dual_corrs = np.load(dual_corr_file)

txs = [
    sorts.radars.eiscat_uhf.tx[0],
    sorts.radars.eiscat_esr.tx[0],
]

dual_inds = list(range(len(dual_corrs['cid'])))
iod_results = {
    'id': [None for ind in dual_inds],
    'event': [None for ind in dual_inds],
    'mcmc_teme_tle_diff': [None for ind in dual_inds],
    'lsq_teme_tle_diff': [None for ind in dual_inds],
}

beam_fwhm = []
for tx in txs:
    _pointing = tx.beam.pointing.copy()
    tx.beam.sph_point(0, 90)
    G0 = tx.beam.sph_gain(0, 90)
    _el_step = 0.01
    _el = 90 - _el_step
    G1 = tx.beam.sph_gain(0, _el)
    while G1 > G0*0.5:
        _el -= _el_step
        G1 = tx.beam.sph_gain(0, _el)
    beam_fwhm.append(2*(90 - _el))
    tx.beam.point(_pointing)

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

r_err = 5.0e2
v_err = 1.0e2
th_samples = 1000

if comm.size > 1:
    my_dual_inds = dual_inds[comm.rank::comm.size]

data_ids = {'uhf': 0, 'esr': 1}
radar_lst = ['uhf', 'esr']
for dual_ind in my_dual_inds:
    
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

    meas_times = []
    for ind, (tx, dat, mid) in enumerate(zip(txs, datas, mids)):
        meas_times.append(dat['times'][mid])

    class fake_args:
        v = 0

    event_names = []
    event_datas = []
    event_peak_ids = []
    for rdi, radar in enumerate(radar_lst):
        meas_sumf = meas_times[rdi].datetime
        meas_sumf = f'{meas_sumf.year}{meas_sumf.month}{meas_sumf.day}_{meas_sumf.hour}'
        for sumf in high_res_files[radar]['summaries']:
            if sumf.parent.name != meas_sumf:
                continue

            print(f'using summary file {sumf}...')

            ext_data = read_extended_event_data([sumf], fake_args)
            t0 = Time(ext_data['t'], format='unix', scale='utc')
            _dt = np.abs((t0 - meas_times[rdi]).sec)
            _match_ind = np.argmin(_dt)
            event_name = ext_data.iloc[_match_ind]['event_name']
            event_names.append(event_name)
            print(f'matched event (ind={_match_ind}): dt={_dt[_match_ind]} for {event_name}')
            
            event_data = None
            for evpath in high_res_files[radar]['files']:
                if evpath.parent.name == event_name:
                    print(f'loading {evpath.name}...')
                    event_data = load_spade_extended(evpath)
                    break

            event_datas.append(event_data)

            SNR_db = 10*np.log10(event_data['SNR'])
            event_data['select'] = SNR_db > np.log10(MIN_SNR)*10
            event_peak = np.argmax(event_data['SNR'][event_data['select']])
            event_peak_ids.append(event_peak)

            break
    iod_results['event'][dual_ind] = event_name
    iod_results['id'][dual_ind] = dual_ind

    tvs = []
    states = []
    meas_t = []
    prop.set(out_frame='ITRS')
    for ind, dat in enumerate(event_datas):
        t_off = t_jitter[dual_corrs[f'jitter{ind}'][dual_ind]]
        tm = Time(dat['unix'][dat['select']].values, format='unix', scale='utc')
        meas_t.append(tm)
        tv = (tm - epoch0).sec + t_off

        tvs.append(tv)

        print(f'Propagating {len(tv)} points for {radar_lst[ind]} with jitter {t_off} s')
        print('t_check', (tm - meas_times[ind]).sec)

        state = prop.propagate(
            tv, [dual_corrs['line1'][dual_ind], dual_corrs['line2'][dual_ind]]
        )
        states.append(state)

    sim_r = []
    meas_r = []
    sim_v = []
    meas_v = []
    for ind, (tx, state, dat) in enumerate(zip(txs, states, event_datas)):
        r_s, v_s = get_range_and_range_rate(tx, state)
        sim_r.append(r_s)
        sim_v.append(v_s)
        meas_r.append(dat['r'][dat['select']].values)
        meas_v.append(dat['v'][dat['select']].values)

    meas_r0 = []
    meas_v0 = []
    for ind, (dat, mid) in enumerate(zip(datas, mids)):
        meas_r0.append(dat['r'][mid])
        meas_v0.append(dat['v'][mid])
    meas_r0 = np.array(meas_r0)
    meas_v0 = np.array(meas_v0)
    
    for ind in range(len(radar_lst)):
        print(radar_lst[ind] + '\n' + '='*len(radar_lst[ind]))
        print('corr diffs')
        # two way and reverse defined
        print('r-err [m  ]: ', -0.5*dual_corrs[f'dr{ind}'][dual_ind]) 
        print('v-err [m/s]: ', -0.5*dual_corrs[f'dv{ind}'][dual_ind])
        print('events.txt diff')
        print('r-err [m  ]: ', meas_r0[ind] - sim_r[ind][event_peak_ids[ind]])
        print('v-err [m/s]: ', meas_v0[ind] - sim_v[ind][event_peak_ids[ind]])
        print('hlist diff')
        print('r-err [m  ]: ', meas_r[ind] - sim_r[ind])
        print('v-err [m/s]: ', meas_v[ind] - sim_v[ind])

    # fig, axes = plt.subplots(2, 2)
    # axes[0, 0].plot(tvs[0], meas_r[0], 'r')
    # axes[0, 0].plot(tvs[0], sim_r[0], 'b')

    # axes[1, 0].plot(tvs[0], meas_v[0], 'r')
    # axes[1, 0].plot(tvs[0], sim_v[0], 'b')

    # axes[0, 1].plot(tvs[1], meas_r[1], 'r')
    # axes[0, 1].plot(tvs[1], sim_r[1], 'b')

    # axes[1, 1].plot(tvs[1], meas_v[1], 'r')
    # axes[1, 1].plot(tvs[1], sim_v[1], 'b')

    # plt.show()

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

        r_ecef = ecef_st + meas_r[ind][event_peak_ids[ind]]*ecef_point

        r_teme = sorts.frames.convert(
            meas_times[ind], 
            np.hstack([r_ecef, np.ones_like(r_ecef)]), 
            in_frame='ITRS', 
            out_frame='TEME',
        )
        r_teme = r_teme[:3]
        pos_vecs.append(r_teme)

    dt = (meas_times[1] - meas_times[0]).sec
    print('times: ', [dat['times'][mid].iso for mid, dat in zip(mids, datas)])
    print('dt: ', dt)

    if dt < 0:
        orb = solve_lambert(pos_vecs[1], pos_vecs[0], -dt)
        date0 = odlab.times.mjd2npdt(meas_times[1].mjd)
        epoch_iod = meas_times[1]
    else:
        orb = solve_lambert(pos_vecs[0], pos_vecs[1], dt)
        date0 = odlab.times.mjd2npdt(meas_times[0].mjd)
        epoch_iod = meas_times[0]

    print('inital lambert guess: ')
    print(orb)

    if orb.e > 1:
        print('lambert solver fails to find bound orbit, reverting to circular optimization')
        theta = np.linspace(0, np.pi*2, th_samples)
        sample_orbs_0 = theta_orbit(pos_vecs[0], meas_times[0], theta)
        sample_orbs_0.calculate_kepler()
        sample_orbs = sample_orbs_0.copy()
        sample_orbs.propagate((meas_times[1] - meas_times[0]).sec)

        sample_itrs = sorts.frames.convert(
            meas_times[1], 
            sample_orbs.cartesian, 
            in_frame='TEME', 
            out_frame='ITRS',
        )

        sample_r, sample_v = get_range_and_range_rate(txs[1], sample_itrs)

        metric = ((sample_r - meas_r[1][event_peak_ids[1]])/r_err)**2 + ((sample_v - meas_v[1][event_peak_ids[1]])/v_err)**2
        best_th = np.argmin(metric)

        orb = sample_orbs_0[best_th]

        print(f'circular opt metric: \n\
            [dr={sample_r[best_th] - meas_r[1][event_peak_ids[1]]} m] \n\
            [dv={sample_v[best_th] - meas_v[1][event_peak_ids[1]]} m/s] \n\
            [metric={metric[best_th]}]')
        print('inital circular guess: ')
        print(orb)
        date0 = odlab.times.mjd2npdt(meas_times[0].mjd)
        epoch_iod = meas_times[0]
    
    for ind, radar in enumerate(radar_lst):
        orb_prop = orb.copy()
        orb_prop.allocate(len(meas_t[ind]))
        for var_ind in range(6):
            orb_prop._kep[var_ind, :] = orb._kep[var_ind, 0]
        orb_prop.propagate((meas_t[ind] - epoch_iod).sec)
        iod_start_cart = sorts.frames.convert(
            meas_t[ind], 
            orb_prop.cartesian, 
            in_frame='TEME', 
            out_frame='ITRS',
        )
        r_s, v_s = get_range_and_range_rate(txs[ind], iod_start_cart)
        print('IOD start diff')
        print('r-err [m  ]: ', meas_r[ind] - r_s)
        print('v-err [m/s]: ', meas_v[ind] - v_s)

    prop.set(out_frame='TEME')
    true_prior = prop.propagate(
        (epoch_iod - epoch0).sec, [dual_corrs['line1'][dual_ind], dual_corrs['line2'][dual_ind]],
    )
    prop.set(out_frame='ITRS')

    odlab.profiler.enable('odlab')

    od_prop = sorts.propagator.Kepler(
        settings=dict(
            in_frame='TEME',
            out_frame='ITRS',
        )
    )

    source_data = []
    for ind, tx in enumerate(txs):
        radar_data = np.empty((len(meas_t[ind]),), dtype=odlab.sources.RadarTracklet.dtype)
        radar_data['date'][:] = odlab.times.mjd2npdt(meas_t[ind].mjd)
        radar_data['r'][:] = meas_r[ind]*2  # two way
        radar_data['v'][:] = meas_v[ind]*2  # two way
        radar_data['r_sd'][:] = r_err
        radar_data['v_sd'][:] = v_err

        source_data.append({
            'data': radar_data,
            'meta': dict(
                tx_ecef = tx.ecef,
                rx_ecef = tx.ecef,
            ),
            'index': 1,
        })

    # pprint(source_data)
    paths = odlab.SourcePath.from_list(source_data, 'ram')

    sources = odlab.SourceCollection(paths = paths)
    sources.details()

    state_variables = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    dtype = [(name, 'float64') for name in state_variables]

    step = np.empty((1,), dtype=dtype)
    for ind, name in enumerate(state_variables):
        step[name] = step_arr[ind]

    # kep_state_variables = ['a', 'e', 'inc', 'apo', 'raan', 'mu0']
    # kep_dtype = [(name, 'float64') for name in kep_state_variables]

    state0_named = np.empty((1,), dtype=dtype)
    true_prior_named = np.empty((1,), dtype=dtype)
    for ind, name in enumerate(state_variables):
        state0_named[name] = orb.cartesian[ind, 0]
        true_prior_named[name] = true_prior[ind]

    input_data_state = {
        'sources': sources,
        'Models': [odlab.RadarPair]*len(sources),
        'date0': date0,
        'params': dict(
            M0 = pyorb.M_earth,
        ),
    }

    post_grad = odlab.OptimizeLeastSquares(
        data = input_data_state,
        variables = state_variables,
        state_variables = state_variables,
        start = state0_named,
        prior = None,
        propagator = od_prop,
        method = 'Nelder-Mead',
        options = dict(
            maxiter = 10000,
            disp = False,
            xatol = 1e1,
        ),
    )

    post_path = out_pth / f'lsq_results_dual_obj{dual_ind}.h5'
    if post_path.is_file() and not clobber:
        post_grad_results = odlab.PosteriorParameters.load_h5(post_path)
    else:
        post_grad.run()
        post_grad.results.save(post_path)
        post_grad_results = post_grad.results

    mcmc_start = post_grad_results.MAP
    post = odlab.MCMCLeastSquares(
        data = input_data_state,
        variables = state_variables,
        state_variables = state_variables,
        start = mcmc_start,
        prior = None,
        propagator = od_prop,
        method = 'SCAM',
        proposal = 'LinSigma',
        # proposal = 'normal',
        method_options = dict(
            accept_max = 0.5,
            accept_min = 0.3,
            adapt_interval = 500,
        ),
        steps = steps,
        step = step,
        tune = 0,
        MPI = False,
    )

    post_mcmc_path = out_pth / f'mcmc_results_dual_obj{dual_ind}.h5'
    if post_mcmc_path.is_file() and not clobber:
        post_results = odlab.PosteriorParameters.load_h5(post_mcmc_path)
    else:
        post.run()
        post.results.save(post_mcmc_path)
        post_results = post.results

    print(odlab.profiler)

    print(' POST GRADIENT ASCENT \n')
    print(post_grad_results)

    print('==== Start variables [m, m/s] ====')
    for key in state_variables:
        print(f'{key}: {state0_named[key]}')

    print('==== TLE prior [m, m/s] ====')
    for ind, key in enumerate(state_variables):
        print(f'{key}: {true_prior[ind]}')

    print('==== IOD-LSQ result [m, m/s] ====')
    for key in state_variables:
        print(f'{key}: {post_grad_results.MAP[key]}')

    print('==== (IOD-LSQ - TLE) diff [km, km/s] ====')
    iod_results['lsq_teme_tle_diff'][dual_ind] = np.empty_like(true_prior)
    for ind, key in enumerate(state_variables):
        print(f'{key}: {(post_grad_results.MAP[key] - true_prior[ind])*1e-3}')
        iod_results['lsq_teme_tle_diff'][dual_ind][ind] = post_grad_results.MAP[key] - true_prior[ind]

    print('==== IOD-MCMC result [m, m/s] ====')
    for key in state_variables:
        print(f'{key}: {post_results.MAP[key]}')

    print('==== (IOD-MCMC - TLE) diff [km, km/s] ====')
    iod_results['mcmc_teme_tle_diff'][dual_ind] = np.empty_like(true_prior)
    for ind, key in enumerate(state_variables):
        print(f'{key}: {(post_results.MAP[key] - true_prior[ind])*1e-3}')
        iod_results['mcmc_teme_tle_diff'][dual_ind][ind] = post_results.MAP[key] - true_prior[ind]

    post_grad_state = [post_results.MAP[key][0] for key in state_variables]
    post_grad_kep_orb = pyorb.Orbit(
        M0=pyorb.M_earth, 
        m=0, 
        num=1, 
        radians=False,
        cartesian = np.array(post_grad_state).reshape(6, 1),
    )
    true_prior_kep_orb = pyorb.Orbit(
        M0=pyorb.M_earth, 
        m=0, 
        num=1, 
        radians=False,
        cartesian = true_prior.reshape(6, 1),
    )

    od_prop.out_frame = 'TEME'
    prop.out_frame = 'TEME'

    tle_prop = prop.get_TLE_parameters(
        dual_corrs['line1'][dual_ind], 
        dual_corrs['line2'][dual_ind],
    )
    satnum = tle_prop['satnum']

    tle_states = od_prop.propagate(
        err_t_samp,
        true_prior_kep_orb,
        epoch = epoch_iod,
    )
    iod_states = od_prop.propagate(
        err_t_samp,
        post_grad_kep_orb,
        epoch = epoch_iod,
    )

    iod_pos_err = np.linalg.norm(tle_states[:3, :] - iod_states[:3, :], axis=0)
    tle_states_itrs = sorts.frames.convert(
        epoch_iod + TimeDelta(err_t_samp, format='sec'), 
        tle_states, 
        in_frame='TEME', 
        out_frame='ITRS',
    )
    iod_ang_errs = [
        multi_vector_angle(
            tle_states_itrs[:3, :] - tx.ecef[:, None],
            sorts.frames.convert(
                epoch_iod + TimeDelta(err_t_samp, format='sec'), 
                iod_states, 
                in_frame='TEME', 
                out_frame='ITRS',
            )[:3, :] - tx.ecef[:, None],
        )
        for tx in txs
    ]
    for txi in range(len(txs)):
        sel = np.logical_not(txs[txi].field_of_view(tle_states_itrs))
        iod_ang_errs[txi][sel] = np.nan

    od_prop.out_frame = 'ITRS'
    prop.out_frame = 'ITRS'

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.plot(err_t_samp/3600.0, iod_pos_err*1e-3, "-b")
    ax.set_xlabel('Time past IOD epoch [h]')
    ax.set_ylabel('Position difference [km]')
    ax.set_title(f'Dual beampark IOD vs TLE, NORAD-ID {satnum}')
    fig.savefig(out_pth / f'IOD_vs_TLE_pos_error_dual_obj{dual_ind}.png')
    plt.close(fig)

    cols = ['b', 'm']

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    for rdi in range(len(radar_lst)):
        ax.plot(err_t_samp/3600.0, iod_ang_errs[rdi], c=cols[rdi], label=f'From {radar_lst[rdi].upper()}')
        ax.axhline(beam_fwhm[rdi], ls='--', c=cols[rdi], label=f'{radar_lst[rdi].upper()} FWHM')
    ax.legend()
    ax.set_xlabel('Time past IOD epoch [h]')
    ax.set_ylabel('Angle difference [deg]')
    ax.set_title(f'Dual beampark IOD vs TLE, NORAD-ID {satnum}')
    fig.savefig(out_pth / f'IOD_vs_TLE_ang_error_dual_obj{dual_ind}.png')
    plt.close(fig)

    figs, axes = odlab.plot.trace(post_results, reference=true_prior_named)
    for ind, fig in enumerate(figs):
        fig.savefig(out_pth / f'IOD_MCMC_trace_p{ind}_dual_obj{dual_ind}.png')
        plt.close(fig)
    fig, axes = odlab.plot.scatter_trace(
        post_results, 
        reference=true_prior_named,
        alpha = 0.1,
        size = 4,
    )
    fig.savefig(out_pth / f'IOD_MCMC_scatter_ref_dual_obj{dual_ind}.png')
    plt.close(fig)

    fig, axes = odlab.plot.scatter_trace(
        post_results,
        alpha = 0.1,
    )
    fig.savefig(out_pth / f'IOD_MCMC_scatter_dual_obj{dual_ind}.png')
    plt.close(fig)

    for absolute in [True, False]:
        figs, axes = odlab.plot.residuals(
            post, 
            [state0_named, post_grad_results.MAP, post_results.MAP, true_prior_named], 
            ['IOD-start', 'IOD-LSQ-MAP', 'IOD-MCMC-MAP', 'TLE'], 
            ['-b', '-g', '-m', '-r'], 
            units = {'r': 'm', 'v': 'm/s'},
            absolute=absolute,
        )
        for ind, fig in enumerate(figs):
            fig.savefig(out_pth / f'IOD_{"abs_" if absolute else ""}residual_p{ind}_dual_obj{dual_ind}.png')
            plt.close(fig)

    
if comm.size > 1:
    mpi_inds = []
    for thr_id in range(comm.size):
        mpi_inds.append(dual_inds[thr_id::comm.size])

    if comm.rank == 0:
        for thr_id in range(comm.size):
            if thr_id != 0:
                for ind in mpi_inds[thr_id]:
                    for kind, key in enumerate(iod_results):
                        iod_results[key][ind] = comm.recv(source=thr_id, tag=ind*10 + kind)

    else:
        for ind in mpi_inds[comm.rank]:
            for kind, key in enumerate(iod_results):
                comm.send(iod_results[key][ind], dest=0, tag=ind*10 + kind)

    if comm.rank != 0:
        exit()

with open(results_data_file, 'wb') as fh:
    pickle.dump(iod_results, fh)
