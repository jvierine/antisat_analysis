from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time, TimeDelta
import scipy.optimize as sciopt
import scipy.stats as scist
from tqdm import tqdm
import multiprocessing as mp

import sorts
import pyant
import pyorb

CORES = 5
G_db_lim = 10.0
dt = 1.0
sample_nums = [200, 50]
# sample_nums = [100, 10]
INC_LIMS = (-0.008, 0.008)
ANOM_LIMS = (-0.005, 0.005)
MIN_SEP = 0.003
MIN_PEAK = 0.8

event_name = 'uhf_20211123_120429_800000'
# event_name = 'uhf_20211123_120016_600000'

data_path = Path('/home/danielk/data/spade/beamparks/uhf/')
data_path /= f'2021.11.23/leo_bpark_2.1u_NO@uhf_extended/20211123_12/{event_name}/'
data_path /= f'{event_name}.hlist'
out_folder = Path(__file__).resolve().parent / 'output' / 'RCS_test_plots2'
out_folder.mkdir(exist_ok=True)
CACHE_FILE = Path(__file__).resolve().parent / 'output' / 'RCS_test_cache.pickle'

close_plot = True


def load_spade_extended(path):
    names = [f'{x}' for x in range(20)]
    names[8] = 'r'
    names[10] = 'SNR'
    names[14] = 't'
    names[9] = 'v'

    data = pd.read_csv(
        path, 
        sep=r'[ ]+', 
        comment='%', 
        skip_blank_lines=True, 
        names=names, 
        skiprows=45,
        engine='python',
    )
    data['SNR'] = data['SNR']**2
    for ti in range(len(data['t']) - 1):
        if data['t'].values[ti] > data['t'].values[ti + 1]:
            data['t'].values[(ti + 1):] += data['t'].values[ti]
    data['t'] = (data['t'] - np.min(data['t']))*1e-6
    data['r'] = data['r']*1e3

    return data


def locate_in_beam(x, orb, r_ecef):
    orb0 = orb.copy()
    orb0.i = x[0]
    orb0.Omega = x[1]
    orb0.anom = x[2]
    pos = sorts.frames.convert(
        epoch, 
        orb0.cartesian, 
        in_frame='TEME', 
        out_frame='ITRS',
    ).flatten()[:3]
    dist = np.linalg.norm(pos - r_ecef)
    return dist


def evaluate_sample(ind, data, orb, incs, nus, t_, epoch, radar, ecef_st, dt):
    orb_pert = orb.copy()
    orb_pert.i = orb_pert.i + incs[ind]
    orb_pert.anom = orb_pert.anom + nus[ind]

    states_ecef = prop.propagate(t_, orb_pert, epoch=epoch)
    local_pos = sorts.frames.ecef_to_enu(
        radar.tx[0].lat, 
        radar.tx[0].lon, 
        radar.tx[0].alt, 
        states_ecef[:3, :] - ecef_st[:, None],
        radians=False,
    )

    pth = local_pos/np.linalg.norm(local_pos, axis=0)
    G_pth = radar.tx[0].beam.gain(pth)
    G_pth_db = 10*np.log10(G_pth)
    diam = sorts.signals.hard_target_diameter(
        G_pth, 
        G_pth,
        radar.tx[0].wavelength,
        radar.tx[0].power,
        data['r'].values, 
        data['r'].values,
        data['SNR'].values, 
        bandwidth=radar.tx[0].coh_int_bandwidth,
        rx_noise_temp=radar.rx[0].noise,
        radar_albedo=1.0,
    )
    SNR_sim = sorts.signals.hard_target_snr(
        G_pth, 
        G_pth,
        radar.tx[0].wavelength,
        radar.tx[0].power,
        data['r'].values, 
        data['r'].values,
        diameter=1, 
        bandwidth=radar.tx[0].coh_int_bandwidth,
        rx_noise_temp=radar.rx[0].noise,
        radar_albedo=1.0,
    )
    diam[G_pth_db < G_db_lim] = np.nan
    SNR_sim[G_pth_db < G_db_lim] = 0
    
    xsn = data['SNR'].values.copy()
    ysn = SNR_sim.copy()

    norm_coef = np.sqrt(np.sum(xsn**2))*np.sqrt(np.sum(ysn**2))
    match = np.sum(xsn*ysn)/norm_coef
    
    inds = np.logical_not(np.isnan(diam))
    inds = np.logical_and(inds, np.abs(t_) < dt)
    inds = np.where(inds)[0]

    if len(inds) < 3:
        diam_match = 0
        # scale = 0
    else:
        std_est = np.std(diam[inds])/np.mean(diam[inds]) + 1
        diam_match = 1.0/std_est
        # scale = np.sum(G_pth_db[inds] >= G_db_lim)/len(G_pth_db[inds])
    
    if np.sum(SNR_sim) < 0.001:
        match = 0.0

    # match *= scale

    match_v2 = match
    match_v2 *= diam_match

    return pth, diam, SNR_sim, match, match_v2, ind


def wrapped_evaluate_sample(f_args):
    return evaluate_sample(*f_args)


if __name__ == '__main__':

    print(f'sorts=={sorts.__version__}')
    print(f'pyorb=={pyorb.__version__}')
    print(f'pyant=={pyant.__version__}')

    radar = sorts.radars.eiscat_uhf
    radar.tx[0].beam.sph_point(azimuth=90, elevation=75)

    data = load_spade_extended(data_path)

    prop = sorts.propagator.Kepler(
        settings=dict(
            in_frame='TEME',
            out_frame='ITRS',
        ),
    )
    ecef_point = radar.tx[0].pointing_ecef
    ecef_st = radar.tx[0].ecef

    snr_max = np.argmax(data['SNR'].values)
    r_obj = data['r'].values[snr_max]
    r_ecef = ecef_st + r_obj*ecef_point

    epoch = Time('J2000')
    t_ = data['t'].values - data['t'].values[snr_max]

    orb = pyorb.Orbit(M0=pyorb.M_earth, m=0, num=1, epoch=epoch, radians=False)
    orb.a = np.linalg.norm(r_ecef)
    orb.e = 0
    orb.omega = 0

    it = 1
    min_res = sciopt.minimize(
        locate_in_beam, 
        [51, 79, 158], 
        args=(orb, r_ecef), 
        method='Nelder-Mead',
    )
    while it < 10 and min_res.fun > 1e3:
        it += 1
        x0 = np.random.rand(3)
        x0[0] *= 90
        x0[1:] *= 360
        min_res0 = sciopt.minimize(locate_in_beam, x0, args=(orb, r_ecef), method='Nelder-Mead')
        print(f'{it}: [{min_res0.fun}] {x0} -> {min_res0.x}')
        if min_res0.fun < min_res.fun:
            min_res = min_res0
        
        if min_res.fun < 1e3:
            break

    print(min_res)
    orb.i = min_res.x[0]
    orb.Omega = min_res.x[1]
    orb.anom = min_res.x[2]
    print(orb)

    incs_mat, nus_mat = np.meshgrid(
        np.linspace(INC_LIMS[0], INC_LIMS[1], sample_nums[0]),
        np.linspace(ANOM_LIMS[0], ANOM_LIMS[1], sample_nums[1]),
    )
    incs = incs_mat.reshape((incs_mat.size, ))
    nus = nus_mat.reshape((nus_mat.size, ))
    samps = len(incs)

    diams = [None for x in range(samps)]
    SNR_sims = [None for x in range(samps)]
    pths = [None for x in range(samps)]
    matches = [None for x in range(samps)]
    matches_v2 = [None for x in range(samps)]

    if CACHE_FILE.is_file():
        with open(CACHE_FILE, 'rb') as fh:
            reses = pickle.load(fh)
    else:
        with mp.Pool(CORES) as pool:
            reses = []
            with tqdm(total=samps) as pbar:
                f_args_list = [(
                    ind, data, orb, 
                    incs, nus, t_, 
                    epoch, radar, ecef_st, 
                    dt,
                    ) for ind in range(samps)
                ]
                for i, res in enumerate(pool.imap_unordered(wrapped_evaluate_sample, f_args_list)):
                    reses.append(res)
                    pbar.update()
        with open(CACHE_FILE, 'wb') as fh:
            pickle.dump(reses, fh)

    for ind in range(samps):
        pth, diam, SNR_sim, match, match_v2, ind = reses[ind]
        pths[ind] = pth
        diams[ind] = diam
        SNR_sims[ind] = SNR_sim
        matches[ind] = match
        matches_v2[ind] = match_v2

    print('Results distributed...')

    del reses

    matches = np.array(matches)
    matches_v2 = np.array(matches_v2)

    samp_inds = np.arange(samps)

    matches_mat = matches.reshape(nus_mat.shape)

    sn_m_max = np.max(data['SNR'].values)
    sn_m = data['SNR'].values/sn_m_max

    print('Plotting...')

    fig1, ax1 = plt.subplots(figsize=(12, 8))
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    for pth in pths:
        ax1.plot(pth[0, :], pth[1, :])
        ax2.plot(pth[0, :], pth[1, :], '-w')
    pyant.plotting.gain_heatmap(radar.tx[0].beam, min_elevation=85.0, ax=ax2)

    fig1.savefig(out_folder / 'samples_trajs.png')
    if close_plot:
        plt.close(fig1)

    fig2.savefig(out_folder / 'sample_trajs_gain.png')
    if close_plot:
        plt.close(fig2)

    best_matches = np.argsort(matches)[::-1]
    best_peaks = [best_matches[0]]
    for ind in best_matches[1:]:
        if matches[ind] < MIN_PEAK:
            continue
        ds = np.sqrt([
            (incs[ind] - incs[peak_ind])**2 + (nus[ind] - nus[peak_ind])**2
            for peak_ind in best_peaks
        ])
        if np.all(ds > MIN_SEP):
            best_peaks.append(ind)

    peak_diams = np.array([diams[ind][snr_max] for ind in best_peaks])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(peak_diams*1e2, matches[best_peaks], '.')
    ax.set_xlabel('Diameter [cm]')
    ax.set_ylabel('Matching function [1]')
    ax.set_title('Local matching peak distribution')
    fig.savefig(out_folder / 'peak_diam_dist_v0.png')
    if close_plot:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(diams[best_matches[0]]*1e2)
    ax.axvline(diams[best_matches[0]][snr_max]*1e2, color='r')
    ax.set_xlabel('Diameter [log10(cm)]')
    ax.set_ylabel('Frequency [1]')
    ax.set_title('Best match diameter distribution')
    fig.savefig(out_folder / 'diam_dist_best_v0.png')
    if close_plot:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8))
    pmesh = ax.pcolormesh(incs_mat, nus_mat, matches_mat)
    cbar = fig.colorbar(pmesh, ax=ax)
    ax.plot(incs[best_peaks], nus[best_peaks], 'or')
    cbar.set_label('Matching function [1]')
    ax.set_xlabel('Inclination perturbation [deg]')
    ax.set_ylabel('Anomaly perturbation [deg]')
    fig.savefig(out_folder / 'peaks_heat_v0.png')
    if close_plot:
        plt.close(fig)

    for rank, best_ind in enumerate(best_peaks):
        diam = diams[best_ind]
        SNR_sim = SNR_sims[best_ind]
        xp = samp_inds[best_ind]
        sn_s = SNR_sim/np.max(SNR_sim)
        sn_s[sn_s <= 0] = np.nan
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True)

        ax2 = axes[0, 0].twinx()
        axes[0, 0].plot(data['t'], diam*1e2, '-b')
        axes[0, 0].set_ylabel('Diameter [cm]', color='b')

        ax2.plot(data['t'][np.abs(t_) < dt], diam[np.abs(t_) < dt]*1e2, '-g')
        ax2.set_ylabel('Diameter around peak SNR [cm]', color='g')

        axes[0, 1].plot(data['t'], np.log10(diam*1e2))
        axes[0, 1].set_ylabel('Diameter [log10(cm)]')
        axes[1, 0].set_xlabel('Time [s]')
        
        axes[1, 0].plot(data['t'], sn_s, label='Estimated')
        axes[1, 0].plot(data['t'], sn_m, label='Measured')
        axes[1, 1].plot(data['t'], 10*np.log10(sn_s), label='Estimated')
        axes[1, 1].plot(data['t'], 10*np.log10(sn_m), label='Measured')
        axes[1, 0].legend()
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 0].set_ylabel('Normalized SNR [1]')
        axes[1, 1].set_ylabel('Normalized SNR [dB]')

        fig.savefig(out_folder / f'peak{rank}_data_v0.png')
        if close_plot:
            plt.close(fig)

    # for j in range(2):
    #     if j == 0:
    #         best_inds = np.argsort(matches)[::-1][:5]
    #         mch = matches.copy()
    #     elif j == 1:
    #         best_inds = np.argsort(matches_v2)[::-1][:5]
    #         mch = matches_v2.copy()

    j = 0
    best_inds = np.argsort(matches)[::-1][:5]
    mch = matches.copy()

    mch_mat = mch.reshape(nus_mat.shape)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(samp_inds, mch)
    ax.plot(samp_inds[best_inds], mch[best_inds], 'or')
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Matching function [1]')
    fig.savefig(out_folder / f'matches_v{j}.png')
    if close_plot:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8))
    pmesh = ax.pcolormesh(incs_mat, nus_mat, mch_mat)
    cbar = fig.colorbar(pmesh, ax=ax)
    ax.plot(incs[best_inds], nus[best_inds], 'or')
    cbar.set_label('Matching function [1]')
    ax.set_xlabel('Inclination perturbation [deg]')
    ax.set_ylabel('Anomaly perturbation [deg]')
    fig.savefig(out_folder / f'matches_heat_v{j}.png')
    if close_plot:
        plt.close(fig)

    fig_heat, ax_heat = plt.subplots(figsize=(12, 8))
    pyant.plotting.gain_heatmap(radar.tx[0].beam, min_elevation=85.0, ax=ax_heat)
    for ind, best_ind in enumerate(best_inds):
        pth = pths[best_ind]
        ax_heat.plot(pth[0, :], pth[1, :], '-r' if ind > 0 else '-g')
    ax_heat.set_xlabel('kx [1]')
    ax_heat.set_ylabel('ky [1]')
    fig_heat.savefig(out_folder / f'matches_traj_gain_v{j}.png')
    if close_plot:
        plt.close(fig_heat)

    print(f'inclination offset {incs[best_inds[0]]} deg')
    print(f'anom offset {nus[best_inds[0]]} deg')
    off_angles = np.zeros_like(mch)
    for ind in range(samps):
        off_angles[ind] = pyant.coordinates.vector_angle(
             pths[ind][:, snr_max], 
             radar.tx[0].beam.pointing, 
             radians=False,
        )
    off_angles = off_angles.reshape(mch_mat.shape)
    offset_angle = pyant.coordinates.vector_angle(
         pths[best_inds[0]][:, snr_max], 
         radar.tx[0].beam.pointing, 
         radians=False,
    )
    print(f'offaxis angle {offset_angle} deg')

    fig, ax = plt.subplots(figsize=(12, 8))
    pmesh = ax.pcolormesh(incs_mat, nus_mat, off_angles)
    cbar = fig.colorbar(pmesh, ax=ax)
    ax.plot(incs[best_inds], nus[best_inds], 'or')
    cbar.set_label('Peak SNR angle from boresight [deg]')
    ax.set_xlabel('Inclination perturbation [deg]')
    ax.set_ylabel('Anomaly perturbation [deg]')
    fig.savefig(out_folder / f'offaxis_heat_v{j}.png')
    if close_plot:
        plt.close(fig)

    for rank, best_ind in enumerate(best_inds):
        diam = diams[best_ind]
        SNR_sim = SNR_sims[best_ind]
        xp = samp_inds[best_ind]
        sn_s = SNR_sim/np.max(SNR_sim)
        sn_s[sn_s <= 0] = np.nan
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True)

        ax2 = axes[0, 0].twinx()
        axes[0, 0].plot(data['t'], diam*1e2, '-b')
        axes[0, 0].set_ylabel('Diameter [cm]', color='b')

        ax2.plot(data['t'][np.abs(t_) < dt], diam[np.abs(t_) < dt]*1e2, '-g')
        ax2.set_ylabel('Diameter around peak SNR [cm]', color='g')

        axes[0, 1].plot(data['t'], np.log10(diam*1e2))
        axes[0, 1].set_ylabel('Diameter [log10(cm)]')
        axes[1, 0].set_xlabel('Time [s]')
        
        axes[1, 0].plot(data['t'], sn_s, label='Estimated')
        axes[1, 0].plot(data['t'], sn_m, label='Measured')
        axes[1, 1].plot(data['t'], 10*np.log10(sn_s), label='Estimated')
        axes[1, 1].plot(data['t'], 10*np.log10(sn_m), label='Measured')
        axes[1, 0].legend()
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 0].set_ylabel('Normalized SNR [1]')
        axes[1, 1].set_ylabel('Normalized SNR [dB]')

        fig.savefig(out_folder / f'match{rank}_data_v{j}.png')
        if close_plot:
            plt.close(fig)

    plt.show()
