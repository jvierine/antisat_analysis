#!/usr/bin/env python


from pathlib import Path
import pickle
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import scipy.spatial as scisp
from astropy.time import Time, TimeDelta
from tqdm import tqdm
import h5py

import sorts
import pyant
import pyorb

from convert_spade_events_to_h5 import get_spade_data
from convert_spade_events_to_h5 import read_spade, date2unix
from convert_spade_events_to_h5 import MIN_SNR

from discosweb_download import main as discosweb_download_main

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    class COMM_WORLD:
        rank = 0
        size = 1
        
        def Barrier(self):
            pass

    comm = COMM_WORLD()

'''
Example execution:

mpirun -np 6 ./beampark_rcs_estimation.py estimate \
    --matches-plotted 3 --min-gain 10.0 --min-snr 15.0 \
    --inclination-samples 150 --anomaly-samples 100 \
    eiscat_uhf \
    ~/data/spade/beamparks/uhf/2021.11.23/leo_bpark_2.1u_NO@uhf_extended/ \
    ./projects/output/russian_asat/2021.11.23_uhf_rcs

./beampark_rcs_estimation.py predict eiscat_uhf \
    projects/output/russian_asat/{\
    2021_11_23_spacetrack.tle,\
    uhf/2021.11.23.h5,\
    2021.11.23_uhf_correlation.pickle,\
    2021.11.23_uhf_correlation_plots/eiscat_uhf_selected_correlations.npy} \
    ~/data/spade/beamparks/uhf/2021.11.23/leo_bpark_2.1u_NO@uhf_extended/ \
    projects/output/russian_asat/2021.11.23_uhf_rcs

./beampark_rcs_estimation.py discos -k discos \
    projects/output/russian_asat/{\
    2021_11_23_spacetrack.tle,\
    2021.11.23_uhf_correlation.pickle,\
    2021.11.23_uhf_correlation_plots/eiscat_uhf_selected_correlations.npy,\
    2021.11.23_uhf_rcs/}

./beampark_rcs_estimation.py collect -L 0.3 \
    eiscat_uhf \
    projects/output/russian_asat/2021_11_23_spacetrack.tle \
    ~/data/spade/beamparks/uhf/2021.11.23/leo_bpark_2.1u_NO@uhf_extended/ \
    ./projects/output/russian_asat/2021.11.23_uhf_rcs

'''


def matching_function(data, SNR_sim, low_gain_inds, args):
    # Filter measurnments
    sndb = np.log10(data['SNR'].values)*10
    use_data = sndb > args.min_snr
    limit_ratio = args.min_snr/np.max(sndb)

    sn = data['SNR'].values[use_data]
    xsn = np.full(sn.shape, 0, dtype=sn.dtype)
    idx = sn > 0
    xsn[idx] = np.log10(sn[idx])

    sns = SNR_sim[use_data]
    ysn = np.full(sns.shape, 0, dtype=sn.dtype)
    idx = sns > 0
    ysn[idx] = np.log10(sns[idx])

    norm_coef = np.sqrt(np.sum(xsn**2))*np.sqrt(np.sum(ysn**2))
    if np.sum(norm_coef) < 0.0001:
        match = 0.0
    else:
        match = np.sum(xsn*ysn)/norm_coef

    not_used = np.logical_not(use_data)
    if not np.any(not_used):
        low_sn_weight = 1
    elif np.any(SNR_sim > 0):
        ysndb = np.full(SNR_sim.shape, np.nan, dtype=SNR_sim.dtype)
        ysndb[SNR_sim > 0] = np.log10(SNR_sim[SNR_sim > 0])*10
        ysndb /= np.nanmax(ysndb)

        matching_low_snr = np.logical_and(
            not_used,
            ysndb < limit_ratio,
        )
        low_sn_weight = np.sum(matching_low_snr)/np.sum(not_used)
    else:
        low_sn_weight = 0

    # Apply weighting based on amount of data used and low snr portion
    points = np.sum(use_data)
    points_weight = (points - np.sum(low_gain_inds[use_data]))/points
    # match *= points_weight
    # match *= low_sn_weight

    meta = [points_weight, low_sn_weight]

    return match, meta


def load_spade_extended(path):
    names = [f'{x}' for x in range(20)]
    names[0] = 'hit'
    names[1] = 'Y'
    names[2] = 'M'
    names[3] = 'D'
    names[4] = 'h'
    names[5] = 'm'
    names[6] = 's'
    names[7] = 'us'
    
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
    t_strs = []
    for ind in range(len(data)):
        t_strs.append(
            '-'.join([f'{x:02}' for x in [data['Y'].values[ind], data['M'].values[ind], data['D'].values[ind]]]) 
            + 'T' 
            + ':'.join([f'{x:02}' for x in [data['h'].values[ind], data['m'].values[ind], data['s'].values[ind]]]) 
            + f'.{data["us"].values[ind]}'
        )
    unix0 = Time(t_strs, format='isot', scale='utc')
    data['unix'] = unix0.unix
    data['r'] = data['r']*1e3

    return data


def locate_in_beam(x, orb, r_ecef, epoch):
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


def evaluate_sample(
                data, orb, 
                inc, nu, t_, 
                epoch, radar, ecef_st, 
                prop,
            ):
    orb_pert = orb.copy()
    orb_pert.i = orb_pert.i + inc
    orb_pert.anom = orb_pert.anom + nu

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

    return pth, diam, SNR_sim, G_pth


def plot_selection(ax, x, y, selector, styles, labels=None, colors=None):
    not_selector = np.logical_not(selector)
    x_sel = np.copy(x)
    x_sel[not_selector] = np.nan
    x_not_sel = np.copy(x)
    x_not_sel[selector] = np.nan
    y_sel = np.copy(y)
    y_sel[not_selector] = np.nan
    y_not_sel = np.copy(y)
    y_not_sel[selector] = np.nan

    if isinstance(styles, str):
        styles = [styles, styles]
    if isinstance(labels, str):
        labels = [labels, labels]

    sel_kw = {}
    not_sel_kw = {}
    if labels is not None:
        if labels[0] is not None:
            sel_kw['label'] = labels[0]
        if labels[1] is not None:
            not_sel_kw['label'] = labels[1]
    if colors is not None:
        if isinstance(colors, str):
            sel_kw['color'] = colors
            not_sel_kw['color'] = colors
        else:
            if colors[0] is not None:
                sel_kw['color'] = colors[0]
            if colors[1] is not None:
                not_sel_kw['color'] = colors[1]

    ax.plot(x_sel, y_sel, styles[0], **sel_kw)
    ax.plot(x_not_sel, y_not_sel, styles[1], **not_sel_kw)

    return ax


def plot_match(
            data, diams, 
            t_, dt, use_data, 
            sn_s, sn_m, gain,
            mch_rank, results_folder, radar, 
            pth, args,
        ):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)

    styles = ['-', '.-']

    plot_selection(
        ax=axes[0, 0], 
        x=data['t'].values, 
        y=diams*1e2, 
        selector=use_data, 
        styles=styles, 
        labels=['Used measurements', 'Not used measurements'], 
        colors='b',
    )
    axes[0, 0].set_ylabel('Diameter [cm]', color='b')

    plot_selection(
        ax=axes[0, 1], 
        x=data['t'].values, 
        y=np.log10(diams*1e2), 
        selector=use_data, 
        styles=styles, 
        labels=['Used measurements', 'Not used measurements'], 
        colors='b',
    )
    axes[0, 1].set_ylabel('Diameter [log10(cm)]')

    axes[1, 0].set_xlabel('Time [s]')

    plot_selection(
        ax=axes[1, 0], 
        x=data['t'].values, 
        y=sn_s, 
        selector=use_data, 
        styles=styles, 
        labels=['Estimated', None], 
        colors='b',
    )
    plot_selection(
        ax=axes[1, 0], 
        x=data['t'].values, 
        y=sn_m, 
        selector=use_data, 
        styles=styles, 
        labels=['Measured', None], 
        colors='r',
    )

    plot_selection(
        ax=axes[1, 1], 
        x=data['t'].values, 
        y=10*np.log10(sn_s), 
        selector=use_data, 
        styles=styles, 
        labels=['Estimated', None], 
        colors='b',
    )
    plot_selection(
        ax=axes[1, 1], 
        x=data['t'].values, 
        y=10*np.log10(sn_m), 
        selector=use_data, 
        styles=styles, 
        labels=['Measured', None], 
        colors='r',
    )

    axes[0, 2].axhline(args.min_snr, color='g', label='Minimum SNR')
    plot_selection(
        ax=axes[0, 2], 
        x=data['t'].values, 
        y=10*np.log10(data['SNR']), 
        selector=use_data, 
        styles=styles, 
        labels=['Used measurements', 'Not used measurements'], 
        colors='r',
    )
    axes[0, 2].legend()

    axes[1, 2].axhline(args.min_gain, color='g', label='Minimum gain')
    plot_selection(
        ax=axes[1, 2], 
        x=data['t'].values, 
        y=gain, 
        selector=use_data, 
        styles=styles, 
        labels=[None, None], 
        colors='b',
    )
    axes[1, 2].legend()

    axes[1, 0].legend()
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 2].set_xlabel('Time [s]')
    axes[0, 2].set_ylabel('Measured SNR [dB]')
    axes[1, 2].set_ylabel('Estimated Gain [dB]')
    axes[1, 0].set_ylabel('Normalized SNR [1]')
    axes[1, 1].set_ylabel('Normalized SNR [dB]')
    fig.savefig(results_folder / f'match{mch_rank}_data.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8))
    pyant.plotting.gain_heatmap(
        radar.tx[0].beam, 
        min_elevation=85.0, 
        ax=ax,
    )
    ax.plot(
        pth[0, :], 
        pth[1, :], 
        '-r'
    )
    ax.set_xlabel('kx [1]')
    ax.set_ylabel('ky [1]')
    fig.savefig(results_folder / f'match{mch_rank}_traj_gain.png')
    plt.close(fig)


def read_extended_event_data(input_summaries, args):
    _event_datas = []
    for events_file in input_summaries:
        _event_data = get_spade_data([events_file], verbose=args.v)

        _extended_files = []
        with open(events_file, 'r') as fh:
            for ind, line in enumerate(fh):
                if ind < 33 or len(line.strip()) == 0:
                    continue
                ev_name = line[181:].strip()
                _extended_files.append(ev_name)
        _event_data['event_name'] = _extended_files

        # Filter like the h5 generator filters
        disc = np.argwhere(_event_data['RT'].values**2.0 <= MIN_SNR).flatten()
        _event_data.drop(disc, inplace=True)
        _event_datas.append(_event_data)

    event_data = pd.concat(_event_datas)

    dts = []
    for ind in range(len(event_data)):
        t0 = date2unix(
            int(event_data['YYYY'].values[ind]), 
            int(event_data['MM'].values[ind]), 
            int(event_data['DD'].values[ind]), 
            int(event_data['hh'].values[ind]), 
            int(event_data['mm'].values[ind]),
            0,
        )
        t0 += event_data['ss.s'].values[ind]
        dts.append(t0)
    event_data['t'] = dts

    return event_data


def main_estimate(args):

    dt = 1.0

    radar = getattr(sorts.radars, args.radar)

    output_pth = Path(args.output).resolve()
    input_pth = Path(args.input).resolve()

    input_pths = list(input_pth.rglob('*.hlist'))
    input_summaries = list(input_pth.rglob('events.txt'))
    event_names = [x.stem for x in input_pths]
    output_pth.mkdir(exist_ok=True, parents=True)

    prop = sorts.propagator.Kepler(
        settings=dict(
            in_frame='TEME',
            out_frame='ITRS',
        ),
    )

    incs_mat, nus_mat = np.meshgrid(
        np.linspace(
            args.inclination_limits[0], 
            args.inclination_limits[1], 
            args.inclination_samples,
        ),
        np.linspace(
            args.anomaly_limits[0], 
            args.anomaly_limits[1], 
            args.anomaly_samples,
        ),
    )
    incs = incs_mat.reshape((incs_mat.size, ))
    nus = nus_mat.reshape((nus_mat.size, ))
    samps = len(incs)

    event_data = read_extended_event_data(input_summaries, args)

    inds = list(range(len(event_names)))
    step = int(len(event_names)/comm.size + 1)
    all_inds = [
        np.array(inds[i:(i + step)], dtype=np.int64) 
        for i in range(0, len(inds), step)
    ]

    assert all_inds[-1][-1] == inds[-1]

    comm.Barrier()

    pbar = tqdm(
        total=len(all_inds[comm.rank]), 
        position=comm.rank*2, 
        desc=f'Rank-{comm.rank} events',
    )
    worker_inds = all_inds[comm.rank]

    for evi in worker_inds:
        event_name = event_names[evi]
        pbar.set_description(f'Rank-{comm.rank} events [{event_name}]')

        event_indexing = event_data['event_name'] == event_name
        if not np.any(event_indexing):
            if args.v:
                print(f'Warning: "{event_name}" not found in events.txt, skipping')
            continue
        event_row = event_data[event_indexing]
        if args.v:
            print('Processing:')
            print(event_row)

        results_folder = output_pth / event_name
        results_folder.mkdir(exist_ok=True)
        cache_file = results_folder / f'{event_name}_rcs.pickle'
        match_file = results_folder / f'{event_name}_results.pickle'
        data_path = input_pths[evi]

        data = load_spade_extended(data_path)

        radar.tx[0].beam.sph_point(
            azimuth=event_row['AZ'].values[0], 
            elevation=event_row['EL'].values[0],
        )
        if args.v:
            print(f'Setting pointing to: \n\
                azimuth = {radar.tx[0].beam.azimuth} deg\n\
                elevation = {radar.tx[0].beam.elevation} deg')

        ecef_point = radar.tx[0].pointing_ecef
        ecef_st = radar.tx[0].ecef

        snr_max = np.argmax(data['SNR'].values)
        r_obj = data['r'].values[snr_max]
        r_ecef = ecef_st + r_obj*ecef_point

        epoch = Time(data['unix'].values[snr_max], format='unix', scale='utc')
        if args.v:
            print(f'Epoch {epoch.iso}')

        t_ = data['t'].values - data['t'].values[snr_max]

        r_teme = sorts.frames.convert(
            epoch, 
            np.hstack([r_ecef, np.ones_like(r_ecef)]), 
            in_frame='ITRS', 
            out_frame='TEME',
        )
        r_teme = r_teme[:3]

        orb = pyorb.Orbit(
            M0=pyorb.M_earth, m=0, 
            num=1, epoch=epoch, 
            degrees=True,
        )
        orb.update(
            a = np.linalg.norm(r_teme),
            e = 0,
            omega = 0,
            i = 0,
            Omega = 0,
            anom = 0,
        )
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
        v_temes = v_norm*b1
        orb.update(
            x = r_teme[0],
            y = r_teme[1],
            z = r_teme[2],
            vx = v_temes[0],
            vy = v_temes[1],
            vz = v_temes[2],
        )
        if args.v:
            print(orb)

        if cache_file.is_file() and not args.clobber:
            if args.v:
                print('Loading from cache...')
            with open(cache_file, 'rb') as fh:
                reses = pickle.load(fh)
        else:
            if args.v:
                print('Simulating SNR curves...')

            subpbar = tqdm(
                total=samps, 
                position=comm.rank*2 + 1,
                desc=f'Rank-{comm.rank} SNR sampling',
            )
            reses = []
            for ind in range(samps):
                res = evaluate_sample(
                    data, orb, 
                    incs[ind], nus[ind], t_, 
                    epoch, radar, ecef_st, 
                    prop, 
                )
                reses.append(res)
                subpbar.update(1)
            subpbar.close()

            with open(cache_file, 'wb') as fh:
                pickle.dump(reses, fh)

        diams = [None for x in range(samps)]
        SNR_sims = [None for x in range(samps)]
        pths = [None for x in range(samps)]
        gains = [None for x in range(samps)]
        low_gains = [None for x in range(samps)]
        for ind in range(samps):
            pth, diam, SNR_sim, G_pth = reses[ind]

            G_pth_db = 10*np.log10(G_pth)
            gains[ind] = G_pth_db
            low_gain_inds = G_pth_db < args.min_gain
            diam[low_gain_inds] = np.nan
            SNR_sim[low_gain_inds] = 0

            pths[ind] = pth
            diams[ind] = diam
            SNR_sims[ind] = SNR_sim
            low_gains[ind] = low_gain_inds
        
        del reses

        if args.v:
            print('Calculating match values...')

        matches = [None for x in range(samps)]
        metas = [None for x in range(samps)]
        for ind in range(samps):
            matches[ind], metas[ind] = matching_function(
                data, SNR_sims[ind], 
                low_gains[ind], args, 
            )

        matches = np.array(matches)
        matches_mat = matches.reshape(nus_mat.shape)
        metas = np.array(metas)
        points_weight_mat = metas[:, 0].reshape(nus_mat.shape)
        low_sn_weight_mat = metas[:, 1].reshape(nus_mat.shape)

        use_data = np.log10(data['SNR'].values)*10 > args.min_snr

        sn_m_max = np.max(data['SNR'].values)
        sn_m = data['SNR'].values/sn_m_max

        best_matches_inds = np.argsort(matches)[::-1]
        best_matches = matches[best_matches_inds]
        best_diams = np.array([diams[ind][snr_max] for ind in best_matches_inds])
        best_gains = np.array([gains[ind][snr_max] for ind in best_matches_inds])
        best_ind = best_matches_inds[0]

        diams_at_peak = np.array([diams[ind][snr_max] for ind in np.arange(len(diams))])
        diams_at_peak_mat = diams_at_peak.reshape(nus_mat.shape)

        off_angles = np.zeros_like(matches)
        for ind in range(samps):
            off_angles[ind] = pyant.coordinates.vector_angle(
                 pths[ind][:, snr_max], 
                 radar.tx[0].beam.pointing, 
                 radians=False,
            )
        best_offaxis = off_angles[best_matches_inds]
        off_angles_lin = off_angles.copy()
        off_angles = off_angles.reshape(matches_mat.shape)
        offset_angle = pyant.coordinates.vector_angle(
             pths[best_ind][:, snr_max], 
             radar.tx[0].beam.pointing, 
             radians=False,
        )

        kv = np.array([pth[:, snr_max] for pth in pths]).T
        azel = pyant.coordinates.cart_to_sph(kv, radians=True)
        
        prob_remover = matches < 0.5*matches.max()
        matches_prob = matches.copy() - 0.5*matches.max()
        matches_prob[prob_remover] = 0
        matches_prob /= np.sum(matches_prob)
        
        zenith_prob = 5.0 - off_angles_lin
        zenith_prob[prob_remover] = 0
        zenith_prob[zenith_prob < 0] = 0
        zenith_prob /= np.sum(zenith_prob)

        kx_mat = kv[0, :].reshape(matches_mat.shape)
        ky_mat = kv[1, :].reshape(matches_mat.shape)

        dkx_mat = kx_mat[1:, :-1] - kx_mat[:-1, :-1]
        dky_mat = ky_mat[:-1, 1:] - ky_mat[:-1, :-1]
        kA = dkx_mat*dky_mat

        select_mat = np.full(matches_mat.shape, False, dtype=bool)
        select_mat[:-1, :-1] = True
        select = select_mat.reshape(matches.shape)

        # plt.errorbar(
        #     kx_mat[:-1, :-1].flatten(),
        #     ky_mat[:-1, :-1].flatten(),
        #     xerr=dkx_mat.flatten(),
        #     yerr=dky_mat.flatten(),
        #     fmt='.',
        # )
        # plt.show()

        # vor = scisp.Voronoi(kv[:2, :].T)
        # fig = scisp.voronoi_plot_2d(vor)
        # plt.show()

        # just approximate by the sin
        sph_area0 = kA.reshape((np.sum(select), ))/np.sin(azel[1, select])
        sph_area = np.empty_like(matches)
        sph_area[select] = sph_area0
        sph_area[np.logical_not(select)] = np.nan

        diams_prob = matches_prob*zenith_prob/sph_area
        diams_prob[np.isnan(diams_prob)] = 0
        diams_prob[prob_remover] = 0
        diams_prob /= np.sum(diams_prob)

        diam_prob_dist, diam_prob_bins = np.histogram(
            np.log10(diams_at_peak*1e2), 
            bins=np.linspace(0, 5, int(0.25*np.sqrt(len(diams_prob)))), 
            weights=diams_prob,
        )

        if args.v:
            print('== Best match ==')
            print(f'Inclination offset {incs[best_ind]} deg')
            print(f'Anom offset {nus[best_ind]} deg')
            print(f'Offaxis angle {offset_angle} deg')

        summary_data = dict(
            best_matches_inds = best_matches_inds, 
            best_matches = best_matches,
            best_diams = best_diams,
            best_gains = best_gains, 
            best_offaxis = best_offaxis,
            best_path = pths[best_ind],
            best_inc = incs[best_ind],
            best_anom = nus[best_ind],
            matches = matches_mat,
            points_weight = points_weight_mat,
            low_sn_weight = low_sn_weight_mat,
            diams_at_peak = diams_at_peak_mat,
            diam_prob_dist = diam_prob_dist,
            diam_prob_bins = diam_prob_bins,
        )

        with open(match_file, 'wb') as fh:
            pickle.dump(summary_data, fh)

        if args.v:
            print('Plotting...')

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        pmesh = axes[0, 0].pcolormesh(
            incs_mat, 
            nus_mat, 
            matches_prob.reshape(matches_mat.shape),
        )
        cbar = fig.colorbar(pmesh, ax=axes[0, 0])
        cbar.set_label('Probability from matching  [1]')
        axes[0, 0].set_ylabel('Anomaly perturbation [deg]')

        pmesh = axes[1, 0].pcolormesh(
            incs_mat, 
            nus_mat, 
            zenith_prob.reshape(matches_mat.shape),
        )
        cbar = fig.colorbar(pmesh, ax=axes[1, 0])
        cbar.set_label('Probability from off-axis angle  [1]')
        axes[1, 0].set_xlabel('Inclination perturbation [deg]')
        axes[1, 0].set_ylabel('Anomaly perturbation [deg]')

        pmesh = axes[0, 1].pcolormesh(
            incs_mat, 
            nus_mat, 
            diams_prob.reshape(matches_mat.shape),
        )
        cbar = fig.colorbar(pmesh, ax=axes[0, 1])
        cbar.set_label('Probability for diameter [1]')

        pmesh = axes[1, 1].pcolormesh(
            incs_mat, 
            nus_mat, 
            sph_area.reshape(matches_mat.shape),
        )
        cbar = fig.colorbar(pmesh, ax=axes[1, 1])
        cbar.set_label('Sample angular area [sr]')
        axes[1, 1].set_xlabel('Inclination perturbation [deg]')

        fig.savefig(results_folder / 'diam_prob_funcs.png')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 8))
        pmesh = ax.pcolormesh(
            incs_mat, 
            nus_mat, 
            np.log10(1e2*diams_at_peak).reshape(matches_mat.shape),
        )
        cbar = fig.colorbar(pmesh, ax=ax)
        cbar.set_label('Diameter at peak SNR [log10(cm)]')
        ax.set_xlabel('Inclination perturbation [deg]')
        ax.set_ylabel('Anomaly perturbation [deg]')
        fig.savefig(results_folder / 'diam_peak_heat.png')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(
            (diam_prob_bins[:-1] + diam_prob_bins[1:])*0.5,
            diam_prob_dist, 
            align='center', 
            width=np.diff(diam_prob_bins),
        )
        ax.set_xlabel('Diameter at peak SNR [log10(cm)]')
        ax.set_ylabel('Probability (from matching function) [1]')
        fig.savefig(results_folder / 'diam_prob_dist.png')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(np.log10(best_diams*1e2), best_matches, '.', label='Samples')
        ax.plot(
            np.log10(best_diams[0]*1e2), 
            best_matches[0], 
            'or', 
            label='Best match',
        )
        ax.set_xlabel('Diameter at peak SNR [log10(cm)]')
        ax.set_ylabel('Matching function [1]')
        fig.savefig(results_folder / 'diam_match_dist.png')
        plt.close(fig)

        mch_rank = 0
        for mch_ind in best_matches_inds[:args.matches_plotted]:
            mch_rank += 1

            sn_s = SNR_sims[mch_ind]/np.max(SNR_sims[mch_ind])
            sn_s[sn_s <= 0] = np.nan
            
            plot_match(
                data, diams[mch_ind], 
                t_, dt, use_data, 
                sn_s, sn_m, gains[mch_ind],
                mch_rank, results_folder, radar, 
                pths[mch_ind], args,
            )

        orig = np.argmin(off_angles.flatten())
        plot_match(
            data, diams[orig], 
            t_, dt, use_data, 
            sn_s, sn_m, gains[orig],
            'boresight', 
            results_folder, radar, 
            pths[orig], args,
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        pmesh = ax.pcolormesh(incs_mat, nus_mat, matches_mat)
        cbar = fig.colorbar(pmesh, ax=ax)
        ax.plot(incs[best_ind], nus[best_ind], 'or')
        cbar.set_label('Matching function [1]')
        ax.set_xlabel('Inclination perturbation [deg]')
        ax.set_ylabel('Anomaly perturbation [deg]')
        fig.savefig(results_folder / 'match_heat.png')
        plt.close(fig)

        for weight, name, fname in zip(
                        [points_weight_mat, low_sn_weight_mat],
                        ['Used data points weight', 'Low SNR points weight'],
                        ['points_weight', 'snr_weight'],
                    ):
            fig, ax = plt.subplots(figsize=(12, 8))
            pmesh = ax.pcolormesh(incs_mat, nus_mat, weight)
            cbar = fig.colorbar(pmesh, ax=ax)
            ax.plot(incs[best_ind], nus[best_ind], 'or')
            cbar.set_label(f'{name} [1]')
            ax.set_xlabel('Inclination perturbation [deg]')
            ax.set_ylabel('Anomaly perturbation [deg]')
            fig.savefig(results_folder / f'{fname}_heat.png')
            plt.close(fig)

        wmat = points_weight_mat*low_sn_weight_mat
        wmat[wmat == 0] = np.nan

        fig, ax = plt.subplots(figsize=(12, 8))
        pmesh = ax.pcolormesh(incs_mat, nus_mat, matches_mat/wmat)
        cbar = fig.colorbar(pmesh, ax=ax)
        cbar.set_label('Non-weighted matching function [1]')
        ax.set_xlabel('Inclination perturbation [deg]')
        ax.set_ylabel('Anomaly perturbation [deg]')
        fig.savefig(results_folder / 'match_noweight_heat.png')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 8))
        pmesh = ax.pcolormesh(incs_mat, nus_mat, off_angles)
        cbar = fig.colorbar(pmesh, ax=ax)
        ax.plot(incs[best_ind], nus[best_ind], 'or')
        cbar.set_label('Peak SNR angle from boresight [deg]')
        ax.set_xlabel('Inclination perturbation [deg]')
        ax.set_ylabel('Anomaly perturbation [deg]')
        fig.savefig(results_folder / 'offaxis_heat.png')
        plt.close(fig)

        cmap = cm.get_cmap('plasma')
        vmin, vmax = np.min(matches), np.max(matches)
        sm = plt.cm.ScalarMappable(
            cmap=cmap, 
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
        )
        
        cmap3 = cm.get_cmap('hot_r')
        sm3 = plt.cm.ScalarMappable(
            cmap=cmap3, 
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
        )

        fig1, ax1 = plt.subplots(figsize=(12, 8))
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        for ind, pth in enumerate(pths):
            cval = (matches[ind] - vmin)/vmax
            ax1.plot(pth[0, :], pth[1, :], color=cmap(cval), alpha=0.1)
            ax2.plot(pth[0, :], pth[1, :], color=[0.5, 0.5, 0.5], alpha=0.1)
            ax3.plot(pth[0, snr_max], pth[1, snr_max], '.', color=cmap3(cval))
        cbar = fig.colorbar(sm, ax=ax1)
        cbar.set_label('Matching function [1]')
        cbar = fig.colorbar(sm3, ax=ax3)
        cbar.set_label('Matching function [1]')
        pyant.plotting.gain_heatmap(
            radar.tx[0].beam, 
            min_elevation=85.0, 
            ax=ax2,
        )
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_xlim(ax1.get_xlim())

        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel('kx [1]')
            ax.set_ylabel('ky [1]')

        fig1.savefig(results_folder / 'sample_trajs.png')
        plt.close(fig1)
        fig2.savefig(results_folder / 'sample_trajs_gain.png')
        plt.close(fig2)
        fig3.savefig(results_folder / 'sample_peak_match.png')
        plt.close(fig3)

        pbar.update(1)
    pbar.close()


def generate_prediction(data, t, obj, radar, args):

    states_ecef = obj.get_state(t)
    ecef_r = states_ecef[:3, :] - radar.tx[0].ecef[:, None]

    local_pos = sorts.frames.ecef_to_enu(
        radar.tx[0].lat, 
        radar.tx[0].lon, 
        radar.tx[0].alt, 
        ecef_r,
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

    low_gain_inds = G_pth_db < args.min_gain
    diam[low_gain_inds] = np.nan
    SNR_sim[low_gain_inds] = 0

    return SNR_sim, G_pth, diam, low_gain_inds, ecef_r, pth


def main_predict(args):
    radar = getattr(sorts.radars, args.radar)

    t_jitter = np.arange(
        -args.jitter_width, 
        args.jitter_width, 
        0.1, 
        dtype=np.float64,
    )

    tle_pth = Path(args.catalog).resolve()
    output_pth = Path(args.output).resolve()
    input_pth = Path(args.input).resolve()
    corr_events = Path(args.correlation_events).resolve()
    corr_pth = Path(args.correlation_data).resolve()
    corr_select_pth = Path(args.correlation_results).resolve()

    input_pths = list(input_pth.rglob('*.hlist'))
    input_summaries = list(input_pth.rglob('events.txt'))
    event_names = [x.stem for x in input_pths]
    output_pth.mkdir(exist_ok=True, parents=True)

    correlation_select = np.load(corr_select_pth)

    with h5py.File(corr_pth, 'r') as ds:
        indecies = ds['matched_object_index'][()][0, :]
        measurnment_id = ds['observation_index'][()]
        indecies = indecies[measurnment_id]
        measurnment_id = np.arange(len(measurnment_id))

    event_data = read_extended_event_data(input_summaries, args)

    _input_pths = []
    for event_name in event_data['event_name'].values:
        event_name = str(event_name)
        if event_name not in event_names:
            continue
        event_indexing = event_names.index(event_name)
        _input_pths.append(input_pths[event_indexing])
    event_data['path'] = _input_pths
    event_data.sort_values(by=['t'], inplace=True)

    event_names = event_data['event_name'].values.copy()
    event_paths = event_data['path'].values.copy()

    _ret = read_spade(input_pth, output_h5=None)
    t_extended = _ret[0]
    t_extended = t_extended[np.argsort(t_extended)]
    times_extended = Time(t_extended, format='unix', scale='utc')

    # select only from sub_folder in case correlation is for more data
    t0 = min(times_extended)
    t1 = max(times_extended)

    print('Selecting')
    print(t0.iso)
    print(t1.iso)

    with h5py.File(corr_events, 'r') as h_det:
        r = h_det['r'][()]*1e3  # km -> m, one way
        t = h_det['t'][()]  # Unix seconds
        v = h_det['v'][()]*1e3  # km/s -> m/s
        inds = np.argsort(t)
        t = t[inds]
        r = r[inds]
        v = v[inds]

        times = Time(t, format='unix', scale='utc')
        epoch = times[0]
        t = (times - epoch).sec

    sub_select_inds = np.logical_and(
        times >= t0, 
        times <= t1,
    )

    sub_measurnment_id = measurnment_id[sub_select_inds]
    sub_correlation_select = correlation_select[sub_select_inds]
    sub_indecies = indecies.flatten()[sub_select_inds]

    correlated_measurnment_id = sub_measurnment_id[sub_correlation_select]
    correlated_object_id = sub_indecies[sub_correlation_select]

    correlated_extended_files = event_paths[sub_correlation_select]
    correlated_extended_names = event_names[sub_correlation_select]

    event_names_lst = event_names.tolist()
    
    pop = sorts.population.tle_catalog(tle_pth, cartesian=False)
    pop.unique()

    for select_id in range(len(correlated_measurnment_id)):

        meas_id = correlated_measurnment_id[select_id]
        obj_id = correlated_object_id[select_id]
        event_path = str(correlated_extended_files[select_id])
        event_name = str(correlated_extended_names[select_id])
        event_row = event_data.iloc[event_names_lst.index(event_name)]

        obj = pop.get_object(obj_id)
        print('Correlated TLE object for :')
        print(f' - measurement {meas_id}')
        print(f' - object {obj_id}')
        print(f' - event {event_name}')
        print(obj)
        print(event_path)
        obj.out_frame = 'ITRS'

        results_folder = output_pth / event_name
        results_folder.mkdir(exist_ok=True)

        data = load_spade_extended(event_path)

        radar.tx[0].beam.sph_point(
            azimuth=event_row['AZ'], 
            elevation=event_row['EL'],
        )

        dt = (times[meas_id] - obj.epoch).sec
        peak_snr_id = np.argmax(data['SNR'].values)
        t_vec = data['t'].values - data['t'].values[peak_snr_id]

        matches = np.empty_like(t_jitter)
        pdatas = [None]*len(t_jitter)

        for tind in range(len(t_jitter)):
            _t = t_vec + dt + t_jitter[tind]
            pdatas[tind] = generate_prediction(data, _t, obj, radar, args)
            SNR_sim, G_pth, diam, low_gain_inds, ecef_r, pth = pdatas[tind]

            matches[tind], meta = matching_function(
                data, SNR_sim, 
                low_gain_inds, args, 
            )
        best_match = np.argmax(matches)
        SNR_sim, G_pth, diam, low_gain_inds, ecef_r, pth = pdatas[best_match]
        print(f't-jitter={t_jitter[best_match]}, match={matches[best_match]}')

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(data['t'], np.linalg.norm(ecef_r, axis=0))
        ax.plot(data['t'], data['r'])
        ax.plot([data['t'].values[peak_snr_id]], [r[meas_id]], 'or')

        fig.savefig(results_folder / 'correlated_pass_range_match.png')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(pth[0, :], pth[1, :], '-w')
        pyant.plotting.gain_heatmap(radar.tx[0].beam, min_elevation=85.0, ax=ax)

        fig.savefig(results_folder / 'correlated_pass_pth_gain.png')
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True)
        axes[0, 0].plot(data['t'], diam*1e2)
        axes[0, 0].set_ylabel('Diameter [cm]')
        axes[0, 1].plot(data['t'], np.log10(diam*1e2))
        axes[0, 1].set_ylabel('Diameter [log10(cm)]')
        axes[1, 0].set_xlabel('Time [s]')

        axes[1, 0].plot(data['t'], SNR_sim/np.max(SNR_sim), label='Estimated')
        axes[1, 0].plot(
            data['t'], 
            data['SNR'].values/np.max(data['SNR'].values), 
            'x', label='Measured',
        )
        axes[1, 1].plot(
            data['t'], 
            10*np.log10(SNR_sim/np.max(SNR_sim)), 
            label='Estimated',
        )
        axes[1, 1].plot(
            data['t'], 
            10*np.log10(data['SNR'].values/np.max(data['SNR'].values)), 
            'x', 
            label='Measured',
        )
        axes[1, 0].legend()
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 0].set_ylabel('Normalized SNR [1]')
        axes[1, 1].set_ylabel('Normalized SNR [dB]')
        fig.savefig(results_folder / 'correlated_pass_snr_match.png')
        plt.close(fig)

        offset_angle = pyant.coordinates.vector_angle(
             pth[:, peak_snr_id], 
             radar.tx[0].beam.pointing, 
             radians=False,
        )
        print(f'offaxis angle {offset_angle} deg')

        summary_data = dict(
            SNR_sim = SNR_sim, 
            diam_sim = diam,
            gain_sim = G_pth,
            pth_sim = pth, 
            measurement_id = meas_id,
            object_id = obj_id,
            offset_angle = offset_angle,
        )
        with open(results_folder / 'correlated_snr_prediction.pickle', 'wb') as fh:
            pickle.dump(summary_data, fh)


def main_discos(args):

    tle_pth = Path(args.catalog).resolve()
    corr_pth = Path(args.correlation_data).resolve()
    corr_select_pth = Path(args.correlation_results).resolve()
    
    output_pth = Path(args.output).resolve()
    output_pth.mkdir(exist_ok=True, parents=True)
    output_discos = output_pth / 'discos_entires.json'
    output_data_file = output_pth / 'discos_object_sizes.npz'

    correlation_select = np.load(corr_select_pth)

    with h5py.File(corr_pth, 'r') as ds:
        indecies = ds['matched_object_index'][()][0, :]
        measurnment_id = ds['observation_index'][()]
        indecies = indecies[measurnment_id]
        measurnment_id = np.arange(len(measurnment_id))

    indecies = indecies[correlation_select]
    measurnment_id = measurnment_id[correlation_select]

    print('Loading population...')
    pop = sorts.population.tle_catalog(tle_pth, cartesian=False)
    pop.unique()

    oids = pop['oid'][indecies].tolist()

    if not output_discos.is_file() or args.clobber:
        print(f'Fetching data on {len(oids)} objects...')
        discos_args = [
            '--secret-tool-key', args.secret_tool_key,
            '--output', str(output_discos),
        ]
        discos_args += [str(x) for x in oids]
        print(' '.join(discos_args))
        discosweb_download_main(discos_args)

    print('Loading discos data...')
    assert output_discos.is_file(), f'Something went wrong, "{output_discos}" does not exist'
    with open(output_discos, 'r') as fh:
        discos_data = json.load(fh)

    output_data = {
        key: np.empty((len(discos_data), ), dtype=np.float64)
        for key in ['measurnment_id', 'oid', 'min_area', 'avg_area', 'max_area', 'diam']
    }
    for ind, obj in enumerate(discos_data):
        output_data['oid'][ind] = obj['attributes']['satno']
        mid = oids.index(int(obj['attributes']['satno']))
        output_data['measurnment_id'][ind] = measurnment_id[mid]
        
        for data_key, discos_key in zip(
                    ['min_area', 'avg_area', 'max_area', 'diam'], 
                    ['xSectMin', 'xSectAvg', 'xSectMax', 'diameter'], 
                ):
            val = obj['attributes'][discos_key]
            output_data[data_key][ind] = val if val is not None else np.nan

    print('Writing object size data...')
    np.savez(output_data_file, **output_data)


def area_to_diam(A):
    return np.sqrt(A/np.pi)*2


def main_collect(args):
    radar = getattr(sorts.radars, args.radar)

    data_pth = Path(args.data).resolve()
    input_pth = Path(args.input).resolve()
    tle_pth = Path(args.catalog).resolve()

    data_pths = list(data_pth.rglob('*.hlist'))
    event_names = [x.stem for x in data_pths]

    discos_file = input_pth / 'discos_object_sizes.npz'
    discos_data = np.load(discos_file)

    collected_file = input_pth / 'collected_results.pickle'

    pop = sorts.population.tle_catalog(tle_pth, cartesian=False)
    pop.unique()

    num = len(event_names)

    _init_array = np.full((num,), np.nan, dtype=np.float64)
    summary_data = {
        'measurnment_id': _init_array.copy(),
        't_unix_peak': _init_array.copy(),
        'event_name': event_names,
        'correlated': np.full((num,), False, dtype=bool),
        'object_id': _init_array.copy(),
        'norad_id': _init_array.copy(),
        'SNR': _init_array.copy(),
        'range': _init_array.copy(),
        'doppler': _init_array.copy(),
        'estimated_offset_angle': _init_array.copy(),
        'estimated_diam': _init_array.copy(),
        'estimated_min_diam': _init_array.copy(),
        'estimated_max_diam': _init_array.copy(),
        'estimated_diams': [None]*num,
        'estimated_gain': _init_array.copy(),
        'estimated_diam_prob': [None]*num,
        'estimated_diam_bins': [None]*num,
        'proxy_d_inc': _init_array.copy(),
        'proxy_d_anom': _init_array.copy(),
        'predicted_offset_angle': _init_array.copy(),
        'predicted_diam': _init_array.copy(),
        'predicted_gain': _init_array.copy(),
        'discos_diam': _init_array.copy(),
        'discos_avg_diam': _init_array.copy(),
        'discos_min_diam': _init_array.copy(),
        'discos_max_diam': _init_array.copy(),
    }

    input_summaries = list(data_pth.rglob('events.txt'))
    event_data = read_extended_event_data(input_summaries, args)

    size_dist = None
    size_bins = None

    for ev_id in tqdm(range(num)):
        event_name = event_names[ev_id]
        event_indexing = event_data['event_name'] == event_name

        results_folder = input_pth / event_name
        match_file = results_folder / f'{event_name}_results.pickle'
        predict_file = results_folder / 'correlated_snr_prediction.pickle'
        
        if not match_file.is_file():
            continue

        data = load_spade_extended(data_pths[ev_id])
        snr_max = np.argmax(data['SNR'].values)

        event_row = event_data[event_indexing]
        radar.tx[0].beam.sph_point(
            azimuth=event_row['AZ'].values[0], 
            elevation=event_row['EL'].values[0],
        )

        with open(match_file, 'rb') as fh:
            match_data = pickle.load(fh)
        if predict_file.is_file():
            with open(predict_file, 'rb') as fh:
                predict_data = pickle.load(fh)
        else:
            predict_data = None

        summary_data['SNR'][ev_id] = data['SNR'].values[snr_max]
        summary_data['range'][ev_id] = data['r'].values[snr_max]*1e3
        summary_data['doppler'][ev_id] = data['v'].values[snr_max]*1e3
        summary_data['t_unix_peak'][ev_id] = data['unix'].values[snr_max]

        summary_data['proxy_d_inc'][ev_id] = match_data['best_inc']
        summary_data['proxy_d_anom'][ev_id] = match_data['best_anom']
        
        summary_data['estimated_offset_angle'][ev_id] = match_data['best_offaxis'][0]
        summary_data['estimated_diam'][ev_id] = match_data['best_diams'][0]
        summary_data['estimated_gain'][ev_id] = match_data['best_gains'][0]

        match_limit = np.percentile(match_data['best_matches'], 100 - args.match_limit_percentile)
        probable_matches = match_data['best_matches'] > match_limit
        probable_diams = match_data['best_diams'][probable_matches]

        summary_data['estimated_diams'][ev_id] = probable_diams
        summary_data['estimated_min_diam'][ev_id] = np.min(probable_diams)
        summary_data['estimated_max_diam'][ev_id] = np.max(probable_diams)

        if size_dist is None:
            size_dist = np.zeros_like(match_data['diam_prob_dist'])
            size_bins = np.copy(match_data['diam_prob_bins'])
        size_dist += match_data['diam_prob_dist']

        summary_data['estimated_diam_prob'][ev_id] = match_data['diam_prob_dist']
        summary_data['estimated_diam_bins'][ev_id] = match_data['diam_prob_bins']

        if predict_data is not None:

            summary_data['correlated'][ev_id] = True
            summary_data['measurnment_id'][ev_id] = predict_data['measurement_id']
            summary_data['object_id'][ev_id] = predict_data['object_id']
            summary_data['norad_id'][ev_id] = pop['oid'][predict_data['object_id']]
            
            summary_data['predicted_offset_angle'][ev_id] = predict_data['offset_angle']
            summary_data['predicted_diam'][ev_id] = predict_data['diam_sim'][snr_max]
            summary_data['predicted_gain'][ev_id] = predict_data['gain_sim'][snr_max]
            
            discos_row = np.argwhere(
                discos_data['measurnment_id'] == predict_data['measurement_id']
            )
            discos_row = discos_row.flatten()
            if len(discos_row) > 0:
                discos_row = discos_row[0]

                summary_data['discos_diam'][ev_id] = area_to_diam(discos_data['diam'][discos_row])
                summary_data['discos_avg_diam'][ev_id] = area_to_diam(discos_data['avg_area'][discos_row])
                summary_data['discos_min_diam'][ev_id] = area_to_diam(discos_data['min_area'][discos_row])
                summary_data['discos_max_diam'][ev_id] = area_to_diam(discos_data['max_area'][discos_row])

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(
                match_data['best_path'][0, :], 
                match_data['best_path'][1, :], 
                '-r', 
                label='estimated path',
            )
            ax.plot(
                predict_data['pth_sim'][0, :], 
                predict_data['pth_sim'][1, :], 
                '-g', 
                label='predicted path',
            )
            ax.legend()
            pyant.plotting.gain_heatmap(radar.tx[0].beam, min_elevation=85.0, ax=ax)

            fig.savefig(results_folder / 'compare_pass_pth_gain.png')
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(
        (size_bins[:-1] + size_bins[1:])*0.5,
        size_dist, 
        align='center', 
        width=np.diff(size_bins),
    )
    ax.set_xlabel('Diameter at peak SNR [log10(cm)]')
    ax.set_ylabel('Frequency (from matching function probability) [1]')
    fig.savefig(input_pth / 'diam_prob_dist.png')
    plt.close(fig)

    with open(collected_file, 'wb') as fh:
        pickle.dump(summary_data, fh)


def build_predict_parser(parser):
    parser.add_argument('radar', type=str, help='The observing radar system')
    parser.add_argument('catalog', type=str, help='TLE catalog path')
    parser.add_argument('correlation_events', type=str, help='Path to \
        correlated events')
    parser.add_argument('correlation_data', type=str, help='Path to \
        correlation data')
    parser.add_argument('correlation_results', type=str, help='Path to \
        correlation results')
    parser.add_argument('input', type=str, help='Observation data folder, \
        expects folder with *.hlist files and an events.txt summary file.')
    parser.add_argument('output', type=str, help='Results output location')
    parser.add_argument(
        '--min-gain',
        default=10.0,
        metavar='MIN',
        type=float, 
        help='The minimum amount of dB one way gain at which to evalute \
        SNR and RCS',
    )
    parser.add_argument(
        '--min-snr',
        default=5.0,
        metavar='MIN',
        type=float, 
        help='The minimum amount of measured SNR dB at which to evalute \
        SNR and RCS',
    )
    parser.add_argument(
        '--jitter-width',
        default=5.0,
        metavar='JITTER',
        type=float, 
        help='The number of seconds to jitter back and forth',
    )
    return parser


def build_collect_parser(parser):
    parser.add_argument('radar', type=str, help='The observing radar system')
    parser.add_argument('catalog', type=str, help='TLE catalog path')
    parser.add_argument('data', type=str, help='Observation data folder, \
        expects folder with *.hlist files and an events.txt summary file.')
    parser.add_argument('input', type=str, help='Results output location to \
        collect results from, summary is also saved here')
    parser.add_argument(
        '-L', '--match-limit-percentile',
        default=0.3,
        metavar='LIMIT',
        type=float, 
        help='Percentile of matches at which to consider matches as probable',
    )
    return parser


def build_estimate_parser(parser):
    parser.add_argument('radar', type=str, help='The observing radar system')
    parser.add_argument('input', type=str, help='Observation data folder, \
        expects folder with *.hlist files and an events.txt summary file.')
    parser.add_argument('output', type=str, help='Results output location')

    parser.add_argument(
        '--matches-plotted',
        default=1,
        metavar='NUM',
        type=int, 
        help='The number of best matches to plot.',
    )
    parser.add_argument(
        '-c', '--clobber', 
        action='store_true', 
        help='Override output location if it exists',
    )
    parser.add_argument(
        '--min-gain',
        default=10.0,
        metavar='MIN',
        type=float, 
        help='The minimum amount of dB one way gain at which to evalute \
        SNR and RCS',
    )
    parser.add_argument(
        '--min-snr',
        default=5.0,
        metavar='MIN',
        type=float, 
        help='The minimum amount of measured SNR dB at which to evalute \
        SNR and RCS',
    )
    parser.add_argument(
        '--inclination-limits',
        nargs=2, 
        default=[-0.46, 0.46],
        metavar=('MIN', 'MAX'),
        type=float, 
        help='The width of the inclination perturbation of the proxy orbit, \
        determines the off-track width of the gain sampling',
    )
    parser.add_argument(
        '--inclination-samples',
        default=200,
        metavar='NUM',
        type=int, 
        help='Number of samples to use for inclination perturbations.',
    )
    parser.add_argument(
        '--anomaly-limits',
        nargs=2, 
        default=[-0.29, 0.29],
        metavar=('MIN', 'MAX'),
        type=float, 
        help='The width of the anomaly perturbation of the proxy orbit, \
        determines the off-track width of the gain sampling',
    )
    parser.add_argument(
        '--anomaly-samples',
        default=50,
        metavar='NUM',
        type=int, 
        help='Number of samples to use for anomaly perturbations.',
    )

    return parser


def build_discos_parser(parser):
    parser.add_argument('catalog', type=str, help='TLE catalog path')
    parser.add_argument('correlation_data', type=str, help='Path to \
        correlation data')
    parser.add_argument('correlation_results', type=str, help='Path to \
        correlation results')
    parser.add_argument('output', type=str, help='Results output location')
    parser.add_argument(
        '--secret-tool-key', '-k', type=str, metavar='KEY',
        help='Attribute value (key) to fetch secret from',
    )
    parser.add_argument(
        '-c', '--clobber', 
        action='store_true', 
        help='Override output location if it exists',
    )


def build_parser():

    parser = argparse.ArgumentParser(description='Analyze detailed SNR curves \
        to determine RCS and statistics')
    parser.add_argument(
        '-v',
        action='store_true', 
        help='Verbose output',
    )

    subparsers = parser.add_subparsers(
        help='Available command line interfaces', 
        dest='command',
    )
    subparsers.required = True

    estimate_parser = subparsers.add_parser('estimate', help='Estimate RCS \
        values for target using SNR time series for each event.')
    estimate_parser = build_estimate_parser(estimate_parser)

    predict_parser = subparsers.add_parser('predict', help='Estimate RCS \
        values for target using correlation and TLE.')
    predict_parser = build_predict_parser(predict_parser)

    discos_parser = subparsers.add_parser('discos', help='Fetch size \
        data for all correlated objects from DISCOSweb')
    discos_parser = build_discos_parser(discos_parser)

    collect_parser = subparsers.add_parser('collect', help='Collect RCS \
        statistics from "estimate" and "predict"')
    collect_parser = build_collect_parser(collect_parser)

    return parser


def main(input_args=None):
    parser = build_parser()

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    print(f'Entering sub-system: {args.command.upper()}')

    if args.command == 'estimate':
        main_estimate(args)
    elif args.command == 'predict':
        main_predict(args)
    elif args.command == 'collect':
        main_collect(args)
    elif args.command == 'discos':
        main_discos(args)


if __name__ == '__main__':
    main()
