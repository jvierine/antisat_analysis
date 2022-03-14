#!/usr/bin/env python


from pathlib import Path
import pickle
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time
import scipy.optimize as sciopt
from tqdm import tqdm
import multiprocessing as mp
import h5py

import sorts
import pyant
import pyorb

from convert_spade_events_to_h5 import get_spade_data
from convert_spade_events_to_h5 import read_spade, date2unix
from convert_spade_events_to_h5 import MIN_SNR

from discosweb_download import main as discosweb_download_main

'''
Example execution:

./beampark_rcs_estimation.py -v estimate -n 6 \
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


def matching_function(data, SNR_sim, low_gain_inds, args, ind):
    # Filter measurnments
    use_data = np.log10(data['SNR'].values)*10 > args.min_snr

    xsn = data['SNR'].values[use_data]
    ysn = SNR_sim[use_data]

    norm_coef = np.sqrt(np.sum(xsn**2))*np.sqrt(np.sum(ysn**2))
    if np.sum(norm_coef) < 0.0001:
        match = 0.0
    else:
        match = np.sum(xsn*ysn)/norm_coef

    # Apply weighting based on amount of data used
    points = np.sum(use_data)
    match *= (points - np.sum(low_gain_inds[use_data]))/points

    return match, ind


def wrapped_matching_function(f_args):
    return matching_function(*f_args)


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
            ind, data, orb, 
            incs, nus, t_, 
            epoch, radar, ecef_st, 
            prop,
        ):
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

    return pth, diam, SNR_sim, G_pth, ind


def wrapped_evaluate_sample(f_args):
    return evaluate_sample(*f_args)


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
    if labels is not None:
        if colors[0] is not None:
            sel_kw['color'] = colors[0]
        if colors[1] is not None:
            not_sel_kw['color'] = colors[1]

    ax.plot(x_sel, y_sel, styles[0], **sel_kw)
    ax.plot(x_not_sel, y_not_sel, styles[1], **not_sel_kw)

    return ax


def plot_match(
            data, mch_ind, diams, 
            t_, dt, use_data, 
            sn_s, sn_m, gain,
            mch_rank, results_folder, radar, 
            pths,
        ):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    ax2 = axes[0, 0].twinx()
    axes[0, 0].plot(data['t'], diams[mch_ind]*1e2, '-b')
    axes[0, 0].set_ylabel('Diameter [cm]', color='b')
    ax2.plot(
        data['t'][np.abs(t_) < dt], 
        diams[mch_ind][np.abs(t_) < dt]*1e2, 
        'xg',
    )
    ax2.set_ylabel('Diameter around peak SNR [cm]', color='g')

    plot_selection(
        ax=axes[0, 1], 
        x=data['t'].values, 
        y=np.log10(diams[mch_ind]*1e2), 
        selector=use_data, 
        styles=['-', '--'], 
        labels=['Used measurements', 'Not used measurements'], 
    )
    axes[0, 1].legend()
    axes[0, 1].set_ylabel('Diameter [log10(cm)]')

    axes[1, 0].set_xlabel('Time [s]')

    plot_selection(
        ax=axes[1, 0], 
        x=data['t'].values, 
        y=sn_s, 
        selector=use_data, 
        styles=['-', '--'], 
        labels=['Estimated', None], 
        colors='b',
    )

    axes[1, 0].plot(
        data['t'][use_data], sn_s[use_data], 
        '-', color='b', label='Estimated')
    axes[1, 0].plot(
        data['t'][use_data], sn_m[use_data], 
        '-', color='r', label='Measured')
    axes[1, 0].plot(
        data['t'][not_use_data], sn_s[not_use_data], 
        '--', color='b')
    axes[1, 0].plot(
        data['t'][not_use_data], sn_m[not_use_data], 
        '--', color='r')
    axes[1, 1].plot(
        data['t'][use_data], 10*np.log10(sn_s)[use_data], 
        '-', color='b', label='Estimated')
    axes[1, 1].plot(
        data['t'][use_data], 10*np.log10(sn_m)[use_data], 
        '-', color='r', label='Measured')
    axes[1, 1].plot(
        data['t'][not_use_data], 10*np.log10(sn_s)[not_use_data], 
        '--', color='b')
    axes[1, 1].plot(
        data['t'][not_use_data], 10*np.log10(sn_m)[not_use_data], 
        '--', color='r')
    axes[0, 2].plot(
        data['t'][use_data], 10*np.log10(data['SNR'])[use_data], 
        '-', color='b')
    axes[0, 2].plot(
        data['t'][not_use_data], 10*np.log10(data['SNR'])[not_use_data], 
        '--', color='b')
    axes[1, 2].plot(
        data['t'][use_data], gain[use_data], 
        '-', color='b')
    axes[1, 2].plot(
        data['t'][not_use_data], gain[not_use_data], 
        '--', color='b')
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
        pths[mch_ind][0, :], 
        pths[mch_ind][1, :], 
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

    for evi, event_name in enumerate(event_names):
        event_indexing = event_data['event_name'] == event_name
        if not np.any(event_indexing):
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
        print(f'Setting pointing to: \n\
            azimuth = {radar.tx[0].beam.azimuth} deg\n\
            elevation = {radar.tx[0].beam.elevation} deg')

        ecef_point = radar.tx[0].pointing_ecef
        ecef_st = radar.tx[0].ecef

        snr_max = np.argmax(data['SNR'].values)
        r_obj = data['r'].values[snr_max]
        r_ecef = ecef_st + r_obj*ecef_point

        epoch = Time(data['t'].values[snr_max], format='unix', scale='utc')
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
            with mp.Pool(args.processes) as pool:
                reses = []
                with tqdm(total=samps) as pbar:
                    f_args_list = [(
                        ind, data, orb, 
                        incs, nus, t_, 
                        epoch, radar, ecef_st, 
                        prop, 
                        ) for ind in range(samps)
                    ]
                    for i, res in enumerate(pool.imap_unordered(
                                wrapped_evaluate_sample, 
                                f_args_list,
                            )):
                        reses.append(res)
                        pbar.update()
            with open(cache_file, 'wb') as fh:
                pickle.dump(reses, fh)

        diams = [None for x in range(samps)]
        SNR_sims = [None for x in range(samps)]
        pths = [None for x in range(samps)]
        gains = [None for x in range(samps)]
        low_gains = [None for x in range(samps)]
        for ind in range(samps):
            pth, diam, SNR_sim, G_pth, ind = reses[ind]

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
        with mp.Pool(args.processes) as pool:
            with tqdm(total=samps) as pbar:
                f_args_list = [(
                    data, SNR_sims[ind], 
                    low_gains[ind], args, 
                    ind, 
                    ) for ind in range(samps)
                ]
                wrapped_matching_function(f_args_list[0])
                for match, ind in pool.imap_unordered(
                            wrapped_matching_function, 
                            f_args_list,
                        ):
                    matches[ind] = match
                    pbar.update()

        matches = np.array(matches)
        matches_mat = matches.reshape(nus_mat.shape)

        use_data = np.log10(data['SNR'].values)*10 > args.min_snr
        not_use_data = np.logical_not(use_data)

        sn_m_max = np.max(data['SNR'].values)
        sn_m = data['SNR'].values/sn_m_max

        best_matches_inds = np.argsort(matches)[::-1]
        best_matches = matches[best_matches_inds]
        best_diams = np.array([diams[ind][snr_max] for ind in best_matches_inds])
        best_gains = np.array([gains[ind][snr_max] for ind in best_matches_inds])
        best_ind = best_matches_inds[0]

        off_angles = np.zeros_like(matches)
        for ind in range(samps):
            off_angles[ind] = pyant.coordinates.vector_angle(
                 pths[ind][:, snr_max], 
                 radar.tx[0].beam.pointing, 
                 radians=False,
            )
        best_offaxis = off_angles[best_matches_inds]
        off_angles = off_angles.reshape(matches_mat.shape)
        offset_angle = pyant.coordinates.vector_angle(
             pths[best_ind][:, snr_max], 
             radar.tx[0].beam.pointing, 
             radians=False,
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
        )

        with open(match_file, 'wb') as fh:
            pickle.dump(summary_data, fh)

        if args.v:
            print('Plotting...')

        fig1, ax1 = plt.subplots(figsize=(12, 8))
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        for pth in pths:
            ax1.plot(pth[0, :], pth[1, :])
            ax2.plot(pth[0, :], pth[1, :], color=[0.5, 0.5, 0.5], alpha=0.1)

        pyant.plotting.gain_heatmap(
            radar.tx[0].beam, 
            min_elevation=85.0, 
            ax=ax2,
        )
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_xlim(ax1.get_xlim())

        fig1.savefig(results_folder / 'sample_trajs.png')
        plt.close(fig1)
        fig2.savefig(results_folder / 'sample_trajs_gain.png')
        plt.close(fig2)

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
                data, mch_ind, diams, 
                t_, dt, use_data, 
                not_use_data, sn_s, sn_m, gains[mch_ind],
                mch_rank, results_folder, radar, 
                pths,
            )

        orig = np.argmin(np.abs(incs) + np.abs(nus))
        plot_match(
            data, orig, diams, 
            t_, dt, use_data, 
            not_use_data, sn_s, sn_m, gains[orig],
            np.argwhere(best_matches_inds == orig).flatten()[0], 
            results_folder, radar, pths,
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

        fig, ax = plt.subplots(figsize=(12, 8))
        pmesh = ax.pcolormesh(incs_mat, nus_mat, off_angles)
        cbar = fig.colorbar(pmesh, ax=ax)
        ax.plot(incs[best_ind], nus[best_ind], 'or')
        cbar.set_label('Peak SNR angle from boresight [deg]')
        ax.set_xlabel('Inclination perturbation [deg]')
        ax.set_ylabel('Anomaly perturbation [deg]')
        fig.savefig(results_folder / 'offaxis_heat.png')
        plt.close(fig)


def main_predict(args):
    radar = getattr(sorts.radars, args.radar)

    G_db_lim = 1
    est_diam = 1e0

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
    
    pop = sorts.population.tle_catalog(tle_pth, cartesian=False)
    pop.unique()

    for select_id in range(len(correlated_measurnment_id)):

        meas_id = correlated_measurnment_id[select_id]
        obj_id = correlated_object_id[select_id]
        event_path = str(correlated_extended_files[select_id])
        event_name = str(correlated_extended_names[select_id])
        event_row = event_data[event_names.index(event_name)]

        # ADD JITTER ITERATOR HERE

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
            azimuth=event_row['AZ'].values[0], 
            elevation=event_row['EL'].values[0],
        )

        dt = (times[meas_id] - obj.epoch).sec
        peak_snr_id = np.argmax(data['SNR'].values)
        t_vec = data['t'].values - data['t'].values[peak_snr_id]

        states_ecef = obj.get_state(dt + t_vec)
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
            diameter=est_diam, 
            bandwidth=radar.tx[0].coh_int_bandwidth,
            rx_noise_temp=radar.rx[0].noise,
            radar_albedo=1.0,
        )

        diam[G_pth_db < G_db_lim] = np.nan
        SNR_sim[G_pth_db < G_db_lim] = 0

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

    # TODO This is hardcoded for now
    radar.tx[0].beam.sph_point(
        azimuth=90, 
        elevation=75,
    )

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

    for ev_id in tqdm(range(num)):
        event_name = event_names[ev_id]

        results_folder = input_pth / event_name
        match_file = results_folder / f'{event_name}_results.pickle'
        predict_file = results_folder / 'correlated_snr_prediction.pickle'
        
        if not match_file.is_file():
            continue

        data = load_spade_extended(data_pths[ev_id])
        snr_max = np.argmax(data['SNR'].values)

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
        '-n', '--processes',
        default=1,
        metavar='NUM',
        type=int,
        help='Number of parallel processes used for calculations.',
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
        default=[-0.008, 0.008],
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
        default=[-0.005, 0.005],
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
