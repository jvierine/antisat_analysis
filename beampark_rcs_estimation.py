#!/usr/bin/env python


from pathlib import Path
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time, TimeDelta
import scipy.optimize as sciopt
from tqdm import tqdm
import multiprocessing as mp

import sorts
import pyant
import pyorb

from convert_spade_events_to_h5 import get_spade_data
from convert_spade_events_to_h5 import MIN_SNR

'''
Example execution:

./beampark_rcs_estimation.py estimate -v -n 6 \
    eiscat_uhf \
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


def plot_match(
            data, mch_ind, diams, 
            t_, dt, use_data, 
            not_use_data, sn_s, sn_m, 
            mch_rank, results_folder, radar, 
            pths,
        ):
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True)
    ax2 = axes[0, 0].twinx()
    axes[0, 0].plot(data['t'], diams[mch_ind]*1e2, '-b')
    axes[0, 0].set_ylabel('Diameter [cm]', color='b')
    ax2.plot(
        data['t'][np.abs(t_) < dt], 
        diams[mch_ind][np.abs(t_) < dt]*1e2, 
        '-g',
    )
    ax2.set_ylabel('Diameter around peak SNR [cm]', color='g')

    axes[0, 1].plot(
        data['t'][use_data], 
        np.log10(diams[mch_ind][use_data]*1e2),
        label='Used measurements'
    )
    axes[0, 1].plot(
        data['t'][not_use_data], 
        np.log10(diams[mch_ind][not_use_data]*1e2),
        label='Not used measurements'
    )
    axes[0, 1].legend()
    axes[0, 1].set_ylabel('Diameter [log10(cm)]')

    axes[1, 0].set_xlabel('Time [s]')
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
    axes[1, 0].legend()
    axes[1, 1].set_xlabel('Time [s]')
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
        discard = np.argwhere(_event_data['RT'].values**2.0 <= MIN_SNR).flatten()
        _event_data.drop(discard)
        _event_datas.append(_event_data)

    event_data = pd.concat(_event_datas)
    del _event_datas
    del _event_data

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

        mock_epoch = Time('J2000')
        t_ = data['t'].values - data['t'].values[snr_max]

        orb = pyorb.Orbit(
            M0=pyorb.M_earth, m=0, 
            num=1, epoch=mock_epoch, 
            radians=False,
        )
        orb.a = np.linalg.norm(r_ecef)
        orb.e = 0
        orb.omega = 0

        it = 1
        x0 = np.array([51, 79, 158])
        min_res = sciopt.minimize(
            locate_in_beam, 
            x0, 
            args=(orb, r_ecef, mock_epoch), 
            method='Nelder-Mead',
        )
        if args.v:
            print(f'{it}: [{min_res.fun}] {x0}')
        while it < 10 and min_res.fun > 1e3:
            it += 1
            x0 = np.random.rand(3)
            x0[0] *= 90
            x0[1:] *= 360
            min_res0 = sciopt.minimize(
                locate_in_beam, x0, 
                args=(orb, r_ecef, mock_epoch), 
                method='Nelder-Mead',
            )

            if args.v:
                print(f'{it}: [{min_res0.fun}] {x0} -> {min_res0.x}')
            if min_res0.fun < min_res.fun:
                min_res = min_res0
            if min_res.fun < 1e3:
                break
        if args.v:
            print(min_res)

        orb.i = min_res.x[0]
        orb.Omega = min_res.x[1]
        orb.anom = min_res.x[2]
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
                        mock_epoch, radar, ecef_st, 
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
                not_use_data, sn_s, sn_m, 
                mch_rank, results_folder, radar, 
                pths,
            )

        orig = np.argmin(np.abs(incs) + np.abs(nus))
        plot_match(
            data, orig, diams, 
            t_, dt, use_data, 
            not_use_data, sn_s, sn_m, 
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

    tle_pth = None if len(args.catalog) == 0 else Path(args.catalog).resolve()
    output_pth = Path(args.output).resolve()
    input_pth = Path(args.input).resolve()
    corr_pth = Path(args.correlation_data).resolve()
    corr_select_pth = Path(args.correlation_results).resolve()

    input_pths = list(input_pth.rglob('*.hlist'))
    input_summaries = list(input_pth.rglob('events.txt'))
    event_names = [x.stem for x in input_pths]
    output_pth.mkdir(exist_ok=True, parents=True)


def build_predict_parser(parser):
    parser.add_argument('radar', type=str, help='The observing radar system')
    parser.add_argument('catalog', type=str, help='TLE catalog path')
    parser.add_argument('correlation-data', type=str, help='Path to \
        correlation data')
    parser.add_argument('correlation-results', type=str, help='Path to \
        correlation results')
    parser.add_argument('input', type=str, help='Observation data folder, \
        expects folder with *.hlist files and an events.txt summary file.')
    parser.add_argument('output', type=str, help='Results output location')

    return parser


def build_collect_parser(parser):
    parser.add_argument('input', type=str, help='Results output location to \
        collect results from')
    parser.add_argument('output', type=str, help='Folder to save collection \
        output to')
    return parser


def build_estimate_parser(parser):
    parser.add_argument('radar', type=str, help='The observing radar system')
    parser.add_argument('input', type=str, help='Observation data folder, \
        expects folder with *.hlist files and an events.txt summary file.')
    parser.add_argument('output', type=str, help='Results output location')
    parser.add_argument(
        '-v',
        action='store_true', 
        help='Verbose output',
    )

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


def build_parser():

    parser = argparse.ArgumentParser(description='Analyze detailed SNR curves \
        to determine RCS and statistics')

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
        pass


if __name__ == '__main__':
    main()
