#!/usr/bin/env python

import argparse
import pathlib
from scipy.optimize import curve_fit
import scipy.stats as st

import matplotlib.pyplot as plt
import numpy as np
import h5py
from astropy.time import Time


'''

Example run

python beampark_correlation_analysis.py ~/data/spade/beamparks/{uhf,esr}/2021.11.23/correlation.h5

'''


def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return st.norm.pdf(x, mu1, sigma1)*np.abs(A1) + st.norm.pdf(x, mu2, sigma2)*np.abs(A2)


def plot_measurement_data(observed_data, simulated_data, axes=None):
    '''Plot the observed and simulated population object measurment parameters.
    '''
    t = observed_data['t']
    r = observed_data['r']
    v = observed_data['v']
    r_ref = simulated_data['r_ref']
    v_ref = simulated_data['v_ref']

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 15))
    else:
        fig = None

    axes[0].plot(t - t[0], r*1e-3, label='measurement')
    axes[0].plot(t - t[0], r_ref*1e-3, label='simulation')
    axes[0].set_ylabel('Range [km]')
    axes[0].set_xlabel('Time [s]')

    axes[1].plot(t - t[0], v*1e-3, label='measurement')
    axes[1].plot(t - t[0], v_ref*1e-3, label='simulation')
    axes[1].set_ylabel('Velocity [km/s]')
    axes[1].set_xlabel('Time [s]')

    axes[1].legend()

    return fig, axes


def plot_correlation_residuals(observed_data, simulated_data, axes=None):
    '''Plot the correlation residuals between the measurement and simulated population object.
    '''
    t = observed_data['t']
    r = observed_data['r']
    v = observed_data['v']
    r_ref = simulated_data['r_ref']
    v_ref = simulated_data['v_ref']

    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    else:
        fig = None
    
    axes[0, 0].hist((r_ref - r)*1e-3)
    axes[0, 0].set_xlabel('Range residuals [km]')

    axes[0, 1].hist((v_ref - v)*1e-3)
    axes[0, 1].set_xlabel('Velocity residuals [km/s]')
    
    axes[1, 0].plot(t - t[0], (r_ref - r)*1e-3)
    axes[1, 0].set_ylabel('Range residuals [km]')
    axes[1, 0].set_xlabel('Time [s]')

    axes[1, 1].plot(t - t[0], (v_ref - v)*1e-3)
    axes[1, 1].set_ylabel('Velocity residuals [km/s]')
    axes[1, 1].set_xlabel('Time [s]')

    return fig, axes


def draw_ellipse(x_size, y_size, ax, res=100, style='-r', log=False):
    th = np.linspace(0, np.pi*2, res)
    ex = np.cos(th)*x_size
    ey = np.sin(th)*y_size

    if log:
        ex = np.log10(np.abs(ex))
        ey = np.log10(np.abs(ey))

    ax.plot(ex, ey, style)
    return ax


def main(input_args=None):
    parser = argparse.ArgumentParser(description='Analyse beampark correlation for a beampark')
    parser.add_argument('input', type=str, help='Input correlation data')
    parser.add_argument('measurements', type=str, help='Input measurement data')
    parser.add_argument('--output', type=str, default='', help='Output folder for plots and results')
    parser.add_argument('--threshold', type=float, default=None, help='Treshold in elliptical distance to choose correlations')
    parser.add_argument('--range-rate-scaling', default=0.2, type=float, help='Scaling used on range rate in the sorting function of the correlator')
    parser.add_argument('--range-scaling', default=1.0, type=float, help='Scaling used on range in the sorting function of the correlator')
    parser.add_argument(
        '-f', '--format',
        default='png',
        help='Plot format',
    )
    args = parser.parse_args()

    arg_input = args.input
    meas_file = args.measurements

    scale_x = args.range_scaling
    scale_y = args.range_rate_scaling
    threshold = args.threshold
    if threshold is not None:
        threshold *= 1e-3

    if len(args.output) > 0:
        out_path = pathlib.Path(args.output).resolve()
        out_path.mkdir(exist_ok=True)
    else:
        out_path = None

    input_pth = pathlib.Path(arg_input).resolve()
    input_meas_pth = pathlib.Path(meas_file).resolve()

    print('Loading monostatic measurements')
    with h5py.File(str(input_meas_pth), 'r') as h_det:
        r = h_det['r'][()]
        t = h_det['t'][()] # Unix seconds
        v = h_det['v'][()]

        inds_sort = np.argsort(t)

        epoch = Time(t.min(), format='unix')
        t = t[inds_sort] - t.min()
        r = r[inds_sort]
        v = v[inds_sort]

    with h5py.File(input_pth, 'r') as ds:
        indecies = ds['matched_object_index'][()]
        metric = ds['matched_object_metric'][()]
        name = ds.attrs['radar_name']

    x = metric['dr']*1e-3
    y = metric['dv']*1e-3
    m = metric['metric']*1e-3
    inds = np.logical_not(np.logical_or(np.isnan(x), np.isnan(y)))
    xp = x[inds]
    yp = y[inds]

    log10_elip_dst = np.log10(m[np.logical_not(np.isnan(m))])
    num = int(np.round(np.sqrt(len(log10_elip_dst))))

    if num < 10:
        if threshold is None:
            print('treshold cannot be automatically determined: defaulting to 1')
            threshold = 1.0
        threshold_est = None
    else:
        count, bins = np.histogram(log10_elip_dst, num)
        bin_centers = (bins[1:] + bins[:-1])*0.5

        start = (0, np.std(log10_elip_dst)*0.5, np.max(count)*0.5, np.max(bin_centers), np.std(log10_elip_dst)*0.5, np.max(count)*0.5)
        params, cov = curve_fit(bimodal, bin_centers, count, start)

        # this is intersection and is better estimate i think
        log_threshold_sample = np.linspace(params[0], params[3], 1000)
        log_threshold_intersection = np.argmin(bimodal(log_threshold_sample, *params))
        threshold_est = 10**log_threshold_sample[log_threshold_intersection]
        # threshold_est = 10**((params[0] + params[3])*0.5)

        print(f'ESTIMATED threshold: {threshold_est*1e3}')

    fig, axes = plt.subplots(1, 2, figsize=(15, 15))

    if threshold is None:
        threshold_ = threshold_est
    else:
        threshold_ = threshold

    title_str = f'{name.upper().replace("_", " ")} - {epoch.datetime.date()}'

    for ax in axes:
        ax.plot(xp, yp, '.b')
        ax.plot([0, 0], [yp.min(), yp.max()], '-r')
        ax.plot([xp.min(), xp.max()], [0, 0], '-r')

        draw_ellipse(scale_x*threshold_, scale_y*threshold_, ax)

        ax.set_xlabel('Range residuals [km]')
        ax.set_ylabel('Range-rate residuals [km/s]')
        ax.set_title(title_str)

    axes[1].set_xlim([-scale_x*threshold_, scale_x*threshold_])
    axes[1].set_ylim([-scale_y*threshold_, scale_y*threshold_])

    if out_path is not None:
        fig.savefig(out_path / f'{name}_residuals.{args.format}')
        plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.hist(log10_elip_dst, num)
    if threshold is not None:
        ax.plot([np.log10(threshold), np.log10(threshold)], ax.get_ylim(), '-g', label='Input threshold')
    if threshold_est is not None:
        ax.plot([np.log10(threshold_est), np.log10(threshold_est)], ax.get_ylim(), '--g', label='Estimated threshold')
        ax.plot(np.linspace(bins[0], bins[-1], 1000), bimodal(np.linspace(bins[0], bins[-1], 1000), *params), '-r', label='Fit')

    ax.set_xlabel('Distance function [log10(1)]')
    ax.set_ylabel('Frequency [1]')
    ax.set_title(title_str)
    ax.legend()

    if out_path is not None:
        fig.savefig(out_path / f'{name}_ellipse_distance.{args.format}')
        plt.close(fig)

    select = np.logical_and(
        m < threshold_,
        np.logical_not(np.isnan(m)),
    ).flatten()
    not_select = np.logical_not(select)

    if out_path is not None:
        res_file = out_path / f'{name}_selected_correlations.npy'
        np.save(res_file, select)

    fig, axes = plt.subplots(2, 1, figsize=(15, 15))

    ax = axes[0]
    ax.plot(t[select]/3600.0, r[select], '.r', label='Correlated')
    ax.plot(t[not_select]/3600.0, r[not_select], '.b', label='Uncorrelated')
    ax.legend()
    ax.set_ylabel('Range [km]')
    ax.set_xlabel('Time [h]')
    ax.set_title(title_str)

    ax = axes[1]
    ax.plot(t[select]/3600.0, v[select], '.r')
    ax.plot(t[not_select]/3600.0, v[not_select], '.b')

    ax.set_ylabel('Range-rate [km/s]')
    ax.set_xlabel('Time [h]')

    if out_path is not None:
        fig.savefig(out_path / f'{name}_rv_t_correlations.{args.format}')
        plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.plot(r[select], v[select], '.r', label='Correlated')
    ax.plot(r[not_select], v[not_select], '.b', label='Uncorrelated')
    ax.legend()
    ax.set_xlabel('Range [km]')
    ax.set_ylabel('Range-rate [km/s]')
    ax.set_title(title_str)

    if out_path is not None:
        fig.savefig(out_path / f'{name}_rv_correlations.{args.format}')
        plt.close(fig)

    if 'jitter_index' in metric.dtype.names:
        ji = metric['jitter_index']
        if not np.all(np.isnan(ji)):
            fig, ax = plt.subplots(1, 1, figsize=(15, 15))
            ax.hist([ji.flatten()[select], ji.flatten()[not_select]], stacked=True)
            ax.set_xlabel('Jitter index [1]')
            ax.set_ylabel('Frequency [1]')
            ax.set_title(title_str)

            if out_path is not None:
                fig.savefig(out_path / f'{name}_jitter.{args.format}')
                plt.close(fig)

    if out_path is None:
        plt.show()


if __name__ == '__main__':
    main()
