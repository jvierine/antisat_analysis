
import argparse
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import h5py
from astropy.time import Time

'''

Example run

python beampark_correlation_analysis.py ~/data/spade/beamparks/{uhf,esr}/2021.11.23/correlation.h5

'''

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyse beampark correlation for a beampark')
    parser.add_argument('input', type=str, help='Input correlation data')
    parser.add_argument('measurements', type=str, help='Input measurement data')
    parser.add_argument('--output', type=str, default='', help='Output folder for plots')

    args = parser.parse_args()

    arg_input = args.input
    meas_file = args.measurements

    scale_x = 1.0
    scale_y = 0.2

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

        inds = np.argsort(t)

        t = t[inds] - t.min()
        r = r[inds]
        v = v[inds]

    with h5py.File(input_pth, 'r') as ds:
        indecies = ds['matched_object_index'][()]
        metric = ds['matched_object_metric'][()]
        name = ds.attrs['radar_name']

    x = metric['dr']*1e-3
    y = metric['dv']*1e-3
    m = metric['metric']*1e-3
    ji = metric['jitter_index']
    inds = np.logical_not(np.logical_or(np.isnan(x), np.isnan(y)))
    xp = x[inds]
    yp = y[inds]

    fig, axes = plt.subplots(1, 2, figsize=(15, 15))

    for ax in axes:
        ax.plot(xp, yp, '.b')
        ax.plot([0, 0], [yp.min(), yp.max()], '-r')
        ax.plot([xp.min(), xp.max()], [0, 0], '-r')

        draw_ellipse(scale_x, scale_y, ax)

        ax.set_xlabel('Range residuals [km]')
        ax.set_ylabel('Range-rate residuals [km/s]')
        ax.set_title(name)

    axes[1].set_xlim([-scale_x, scale_x])
    axes[1].set_ylim([-scale_y, scale_y])

    if out_path is not None:
        fig.savefig(out_path / f'{name}_residuals.png')

    log10_elip_dst = np.log10(m[np.logical_not(np.isnan(m))])

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.hist(log10_elip_dst, 100)
    ax.plot([1, 1], ax.get_ylim(), '-r')

    ax.set_xlabel('Elliptical compound residual [log10(1)]')
    ax.set_ylabel('Frequency [1]')
    ax.set_title(name)

    if out_path is not None:
        fig.savefig(out_path / f'{name}_ellipse_distance.png')

    select = np.logical_and(
        m < 1.0,
        np.logical_not(np.isnan(m)),
    ).flatten()
    not_select = np.logical_not(select)

    fig, axes = plt.subplots(2, 1, figsize=(15, 15))

    ax = axes[0]
    ax.plot(t[select]/3600.0, r[select], '.r', label='Correlated')
    ax.plot(t[not_select]/3600.0, r[not_select], '.b', label='Uncorrelated')
    ax.legend()
    ax.set_ylabel('Range [km]')
    ax.set_xlabel('Time [h]')
    ax.set_title(name)

    ax = axes[1]
    ax.plot(t[select]/3600.0, v[select], '.r')
    ax.plot(t[not_select]/3600.0, v[not_select], '.b')

    ax.set_ylabel('Range-rate [km/s]')
    ax.set_xlabel('Time [h]')

    if out_path is not None:
        fig.savefig(out_path / f'{name}_rv_t_correlations.png')

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.plot(r[select], v[select], '.r', label='Correlated')
    ax.plot(r[not_select], v[not_select], '.b', label='Uncorrelated')
    ax.legend()
    ax.set_xlabel('Range [km]')
    ax.set_ylabel('Range-rate [km/s]')
    ax.set_title(name)

    if out_path is not None:
        fig.savefig(out_path / f'{name}_rv_correlations.png')

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.hist([ji.flatten()[select], ji.flatten()[not_select]], stacked=True)
    ax.set_xlabel('Jitter index [1]')
    ax.set_ylabel('Frequency [1]')
    ax.set_title(name)

    if out_path is not None:
        fig.savefig(out_path / f'{name}_jitter.png')

    plt.show()
