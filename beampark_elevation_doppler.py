import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import Time, TimeDelta

import scipy.optimize as sio
import numpy as np
import h5py

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    class COMM_WORLD:
        rank = 0
        size = 1
    comm = COMM_WORLD()

import sorts
import pyorb


def get_on_axis_angle(station, states, ecef_k_vector):
    """ 
    one dimensional angle between on-axis position and target
    """
    
    dstate = states[:3, :] - station.ecef[:, None]
    
    k_dot_p = np.sum(dstate*ecef_k_vector[:, None], axis=0)

    p_dot_p = np.sqrt(np.sum(dstate**2.0, axis=0))
    angle = np.arccos(k_dot_p/p_dot_p)
    return angle


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


def plot_inclination_results(anom, Omega, angle, ranges, range_rates, samples, orb_hits=None):
    fig, ax = plt.subplots()
    pc = ax.pcolormesh(anom.reshape(samples), Omega.reshape(samples), angle.reshape(samples), shading='gouraud')
    fig.colorbar(pc, ax=ax)
    if orb_hits is not None:
        for x in orb_hits:
            ax.plot([np.mod(x[0] + 360, 360)], [np.mod(x[1] + 360, 360)], 'or')

    figs = [fig]
    axes = [ax]

    fig, ax = plt.subplots(1, 2)
    pc = ax[0].pcolormesh(anom.reshape(samples), Omega.reshape(samples), ranges.reshape(samples), shading='gouraud')
    fig.colorbar(pc, ax=ax[0])
    pc = ax[1].pcolormesh(anom.reshape(samples), Omega.reshape(samples), range_rates.reshape(samples), shading='gouraud')
    fig.colorbar(pc, ax=ax[1])

    figs += [fig]
    axes += [ax]
    return figs, axes


def optim_on_axis_angle(x, epoch, st, k_ecef, semi_major_axis, inclination):
    orbit = pyorb.Orbit(
        M0 = pyorb.M_earth, 
        degrees = True,
        a = semi_major_axis, 
        e = 0, 
        i = inclination, 
        omega = 0,
        Omega = x[1],
        anom = x[0],
        num = 1,
        type = 'mean',
    )
    states = orbit.cartesian
    ecef = sorts.frames.convert(epoch, states, in_frame='GCRS', out_frame='ITRS')
    angle = np.degrees(get_on_axis_angle(st, ecef, k_ecef))

    return angle


def orb_to_range_and_range_rate(x, epoch, st, semi_major_axis, inclination):
    orbit = pyorb.Orbit(
        M0 = pyorb.M_earth, 
        degrees = True,
        a = semi_major_axis, 
        e = 0, 
        i = inclination, 
        omega = 0,
        Omega = x[1],
        anom = x[0],
        num = 1,
        type = 'mean',
    )
    states = orbit.cartesian
    ecef = sorts.frames.convert(epoch, states, in_frame='GCRS', out_frame='ITRS')

    return get_range_and_range_rate(st, ecef)


def main():

    parser = argparse.ArgumentParser(description='Calculate the doppler-inclination correlation for a beampark')
    parser.add_argument('radar', type=str, help='The observing radar system')
    parser.add_argument('azimuth', type=float, help='Azimuth of beampark')
    parser.add_argument('elevation', type=float, help='Elevation of beampark')

    args = parser.parse_args()

    radar = getattr(sorts.radars, args.radar)

    epoch = Time("2000-01-01T00:10:00Z", format='isot')

    samples = (100, 100)
    a_samples = 20
    optim = True
    R_earth = 6371e3

    semi_major_axis = R_earth + np.linspace(300e3, 3000e3, num=a_samples)

    st = radar.tx[0]
    st.beam.sph_point(args.azimuth, args.elevation, radians=False)
    
    inclination = 80.0

    k_ecef = st.pointing_ecef

    anom, Omega = np.meshgrid(
        np.linspace(0, 360, num=samples[0]),
        np.linspace(0, 360, num=samples[1]),
        indexing='ij',
    )
    anom = anom.reshape(anom.size)
    Omega = Omega.reshape(Omega.size)

    all_hits = [[], []]
    all_obs = [[], []]

    for a in semi_major_axis:

        orbit = pyorb.Orbit(
            M0 = pyorb.M_earth, 
            degrees = True,
            a = a, 
            e = 0, 
            i = 0, 
            omega = 0,
            Omega = Omega,
            anom = anom,
            num = anom.size,
            type = 'mean',
        )

        print(orbit)

        orbit.i = inclination
        states = orbit.cartesian
        ecef = sorts.frames.convert(epoch, states, in_frame='GCRS', out_frame='ITRS')
        angle = np.degrees(get_on_axis_angle(st, ecef, k_ecef))

        ranges, range_rates = get_range_and_range_rate(st, ecef)

        inds = np.argsort(angle)

        x_start = [anom[inds[0]], Omega[inds[0]]]
        orb_hits = []
        xhats = [] 
        if optim:
            xhat1 = sio.minimize(
                lambda x: optim_on_axis_angle(x, epoch, st, k_ecef, a, inclination),
                x_start,
                method='Nelder-Mead',
            )
            print(x_start)
            print(xhat1)
            xhats += [xhat1]
            all_hits[0] += [xhat1.x]
            all_obs[0] += [list(orb_to_range_and_range_rate(xhat1.x, epoch, st, a, inclination))]
            orb_hits += [xhat1.x]
        else:
            all_hits[0] += [x_start]
            all_obs[0] += [list(orb_to_range_and_range_rate(x_start, epoch, st, a, inclination))]
            orb_hits += [x_start]

        other_ind = np.argmax(np.abs(anom[inds] - anom[inds[0]]) > 5.0)
        
        if other_ind.size > 0:
            x_start2 = [anom[inds[other_ind]], Omega[inds[other_ind]]]
            if optim:
                xhat2 = sio.minimize(
                    lambda x: optim_on_axis_angle(x, epoch, st, k_ecef, a, inclination),
                    x_start2,
                    method='Nelder-Mead',
                )
                print(x_start2)
                print(xhat2)
                xhats += [xhat2]
                xhats = [xhat1, xhat2]
                all_hits[1] += [xhat2.x]
                all_obs[1] += [list(orb_to_range_and_range_rate(xhat2.x, epoch, st, a, inclination))]
                orb_hits += [xhat2.x]
            else:
                all_hits[1] += [x_start2]
                all_obs[1] += [list(orb_to_range_and_range_rate(x_start2, epoch, st, a, inclination))]
                orb_hits += [x_start2]

            if all_obs[1][-1][1] > all_obs[0][-1][1]:
               _x = all_obs[1][-1]
               all_obs[1][-1] = all_obs[0][-1]
               all_obs[0][-1] = _x

               _x = all_hits[1][-1] 
               all_hits[1][-1] = all_hits[0][-1]
               all_hits[0][-1] = _x

        # figs, axes = plot_inclination_results(anom, Omega, angle, ranges, range_rates, samples, orb_hits=orb_hits)
        # plt.show()

    fig, axes = plt.subplots(3, 1)

    for node, col in zip(range(2), ['b', 'r']):
        obs_data = np.squeeze(np.array(all_obs[node])).T
        orb_data = np.squeeze(np.array(all_hits[node])).T

        axes[0].plot(obs_data[0, :], obs_data[1, :], '-' + col)
        axes[1].plot(obs_data[0, :], orb_data[0, :], '-' + col)
        axes[2].plot(obs_data[0, :], orb_data[1, :], '-' + col)
    plt.show()


if __name__ == '__main__':
    main()

