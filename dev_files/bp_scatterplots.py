
from pathlib import Path

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
plt.style.use('tableau-colorblind10')

from astropy.time import Time

import h5py

import sorts
from sorts.population import tles

# LEO.H5 ncdump format:

# netcdf leo {
# dimensions:
#         phony_dim_0 = 572 ;
# variables:
#         double diams(phony_dim_0) ;           # object diameter [m]
#         double dur(phony_dim_0) ;             # duration of object tracklet [s]
#         double r(phony_dim_0) ;               # range to object at peak SNR [km]
#         double snr(phony_dim_0) ;             # SNR of object at peak [linear units]
#         double t(phony_dim_0) ;               # time of peak SNR [Unix time, seconds]
#         double v(phony_dim_0) ;               # range rate of object at peak SNR [km/s] positive away
# }

# CORRELATION.H5 ncdump format:

# netcdf correlation {
# dimensions:
#         cartesian_pos_axis = 3 ;
#         cartesian_vel_axis = 3 ;
#         maching_rank = 1 ;
#         object_index = 471 ;
#         observation_index = 572 ;
# variables:
#         string cartesian_pos_axis(cartesian_pos_axis) ;
#                 string cartesian_pos_axis:long_name = "Cartesian position axis names" ;
#         string cartesian_vel_axis(cartesian_vel_axis) ;
#                 string cartesian_vel_axis:long_name = "Cartesian velocity axis names" ;
#         int64 maching_rank(maching_rank) ;
#                 string maching_rank:long_name = "Matching rank numbering from best as lowest rank to worst at hightest rank" ;
#         int64 matched_object_index(maching_rank, observation_index) ;
#                 string matched_object_index:long_name = "Index of the correlated object" ;
#         int64 object_index(object_index) ;
#                 string object_index:long_name = "Object index in the used population" ;
#         int64 observation_index(observation_index) ;
#                 string observation_index:long_name = "Observation index in the input radar data" ;
#         double simulated_position(object_index, observation_index, cartesian_pos_axis) ;
#                 string simulated_position:long_name = "Simulated ITRS positions" ;
#                 string simulated_position:units = "m" ;
#         double simulated_range(object_index, observation_index) ;
#                 string simulated_range:long_name = "Simulated range" ;
#                 string simulated_range:units = "m" ;
#         double simulated_range_rate(object_index, observation_index) ;
#                 string simulated_range_rate:long_name = "Simulated range rate" ;
#                 string simulated_range_rate:units = "m/s" ;
#         double simulated_velocity(object_index, observation_index, cartesian_vel_axis) ;
#                 string simulated_velocity:long_name = "Simulated ITRS velocities" ;
#                 string simulated_velocity:units = "m/s" ;

#         // also the following structured variables:
#         matched_object_metric(maching_rank, observation_index) -> (dr, dv)
#         simulated_correlation_metric(object_index, observation_index) -> (dr, dv)
# 
# // global attributes:
#                 string :radar_name = "eiscat_uhf" ;
#                 :tx_lat = 69.5863888888889 ;
#                 :tx_lon = 19.2272222222222 ;
#                 :tx_alt = 86. ;
#                 :rx_lat = 69.5863888888889 ;
#                 :rx_lon = 19.2272222222222 ;
#                 :rx_alt = 86. ;
# }

cache_dir = Path('../cache')

if 1:
    site = 'eiscat_uhf'
    date = '2021.11.23'
    kdate = '2021.12.27'    # Date of kosmos-debris catalogue download
    overall_title = 'EISCAT UHF 2021-11-23'
else:
    site = 'eiscat_esr'
    date = '2021.11.23'
    kdate = '2021.12.27'    # Date of kosmos-debris catalogue download
    overall_title = 'EISCAT ESR 2021-11-23'

datadir = cache_dir / site / date
trkdir  = cache_dir / 'space-track' / date
kosdir = cache_dir / 'kosmos' / kdate

bcor = h5py.File(datadir / 'correlation.h5', 'r')
kcor = h5py.File(datadir / 'kosmos-correlation.h5', 'r')
obs = h5py.File(datadir / 'leo.h5', 'r')
bpop = tles.tle_catalog(trkdir / 'space-track.tles', cartesian=False)
kpop = tles.tle_catalog(kosdir / 'space-track-kosmos.tles', cartesian=False)


# Task 1:

# For each observation (in LEO), find out if the object is either
# a) correlated to an "old" known object
# b) correlated to a known Kosmos-1408 fragment, or
# c) not correlated to a known object

assert len(bcor['observation_index']) == len(kcor['observation_index']) == len(obs['r']), \
    "mismatch in number of observations"

n_obs = len(obs['r'])

# Create mapping from population ID to object index -- Does not give reasonable results!
bobj_ix = dict((y, x) for x, y in enumerate(bcor['matched_object_index'][0, :]))
kobj_ix = dict((y, x) for x, y in enumerate(kcor['matched_object_index'][0, :]))


def _find_ix(_cor, oix, val):
    ix = np.where(_cor['simulated_correlation_metric'][:,oix] == val)[0]
    if len(ix) == 0:
        return None
    if len(ix) == 1:
        return ix[0]

    raise RuntimeError('Need no more than one match here')


oix_k = []
oix_b = []
oix_n = []

for obs_ix in range(n_obs):
    bpop_id = bcor['matched_object_index'][0, obs_ix]
    kpop_id = kcor['matched_object_index'][0, obs_ix]

    # These don't work, and I don't know why
    # bix = bobj_ix[bpop_id]
    # kix = kobj_ix[kpop_id]

    bmval = bcor['matched_object_metric'][0, obs_ix]
    kmval = kcor['matched_object_metric'][0, obs_ix]

    # b_err = np.hypot(*bmval)
    # k_err = np.hypot(*kmval)
    b_err = np.hypot(bmval[0]/15e3, bmval[1]/1e3)
    k_err = np.hypot(kmval[0]/15e3, kmval[1]/1e3)

    if np.isnan(b_err) and np.isnan(k_err):
        # uncorrelated
        oix_n.append(obs_ix)
        continue

    best = np.nanargmin([b_err, k_err])

    if [b_err, k_err][best] > 1:
        # not well enough correlated to matter
        oix_n.append(obs_ix)
        continue

    # best == 0 -> background; best == 1 -> kosmos
    if best == 0:
        #oix_b.append(_find_ix(bcor, obs_ix, bmval))
        oix_b.append(obs_ix)
    else:
        #oix_k.append(_find_ix(kcor, obs_ix, kmval))
        oix_k.append(obs_ix)


obs_t = obs['t'][:]
ax_t = (obs_t - obs_t[0])/60    # minutes
obs_v = obs['v'][:]
obs_r = obs['r'][:]

# bcor.close()
# kcor.close()
# obs.close()

t0 = Time(obs_t[0], format='unix').iso

fh, ah = plt.subplots(2,1, sharex='all')

ah[0].plot(ax_t[oix_b], obs_r[oix_b], 'bo', \
           ax_t[oix_k], obs_r[oix_k], 'rd', \
           ax_t[oix_n], obs_r[oix_n], 'kx')
ah[0].legend(['background', 'kosmos', 'uncorrelated'])
ah[0].set_ylabel('Range [km]')
ah[0].set_ylim([300, 2000])

ah[1].plot(ax_t[oix_b], obs_v[oix_b], 'bo', \
           ax_t[oix_k], obs_v[oix_k], 'rd', \
           ax_t[oix_n], obs_v[oix_n], 'kx')
ah[1].legend(['background', 'kosmos', 'uncorrelated'])
ah[1].set_ylabel('Velocity [km/s]')
ah[1].set_xlabel(f'Time [minutes since {t0}]')
ah[1].set_ylim([-2, 2])

if overall_title:
    fh.suptitle(overall_title + ' Correlations')




##  Residual scatterplots

# Kosmos correlates
#for obs_ix in oix_k:
#    kmval = kcor['matched_object_metric'][0, obs_ix]
#    kcid = _find_ix(kcor, obs_ix, kmval)
# oix_notk = np.fromiter(set(range(n_obs)).difference(oix_k), dtype=int)

mom = kcor['matched_object_metric'][0, :]
r_errk = np.array([x[0] for x in mom]) / 1e3
v_errk = np.array([x[1] for x in mom]) / 1e3

mom = bcor['matched_object_metric'][0, :]
r_errb = np.array([x[0] for x in mom]) / 1e3
v_errb = np.array([x[1] for x in mom]) / 1e3

fh, ah = plt.subplots(1,2)

ah[0].plot(r_errb[oix_b], v_errb[oix_b], 'bo', \
           r_errk[oix_k], v_errk[oix_k], 'rd', \
           r_errb[oix_n], v_errk[oix_n], 'kx')
ah[0].legend(['background', 'kosmos', 'uncorrelated'])
ah[0].set_xlim([-100, 100])
ah[0].set_ylim([-5, 5])
ah[0].set_xlabel('residual range [km]')
ah[0].set_ylabel('residual velocity [km/s]')

ah[1].plot(r_errb[oix_b], v_errb[oix_b], 'bo', \
           r_errk[oix_k], v_errk[oix_k], 'rd', \
           r_errb[oix_n], v_errk[oix_n], 'kx')
ah[1].set_xlim([-15, 15])
ah[1].set_ylim([-3, 3])
ah[1].set_xlabel('residual range [km]')
ah[1].set_ylabel('residual velocity [km/s]')
fh.suptitle(overall_title + ' Residuals')

