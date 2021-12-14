import pathlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import Time, TimeDelta
import h5py

import sorts
import pyorb

HERE = pathlib.Path(__file__).parent

cachefilename = HERE / '..' / 'cache' / 'eiscat_uhf' / 'az90.0_el45.0.h5'

inc_ind = 80
sema_ind = 0

with h5py.File(cachefilename, 'r') as ds:

    dsincl = ds['incl'][inc_ind]
    sma = ds['sema'][sema_ind]

    theta_a = ds['theta_a'][inc_ind, sema_ind]
    Om_a = ds['Om_a'][inc_ind, sema_ind]
    nu_a = ds['nu_a'][inc_ind, sema_ind]
    r_a = ds['r_a'][inc_ind, sema_ind]
    rdot_a = ds['rdot_a'][inc_ind, sema_ind]

    print(f'Value theta_a = {theta_a} deg')

print(f'r    = {r_a*1e-3} km')
print(f'rdot = {rdot_a*1e-3} km/s')

res = 100

anom = np.hstack((np.linspace(0, 360.0, res), np.array([nu_a])))

orbit = pyorb.Orbit(
    M0 = pyorb.M_earth, 
    degrees = True,
    a = sma, 
    e = 0, 
    i = dsincl, 
    omega = 0,
    Omega = Om_a,
    anom = anom,
    num = res+1,
    type = 'mean',
)

radar = getattr(sorts.radars, 'eiscat_uhf')

epoch = Time("2000-01-01T00:10:00Z", format='isot')

st = radar.tx[0]
st.beam.sph_point(90.0, 45.0, radians=False)

ecef_st = st.ecef
ecef_beam = ecef_st + st.pointing_ecef*3000e3

print(ecef_st, ecef_beam)

gcrs_st = sorts.frames.convert(
    epoch, 
    np.hstack((ecef_st, np.zeros((3,)))), 
    in_frame='ITRS', 
    out_frame='GCRS',
)[:3]

gcrs_beam = sorts.frames.convert(
    epoch, 
    np.hstack((ecef_beam, np.zeros((3,)))), 
    in_frame='ITRS', 
    out_frame='GCRS',
)[:3]

states = orbit.cartesian

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
ax.plot(states[0, :res], states[1, :res], states[2, :res], "-b")
ax.plot(states[0, res], states[1, res], states[2, res], "or")
ax.plot(
    [gcrs_st[0], gcrs_beam[0]], 
    [gcrs_st[1], gcrs_beam[1]], 
    [gcrs_st[2], gcrs_beam[2]], 
    "-g",
)

sorts.plotting.grid_earth(ax)

max_range = np.linalg.norm(states[0:3, 0])*2.1

ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)

plt.show()
