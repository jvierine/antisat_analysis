from pathlib import Path
import numpy as n
import h5py
import glob
import matplotlib.pyplot as plt

from beampark_elevation_doppler import cache_path

def get_data():
    fl=glob.glob("../data/range_range_rate*.h5")

    pars=[]
    for f in fl:
        #print(f)
        h=h5py.File(f,"r")
        #print(h.keys())
        if h["range_rate0"][()] < h["range_rate1"][()]:
            rr0=h["range_rate0"][()]
            rr1=h["range_rate1"][()]
        else:
            rr0=h["range_rate1"][()]
            rr1=h["range_rate0"][()]
        
        pars.append([h["a"][()],h["inc"][()],h["range"][()],rr0,rr1])
        h.close()
    pars=n.array(pars)
    return(pars)

def plot_range_rates(pars,pincs=n.array([70,75,80,85,90,95,100,105,110])):
    """
     Given inclination, plot doppler inclinations based on circular orbit crossing on-axis of beam
    """
    incs=n.unique(pars[:,1])
    pi=0
    for inc in incs:
        if n.min(n.abs(inc-pincs)) < 0.1:
            idx=n.where(pars[:,1] == inc)[0]
            ridx=n.argsort(pars[idx,2])
            plt.plot(pars[idx[ridx],2]/1e3,pars[idx[ridx],3]/1e3,color="C%d"%(pi),label=inc)
            plt.plot(pars[idx[ridx],2]/1e3,pars[idx[ridx],4]/1e3,color="C%d"%(pi))
            pi+=1

def plot_r_vs_rdot(station, radar_name='generic', topdir='cache', azim=None, elev=None, degrees=True,
                   incl=None, ax=None, cmap=None, legend=True):

    try:
        cpath = Path(station)
    except TypeError:
        cpath = cache_path(station, radar_name=radar_name,
                        topdir=cache_dir, azim=azim, elev=elev, degrees=degrees)

    if not cpath.exists():
        # TODO: If we can find cache file with az/el within 0.1 degree, use that
        raise FileNotFoundError(f'Cache file {cpath.name} not found')

    if ax is None:
        ax = plt.gca()

    if cmap is None:
        cmap = plt.get_cmap('tab10')

    with h5py.File(cpath, 'r') as ds:
        site_lat = ds['site_lat'][()]

        if incl is None:
            # Set some default inclination values, if not given
            if site_lat > 75:
                incl = n.arange(90-16, 90+16+1, 4)
            elif site_lat > 65:
                incl = n.arange(90-24, 90+24+1, 6)
            else:
                raise RuntimeError('Define inclinations for low-latitude stations')

        lh = []
        lab = []
        for ii, inc in enumerate(incl):
            ix = n.where(ds['incl'][:] == inc)

            theta_a = ds['theta_a'][ix,:]
            r_a = ds['r_a'][ix,:]
            rdot_a = ds['rdot_a'][ix,:]

            theta_d = ds['theta_d'][ix,:]
            r_d = ds['r_d'][ix,:]
            rdot_d = ds['rdot_d'][ix,:]

            ok_a = n.where(theta_a <= 1.0)
            nok_a = n.where(theta_a >= 1.0)
            ok_d = n.where(theta_d <= 1.0)
            nok_d = n.where(theta_d >= 1.0)

            hh = ax.plot(r_a[ok_a], rdot_a[ok_a], 'd-',
                          r_d[ok_d], rdot_d[ok_d], 'd--',
                          r_a[nok_a], rdot_a[nok_a], 'd:',
                          r_d[nok_d], rdot_d[nok_d], 'd:', color=cmap(ii))
            lh.append(hh[0])
            lab.append(f'incl = {inc}')

        if legend:
            ax.legend(lh, lab)





if __name__ == "__main__":
    pars=get_data()
    plot_range_rates(pars)
    plt.legend()
    plt.xlabel("Range (km)")
    plt.ylabel("Range-rate (km/s)")    
    plt.show()
