import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import Time

from sorts.propagator import Kepler
from sorts import SpaceObject
import scipy.optimize as sio
import jcoord
import numpy as n
import sklearn.cluster as scl
import h5py
from mpi4py import MPI
comm = MPI.COMM_WORLD

options = dict(
    settings=dict(
        in_frame='GCRS',
        out_frame='ITRS',
    ),
)

class beampark_radar:
    """ 
    a class which includes coordinates for a radar performing a beampark observations
    """
    def __init__(self,
                 lat=69.58,  
                 lon=19.23,
                 h=86.0,
                 az=90.0,
                 el=70.0,
                 obstime="2019-06-06",
                 name="EISCAT UHF"):
        
        self.lat=lat
        self.lon=lon
        self.h=h
        self.az=az
        self.el=el

        # ecef position of radar
        self.loc = jcoord.geodetic2ecef(self.lat, self.lon, self.h)
        
        # unit vector pointing towards radar on axis
        self.k = jcoord.azel_ecef(self.lat, self.lon, self.h, self.az, self.el)
        

    def get_on_axis_angle(self,states):
        """ 
        one dimensional angle between on-axis position and target
        """
        # dot product
        v0=states[0,:]-self.loc[0]
        v1=states[1,:]-self.loc[1]
        v2=states[2,:]-self.loc[2]        
        k_dot_p=v0*self.k[0] + v1*self.k[1] + v2*self.k[2]
        p_dot_p=n.sqrt(v0**2.0+v1**2.0+v2**2.0)
        angle=n.arccos(k_dot_p/p_dot_p)
        return(angle)

    def get_range_and_range_rate(self,states):
        """
        return the range and range rate of a radar measurement
        in units of meters and meters per second.

         states is an array 6xN with N state vectors 
         dims 0,1,2 = x,y,z  (m)
              3,4,5 = vx,vy,vz (m/s)
        """
        k=n.zeros([3,states.shape[1]],dtype=n.float64)
        
        k[0,:]=states[0,:]-self.loc[0]
        k[1,:]=states[1,:]-self.loc[1]
        k[2,:]=states[2,:]-self.loc[2]
        
        # ranges (distance from radar to satellite)
        ranges=n.sqrt(k[0,:]**2.0+k[1,:]**2.0+k[2,:]**2.0)
        
        k_unit=k/ranges

        # k dot v where k in unit normalized
        range_rates=k_unit[0,:]*states[3,:] + k_unit[1,:]*states[4,:] + k_unit[2,:]*states[5,:]
        
        return(ranges,range_rates)

    def find_passes(self,
                    obj, 
                    t0=0,
                    t1=24*3600.0,
                    threshold_angle=15.0,
                    use_threshold=True,
                    dt_coarse=10.0,  # first search uses this
                    dt=0.2):       # values reported using this resolution
        """
        Figure out when the object is within threshold_angle of the on-axis position of the radar,
        and to report the on-axis angle, range, and range-rate when this occurs with a time resolution 
        of dt. 

        In order to speed up calculations, we first use a coarse sampling of dt_coarse
        """

        # first sample coarse in time to identify regions of interest
        # time since epoch (coarsely sampled in time)
        t = n.arange(t0,t1,dt_coarse)
        # this gives you
        states = obj.get_state(t)
        # on-axis angle in degrees
        angles=180.0*self.get_on_axis_angle(states)/n.pi

        if use_threshold:
            good_times = n.where(angles < threshold_angle)[0]
        else:
            mi=n.argmin(angles)
            return(n.array([t[mi]]))

        if len(good_times) < 2:
            return(n.array([],dtype=n.int))

        end_idx0=n.where(n.diff(t[good_times]) > dt_coarse)[0]
        end_idx = n.concatenate((end_idx0,[len(good_times)-1]))
        start_idx = n.concatenate(([0],end_idx0+1))

        n_passes = len(start_idx)
        t_all=[]
        for pi in range(n_passes):
#            print("pass %d %1.2f-%1.2f"%(pi,t[good_times[start_idx[pi]]],t[good_times[end_idx[pi]]]))
            # high res time sampling
            t_hr = n.arange(t[good_times[start_idx[pi]]],t[good_times[end_idx[pi]]],dt)
            t_all=n.concatenate((t_all,t_hr))
        return(t_all)

def test_pass(obj):

    # create a radar
    radar=beampark_radar()

    t_hr=radar.find_passes(obj,t0=0,t1=24*3600,dt_coarse=10.0)

    # this gives you the state vector of the space object at times t
    t = n.arange(0,24*3600,10)
    states = obj.get_state(t)

    if len(t_hr)>0:
        states_hr = obj.get_state(t_hr)    
    
    angles=radar.get_on_axis_angle(states)
    
    if len(t_hr)>0:    
        angles_hr=radar.get_on_axis_angle(states_hr)    

    plt.plot(t,180.0*angles/n.pi,".")
    if len(t_hr)>0:    
        plt.plot(t_hr,180.0*angles_hr/n.pi,".")    
    plt.xlabel("Time (seconds since epoch)")
    plt.ylabel("On-axis angle (deg)")
    plt.show()
    
    ranges,range_rates=radar.get_range_and_range_rate(states)
    range_km=ranges/1e3
    range_rate_km=range_rates/1e3
    plt.plot(t,range_km,".")

    if len(t_hr)>0:        
        ranges_hr,range_rates_hr=radar.get_range_and_range_rate(states_hr)
        range_km_hr=ranges_hr/1e3
        range_rate_km_hr=range_rates_hr/1e3
        plt.plot(t_hr,range_km_hr,".")
    
    plt.xlabel("Time (seconds since epoch)")
    plt.ylabel("Range (km)")
    plt.show()
    
    plt.plot(t,range_rate_km,".")
    if len(t_hr)>0:    
        plt.plot(t_hr,range_rate_km_hr,".")    
    # compare numerical derivative of range with analytic range-rate
    plt.plot(0.5*(t[0:len(t)-1]+t[1:len(t)]),n.diff(range_km)/10.0,"x")    
    plt.xlabel("Time (seconds since epoch)")
    plt.ylabel("Range (km/s)")
    plt.show()


def get_on_axis_radar_obs(radar,i=74.0,a=7000e3):
    """
    Use optimization search to find range and doppler corresponding to on-axis crossing.
    There are two solutions to Doppler!
    """
    def ss(x):
        raan = n.mod(x[0],360.0)
        mu0 = n.mod(x[1],360.0)
        t_hr = n.array([x[2]])
        
        obj = SpaceObject(
            Kepler,
            propagator_options = options,
            a = a, 
            e = 0.0, 
            i = i, 
            raan = raan,
            aop = 0,
            mu0 = mu0, 
            epoch = Time("2019-06-01T00:00:00Z"), # utc date for epoch
            parameters = dict(
                d = 0.2,
            )
        )
        states_hr = obj.get_state(t_hr)    
        angles_hr=radar.get_on_axis_angle(states_hr)
        ang=n.min(angles_hr)*180.0/n.pi
        return(ang)
    
    x0=n.array([360.0*n.random.rand(1)[0],360.0*n.random.rand(1)[0]])
    obj = SpaceObject(
        Kepler,
        propagator_options = options,
        a = a, 
        e = 0.0, 
        i = i, 
        raan = x0[0],
        aop = 0,
        mu0 = x0[1], 
        epoch = Time("2019-06-01T00:00:00Z"), # utc date for epoch
        parameters = dict(
            d = 0.2,
        )
    )
    print("finding candidate pass")
    t_hr=radar.find_passes(obj,t0=0.0,t1=24*3600.0,dt_coarse=60.0,use_threshold=False)
    states_hr = obj.get_state(t_hr)            
    angles_hr=180.0*radar.get_on_axis_angle(states_hr)/n.pi
    t0=t_hr[n.argmin(angles_hr)]
    print("starting with %1.2f"%(angles_hr[0]))
    xhat=sio.fmin(ss,[x0[0],x0[1],t0])
    
    obj = SpaceObject(
        Kepler,
        propagator_options = options,
        a = a, 
        e = 0.0, 
        i = i, 
        raan = n.mod(xhat[0],360.0),
        aop = 0,
        mu0 = n.mod(xhat[1],360.0), 
        epoch = Time("2019-06-01T00:00:00Z"), # utc date for epoch
        parameters = dict(
            d = 0.2,
        )
    )
    t_hr=n.array([xhat[2]])
    states_hr = obj.get_state(t_hr)            
    angles_hr=180.0*radar.get_on_axis_angle(states_hr)/n.pi
    ranges,range_rates=radar.get_range_and_range_rate(states_hr)
    return(angles_hr[0],ranges[0],range_rates[0])

def get_range_dop_dist(radar,a=7000e3,i=71.0,N_sim=10):
    """
    Propagate objects in orbit and determine the
    range and range-rate at smallest on-axis angle 
    """
    pars=[]
    for oi in range(N_sim):
        xhat=get_on_axis_radar_obs(radar,i=i,a=a)
        print(xhat)
        pars.append(xhat)
    pars=n.array(pars)
    ka=scl.KMeans(n_clusters=2)
    ka.fit(pars)
    return(ka.cluster_centers_)


def range_doppler_search(radar,i=69.0,a=7000e3):
    """
    look for ranges and range-rates corresponding to 
    """
    # create a radar

    ret=get_range_dop_dist(radar,a=a,i=i,N_sim=20)
    angle=ret[0,0]
    range_m=ret[0,1]
    range_rate_0=ret[0,2]
    range_rate_1=ret[1,2]
    print("found angle %1.2f range %1.2f range-rate %1.2f,%1.2f"%(angle,range_m,range_rate_0,range_rate_1))
    return(angle,range_m,range_rate_0,range_rate_1)

def eiscat_uhf_range_dop_search(el=75.0):
    radar=beampark_radar(el=el)
    R_e=6371e3
    apogees=n.linspace(300e3,3000e3,num=20)+R_e
    incs=n.arange(68,113)
#    incs=[69,70,77]#n.arange(68,113)    
    n_apog=len(apogees)
    n_incs=len(incs)
    for ai in range(n_apog):
        a=apogees[ai]
        for inci in range(comm.rank,n_incs,comm.size):
            inc=incs[inci]
            print("Searching a=%1.2f i=%1.2f"%(a,inc))
            angle,rg,rg_rt0,rg_rt1=range_doppler_search(radar,a=a,i=inc)
            ho=h5py.File("data/range_range_rate-el%1.2f-a%1.2f-i%1.2f.h5"%(el,a,inc),"w")
            ho["a"]=a
            ho["inc"]=inc
            ho["range"]=rg
            ho["angle"]=angle
            ho["range_rate0"]=rg_rt0
            ho["range_rate1"]=rg_rt1
            ho.close()
            

def plot_3d_orbit(obj,radar):
    """
    plot position of space object when it is "close" to the on axis position
    """

    # create a space object
    t_hr=radar.find_passes(obj,t0=0,t1=24*3600)
    t_lr=n.arange(0,24*3600,100)
    # high resolution close to zero on-axis angles only
    
    # low time resolution
    states_lr = obj.get_state(t_lr)        
    
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states_lr[0,:], states_lr[1,:], states_lr[2,:],".")
    ax.plot([radar.loc[0],radar.loc[0]+radar.k[0]*2000e3],
            [radar.loc[1],radar.loc[1]+radar.k[1]*2000e3],
            [radar.loc[2],radar.loc[2]+radar.k[2]*2000e3],color="red")

    if len(t_hr)>0:
        states = obj.get_state(t_hr)
        ax.plot(states[0,:], states[1,:], states[2,:],".")
        
        max_range = np.linalg.norm(states[0:3,0])*2
        
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
    plt.show()


if __name__ == "__main__":

    #get_range_dop_dist(a=7000e3,i=89.0,N_sim=100)
#    range_doppler_search()
    eiscat_uhf_range_dop_search()
    exit(0)
    
    obj = SpaceObject(
        Kepler,
        propagator_options = options,
        a = 7000e3, 
        e = 0.0, 
        i = 74, 
        raan = 0, 
        aop = 0, 
        mu0 = 0, 
        epoch = Time("2019-06-01T00:00:00Z"), # utc date for epoch
        parameters = dict(
            d = 0.2,
        )
    )
    
    test_pass(obj)
    plot_3d_orbit(obj)
