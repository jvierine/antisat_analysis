import numpy as n
import h5py
import glob
import matplotlib.pyplot as plt


def get_data():
    fl=glob.glob("data/range_range_rate*.h5")

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

if __name__ == "__main__":
    pars=get_data()
    plot_range_rates(pars)
    plt.legend()
    plt.xlabel("Range (km)")
    plt.ylabel("Range-rate (km/s)")    
    plt.show()
