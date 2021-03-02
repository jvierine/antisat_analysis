import glob
import numpy as n
import stuffr
import matplotlib.pyplot as plt
import h5py


def read_spade(dirname="leo_bpark_2.1u_NO@uhf/2019040[5]_*/*.txt",output_h5="out.h5"):

    fl=glob.glob(dirname)
    fl.sort()

    t0s=[]
    rgs=[]
    vdops=[]
    rts=[]
    durs=[]
    diams=[]
    nlines=0
    
    oi=0
    for f in fl:
        a=n.genfromtxt(f,comments="%")
        nlines+=a.shape[0]
        for ai in range(a.shape[0]):
            t0=stuffr.date2unix(int(a[ai,0]),int(a[ai,1]),int(a[ai,2]),int(a[ai,3]),int(a[ai,4]),0)+float(a[ai,5])
            cid=a[ai,6]
            tx=a[ai,7]
            st=a[ai,8]
            az=a[ai,9]
            el=a[ai,10]
            ht=a[ai,11]
            rt=a[ai,12]
            rg=a[ai,13]
            rr=a[ai,14]                
            vdop=a[ai,15]
            adop=a[ai,16]
            diam=a[ai,17]
            rcs=a[ai,18]
            ts=a[ai,19]
            en=a[ai,20]
            edur=a[ai,21]
            tp=a[ai,22]
            flag=a[ai,23]
            print("n %d %f"%(oi,vdop))
            oi+=1
            #        print(flag)
            #       if flag == 0.0:
            t0s.append(t0)
            rgs.append(rg)
            vdops.append(vdop)
            rts.append(rt)
            durs.append(edur)
            diams.append(diam)

    t=n.array(t0s)
    r=n.array(rgs)
    v=n.array(vdops)
    snr=n.array(rts)**2.0
    dur=n.array(durs)
    diams=n.array(diams)    
    # store in hdf5 format
    ho=h5py.File(output_h5,"w")
    ho["t"]=t    # t
    ho["r"]=r    # r  
    ho["v"]=v    # vel
    ho["snr"]=snr  # snr
    ho["dur"]=dur  # duration
    ho["diams"]=diams # minimum diameter
    ho.close()
    


def plot_data(fname):
    h=h5py.File(fname,"r")
    v=h["v"][()]
    t=h["t"][()]
    snr=h["snr"][()]
    dur=h["dur"][()]        
    
    t0s=n.array(t0s)
    t0=t0s-n.min(t0s)
    
    # info on i and a
    H,x,y=n.histogram2d(v,rgs,range=[[-3,3],[0,3000]],bins=(100,100))
    
    plt.subplot(121)
    cdops=n.copy(vdops)
    cdops[cdops>4]=4
    cdops[cdops<-4]=-4
    plt.scatter(rgs,vdops,c=cdops,lw=0,cmap="Spectral")
    cbar=plt.colorbar()
    cbar.set_label('Doppler shift (km/s)')
    plt.xlabel("Range (km)")
    plt.ylabel("Doppler velocity (km/s)")
    plt.title("Detections")
    plt.ylim([-3,3])
    plt.subplot(122)
    plt.pcolormesh(y,x,H)
    plt.title("Histogram")
    plt.xlabel("Range (km)")
    plt.ylabel("Doppler velocity (km/s)")
    plt.colorbar()
    plt.show()
    
    # info on d and r
    
    enrs=n.array(rts)
    print(n.min(enrs))
    H,x,y=n.histogram2d(rgs,10.0*n.log10(enrs**2.0),range=[[0,3000],[0,80]],bins=(20,40))
    ho["enrs"]=enrs
    ho["r_enr"]=H
    
    plt.subplot(121)
    plt.scatter(t0,rgs,c=cdops,lw=0,cmap="nipy_spectral")
    
    cbar=plt.colorbar()
    cbar.set_label('Doppler shift (km/s)')
    plt.xlabel("Time (s since %s)"%(stuffr.unix2datestr(t0s[0])))
    plt.ylabel("Range (km)")
    plt.title("Detections")
    plt.subplot(122)
    n_hours=(n.max(t0)-n.min(t0))/1800.0
    n_bins=int((n.max(t0)-n.min(t0))/1800.0)
    w=n_hours/n_bins
    plt.hist(t0,bins=n_bins,weights=n.repeat(w,len(t0)) )
    plt.xlabel("Time (s since %s)"%(stuffr.unix2datestr(t0s[0])))
    plt.ylabel("Detections per 30 minutes")
    plt.title("Histogram")
    plt.show()
    
    
    
    plt.subplot(121)
    plt.scatter(rgs,10.0*n.log10(enrs**2.0),c=cdops,lw=0)
    plt.xlabel("Range (km)")
    plt.ylabel("Signal to noise ratio (dB)")
    plt.subplot(122)
    plt.pcolormesh(x,y,H.T)
    plt.xlabel("Range (km)")
    plt.ylabel("Signal to noise ratio (dB)")
    plt.colorbar()
    plt.show()
    
    r,x=n.histogram(rgs,bins=n.arange(300,2000,50.0))
    ho["h_r"]=r
    ho.close()
    
    
    plt.plot(0.5*(x[0:(len(x)-1)]+x[1:len(x)]),r)
    plt.xlabel("Range (km)")
    plt.ylabel("Detections (50 km bin)")
    plt.show()
    
    vdops=n.array(vdops)
    H,x,y=n.histogram2d(t0,vdops,range=[[0,86400],[-3,3]],bins=(24,100))
    
    plt.subplot(121)
    plt.scatter(t0,vdops,c=cdops,lw=0)
    plt.xlabel("Time (s since %s)"%(stuffr.unix2datestr(t0s[0])))
    plt.ylabel("Doppler velocity (km/s)")
    
    
    plt.ylim([-3,3])
    plt.subplot(122)
    plt.pcolormesh(x,y,H.T)
    plt.xlabel("Time (s since %s)"%(stuffr.unix2datestr(t0s[0])))
    plt.ylabel("Doppler velocity (km/s)")
    
    plt.colorbar()
    plt.show()