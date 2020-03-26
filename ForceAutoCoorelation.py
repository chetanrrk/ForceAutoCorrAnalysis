#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:53:29 2020

@author: chetanrupakheti
"""
import os,sys
import pickle
import time
import numpy as np
#from matplotlib import pyplot as plt
import multiprocessing as mp

"""
Computes the correlation without mean centering
"""
def calcCoor(dat,lag,dt):
    tmax = int(len(dat) - (lag*1.0/dt*1.0))
    mySum=0.0
    
    for i in range(tmax):
        mySum = mySum + (np.dot(dat[i],np.transpose(dat[i+lag])))
        
    return (1.0/tmax) * mySum  

""" 
def calcCoor(dat,lag,lagA,dt):
    print dat
    for i in range(tmax):
        t0 = dat[str(lagA[i])]
        tt = dat[str(lagA[i]+lag)]
        #print "lag,lagA[i],lagA[i]+lag",lag,str(lagA[i]),str(lagA[i]+lag)
        mySum = mySum + (t0*tt)    
    
    return (1.0/tmax) * mySum   
"""

"""
Computes the pearson correlation
"""
def normAutoCoor(dat_t0,dat_tn,lag):
    normDat_t0 = (dat_t0-np.mean(dat_t0,axis=0)) 
    normDat_tn = (dat_tn-np.mean(dat_tn,axis=0))    
    
    corr = np.average([normDat_t0[i].dot(np.transpose(normDat_tn[i])) 
                                    for i in range(len(dat_t0))])    
    
    corr = corr/(np.std(dat_t0,axis=0).dot(np.std(dat_tn,axis=0)))
    
    return corr,lag


"""
Mean centers the data and computes the correlation
"""
def AutoCoor(dat_t0,dat_tn,lag,axis="x"):
    
    #print "processing lag",str(lag),"in proc",os.getpid()
    
    normDat_t0 = np.array(dat_t0)	#(dat_t0-np.mean(dat_t0,axis=0))
    normDat_tn = np.array(dat_tn)	#(dat_tn-np.mean(dat_tn,axis=0))
    
    if axis=="x":
        #corr= np.average(normDat_t0[:,0]*normDat_tn[:,0]) ### product of the x-component 
        corr= np.average(normDat_t0*normDat_tn) ### product of the x-component 
    elif axis=="y":
        corr= np.average(normDat_t0[:,1]*normDat_tn[:,1]) ### product of the y-component 
    elif axis=="z":
        corr= np.average(normDat_t0[:,2]*normDat_tn[:,2]) ### product of the z-component 
    else: ### all "xyz" and taking trace 
        corr = np.average([normDat_t0[i].dot(np.transpose(normDat_tn[i])) 
            for i in range(len(dat_t0))])
    
    return corr,lag


"""
Initializes the data used to compute the correlation
"""
def getAutoCoor(dat,norm=False,axis="x"):
    autocorrs=[] ## autocorrelations from 0...len(timeseries)
    dt=1 ### spacing of my dcds i.e., 1fs
    for i in range(len(dat)): ###  lags loop goes from 0...len(timeseries)
        lag=i
        tmax = int(len(dat) - (lag*1.0/dt*1.0))
        t0=[]
        tn=[]
        for j in range(tmax):
            t0.append(dat[j])
            tn.append(dat[j+lag])
        if len(t0)<=1 or len(tn)<=1:continue
        if norm:corr = normAutoCoor(t0,tn,lag)
        else:corr = AutoCoor(t0,tn,lag,axis) ### along the frozen axis
        autocorrs.append(corr[0])
    return autocorrs


"""
appends the computed correlation result from a sub-process
"""
lag_corrs={} ## autocorrelations from 0...len(timeseries)
def log_result(result):
    #lag_corrs.append(result)
    lag_corrs[result[1]]=result[0]
    
"""
Initializes the data used to compute the correlation
Using multiprocessing to parallelize the computation 
"""

def getAutoCoorMP(dat,procs,lagRange,norm=False,axis="x"):
    pool = mp.Pool(processes=procs)

    dt=1 ### spacing of my dcds i.e., 1fs
    #for i in range(len(dat)): ###  lags loop goes from 0...len(timeseries)
    for i in range(lagRange[0],lagRange[1]): ###  lags loop goes from 0...len(timeseries)
        lag=i
        tmax = int(len(dat) - (lag*1.0/dt*1.0))
        t0=[]
        tn=[]
        for j in range(tmax):
            t0.append(dat[j])
            tn.append(dat[j+lag])
        if len(t0)<=1 or len(tn)<=1:continue
        if norm:
            pool.apply_async(normAutoCoor, args =(t0,tn,lag),callback=log_result)
            #corr = normAutoCoor(t0,tn)
        else:
            pool.apply_async(AutoCoor, args =(t0,tn,lag,axis),callback=log_result)
            #corr = AutoCoor(t0,tn,axis) ### along the frozen axis
    
    pool.close()
    pool.join()
    
    
if __name__=="__main__":
  
    
    dt = 1; ### 1fs sec saving interval was used to generate MD traj
    
    procs = 16 ### number of procs
    
    #force = np.loadtxt("/Users/chetanrupakheti/Documents/project/Drude/ensemble_test/timecorr/forces_traj_0.out")
    
    #force = force[:20000]

    force = np.loadtxt("../chucks/x.out")	
    
    ### computes the for fO-fD, i.e., the force on the oscillator
    forceDiff = np.array([force[i]-force[i+1] for i in range(0,len(force)-1,2)])

    forceDiff = (forceDiff-np.mean(forceDiff,axis=0))

    #forceDiff = forceDiff[:200000] ### first 200 ps data	

    start = time.time()     
    #corrs = getAutoCoor(forceDiff,norm=False,axis="x")

    startLag = int(sys.argv[1])	
    endLag = int(sys.argv[2])		
    lagRange=[startLag,endLag] 	
    getAutoCoorMP(forceDiff,procs,lagRange,norm=False,axis="x") ## logs result in lags_corrs
    
    end = time.time()
    print ("Autocorr calculation done in "+str((end-start)/60.0)+" mins")

    #corrs=[ lag_corrs[lag] for lag in range(startLag,endLag)] 
    corrs=[]
    for lag in range(startLag,endLag):
	if lag_corrs.has_key(lag):
		corrs.append(lag_corrs[lag])

    #plt.plot(range(len(corrs)),corrs,"o")
    #plt.xlabel("Lag")
    #plt.ylabel("Correlation")
    #    
    #plt.savefig("autocorr.pdf") 
    
    job = str(startLag)
    pickle.dump(corrs,open("corrs"+job+".p","w"))  
    
    
    
    
    
    
    
    
    
    """
    #force = np.loadtxt("force_traj.dat")
    ### computes the autocoor for different time lag
    forceCorrs={}
    for i in range(1,len(forceDiff)-1):
        forceCorrs[i] = calcCoor(forceDiff,i,dt)
    
    corrs = [forceCorrs[k] for k in range(1,len(forceCorrs))]
    
    plt.plot(range(1,len(corrs)+1),corrs,"o")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    """
    
    
    
    """
    ###example1 run
    dt = 0.2
    lag = [i/10.0 for i in range(0,12,2)]
    at = {str(t):np.sin((np.pi/2.0)*t) for t in lag}
    #print lag
    #print at
    
    #print calcCoor(at,0.4,lag,dt)
    
    
    
    coor = {}
    
    for i in range(len(lag)):
        c = calcCoor(at,lag[i],lag,dt)
        print lag[i],c
        coor[lag[i]] = c 
    
    print coor
    """
    
    """
    ###example2 run
    dt=1
    lag=range(6)
    at = [np.sin((np.pi/2.0)*t) for t in lag]
    
    coor = {}
    for i in range(len(lag)):
        c = calcCoor(at,i,dt)
        coor[i] = c
    print coor
    """
    
    
    
    
