#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Fri Feb 28 16:53:29 2020

@author: chetan rupakheti
"""
import os
import sys
import pickle
import time
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing as mp


lag_corrs = {}  # stores the autocorrelations from 0...len(timeseries)


def calc_coor(dat, lag, dt):
    """
    Computes the correlation without mean centering
    """

    tmax = int(len(dat) - (lag * 1.0 / dt * 1.0))
    mySum = 0.0

    for i in range(tmax):
        mySum = mySum + (np.dot(dat[i], np.transpose(dat[i + lag])))

    return (1.0 / tmax) * mySum


def compute_normalized_autocoorelation(dat_t0, dat_tn, lag):
    """
    Computes the pearson correlation
    """

    normDat_t0 = (dat_t0 - np.mean(dat_t0, axis=0))
    normDat_tn = (dat_tn - np.mean(dat_tn, axis=0))

    corr = np.average([normDat_t0[i].dot(np.transpose(normDat_tn[i])) for i in range(len(dat_t0))])

    corr = corr / (np.std(dat_t0, axis=0).dot(np.std(dat_tn, axis=0)))

    return corr, lag


def compute_autocooreration(dat_t0, dat_tn, lag, axis="x"):
    """
    Mean centers the data and computes the correlation without dividing by the standard deviation
    """

    # print "processing lag",str(lag),"in proc",os.getpid()

    normDat_t0 = np.array(dat_t0)  # (dat_t0-np.mean(dat_t0,axis=0))
    normDat_tn = np.array(dat_tn)  # (dat_tn-np.mean(dat_tn,axis=0))

    if axis == "x":
        # corr= np.average(normDat_t0[:,0]*normDat_tn[:,0]) # product of the x-component
        corr = np.average(normDat_t0 * normDat_tn)  # product of the x-component
    elif axis == "y":
        corr = np.average(normDat_t0[:, 1] * normDat_tn[:, 1])  # product of the y-component
    elif axis == "z":
        corr = np.average(normDat_t0[:, 2] * normDat_tn[:, 2])  # product of the z-component
    else:  # all "xyz" and taking trace
        corr = np.average([normDat_t0[i].dot(np.transpose(normDat_tn[i]))
                           for i in range(len(dat_t0))])

    return corr, lag


def get_autocoorelation(dat, norm=False, axis="x"):
    """
    Initializes the data used to compute the correlation
    Uses only one processor to compute the autocorrelation
    """

    autocorrs = []  # autocorrelations from 0...len(timeseries)
    dt = 1  # spacing of my dcds i.e., 1fs
    for i in range(len(dat)):  # lags loop goes from 0...len(timeseries)
        lag = i
        tmax = int(len(dat) - (lag * 1.0 / dt * 1.0))
        t0 = []
        tn = []
        for j in range(tmax):
            t0.append(dat[j])
            tn.append(dat[j + lag])
        if len(t0) <= 1 or len(tn) <= 1: continue
        if norm:
            corr = compute_normalized_autocoorelation(t0, tn, lag)
        else:
            corr = compute_autocooreration(t0, tn, lag, axis)  # along the frozen axis
        autocorrs.append(corr[0])
    return autocorrs


def log_result(result):
    """
    appends the computed correlation result from a sub-process
    """

    lag_corrs[result[1]] = result[0]


def compute_autocoorelation_with_mp(dat, procs, lagRange, norm=False, axis="x"):
    """
    Initializes the data used to compute the correlation
    Using multiprocessing to parallelize the computation
    """

    pool = mp.Pool(processes=procs)

    dt = 1  # spacing of my dcds i.e., 1fs

    for i in range(lagRange[0], lagRange[1]):  # lags loop goes from 0...len(timeseries)
        lag = i
        tmax = int(len(dat) - (lag * 1.0 / dt * 1.0))
        t0 = []
        tn = []
        for j in range(tmax):
            t0.append(dat[j])
            tn.append(dat[j + lag])
        if len(t0) <= 1 or len(tn) <= 1: continue
        if norm:
            pool.apply_async(compute_normalized_autocoorelation, args=(t0, tn, lag), callback=log_result)
            # corr = normAutoCoor(t0,tn)
        else:
            pool.apply_async(compute_autocooreration, args=(t0, tn, lag, axis), callback=log_result)
            # corr = AutoCoor(t0,tn,axis) ### along the frozen axis

    pool.close()
    pool.join()


if __name__ == "__main__":

    dt = 1  # 1fs sec saving interval was used to generate MD traj

    procs = 16  # number of procs

    force = np.loadtxt("./x.out")

    # computes the for fO-fD, i.e., the force on the oscillator
    forceDiff = np.array([force[i] - force[i + 1] for i in range(0, len(force) - 1, 2)])

    forceDiff = (forceDiff - np.mean(forceDiff, axis=0)) # mean centering the data

    start = time.time()

    startLag = int(sys.argv[1])
    endLag = int(sys.argv[2])
    lagRange = [startLag, endLag]
    compute_autocoorelation_with_mp(forceDiff, procs, lagRange, norm=False, axis="x")  # logs result in lags_corrs

    end = time.time()
    print ("Autocorr calculation done in " + str((end - start) / 60.0) + " mins")

    # corrs=[ lag_corrs[lag] for lag in range(startLag,endLag)]
    corrs = []
    for lag in range(startLag, endLag):
        if lag_corrs.has_key(lag):
            corrs.append(lag_corrs[lag])

    """Lets plot the computed correlation"""
    plt.plot(range(len(corrs)), corrs, "o")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")

    plt.savefig("autocorr.pdf")

    job = str(startLag)
    pickle.dump(corrs, open("corrs" + job + ".p", "w"))


