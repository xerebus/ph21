# Ph21 Set 2
# Aritra Biswas

# ls_arecibo.py
# Lomb-Scargle periodogram of various signals

import sys
import numpy as np
import scipy.signal as spsignal
import matplotlib.pyplot as plotter
import math

from crts_extract import *

#def read_1d_file(path):
#    '''Read 1-dimensional plaintext data from a file and return a tuple of
#    two numpy arrays: a time array (assuming timesteps of 1) and a 
#    data array. Drop some data to simulate uneven sampling.'''
#
#    dat = np.array([np.float64(line) for line in open(path, "r")])
#    times = np.arange(np.float64(dat.size))
#
#    # remove 10% of values at random
#    indices_to_drop = np.random.choice(np.arange(dat.size),
#        np.int(dat.size / 2))
#    dat = np.delete(dat, indices_to_drop)
#    times = np.delete(times, indices_to_drop)
#    
#    # randomly perturb times
#    for i in range(times.size):
#        times[i] *= np.random.normal(1.0, 0.001)
#
#    return (times, dat)

def get_max_loc(x, y):
    '''Given corresponding x and y arrays of matching size, return
    the x corresponding to the max y.'''

    return x[np.argmax(y)]

def get_ls(times, dat):
    '''Given time and data arrays, return a Lomb-Scargle periodogram:
    a tuple (freq, Fdat) of a frequency array and an intensity array.'''

    freqs = np.fft.fftfreq(int(times[-1] - times[0])) # interval, not size
    Fdat = spsignal.lombscargle(times, dat, 2*np.pi*freqs) # 1/s -> rad/s

    return (freqs, Fdat)

if __name__ == "__main__":

    name = "Her X1"
    rad = "0.1"

    dat_2d = html_parse(cgi_pull_data(name, rad, "web"))
    dat = dat_2d[:,1]
    times = dat_2d[:,5]

    (freqs, Fdat) = get_ls(times, dat)
    f0 = np.abs(get_max_log(freqs, Fdat))

    # plot periodogram
    plotter.plot(freqs, Fdat)
    plotter.show()

    print "Periodogram maximized at: %f" % f0
