# Ph21 Set 2
# Aritra Biswas

# fft_arecibo.py
# FFT of fictitious data from Arecibo radiotelescope to identify 1420 MHz
# signal

import sys
import numpy as np
import matplotlib.pyplot as plotter
import math

def read_file(path):
    '''Read 1-dimensional plaintext data from a file and return a numpy
    array.'''

    return np.array([np.float64(line) for line in open(path, "r")])

def get_fft(dat):
    '''FFT a 1-D array in millisecond timesteps
    and return a tuple: (frequency_array,
    fft_array.'''

    freq = 10**(-3) * np.fft.fftfreq(dat.size)
    Fdat = np.fft.fft(dat).real

    return (freq, Fdat)

def get_max_loc(x, y):
    '''Given corresponding x and y arrays of matching size, return
    the x corresponding to the max y.'''

    return x[np.argmax(y)]

if __name__ == "__main__":

    dat = read_file(sys.argv[1])
    (freq, Fdat) = get_fft(dat)
    f0 = np.abs(get_max_loc(freq, Fdat))
    
    # plot Fourier transform
    plotter.plot(freq, Fdat, label = r"$\tilde g(f)$")
    plotter.xlim(0, plotter.xlim()[1]) # plot from 0 to auto

    # plot Gaussian transforms to find dt
    t = np.arange(dat.size)
    t0 = dat.size / 2 # L/2 to center the Gaussian
    for dt in [t0/2, t0/4, t0/6, t0/8]:
        h = np.exp(-(t - t0)**2 / (dt)**2)
        Fh = np.fft.fft(h)
        plotter.plot(freq + f0, Fh.real,
            label = (r"$\tilde h(f), \Delta t = %i\;\mathrm{ms}$" % dt))

    plotter.legend(prop = {"size" : 16})
    plotter.show()
    
    print "Fourier transform maximized at: %f" % f0
