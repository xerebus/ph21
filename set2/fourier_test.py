# Ph21 Set 2
# Aritra Biswas

# fourier_test.py
# Demonstration of numpy FFT and inverse FFT with desired normalization

import numpy as np
import matplotlib.pyplot as plotter
import math

# TEST DATA

# parameters
A = 2
C = 3
f = 2
phi = np.pi/2
B = 0.25
L = 50

# test functions
g = lambda t: A * np.cos(f*t + phi) + C
h = lambda t: A * np.exp(-B*(t - (L/2))**2)

# time range
t = np.arange(0, L, 0.25)
g_dat = g(t)
h_dat = h(t)

# FFT TESTING

for dat in [g_dat, h_dat]:

    # plot function
    plotter.plot(t, dat)
    plotter.show()

    # plot 
    Fdat = np.fft.fft(dat)
    Fg_freq = np.fft.fftfreq(t.size, 0.25)
    plotter.plot(Fg_freq, Fdat.real)
    plotter.show()

    FiFdat = np.fft.ifft(Fdat)
    plotter.plot(t, FiFdat.real)
    plotter.show()
