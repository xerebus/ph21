# Ph21 Set 2
# Aritra Biswas

# ls_arecibo.py
# Lomb-Scargle periodogram of various signals

import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plotter
import math

def read_file(path):
    '''Read 1-dimensional plaintext data from a file and return a numpy
    array.'''

    return np.array([np.float64(line) for line in open(path, "r")])


def get_max_loc(x, y):
    '''Given corresponding x and y arrays of matching size, return
    the x corresponding to the max y.'''

    return x[np.argmax(y)]

if __name__ == "__main__":

    dat = read_file(sys.argv[1])
