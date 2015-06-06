# Ph21 Set 6 (Set 5)
# Aritra Biswas

# pca.py
# General PCA (principal component analysis) on multidimensional data

import numpy as np

def get_pca_matrix(X):
    '''Given X, an m x n 2D array (matrix) of n samples of m measurement
    types, return P, an n x m matrix whose rows are the eigenvectors of
    the covariance matrix of X.'''

    X = [row - np.mean(row) for row in X]
    CX = np.cov(X)
    (eigvals, PT) = np.linalg.eig(CX)
    P = np.transpose(PT)

    return P

def project_pca(X):
    '''Given X, an m x n matrix of n samples of m measurements, find the
    PCA basis and project X onto it, returning Y, an m x n matrix of n
    samples of m measurements in the new basis.'''

    P = get_pca_matrix(X)

    return np.dot(P, X)

def test_linear():
    '''Generate a 2D dataset where y is linearly dependent on x and
    perform PCA on it.'''

    # parameters
    n = 10 # number of samples
    a = 2 # y-intercept
    b = 4 # slope
    sig = 0.05 # noise width - samples will be multiplied by norm(1.0, sig)

    xx = range(n)
    yy = [(a + b*x) * np.random.normal(1.0, sig) for x in xx]

    X = np.array([xx, yy])

    return project_pca(X)

def test_multi():
    '''Generate a 5D dataset of linearly dependent variables
    perform PCA on it.'''

    # parameters
    n = 10 # number of samples
    sig = 0.05 # noise width - samples will be multiplied by norm(1.0, sig)

    tt = range(n)
    xx1 = [(2 + 4*t) * np.random.normal(1.0, sig) for t in tt]
    xx2 = [(-2 + 3*t) * np.random.normal(1.0, sig) for t in tt]
    xx3 = [(3 - 0.5*t) * np.random.normal(1.0, sig) for t in tt]
    xx4 = [(-1 + 3.5*t) * np.random.normal(1.0, sig) for t in tt]
    xx5 = [(1 + 2*t) * np.random.normal(1.0, sig) for t in tt]

    X = np.array([xx1, xx2, xx3, xx4, xx5])

    return project_pca(X)

def get_signal_spreads(Y):
    '''From a PCA-transformed data array, return the spread on each axis.'''

    return [np.std(row) for row in Y]
