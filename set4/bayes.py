# Ph21 Set 4
# Aritra Biswas

# bayes.py
# Investigate posterior distributions for parameters of coin tossing
# and lighthouse problems

import numpy as np
import scipy.special as spp
import random
import matplotlib.pyplot as plotter

pi = np.pi

def gen_coin_data(n, H):
    '''Given a probability H in [0, 1] of getting heads, simulate n
    coin tosses. Return (n, h) where h is the number of heads.'''

    assert H >= 0 and H <= 1
    h = 0
    for i in range(0, n):
        if random.random() < H:
            h += 1

    return (n, h)

def calc_coin_logposterior(data, prior):
    '''Using Bayes' Theorem, calculate an (unnormalized) posterior
    probability function. prior must be a single-variable function of the
    parameter H. Returns a function of the parameter H.'''

    log_likelihood = lambda n, h, H: np.log(
        spp.binom(n, h) * H**h * (1 - H)**(n - h)
    )

    (n, h) = data # get data
    log_posterior = lambda H: log_likelihood(n, h, H) + np.log(prior(H))

    return log_posterior

def gen_lighthouse_data(n, alpha, beta):
    '''Generate data points x_k for the lighthouse problem. n is the number
    of pulses fired from the lighthouse, but keep in mind that about
    half of them don't hit the shore.'''

    th = [random.uniform(0, 2*pi) for i in range(0, n)]
    hits_shore = lambda th: th < pi/2 or th > 3*pi/2
    x = [beta * np.tan(th_k) + alpha for th_k in th if hits_shore(th_k)]

    return x

def calc_lighthouse_log1dposterior(data, prior, beta):
    '''Using Bayes' Theorem, calculate an (un-normalized) posterior
    probability function. prior must be a single-variable function of the
    parameter alpha. data should be an array of x_k's from
    gen_lighthouse_data. beta should be known since this is a 1d calculation
    for alpha. Returns a function of the parameter alpha.'''

    # define likelihood function for each x_k
    log_likelihood_k = lambda x_k, alpha, beta: np.log(
        (beta / (2*pi)) / ((x_k - alpha)**2 + beta**2)
    )
    # likelihood function for a whole dataset - product of probabilities
    log_likelihood = lambda x, alpha, beta: (
        np.sum([log_likelihood_k(x_k, alpha, beta) for x_k in x])
    )

    x = data # get array of x_k from gen_lighthouse_data
    log_posterior = lambda alpha: (
        log_likelihood(x, alpha, beta) + np.log(prior(alpha))
    )

    return log_posterior

def calc_lighthouse_log2dposterior(data, prior_alpha, prior_beta):
    '''Using Bayes' Theorem, calculate an (un-normalized) posterior
    probability function. Each prior must be a single-variable function of
    alpha or beta. data should be an array of x_k's from
    gen_lighthouse_data. Returns a function of the parameters alpha and
    beta.'''

    # define likelihood function for each x_k
    log_likelihood_k = lambda x_k, alpha, beta: np.log(
        (beta / (2*pi)) / ((x_k - alpha)**2 + beta**2)
    )
    # likelihood function for a whole dataset - product of probabilities
    log_likelihood = lambda x, alpha, beta: (
        np.sum([log_likelihood_k(x_k, alpha, beta) for x_k in x])
    )

    x = data # get array of x_k from gen_lighthouse_data
    log_posterior = lambda alpha, beta: (
        log_likelihood(x, alpha, beta) + np.log(prior_alpha(alpha)) 
        + np.log(prior_beta(beta))
    )

    return log_posterior

def plot_1dposterior(posterior, p_min, p_max, plot_label = None,
    n_points = 100.0):
    '''Plot the posterior distribution of a single parameter on the range
    (p_min, p_max).'''

    p_grid = np.arange(p_min, p_max, (p_max - p_min) / n_points)
    posterior_grid = [posterior(p) for p in p_grid]
    posterior_grid -= np.max(posterior_grid) # normalize
    plotter.plot(p_grid, posterior_grid, label = plot_label)
    plotter.legend(prop = {"size" : 14})

def plot_2dposterior(posterior, p1_min, p1_max, p2_min, p2_max,
    plot_label = None, n_points = 100.0):
    '''Plot a posterior distribution of 2 parameters as a contour plot.
    p1 is put on the x-axis.'''
    
    p1_grid = np.arange(p1_min, p1_max, (p1_max - p1_min) / n_points)
    p2_grid = np.arange(p2_min, p2_max, (p2_max - p2_min) / n_points)

    posterior_grid = [[posterior(p1, p2) for p1 in p1_grid] for p2 in p2_grid]
    plotter.imshow(posterior_grid,
        extent = [p1_min, p1_max, p2_min, p2_max])
    plotter.title(plot_label)
    plotter.colorbar()

if __name__ == "__main__":

    # COIN TOSSING

    # real parameter value
    H = 0.38

    # priors
    prior_uniform = lambda H: 1
    prior_fair = lambda H: np.exp( -(H - 0.5)**2 / (2 * 0.1**2) )
    prior_veryfair = lambda H: np.exp( -(H - 0.5)**2 / (2 * 0.05**2) )

    for prior in [prior_uniform, prior_fair, prior_veryfair]:
        for n in [2, 5, 10, 250]:
            data = gen_coin_data(n, H)
            log_posterior = calc_coin_logposterior(data, prior)
            plot_1dposterior(log_posterior, 0, 1,
                plot_label = "$H$ = %f, $n$ = %i" % (H, n))
        plotter.show()

    # LIGHTHOUSE, ALPHA

    # real parameter values
    alpha = 1.0
    beta = 1.5

    # priors
    prior_uniform = lambda alpha: 1
    prior_wide = lambda alpha: np.exp( -(H - 1.5)**2 / (2 * 0.3**2) )
    prior_narrow = lambda H: np.exp( -(H - 1.5)**2 / (2 * 0.1**2) )

    for prior in [prior_uniform, prior_wide, prior_narrow]:
        for n in [10, 20, 50, 250]:
            data = gen_lighthouse_data(n, alpha, beta)
            log_posterior = calc_lighthouse_log1dposterior(data, prior, beta)
            plot_1dposterior(log_posterior, 0, 2,
                plot_label = ("$\\alpha$ = %f, $\\beta$ = %f, $n$ = %i"
                % (alpha, beta, n))
            )
        plotter.show()

    # LIGHTHOUSE, ALPHA AND BETA

    # using same parameter values and priors

    for prior in [prior_uniform, prior_wide, prior_narrow]:
        for n in [10, 20, 50, 150]:
            data = gen_lighthouse_data(n, alpha, beta)
            log_posterior = calc_lighthouse_log2dposterior(data, prior, prior)
            plot_2dposterior(log_posterior, 0, 2, 0, 2,
                plot_label = ("$\\alpha$ = %f, $\\beta$ = %f, $n$ = %i"
                % (alpha, beta, n))
            )
            plotter.show()
