# Ph21 Set 5
# Aritra Biswas

# coin_model.py
# Binomial model for coin tossing

import numpy as np
from pymc import Uniform, Cauchy, Normal, deterministic
from bayes import gen_lighthouse_data

# set true values of parameters
alpha_true = 1.0
beta_true = 1.5
alpha_p_true = 1.6
beta_p_true = 0.7

# set dataset size
nflashes = 100000

# gather observed data from true parameters
x_obs = gen_lighthouse_data(nflashes, alpha_true, beta_true)
x_p_obs = gen_lighthouse_data(nflashes, alpha_p_true, beta_p_true)
x_obs += x_p_obs

# parameter prior: distance along shore
alpha = Uniform("alpha", lower = 0.0, upper = 2.0)
#alpha = Normal("alpha", mu = 1.5, tau = (1 / (0.3)**2))
#alpha = Normal("alpha", mu = 1.5, tau = (1 / (0.1)**2))

# parameter prior: distance from shore
beta = Uniform("beta", lower = 0.0, upper = 2.0)
#beta = Normal("beta", mu = 1.5, tau = (1 / (0.3)**2))
#beta = Normal("beta", mu = 1.5, tau = (1 / (0.1)**2))

# model: flash arrival points
x = Cauchy("x", alpha = alpha, beta = beta, value = x_obs, observed = True)
