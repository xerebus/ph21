# Ph21 Set 5
# Aritra Biswas

# coin_model.py
# Binomial model for coin tossing

import numpy as np
from pymc import Uniform, Cauchy, deterministic
from bayes import gen_lighthouse_data

# set true values of parameters
alpha_true = 1.0
beta_true = 1.5

# set dataset size
nflashes = 10000

# gather observed data from true parameters
x_obs = gen_lighthouse_data(nflashes, alpha_true, beta_true)

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
