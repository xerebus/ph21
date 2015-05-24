# Ph21 Set 5
# Aritra Biswas

# coin_model.py
# Binomial model for coin tossing

import numpy as np
from pymc import Uniform, Binomial, Normal, deterministic
from bayes import gen_coin_data

# set true values of parameters
pheads_true = 0.38

# set dataset size
ntosses = 1000

# gather observed data from true parameters
(_, nheads_obs) = gen_coin_data(ntosses, pheads_true)

# parameter prior: probability of heads
#pheads = Uniform("pheads", lower = 0.0, upper = 1.0)
#pheads = Normal("pheads", mu = 0.5, tau = (1 / (0.2)**2))
pheads = Normal("pheads", mu = 0.5, tau = (1 / (0.05)**2))

# model: number of heads
nheads = Binomial("nheads", n = ntosses, p = pheads, value = nheads_obs,
    observed = True) 
