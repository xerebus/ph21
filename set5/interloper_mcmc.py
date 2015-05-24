# Ph21 Set 5
# Aritra Biswas

# coin_mcmc.py
# Run MCMC on coin_model.py

import interloper_model
from pymc import MCMC
from pymc.Matplot import plot

M = MCMC(interloper_model)
M.sample(iter = 10000, burn = 0, thin = 1)
print
plot(M)
print "alpha: %f" % M.alpha
print "beta: %f" % M.beta
